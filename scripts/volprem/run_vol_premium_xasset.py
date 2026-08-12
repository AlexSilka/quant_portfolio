"""VRP edge map across asset classes — is the short-vol premium crypto-only, or does it live in
equities and FX too? The strategy needs an implied-vol index (not price), so the universe is gated
by free vol indices: crypto DVOL (BTC/ETH), equity VIX/VXN (SPX/NDX), FX EVZ (EUR/USD). No free
per-name equity IV or altcoin IV, so those cells are genuinely data-blocked, not skipped.

Each cell: always-short uncapped variance book vs a fair-strike placebo (struck at realised vol, no
premium), vol-targeted 15%, net of vega costs, t+2. Answers "where is this best run".

    python scripts/volprem/run_vol_premium_xasset.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import SEED, VOLPREM_DIR, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.cboe import load_cboe_vol  # noqa: E402
from src.data.deribit import load_dvol  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402
from src.sleeves.vol_premium import realized_vol  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

SEED, TVOL = SEED, VOL_TARGET_ANNUAL


def vt(net: pd.Series, ppy: float) -> pd.Series:
    scale = vol_target_scale(net, TVOL, ppy)
    return (net * scale).clip(lower=-0.999).dropna()


def crypto_close(sym):
    return load_klines(sym, "1d", "2021-01", "2026-08", market="um")["close"]


def naive_dt(s):
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.Series(s.to_numpy(), index=idx.normalize()).groupby(level=0).last()


# (class, label, implied series loader, underlying close loader, ppy)
CELLS = [
    ("crypto", "BTC (DVOL)", lambda: load_dvol("BTC", "2021-01", "2026-08")["close"], lambda: crypto_close("BTCUSDT"), 365),
    ("crypto", "ETH (DVOL)", lambda: load_dvol("ETH", "2021-01", "2026-08")["close"], lambda: crypto_close("ETHUSDT"), 365),
    ("equity", "SPX (VIX->SPY)", lambda: load_cboe_vol("VIX", "2005-01-01"), lambda: load_equity_daily("SPY", start="2005-01-01")["close"], 252),
    ("equity", "NDX (VXN->QQQ)", lambda: load_cboe_vol("VXN", "2009-01-01"), lambda: load_equity_daily("QQQ", start="2009-01-01")["close"], 252),
    ("fx", "EURUSD (EVZ, ->2025-03)", lambda: load_cboe_vol("EVZ"), lambda: load_equity_daily("EURUSD=X", start="2009-01-01")["close"], 252),
]


def evaluate(implied, close, ppy):
    implied, close = naive_dt(implied), naive_dt(close)
    real = vt(vp.short_vol_book(close, implied, timed=False, var_cap=1e9, ppy=ppy)["net"], ppy)
    fair_iv = (realized_vol(close, ppy=ppy) * 100.0).reindex(close.index).ffill()
    plac = vt(vp.short_vol_book(close, fair_iv, timed=False, var_cap=1e9, ppy=ppy)["net"], ppy)
    s = summarise(real, ppy)
    return {"sharpe": s["sharpe_ann"], "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
            "skew": float(real.skew()), "placebo": summarise(plac, ppy)["sharpe_ann"],
            "start": real.index.min().date(), "n": s["n_obs"]}


def main():
    rows = []
    for cls, label, imp_fn, und_fn, ppy in CELLS:
        try:
            r = evaluate(imp_fn(), und_fn(), ppy)
            r.update({"class": cls, "cell": label})
            rows.append(r)
            print(f"  {cls:7s} {label:16s} Sharpe {r['sharpe']:+.2f}  DD {r['max_dd']:+.1%}  "
                  f"skew {r['skew']:+.1f}  placebo {r['placebo']:+.2f}  ({r['start']}, n={r['n']})")
        except Exception as e:
            print(f"  {cls:7s} {label:16s} DATA-BLOCKED: {e}")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(VOLPREM_DIR / "volprem_xasset.csv", index=False)
        print("\n=== VRP edge map by asset class (always-short, vol-targeted 15%, net, t+2) ===")
        print(df[["class", "cell", "sharpe", "placebo", "max_dd", "skew", "months_in_profit", "start"]]
              .round(2).to_string(index=False))
    print("\nXASSET OK")


if __name__ == "__main__":
    main()
