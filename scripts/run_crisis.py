"""Crisis-alpha sleeve — multi-asset time-series-momentum (managed futures) across five liquid classes.

The master book is built from short-gamma risk-premium harvesters (short-vol / carry / momentum /
breakout) that all lose *together* in risk-off crashes — so the book's worst months and multi-month
losing streaks are correlated crashes (2018-Q4, 2019-Q3, COVID-2020, the 2021-22 crypto unwind) with
no offset. This sleeve is the missing long-gamma / crisis-alpha leg: a diversified time-series-momentum
book that goes long uptrends and SHORT downtrends, so it turns positive in sustained sell-offs and is
flat-to-positive otherwise (documented crisis alpha — Hurst-Ooi-Pedersen "A Century of Evidence on Trend
Following"; Moskowitz-Ooi-Pedersen TSMOM).

Breadth is what makes it a *general* crisis hedge: each asset class catches a different crash — equities
+ single-stock leaders the 2018-Q4 / COVID equity sell-offs, bonds + gold the flight-to-safety, crypto
the 2021-22 crypto winter, commodities the 2022 inflation shock, FX the risk-off carry unwind. Pooling
them (equal risk) lifts standalone Sharpe well above the equity-only construction and covers ~6 of the
book's 7 worst windows instead of ~3.

Construction per class: TSMOM sign-blend over three lookback horizons (fast 10/20/40, medium 20/40/63,
slow 40/63/120 days), each horizon per-asset vol-targeted, the three tranches averaged (timeframe
diversification raises the hit rate), then the class vol-targeted to 15%. The five class books are
combined at equal risk and the result vol-targeted to 15%. Signals fill t+2 (never at the bar that generated them) and the per-asset vol scaler is lagged;
~2 bps turnover cost.

    python scripts/run_crisis.py   ->  <BOOK_DIR>/crisis_sleeve.parquet  (+ crash-window diagnostics)
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
from src.config import BOOK_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from src.sleeves.trend_lab import tsmom_panel  # noqa: E402

RAW = ROOT / "data/raw/equity_td"
# five liquid classes — each catches a different crash; top-N kept concentrated (leaders trend cleanest)
EQUITY = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLF", "XLK"]     # indices + intl + financials/tech
COMMOD = ["GLD", "SLV", "USO", "DBC", "DBA", "XLE"]                   # gold/silver/oil/broad/agri/energy
BOND = ["TLT", "IEF", "SHY", "HYG", "LQD"]                            # treasuries + credit (flight-to-safety)
FX = ["AUD-JPY", "EUR-JPY", "GBP-JPY", "USD-JPY", "USD-CHF", "EUR-CHF", "AUD-USD", "NZD-USD",
      "EUR-USD", "GBP-USD", "USD-CAD", "USD-MXN", "USD-ZAR", "USD-NOK", "USD-SEK"]
CRYPTO_TOP = 20
COST_BPS = 2.0        # per unit of turnover on liquid ETFs/FX — the shipped charge
LOOKBACKS = [(10, 20, 40), (20, 40, 63), (40, 63, 120)]   # fast/medium/slow tranches
STOCK_PPY, CRYPTO_PPY = 252, 365


def _etf(t):
    p = RAW / f"{t}_1d.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["close"]
    s.index = pd.to_datetime(s.index)
    return (s.tz_localize(None) if s.index.tz is not None else s).sort_index()


def _fx(pair):
    g = sorted(glob.glob(str(ROOT / f"data/raw/twelvedata/{pair}_1day_2005*.parquet")))
    if not g:
        return None
    s = pd.read_parquet(g[0])["close"]
    s.index = pd.to_datetime(s.index)
    return (s.tz_localize(None) if s.index.tz is not None else s).sort_index()


def _crypto_panel(topn):
    """Top-N crypto by market-cap order, spliced spot(pre-2020)+perp(2020+) via the trend loader."""
    try:
        import scripts.trend.trend_common as T
    except Exception:
        return pd.DataFrame()
    names = (ROOT / "reports/crypto_universe.txt").read_text().strip().split(",")[:topn]
    out = {}
    for sym in names:
        px = T.load_crypto_long(sym, "1d")
        if px is not None and len(px) > 200:
            c = px["close"]
            c.index = pd.to_datetime(c.index)
            out[sym] = c.tz_localize(None) if c.index.tz is not None else c
    return pd.DataFrame(out).sort_index()


def _panel(syms, loader):
    return pd.DataFrame({s: loader(s) for s in syms if loader(s) is not None}).sort_index()


def _vol_target(x, ppy, target=0.15, lb=60):
    lev = vol_target_scale(x, target, ppy, lookback=lb)
    return (x * lev).dropna()


CCY = {"USD": "US", "EUR": "EZ", "JPY": "JP", "GBP": "GB", "CHF": "CH", "AUD": "AU",
       "NZD": "NZ", "CAD": "CA", "MXN": "MX", "ZAR": "ZA", "NOK": "NO", "SEK": "SE"}


def _rate(ccy, index):
    """3-month interbank rate (% p.a.) carried onto daily bars — a rate is a level, not a flow."""
    p = ROOT / "data/raw/rates" / f"IR3TIB01{CCY[ccy]}M156N.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["val"]
    i = pd.DatetimeIndex(s.index)
    s.index = i.tz_convert("UTC").tz_localize(None) if i.tz is not None else i
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def _fx_carry(px):
    """The interest differential a spot cross funds: long BASE-QUOTE earns r_base and pays r_quote.

    A spot FX position is a funded position and the differential is the whole reason its forward
    differs from its spot — trading fifteen crosses on price alone books the depreciation without
    paying for it. Mean |differential| across this set is 1.94% a year and reaches 4.75% on USD-MXN.
    """
    out = {}
    for pair in px.columns:
        base, quote = pair.split("-")
        rb, rq = _rate(base, px.index), _rate(quote, px.index)
        if rb is not None and rq is not None:
            out[pair] = rb - rq
    return pd.DataFrame(out).reindex_like(px) if out else None


def _etf_yield(px):
    """Distributions as an annualised % rate. `load_equity_daily` is split-adjusted, not total-return,
    so every one of these ETFs is traded on a series that drops its own payouts — 2.3% a year across
    this universe, 6.2% on HYG. It is the same omission as FX carry, on the other instrument."""
    out = {}
    for s in px.columns:
        f = ROOT / "data/raw/twelvedata" / f"{s}_div.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        col = next((c for c in ("amount", "dividend", "value") if c in d.columns), None)
        if col is None:
            continue
        a = pd.to_numeric(d[col], errors="coerce").dropna()
        i = pd.DatetimeIndex(a.index)
        a.index = (i.tz_convert("UTC").tz_localize(None) if i.tz is not None else i).normalize()
        out[s] = (a.groupby(level=0).sum().reindex(px.index).fillna(0.0)
                  / px[s].shift(1)).fillna(0.0) * 100.0 * STOCK_PPY
    return pd.DataFrame(out).reindex_like(px).fillna(0.0) if out else None


def _tsmom(close, lookbacks, ppy, cost_bps=COST_BPS, carry_pa=None):
    """`trend_lab.tsmom_panel`, vol-targeted. The construction used to live here as a copy of
    the one in `run_gmacro.py`, and both copies filled at the signal bar's own close and sized
    on an unlagged volatility. One implementation now, with both fixed."""
    return _vol_target(tsmom_panel(close, lookbacks, ppy, cost_bps, carry_pa=carry_pa), ppy)


def _class_book(close, ppy, cost_bps=COST_BPS, carry_pa=None):
    """Average the fast/medium/slow TSMOM tranches (timeframe diversification), vol-target to 15%."""
    if close.empty or close.shape[1] == 0:
        return None
    tranches = [_vol_target(_tsmom(close, lb, ppy, cost_bps, carry_pa), ppy) for lb in LOOKBACKS]
    return _vol_target(pd.concat(tranches, axis=1).mean(axis=1).dropna(), ppy)


def build_crisis(cost_bps=COST_BPS) -> pd.Series:
    eq, cm, bd = _panel(EQUITY, _etf), _panel(COMMOD, _etf), _panel(BOND, _etf)
    fx = _panel(FX, _fx)
    books = {
        "equity": _class_book(eq, STOCK_PPY, cost_bps, _etf_yield(eq)),
        "commod": _class_book(cm, STOCK_PPY, cost_bps, _etf_yield(cm)),
        "bond": _class_book(bd, STOCK_PPY, cost_bps, _etf_yield(bd)),
        "fx": _class_book(fx, STOCK_PPY, cost_bps, _fx_carry(fx)),
        # crypto perps: funding is already charged inside the trend loader that builds this panel
        "crypto": _class_book(_crypto_panel(CRYPTO_TOP), CRYPTO_PPY, cost_bps),
    }
    live = {k: v for k, v in books.items() if v is not None and len(v) > 100}
    df = pd.DataFrame(live).sort_index()
    crisis = _vol_target(df.mean(axis=1, skipna=True).dropna(), CRYPTO_PPY)   # equal risk over live classes
    return crisis.rename("ret")


def main():
    crisis = build_crisis()
    crisis.to_frame().to_parquet(BOOK_DIR / "crisis_sleeve.parquet")
    s = summarise(crisis, CRYPTO_PPY)
    print(f"crisis-alpha sleeve (5-class TSMOM): Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  "
          f"months+ {s['months_in_profit']:.0%}  {crisis.index.min().date()}..{crisis.index.max().date()}")
    print("returns in the book's worst windows (the hedge value — want POSITIVE):")
    for lab, a, b in [("2018-Q4", "2018-10", "2018-12"), ("2019-Q3", "2019-07", "2019-09"),
                      ("COVID 2020Q1", "2020-02", "2020-03"), ("crypto 2021-22", "2021-12", "2022-02"),
                      ("2022 bear", "2022-04", "2022-06"), ("2024-08", "2024-08", "2024-08"),
                      ("2026-06", "2026-06", "2026-06")]:
        print(f"  {lab:14s}: {(1 + crisis.loc[a:b]).prod() - 1:+.1%}")
    print(f"RUN CRISIS OK -> {BOOK_DIR}/crisis_sleeve.parquet")


if __name__ == "__main__":
    main()
