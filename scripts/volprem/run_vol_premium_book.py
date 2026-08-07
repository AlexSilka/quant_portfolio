"""Diversified short-vol BOOK — the deployable form of the VRP sleeve.

Running short-vol on one asset is fragile: one vol spike (skew −11, DD −59% on SPX alone) can ruin
it. A real vol-selling book spreads equal risk across every underlying that has a free implied-vol
index, so idiosyncratic vol spikes diversify — only a *systemic* vol event hits every leg at once.

Universe (18 sleeves) = Cboe vol indices whose underlyings have clean OHLC: VIX/VXN/RVX/VXD/VXEFA
(equity indices), VXAPL/VXAZN/VXGOG/VXGS/VXIBM (single names), VXEEM/VXEWZ/VXFXI (international),
OVX/GVZ/VXSLV/VXGDX (commodities), VXTLT (rates). Crypto (DVOL), FX (EVZ), and energy-sector (VXXLE)
are excluded on frozen ex-ante rules — crypto's 30%-intraday-range days are unhedgeable for a
short-vol delta-hedge, the free EURUSD OHLC carries corrupt prints, and Cboe discontinued VXXLE in
2022-02 — NOT on backtested Sharpe (which would be universe overfitting). Adding more free vol indices
lifts headline Sharpe but leaves the DD/skew tail unchanged: equity/commodity vol is systemic, every
leg spikes together in the crash that sets the drawdown, so breadth cannot fix a systemic-tail book.
The paid leg is OHLC realised variance (intraday path + gap), not close-to-close. Each sleeve is
always-short, uncapped, vol-targeted 15%, net of per-leg vega costs, t+2; the book is their equal-risk
average re-targeted to 15%. Annualised at 252 trading days.

    python scripts/volprem/run_vol_premium_book.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, REPORTS_DIR, TREND_DIR, VOLPREM_DIR, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.cboe import load_cboe_vol  # noqa: E402
from src.data.deribit import load_dvol  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402
from src.sleeves.vol_premium import realized_vol  # noqa: E402

TVOL, PPY_BOOK = VOL_TARGET_ANNUAL, 252

# (vol source, vol symbol, underlying, asset class, ppy) — 18 underlyings with a free implied-vol index.
# Excluded on a frozen ex-ante rule, NOT on backtested Sharpe (which would be universe overfitting):
#   - crypto (BTC/ETH): short-vol is structurally unhedgeable there — 30%-intraday-range days a
#     delta-hedge cannot follow, so the honest OHLC leg makes it a losing trade (BTC −0.41, ETH −0.86).
#   - FX (EVZ→EUR/USD): the free EURUSD=X OHLC has corrupt prints (a −13% "daily" move with a 1% H/L
#     range in 2008) and Cboe discontinued EVZ in 2025-03 — a data-quality exclusion, not a return call.
#   - energy sector (VXXLE→XLE): Cboe discontinued it 2022-02, so it can't run live — same availability
#     rule as EVZ (its 2011-22 standalone Sharpe +4.9 would tempt inclusion; excluded on the frozen rule).
# Weak-but-clean single names (AMZN/IBM ~0) are KEPT: no structural reason to drop them = would be overfit.
# VXGDX (gold-miners) is the one live free index found beyond the original 17 — clean OHLC (GDX), passes
# the same a-priori rule, so it is IN; it lifts Sharpe ~0.1 but leaves the tail flat (systemic vol).
UNIVERSE = [
    ("cboe", "VIX", "SPY", "eq_index", 252), ("cboe", "VXN", "QQQ", "eq_index", 252),
    ("cboe", "RVX", "IWM", "eq_index", 252), ("cboe", "VXD", "DIA", "eq_index", 252),
    ("cboe", "VXEFA", "EFA", "eq_index", 252),
    ("cboe", "VXAPL", "AAPL", "single", 252), ("cboe", "VXAZN", "AMZN", "single", 252),
    ("cboe", "VXGOG", "GOOGL", "single", 252), ("cboe", "VXGS", "GS", "single", 252),
    ("cboe", "VXIBM", "IBM", "single", 252),
    ("cboe", "VXEEM", "EEM", "intl", 252), ("cboe", "VXEWZ", "EWZ", "intl", 252),
    ("cboe", "VXFXI", "FXI", "intl", 252),
    ("cboe", "OVX", "USO", "commodity", 252), ("cboe", "GVZ", "GLD", "commodity", 252),
    ("cboe", "VXSLV", "SLV", "commodity", 252), ("cboe", "VXGDX", "GDX", "commodity", 252),
    ("cboe", "VXTLT", "TLT", "rates", 252),
]


def vt(net, ppy):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).clip(lower=-0.999).dropna()


def naive_dt(s):
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.Series(s.to_numpy(), index=idx.normalize()).groupby(level=0).last()


def implied(src, sym):
    return load_dvol(sym, "2021-01", "2026-08")["close"] if src == "deribit" else load_cboe_vol(sym)


# realistic vol half-spread (points/roll) by leg liquidity — single names/alts/EM trade far wider than SPX/BTC
COST_BY_CLASS = {"crypto": 1.0, "eq_index": 1.0, "single": 2.5, "intl": 2.0,
                 "commodity": 2.0, "rates": 1.5, "fx": 1.5}


def naive_df(df):
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def underlying_bars(sym, cls):
    b = (load_klines(sym, "1d", "2021-01", "2026-08", market="um") if cls == "crypto"
         else load_equity_daily(sym, start="2005-01-01"))
    return naive_df(b[["open", "high", "low", "close"]])


def sleeve(src, sym, und, cls, ppy, fair=False, **kw):
    iv = naive_dt(implied(src, sym))
    bars = underlying_bars(und, cls)                          # OHLC -> realistic paid leg (path + gap)
    px = bars["close"]
    if fair:
        iv = (realized_vol(px, ppy=ppy) * 100.0).reindex(px.index).ffill()
    params = {"timed": False, "var_cap": 1e9, "bars": bars,
              "vega_cost_volpts": COST_BY_CLASS.get(cls, 1.5), **kw}
    return vt(vp.short_vol_book(px, iv, ppy=ppy, **params)["net"], ppy)


def book_from(rets: dict) -> pd.Series:
    """Equal-risk average of vol-targeted sleeves, re-targeted to 15% for comparability."""
    R = pd.DataFrame(rets).sort_index()
    raw = R.mean(axis=1, skipna=True).dropna()          # available-sleeve equal weight each day
    return vt(raw, PPY_BOOK)


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    rets, plac, per = {}, {}, []
    for src, sym, und, cls, ppy in UNIVERSE:
        try:
            r = sleeve(src, sym, und, cls, ppy)
            rets[sym] = r
            plac[sym] = sleeve(src, sym, und, cls, ppy, fair=True)
            s = summarise(r, ppy)
            per.append({"vol_index": sym, "underlying": und, "class": cls, "sharpe": s["sharpe_ann"],
                        "max_dd": s["max_dd"], "skew": float(r.skew()), "start": r.index.min().date()})
            print(f"  {sym:6s} {und:9s} {cls:10s} Sharpe {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.0%}  skew {r.skew():+.1f}")
        except Exception as e:
            print(f"  {sym:6s} {und:9s} {cls:10s} SKIP: {str(e)[:70]}")
    pdf = pd.DataFrame(per)

    # --- the diversified book vs the average single sleeve (diversification benefit) ---
    book = book_from(rets)
    placebo_book = book_from(plac)
    sb = summarise(book, PPY_BOOK)
    avg = pdf[["sharpe", "max_dd", "skew"]].mean()
    corr = pd.DataFrame(rets).corr()
    mean_corr = corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()

    print("\n=== DIVERSIFIED SHORT-VOL BOOK (18 sleeves, equal-risk, vol-targeted 15%, net, t+2) ===")
    print(f"  book:            Sharpe {sb['sharpe_ann']:+.2f}  maxDD {sb['max_dd']:+.1%}  "
          f"skew {book.skew():+.2f}  months+ {sb['months_in_profit']:.0%}  ({book.index.min().date()}..{book.index.max().date()})")
    print(f"  avg SINGLE sleeve: Sharpe {avg['sharpe']:+.2f}  maxDD {avg['max_dd']:+.1%}  skew {avg['skew']:+.2f}")
    print(f"  -> diversification: Sharpe {sb['sharpe_ann']-avg['sharpe']:+.2f}, DD {sb['max_dd']-avg['max_dd']:+.1%}, "
          f"skew {book.skew()-avg['skew']:+.1f}; mean pairwise sleeve corr {mean_corr:+.2f}")
    print(f"  placebo book (fair strike, no premium): Sharpe {summarise(placebo_book, PPY_BOOK)['sharpe_ann']:+.2f}")

    # --- systemic vs idiosyncratic: equity-only vs all-asset breadth ---
    eq = {k: rets[k] for k, c in zip(pdf.vol_index, pdf["class"]) if c in ("eq_index", "single", "intl")}
    eq_book = book_from(eq)
    seq = summarise(eq_book, PPY_BOOK)
    print(f"\n  equity-only book ({len(eq)} sleeves):  Sharpe {seq['sharpe_ann']:+.2f}  maxDD {seq['max_dd']:+.1%}  skew {eq_book.skew():+.2f}")
    print(f"  all-asset book   ({len(rets)} sleeves):  Sharpe {sb['sharpe_ann']:+.2f}  maxDD {sb['max_dd']:+.1%}  skew {book.skew():+.2f}")
    print("  (cross-asset breadth softens the tail vs equity-only; systemic vol events still hit all equity legs)")

    # --- per-year + portfolio value-add vs momentum/carry ---
    py = book.groupby(book.index.year).apply(lambda x: summarise(x, PPY_BOOK)["sharpe_ann"])
    print("\n  per-year book Sharpe: " + "  ".join(f"{y}:{s:+.1f}" for y, s in py.items()))

    ext = {"VRP_book": book}
    if (TREND_DIR / "trend_block_returns.parquet").exists():
        m = pd.read_parquet(TREND_DIR / "trend_block_returns.parquet")
        ext["momentum"] = naive_dt(m["ret"] if "ret" in m else m.iloc[:, 0])
    for cp in (CARRY_DIR / "carry_refined.parquet", CARRY_DIR / "carry_headline.parquet"):
        if Path(cp).exists():
            c = pd.read_parquet(cp)
            ext["carry"] = naive_dt(c["ret"] if "ret" in c else c.select_dtypes("number").iloc[:, 0])
            break
    E = pd.DataFrame(ext).dropna()
    if {"momentum", "carry"} <= set(E.columns):
        E = E.apply(lambda s: vt(s, PPY_BOOK)).dropna()
        base = E[["momentum", "carry"]].mean(axis=1)
        best = max((0.0, 0.1, 0.15, 0.2, 0.3),
                   key=lambda w: summarise((1 - w) * base + w * E["VRP_book"], PPY_BOOK)["sharpe_ann"])
        s0, sw = summarise(base, PPY_BOOK), summarise((1 - best) * base + best * E["VRP_book"], PPY_BOOK)
        print(f"  corr VRP-book to momentum {E['VRP_book'].corr(E['momentum']):+.2f}, to carry {E['VRP_book'].corr(E['carry']):+.2f}")
        print(f"  momentum+carry Sharpe {s0['sharpe_ann']:+.2f} (DD {s0['max_dd']:+.1%}) -> +VRP-book @ w={best:.2f}: "
              f"Sharpe {sw['sharpe_ann']:+.2f} (DD {sw['max_dd']:+.1%})")
        # mandate is on PORTFOLIO drawdown (15%): the -50% standalone book enters small; find the fit
        print("  --- mandate fit: total-portfolio DD vs VRP weight (limit 15%) ---")
        for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            sm = summarise((1 - w) * base + w * E["VRP_book"], PPY_BOOK)
            flag = "" if sm["max_dd"] > -0.15 else "  <-- breaches 15%"
            print(f"    w_VRP={w:.1f}  Sharpe {sm['sharpe_ann']:+.2f}  portfolio DD {sm['max_dd']:+.1%}{flag}")

    # --- per-family §8 risk tool: does a drawdown-responsive de-gross ladder cap the book's own DD? ---
    def degross(ret, trig):
        eq = (1 + ret).cumprod()
        dd = (eq / eq.cummax() - 1.0).to_numpy()
        sc = np.ones(len(ret))
        for thr, keep in trig:
            sc = np.where(dd <= -thr, keep, sc)      # deeper trigger overrides (listed shallow->deep)
        return ret * pd.Series(sc, index=ret.index).shift(1).fillna(1.0)
    print("\n=== standalone book under a drawdown de-gross ladder (the wrong tool for short vol?) ===")
    print(f"  no ladder:            Sharpe {sb['sharpe_ann']:+.2f}  DD {sb['max_dd']:+.1%}")
    for name, trig in [("-15/-25/-35", [(0.15, 0.5), (0.25, 0.25), (0.35, 0.0)]),
                       ("-8/-12/-15",  [(0.08, 0.5), (0.12, 0.25), (0.15, 0.0)])]:
        g = degross(book, trig)
        s = summarise(g, PPY_BOOK)
        print(f"  ladder {name}: Sharpe {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.1%}  "
              f"(caps DD but de-risks into the vol-mean-reversion recovery)")

    pdf.to_csv(VOLPREM_DIR / "volprem_book_sleeves.csv", index=False)
    out = book.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize("UTC")   # tz-aware UTC to match the other family series (master join)
    out.to_frame("ret").to_parquet(VOLPREM_DIR / "volprem_book.parquet")
    print("\nVOLPREM-BOOK OK")


if __name__ == "__main__":
    main()
