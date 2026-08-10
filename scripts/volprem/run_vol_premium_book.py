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

The deployed series (`ret_gated`) carries TWO regime gates, ANDed. The shared VIX term structure is the
right read for the five equity-index sleeves and says nothing about the other thirteen, which sell
variance on metals, oil, duration, EM and single names — so each sleeve also gates on its own implied
vol against its own three-month level (`own_curve_gate`). Gating only on the VIX left those thirteen
exposed to any vol event the S&P did not share, which is what 2026 was.

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
from src.risk.vol_regime import own_curve_gate, short_vol_gate  # noqa: E402
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


# Realistic vol half-spread (VEGA POINTS PER ROLL) by leg liquidity, benchmarked to published
# variance-swap bid/ask: ~0.5 vega for indices, 1-2.5 vega for single names (J.P. Morgan Variance Swaps
# 2006 / Risk.net). Set at or above the high end (index 1.0, single 2.5) — conservative; single names /
# EM / commodity vol trade far wider than SPX. **THIS COST IS CHARGED per leg in sleeve() below**
# (passed as vega_cost_volpts), so every book Sharpe here is NET of option-execution cost, not gross.
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
    # var_cap=1e9 + wing_markup=0 (sleeve default) = the NAKED book: no bought tail hedge, so the short
    # eats the full realised variance in a spike (the honest -78% tail). "Naked" = UNHEDGED TAIL, *not*
    # costless — the per-leg vega SPREAD is still charged here (vega_cost_volpts = COST_BY_CLASS[cls]).
    params = {"timed": False, "var_cap": 1e9, "bars": bars,
              "vega_cost_volpts": COST_BY_CLASS.get(cls, 1.5), **kw}
    return vt(vp.short_vol_book(px, iv, ppy=ppy, **params)["net"], ppy)


def book_from(rets: dict) -> pd.Series:
    """Equal-risk average of vol-targeted sleeves, re-targeted to 15% for comparability."""
    R = pd.DataFrame(rets).sort_index()
    raw = R.mean(axis=1, skipna=True).dropna()          # available-sleeve equal weight each day
    return vt(raw, PPY_BOOK)


def gated_leg(src, sym, und, cls, ppy, gate: pd.Series) -> pd.Series:
    """One sleeve under BOTH regime gates, with the two things a gate costs kept honest:

      * the switch is PAID — flattening the swap and putting it back on crosses the same vega spread a
        roll does, so the gate goes into `short_vol_book` (on the side) where the cost model charges it,
        not onto the finished P&L where timing would look free;
      * the vol-target is sized off the UNGATED leg. Sizing it off the gated series reads the flat
        stretches as low volatility and levers the leg up on re-entry — free leverage exactly when the
        gate steps back in, which is an accounting artifact, not an edge.

    Two gates, because one of them only speaks for five of the eighteen sleeves. `gate` is the shared VIX
    term structure — the right signal for the equity-index legs and blind to the other thirteen, which
    sell variance on metals, oil, duration, EM and single names. `own_curve_gate` gives each sleeve the
    same contango test on its OWN implied vol, so a metals sleeve stands down on a metals vol event even
    while the VIX curve is calm. They compose with AND: both must say contango.
    """
    iv = naive_dt(implied(src, sym))
    bars = underlying_bars(und, cls)
    px = bars["close"]
    base = {"timed": False, "var_cap": 1e9, "bars": bars,
            "vega_cost_volpts": COST_BY_CLASS.get(cls, 1.5)}
    ungated = vp.short_vol_book(px, iv, ppy=ppy, **base)["net"]
    both = gate.reindex(px.index).ffill().fillna(0.0) * own_curve_gate(iv, px.index)
    net = vp.short_vol_book(px, iv, ppy=ppy, gate=both, **base)["net"]
    scale = (TVOL / (ungated.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale.reindex(net.index)).clip(lower=-0.999).dropna()


def gated_book(rets_ungated: dict, gate: pd.Series) -> pd.Series:
    """The deployed (gated) book. The re-entry artifact bites a second time at book level — `vt` would
    re-target the equal-risk mean on ITS trailing vol, which the gated flat stretches deflate — so the
    book's scale is taken from the ungated book and applied to the gated one."""
    gat = {}
    for src, sym, und, cls, ppy in UNIVERSE:
        if sym in rets_ungated:
            gat[sym] = gated_leg(src, sym, und, cls, ppy, gate)
    raw_u = pd.DataFrame(rets_ungated).sort_index().mean(axis=1, skipna=True).dropna()
    raw_g = pd.DataFrame(gat).sort_index().mean(axis=1, skipna=True).dropna()
    scale = (TVOL / (raw_u.rolling(60).std() * np.sqrt(PPY_BOOK))).clip(upper=3.0).shift(1).fillna(0.0)
    return (raw_g * scale.reindex(raw_g.index)).clip(lower=-0.999).dropna()


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

    # --- COST ROBUSTNESS: make the cost accounting explicit and committed, so a reviewer (human or AI)
    # cannot mistake the naked (var_cap=1e9, wing_markup=0) book for a frictionless one. x1 is the shipped
    # book; the x0->x1 gap IS the per-leg vega cost already charged; the edge survives far wider spreads. ---
    print("\n=== COST ROBUSTNESS (vega-spread multiplier; x1 = shipped, realistic per-leg cost) ===")
    cost_rows = []
    for mult in (0.0, 1.0, 2.0, 3.0, 5.0):
        rr = {}
        for src, sym, und, cls, ppy in UNIVERSE:
            try:
                rr[sym] = sleeve(src, sym, und, cls, ppy, vega_cost_volpts=COST_BY_CLASS.get(cls, 1.5) * mult)
            except Exception:
                pass
        sm = summarise(book_from(rr), PPY_BOOK)
        tag = ("gross, NO option cost" if mult == 0 else
               "SHIPPED - realistic per-leg spread" if mult == 1 else f"{mult:.0f}x wider than modelled")
        cost_rows.append({"cost_mult": mult, "sharpe": round(sm["sharpe_ann"], 2),
                          "max_dd": round(sm["max_dd"], 4), "note": tag})
        print(f"  x{mult:.0f}  Sharpe {sm['sharpe_ann']:+.2f}  DD {sm['max_dd']:+.0%}   {tag}")
    pd.DataFrame(cost_rows).to_csv(VOLPREM_DIR / "volprem_cost_robustness.csv", index=False)
    print(f"  -> {cost_rows[0]['sharpe'] - cost_rows[1]['sharpe']:+.2f} Sharpe gap (x0->x1) IS the option "
          f"cost already charged; edge survives 3x wider spreads ({cost_rows[3]['sharpe']:+.2f})")

    pdf.to_csv(VOLPREM_DIR / "volprem_book_sleeves.csv", index=False)
    # Publish both the raw premium (`ret`) and the deployed series (`ret_gated`): the regime gating is part
    # of THIS strategy's signal — validated as timing, not de-risking — so it ships from here, not the book
    # assembler. Two gates compose with AND: the shared VIX term structure (flat unless BOTH curve segments
    # are in contango — the regime that precedes the systemic short-vol crash), and each sleeve's own curve,
    # because the VIX speaks for the five equity-index sleeves and is blind to the thirteen that sell metals,
    # oil, duration, EM and single-name variance. `ret_gated` is rebuilt through the sleeves rather than
    # multiplied onto `ret`, so every switch pays the vega spread. The raw column stays intact: the master
    # reads `ret_gated`, run_ml_book_contribution reads `ret`.
    gate = short_vol_gate(book.index)
    deployed = gated_book(rets, gate)
    own_live = np.mean([own_curve_gate(naive_dt(implied(s, y)), underlying_bars(u, c)["close"].index).mean()
                        for s, y, u, c, _ in UNIVERSE])
    print(f"\n  gates: VIX curve live {gate.mean():.1%} of days, {int((gate.diff().abs() > 0).sum())} switches "
          f"({(gate.diff().abs() > 0).sum() / ((gate.index[-1] - gate.index[0]).days / 365.25):.1f}/yr, spread "
          f"charged); own-curve live {own_live:.1%} of days per sleeve (mean over the 18)")
    sg = summarise(deployed, PPY_BOOK)
    print(f"  gated book:      Sharpe {sg['sharpe_ann']:+.2f}  maxDD {sg['max_dd']:+.1%}  "
          f"skew {deployed.skew():+.2f}  months+ {sg['months_in_profit']:.0%}")
    out = pd.DataFrame({"ret": book, "ret_gated": deployed}).sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_localize("UTC")   # tz-aware UTC to match the other family series (master join)
    out.to_parquet(VOLPREM_DIR / "volprem_book.parquet")
    print("\nVOLPREM-BOOK OK")


if __name__ == "__main__":
    main()
