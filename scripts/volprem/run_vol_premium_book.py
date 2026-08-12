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

The deployed series (`ret_gated`) carries TWO regime gates, ANDed, doing two different jobs. The shared
VIX term structure covers all eighteen sleeves as a systemic-stress read — when the curve inverts the
shock is broad and the sleeves fall together whatever they sell, which is why it stands even the metals
legs down. Each sleeve ALSO gates on its own implied vol against its own three-month level
(`own_curve_gate`), which is what catches the idiosyncratic events the VIX cannot see: in 2026 silver's
realised vol went 32% -> 73% out of a calm S&P, and only the own-curve half stood those sleeves down.

    python scripts/volprem/run_vol_premium_book.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import (BOOK_REBALANCE_BPS, CARRY_DIR, REPORTS_DIR, TREND_DIR,  # noqa: E402
                        VOLPREM_DIR, VOLPREM_TERM_HAIRCUT, VOL_TARGET_ANNUAL)
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.cboe import load_cboe_vol  # noqa: E402
from src.data.deribit import load_dvol  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.vol_regime import own_curve_gate, short_vol_gate  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402
from src.sleeves.vol_premium import realized_vol  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

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


def vt(net, ppy, unit_cost=None, scale=None):
    """Vol-target a finished series, and pay for the re-sizing that does.

    `unit_cost` is what moving ONE unit of this position costs on one bar. A variance swap pays the
    same vega spread to be re-sized that it pays to be rolled, so at sleeve level that is 2·K·spread
    (`vega_unit` below); a layer that moves whole finished sleeves pays the book rebalance rate. Left
    None the re-sizing is free, which is what it used to be everywhere and is only right for a series
    nobody trades.

    `scale` lets a caller size off a different series than the one being scaled — the gated legs size
    off their UNGATED history so a flat stretch does not read as low volatility and lever the leg up on
    re-entry.
    """
    scale = vol_target_scale(net, TVOL, ppy) if scale is None else scale.reindex(net.index)
    out = net * scale
    if unit_cost is not None:
        moved = scale.diff().abs().fillna(0.0)
        u = unit_cost.reindex(net.index).fillna(0.0) if hasattr(unit_cost, "reindex") else unit_cost
        out = out - moved * u
    return out.clip(lower=-0.999).dropna()


def vega_unit(frame, vega_pts):
    """Cost of moving one unit of this variance swap on one bar — the roll's own vega spread, 2·K·s.

    Same formula `vol_premium.short_vol_book` charges when the strike rolls or the side flips, applied
    to the size change the vol target makes. Zero on a bar the leg is flat: there is nothing to re-size.
    """
    return 2.0 * frame["K"].clip(lower=1e-6) * (vega_pts / 100.0) * frame["side"].abs()


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

# WHY THE VIX GATE COVERS ALL EIGHTEEN, INCLUDING GOLD, SILVER, GOLD-MINERS, OIL AND DURATION.
# The obvious objection is that the VIX is the volatility of the S&P 500 and says nothing about the
# volatility of gold — which is true, and is exactly why `own_curve_gate` exists. But the shared gate is
# not being asked to forecast a metals sleeve's own vol. It is a SYSTEMIC-STRESS read: when the VIX curve
# inverts, the shock is broad and the sleeves fall together whatever they sell. That is measurable, and
# it was measured (`make gate-ablation`, the reach ladder): on the leg's ten worst sessions the shared
# gate stands the whole leg down in 7 of 10 and holds the loss to -14.2%, against 3 of 10 and -16.9% when
# its reach stops at the equity sleeves, and 0 of 10 and -19.1% with no shared gate at all. Narrowing the
# reach buys ~2pp of full-window CAGR and gives back exactly the cover this leg exists to need — its
# systemic tail is the book's stated central risk — so the reach stays wide. The own-curve gate is what
# catches the idiosyncratic events the VIX cannot see (2026 metals); the two do different jobs.


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
              "vega_cost_volpts": COST_BY_CLASS.get(cls, 1.5),
              "term_haircut": VOLPREM_TERM_HAIRCUT, **kw}
    f = vp.short_vol_book(px, iv, ppy=ppy, **params)
    return vt(f["net"], ppy, vega_unit(f, params["vega_cost_volpts"]))


def book_from(rets: dict) -> pd.Series:
    """Equal-risk average of vol-targeted sleeves, re-targeted to 15% for comparability.

    The re-target is a trade at the layer that moves whole sleeves, so it pays the same blended rate
    the master book's assembly pays for exactly that (`BOOK_REBALANCE_BPS`) rather than nothing."""
    R = pd.DataFrame(rets).sort_index()
    raw = R.mean(axis=1, skipna=True).dropna()          # available-sleeve equal weight each day
    return vt(raw, PPY_BOOK, BOOK_REBALANCE_BPS / 1e4)


def gated_leg(src, sym, und, cls, ppy, gate: pd.Series | None, own_curve: bool = True,
              vega_mult: float = 1.0) -> pd.Series:
    """One sleeve under BOTH regime gates, with the two things a gate costs kept honest:

      * the switch is PAID — flattening the swap and putting it back on crosses the same vega spread a
        roll does, so the gate goes into `short_vol_book` (on the side) where the cost model charges it,
        not onto the finished P&L where timing would look free;
      * the vol-target is sized off the UNGATED leg. Sizing it off the gated series reads the flat
        stretches as low volatility and levers the leg up on re-entry — free leverage exactly when the
        gate steps back in, which is an accounting artifact, not an edge.

    Two gates, doing two different jobs. `gate` is the shared VIX term structure the caller hands in —
    a systemic-stress read that stands the whole leg down when the curve inverts, whatever each sleeve
    sells. `own_curve_gate` gives every sleeve the same contango test on its OWN implied vol, which is
    what catches the idiosyncratic events the VIX cannot see (a metals vol repricing out of a calm S&P).
    They compose with AND: both must say contango. This function stays mechanical — it applies the gate
    it is given and does not decide reach, so a study can hand it any coverage rule (`gate=None` for
    none at all) and be scored on that rule rather than on the shipped one.

    `own_curve=False` drops the per-sleeve half. The shipped leg keeps both — the flag exists so a
    candidate study can attribute a lift to the VIX rule it names, instead of scoring every VIX variant
    with the per-sleeve gate silently attached and crediting the difference to the label.
    """
    iv = naive_dt(implied(src, sym))
    bars = underlying_bars(und, cls)
    px = bars["close"]
    base = {"timed": False, "var_cap": 1e9, "bars": bars,
            "vega_cost_volpts": COST_BY_CLASS.get(cls, 1.5) * vega_mult,
            "term_haircut": VOLPREM_TERM_HAIRCUT}
    ungated = vp.short_vol_book(px, iv, ppy=ppy, **base)["net"]
    # `gate=None` means this sleeve has no shared regime signal at all — built on the sleeve's own index
    # so there is no fill question, rather than a wide constant series that would read as flat wherever
    # it fails to cover the bars.
    both = pd.Series(1.0, index=px.index) if gate is None else gate.reindex(px.index).ffill().fillna(0.0)
    if own_curve:
        both = both * own_curve_gate(iv, px.index)
    f = vp.short_vol_book(px, iv, ppy=ppy, gate=both, **base)
    return vt(f["net"], ppy, vega_unit(f, base["vega_cost_volpts"]),
              scale=vol_target_scale(ungated, TVOL, ppy))


def gated_book(rets_ungated: dict, gate: pd.Series, own_curve: bool = True,
               vega_mult: float = 1.0) -> pd.Series:
    """The deployed (gated) book: every sleeve under both gates, the shared one included (see the note
    above UNIVERSE's cost table for why the VIX reaches the non-equity sleeves too).

    The re-entry artifact bites a second time at book level — `vt` would re-target the equal-risk mean
    on ITS trailing vol, which the gated flat stretches deflate — so the book's scale is taken from the
    ungated book and applied to the gated one."""
    gat = {}
    for src, sym, und, cls, ppy in UNIVERSE:
        if sym in rets_ungated:
            gat[sym] = gated_leg(src, sym, und, cls, ppy, gate, own_curve=own_curve,
                                 vega_mult=vega_mult)
    raw_u = pd.DataFrame(rets_ungated).sort_index().mean(axis=1, skipna=True).dropna()
    raw_g = pd.DataFrame(gat).sort_index().mean(axis=1, skipna=True).dropna()
    return vt(raw_g, PPY_BOOK, BOOK_REBALANCE_BPS / 1e4,
              scale=vol_target_scale(raw_u, TVOL, PPY_BOOK))


# WHAT THE DEPLOYED LEG SELLS. The research book stays all eighteen sleeves — the breadth study, the
# placebo and the cost ladder are all measured on it — while the leg the master book holds sells
# variance on WHOLE CLASSES: index, international and rates. Not a list of tickers, which is not
# something §2 would accept as "the rule by which an asset enters your universe", but a sentence a desk
# can defend: this book sells index, international and rates variance and leaves single names and
# commodities alone. Single names carry the widest vega spreads in this universe, and the commodity
# sleeves are the ones whose implied vol reprices out of a calm S&P, which is exactly the event the
# shared VIX gate cannot see.
SHIPPED_CLASSES = ("eq_index", "intl", "rates")
SHIPPED_SLEEVES = [sym for _, sym, _, cls, _ in UNIVERSE if cls in SHIPPED_CLASSES]


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    rets, plac, per = {}, {}, []
    for src, sym, und, cls, ppy in UNIVERSE:
        try:
            r = sleeve(src, sym, und, cls, ppy)
            rets[sym] = r
            plac[sym] = sleeve(src, sym, und, cls, ppy, fair=True)
            s = summarise(r, ppy)
            # `vt` floors the day at -99.9%, which is the difference between a bad day and a wiped-out
            # sleeve — and short variance CAN lose more than its notional. Eight of these eighteen have
            # such a day in the NAKED research series (VXN reaches -259.6%), so the count is published
            # rather than left inside the clip. The deployed gated series has none.
            per.append({"vol_index": sym, "underlying": und, "class": cls, "sharpe": s["sharpe_ann"],
                        "max_dd": s["max_dd"], "skew": float(r.skew()), "start": r.index.min().date(),
                        "days_at_ruin": int((r <= -0.999).sum())})
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

    # --- COST ROBUSTNESS, for BOTH constructions. x1 is what ships; the x0->x1 gap IS the per-leg vega
    # cost already charged. The research (ungated, all-sleeve) book is the one that used to be measured
    # here, and the §9 table then quoted its number for the leg the book actually holds — a different
    # construction with a different cost profile, because the gate flattens and re-enters the swap and
    # every one of those switches crosses the spread. Both are measured now and labelled by name. ---
    gate = short_vol_gate(book.index)
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
        sg = summarise(gated_book({k: v for k, v in rr.items() if k in SHIPPED_SLEEVES}, gate,
                                  vega_mult=mult), PPY_BOOK)
        tag = ("gross, NO option cost" if mult == 0 else
               "SHIPPED - realistic per-leg spread" if mult == 1 else f"{mult:.0f}x wider than modelled")
        cost_rows.append({"cost_mult": mult, "sharpe": round(sm["sharpe_ann"], 2),
                          "max_dd": round(sm["max_dd"], 4),
                          "sharpe_deployed": round(sg["sharpe_ann"], 2),
                          "max_dd_deployed": round(sg["max_dd"], 4), "note": tag})
        print(f"  x{mult:.0f}  research book Sharpe {sm['sharpe_ann']:+.2f} (DD {sm['max_dd']:+.0%})   "
              f"DEPLOYED leg Sharpe {sg['sharpe_ann']:+.2f} (DD {sg['max_dd']:+.0%})   {tag}")
    pd.DataFrame(cost_rows).to_csv(VOLPREM_DIR / "volprem_cost_robustness.csv", index=False)
    print(f"  -> {cost_rows[0]['sharpe'] - cost_rows[1]['sharpe']:+.2f} / "
          f"{cost_rows[0]['sharpe_deployed'] - cost_rows[1]['sharpe_deployed']:+.2f} Sharpe gap (x0->x1) IS "
          f"the option cost already charged, research / deployed")

    pdf.to_csv(VOLPREM_DIR / "volprem_book_sleeves.csv", index=False)
    # Publish both the raw premium (`ret`) and the deployed series (`ret_gated`): the regime gating is part
    # of THIS strategy's signal — validated as timing, not de-risking — so it ships from here, not the book
    # assembler. Two gates compose with AND: the shared VIX term structure (flat unless BOTH curve
    # segments are in contango — the regime that precedes the systemic short-vol crash) over all
    # eighteen sleeves, and each sleeve's own curve for the events the VIX cannot see. `ret_gated` is
    # rebuilt through the sleeves rather than multiplied onto `ret`, so every switch pays the vega spread.
    # The raw column stays intact: the master reads `ret_gated`, run_ml_book_contribution reads `ret`.
    deployed = gated_book({k: v for k, v in rets.items() if k in SHIPPED_SLEEVES}, gate)
    own_live = np.mean([own_curve_gate(naive_dt(implied(s, y)), underlying_bars(u, c)["close"].index).mean()
                        for s, y, u, c, _ in UNIVERSE])
    print(f"\n  gates: VIX curve live {gate.mean():.1%} of days, {int((gate.diff().abs() > 0).sum())} switches "
          f"({(gate.diff().abs() > 0).sum() / ((gate.index[-1] - gate.index[0]).days / 365.25):.1f}/yr, spread "
          f"charged), applied to all {len(UNIVERSE)} sleeves; "
          f"own-curve live {own_live:.1%} of days per sleeve (mean over the 18)")
    sg = summarise(deployed, PPY_BOOK)
    print(f"  gated book:      Sharpe {sg['sharpe_ann']:+.2f}  maxDD {sg['max_dd']:+.1%}  "
          f"skew {deployed.skew():+.2f}  months+ {sg['months_in_profit']:.0%}")
    out = pd.DataFrame({"ret": book, "ret_gated": deployed}).sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_localize("UTC")   # tz-aware UTC to match the other family series (master join)
    out.to_parquet(VOLPREM_DIR / "volprem_book.parquet")
    print("\nVOLPREM-BOOK OK")


if __name__ == "__main__":
    main()
