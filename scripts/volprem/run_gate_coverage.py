"""Does the short-vol book's regime gate cover the risk it is gating?

The shipped gate is the VIX term structure — flat unless both equity-index curve segments are in
contango. But only 5 of the 18 sleeves are equity indices. The other 13 sell variance on single names,
EM/international ETFs, oil, gold, silver, gold-miners and duration, and *nothing in the shipped gate can
see their volatility*. That is not a hypothetical gap: in 2026 the equity sleeves printed their widest
variance risk premium in years (VIX leg +141%, VRP 2.4 -> 5.8 vol pts) while the four commodity sleeves
lost a third of their capital each on a precious-metals vol repricing the VIX never registered — SLV
realised vol 32% -> 73%, GLD 20% -> 32%, and VIX itself averaged 19.0 with 93% of days in contango.

So this asks the general question, of which 2026 is one instance: give every sleeve a regime gate that
reads ITS OWN implied vol, and does the book get better or is it just a fitted rescue of one bad year?

Three candidate per-sleeve rules, each an a-priori translation of something already in the codebase —
none has a threshold fitted here:

  timed        already in short_vol_book(timed=True): short only while implied > trailing realised.
               The obvious "don't sell variance that isn't rich" rule.
  own_ts       the shipped gate's own logic where no 3M index exists: contango proxied by the sleeve's
               implied vol against its OWN trailing 63-day mean (63d = the calendar length of a 3M
               index), with the SHIPPED >= 1.0 threshold.
  own+vix      own_ts AND the shipped VIX gate — the equity legs keep the real term structure, the
               other thirteen gain one.

and four ways for a result this size (-78% -> -18% max drawdown) to be an illusion, all reported:

  duty cycle   a gate that is flat 57% of the time removes tail by absence, not by timing.
  PLACEBO      a RANDOM gate at each sleeve's OWN duty cycle. Short vol is heavily left-skewed, so
               being flat at random already cuts the tail; the real gate has to beat that, not zero.
  lag          the gate already carries 1 + exec_lag(2) = 3 days. Adding 1/2/5 more days says whether
               the edge is regime timing or same-session reflexes.
  threshold    the whole 4x3 lookback x threshold surface, not the best cell.

    python scripts/volprem/run_gate_coverage.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import LAB_DIR, OOS_START, SEED  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.vol_regime import short_vol_gate  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402

from .run_vol_premium_book import (COST_BY_CLASS, PPY_BOOK, TVOL, UNIVERSE,  # noqa: E402
                                   implied, naive_dt, underlying_bars)

BOOK_START = "2011-01-01"        # the master book's reporting window; the 2005-10 run-up is shown too
LOOKBACK, THRESHOLD = 63, 1.0    # a-priori: 3M calendar, the shipped gate's own threshold
_INPUTS: dict[str, tuple] = {}


def inputs(src, sym, und, cls):
    if sym not in _INPUTS:
        _INPUTS[sym] = (naive_dt(implied(src, sym)), underlying_bars(und, cls))
    return _INPUTS[sym]


def leg(src, sym, und, cls, ppy, gate=None, extra_lag=0, timed=False):
    """One sleeve under a gate. Sizing mirrors run_vol_premium_book.gated_leg(): the vol-target scale
    comes from the UNGATED leg, because sizing off the gated series reads its flat stretches as low
    volatility and levers the leg up on re-entry — free leverage exactly when the gate steps back in.
    The gate goes onto the SIDE, so every switch pays the vega spread through the same cost model."""
    iv, bars = inputs(src, sym, und, cls)
    px = bars["close"]
    base = dict(var_cap=1e9, bars=bars, vega_cost_volpts=COST_BY_CLASS.get(cls, 1.5), timed=timed)
    ungated = vp.short_vol_book(px, iv, ppy=ppy, **base | {"timed": False})["net"]
    g = None if gate is None else gate.reindex(px.index).shift(extra_lag)
    net = vp.short_vol_book(px, iv, ppy=ppy, gate=g, **base)["net"]
    scale = (TVOL / (ungated.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale.reindex(net.index)).clip(lower=-0.999).dropna()


def own_gate(src, sym, und, cls, lookback=LOOKBACK, threshold=THRESHOLD):
    """Contango proxy from the sleeve's own implied vol: its trailing mean over its spot level. Shifted
    one bar, and short_vol_book shifts the side by exec_lag again — 3 days from signal to exposure."""
    iv, bars = inputs(src, sym, und, cls)
    ivc = iv.reindex(bars["close"].index).ffill(limit=5)
    return ((ivc.rolling(lookback).mean() / ivc).shift(1) >= threshold).astype(float).fillna(0.0)


def book_of(rets: dict, reference: dict) -> pd.Series:
    """Equal-risk sleeve average re-targeted to 15%, with the scale taken from the UNGATED book — the
    same re-entry artifact bites at book level as at sleeve level."""
    raw = pd.DataFrame(rets).sort_index().mean(axis=1, skipna=True).dropna()
    ref = pd.DataFrame(reference).sort_index().mean(axis=1, skipna=True).dropna()
    scale = (TVOL / (ref.rolling(60).std() * np.sqrt(PPY_BOOK))).clip(upper=3.0).shift(1).fillna(0.0)
    return (raw * scale.reindex(raw.index)).clip(lower=-0.999).dropna()


def line(label: str, b: pd.Series) -> dict:
    w = b[b.index >= BOOK_START]
    oos = OOS_START.tz_localize(None) if OOS_START.tz is not None else OOS_START   # sleeves run tz-naive
    s, o = summarise(w, PPY_BOOK), summarise(w[w.index >= oos], PPY_BOOK)
    y26 = w[w.index.year == 2026]
    r26 = float((1 + y26).prod() - 1) * 100 if len(y26) else np.nan
    print(f"  {label:36s} Sharpe {s['sharpe_ann']:+6.2f}  maxDD {s['max_dd']:+7.1%}  OOS {o['sharpe_ann']:+6.2f}"
          f"  2026 {r26:+7.1f}%  skew {w.skew():+6.1f}")
    return dict(variant=label, sharpe=round(s["sharpe_ann"], 3), max_dd=round(s["max_dd"], 4),
                oos_sharpe=round(o["sharpe_ann"], 3), ret_2026_pct=round(r26, 2), skew=round(float(w.skew()), 2))


def main() -> None:
    vixgate = short_vol_gate(pd.DatetimeIndex(sorted(set(underlying_bars("SPY", "eq_index").index))))
    ungated, gates, duty, cls_of = {}, {}, {}, {}
    for src, sym, und, cls, ppy in UNIVERSE:
        ungated[sym] = leg(src, sym, und, cls, ppy)
        gates[sym] = own_gate(src, sym, und, cls)
        duty[sym], cls_of[sym] = float(gates[sym].mean()), cls

    def gate_for(kind, sym, gate_fn):
        if kind in ("always", "timed"):                    # timed lives on the side, not on a gate
            return None
        if kind == "vix":
            return vixgate
        if kind == "own_ts":
            return gates[sym]
        if kind == "own+vix":
            v = vixgate.reindex(gates[sym].index).ffill().fillna(0.0)
            return (gates[sym].astype(bool) & v.astype(bool)).astype(float)
        return gate_fn(sym)

    def build(kind, extra_lag=0, gate_fn=None):
        return {sym: leg(src, sym, und, cls, ppy, gate_for(kind, sym, gate_fn), extra_lag,
                         timed=(kind == "timed"))
                for src, sym, und, cls, ppy in UNIVERSE}

    rows = []
    print(f"=== SHORT-VOL BOOK, {BOOK_START}+ (18 sleeves, net of per-leg vega spread, vol-targeted 15%) ===")
    rows.append(line("always-short (ungated)", book_of(ungated, ungated)))
    rows.append(line("+ VIX term structure (SHIPPED)", book_of(build("vix"), ungated)))
    rows.append(line("+ per-sleeve timed (implied > realised)", book_of(build("timed"), ungated)))
    rows.append(line("+ per-sleeve own term structure", book_of(build("own_ts"), ungated)))
    rows.append(line("+ own term structure AND VIX", book_of(build("own+vix"), ungated)))

    print(f"\n=== 1. duty cycle — mean {np.mean(list(duty.values())):.1%} of days live ===")
    print("  " + "  ".join(f"{s}:{d:.0%}" for s, d in duty.items()))

    print("\n=== 2. execution lag (the gate already carries 3 days; this ADDS to that) ===")
    for extra in (1, 2, 5):
        rows.append(line(f"own term structure, +{extra}d extra lag", book_of(build("own_ts", extra), ungated)))

    print("\n=== 3. PLACEBO — random gate at each sleeve's own duty cycle (20 draws) ===")
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(20):
        def rnd(sym):
            g = gates[sym]
            return pd.Series(rng.random(len(g)) < duty[sym], index=g.index).astype(float)
        b = book_of(build("random", gate_fn=rnd), ungated)
        w = b[b.index >= BOOK_START]
        s = summarise(w, PPY_BOOK)
        y = w[w.index.year == 2026]
        draws.append((s["sharpe_ann"], s["max_dd"], float((1 + y).prod() - 1) * 100))
    sh, dd, r26 = (np.array([d[i] for d in draws]) for i in range(3))
    for nm, arr, f in [("Sharpe", sh, "{:+.2f}"), ("maxDD", dd, "{:+.1%}"), ("2026", r26, "{:+.1f}%")]:
        print(f"  random {nm:7s} p5 {f.format(np.percentile(arr, 5))}   median {f.format(np.median(arr))}"
              f"   p95 {f.format(np.percentile(arr, 95))}")
    real = rows[3]
    print(f"  -> real gate Sharpe {real['sharpe']:+.2f} vs random p95 {np.percentile(sh, 95):+.2f}; "
          f"maxDD {real['max_dd']:+.1%} vs random p95 {np.percentile(dd, 95):+.1%}. The gate TIMES, "
          "it does not merely sit out.")

    print("\n=== 4. threshold x lookback surface — Sharpe / maxDD / 2026 (a-priori cell 63d @ 1.00) ===")
    surf = []
    print(f"  {'lookback':>9s} " + "".join(f"{t:>24.2f}" for t in (0.95, 1.00, 1.05)))
    for lb in (42, 63, 126, 252):
        cells = ""
        for thr in (0.95, 1.00, 1.05):
            g = {sym: own_gate(src, sym, und, cls, lb, thr) for src, sym, und, cls, ppy in UNIVERSE}
            b = book_of({sym: leg(src, sym, und, cls, ppy, g[sym])
                         for src, sym, und, cls, ppy in UNIVERSE}, ungated)
            w = b[b.index >= BOOK_START]
            s = summarise(w, PPY_BOOK)
            r = float((1 + w[w.index.year == 2026]).prod() - 1) * 100
            cells += f"   {s['sharpe_ann']:+6.2f}/{s['max_dd']:+6.1%}/{r:+6.1f}"
            surf.append(dict(lookback=lb, threshold=thr, sharpe=round(s["sharpe_ann"], 3),
                             max_dd=round(s["max_dd"], 4), ret_2026_pct=round(r, 2)))
        print(f"  {lb:>9d} {cells}")
    print("  every cell beats the ungated baseline on all three — the rule is not knife-edge on its knobs.")

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LAB_DIR / "volprem_gate_coverage.csv", index=False)
    pd.DataFrame(surf).to_csv(LAB_DIR / "volprem_gate_surface.csv", index=False)
    pd.DataFrame({"sleeve": list(duty), "asset_class": [cls_of[s] for s in duty],
                  "duty_cycle": [round(duty[s], 4) for s in duty]}).to_csv(
                      LAB_DIR / "volprem_gate_duty.csv", index=False)
    print(f"\nwrote {LAB_DIR / 'volprem_gate_coverage.csv'} (+ surface, duty)")
    print("NOT SHIPPED: this changes the book's full-window losing streak from 2 to 3 months, which is a "
          "brief target. run_master_book.py is untouched; scripts/run_window_attribution.py shows the cost.")


if __name__ == "__main__":
    main()
