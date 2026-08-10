"""Honest walk-forward for the breakout book (Task A §10). The in-sample 1.5 Sharpe look-aheads its
survivor selection; here, at every rebalance date the book holds ONLY breakout sleeves that were
robust on data strictly BEFORE that date, so the stitched series is genuinely out-of-sample.

Two independent walk-forwards:
  (1) SLEEVE-SELECTION WF — pick sleeves by trailing Sharpe on past data only; run under
      anchored/rolling x {annual, semiannual, quarterly} refit cadences to prove the OOS result
      does not depend on the window/cadence choice ("show results do not depend on that choice").
  (2) PARAMETER WF — on each train block pick the best breakout config from an a-priori grid, apply
      it OOS on the next block, stitch. This pays the honest cost of choosing the construction,
      unlike peak-picking the config on the full sample.

    python scripts/breakout/run_bo_walkforward.py [config_id]
"""
import sys

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_sweep import build_pos, SLOW_CFGS  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

PPY = 365
SEL_SHARPE, MIN_OBS = 0.5, 252   # hold a sleeve if trailing Sharpe clears the bar on >= 1y of data
CFG_ID = sys.argv[1] if len(sys.argv) > 1 else "d55_atr3"


def walk_forward(rets, dates, window_years):
    """At each date pick sleeves robust on strictly-prior data, hold equal-risk to the next date."""
    port, picks = [], []
    for i in range(len(dates) - 1):
        T, Tn = dates[i], dates[i + 1]
        lo = T - pd.DateOffset(years=window_years) if window_years else rets.index[0]
        win = rets.loc[lo:T]
        win = win[win.index < T]
        sh = np.sqrt(PPY) * win.mean() / win.std(ddof=1)
        keep = sh.index[(sh > SEL_SHARPE) & (win.count() >= MIN_OBS)]
        held = rets.loc[T:Tn, keep]
        held = held[held.index < Tn]
        port.append(held.mean(axis=1) if len(keep) else pd.Series(0.0, index=held.index))
        picks.append(len(keep))
    return pd.concat(port).sort_index().dropna(), picks


def selection_wf():
    rets = pd.read_parquet(bo.BREAKOUT / f"bo_all_returns_{CFG_ID}.parquet")
    tz = rets.index.tz
    print(f"[1] SLEEVE-SELECTION WF  ({CFG_ID}: {rets.shape[1]} candidate sleeves, "
          f"{rets.index.min().date()}..{rets.index.max().date()})")
    results = {}
    for wlab, wy in [("anchored", None), ("rolling-2y", 2)]:
        for clab, freq in [("annual", "YS"), ("semiannual", "6MS"), ("quarterly", "QS")]:
            dates = pd.date_range("2021-07-01", "2026-07-01", freq=freq, tz=tz)
            wf, picks = walk_forward(rets, dates, wy)
            s = summarise(wf, PPY)
            results[(wlab, clab)] = (s, wf)
            print(f"    {wlab:10s} {clab:11s}: OOS Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  "
                  f"months+ {s['months_in_profit']:.0%}  avg sleeves {np.mean(picks):.0f}")
    shs = [v[0]["sharpe_ann"] for v in results.values()]
    span = max(shs) - min(shs)
    print(f"  OOS Sharpe across the 6 (window x cadence) configs: {min(shs):+.2f}..{max(shs):+.2f}  "
          + ("-> robust to the choice (§10)" if span < 0.6 else f"-> sensitive (span {span:.2f})"))
    # primary series (anchored, annual) + its Monte Carlo
    s0, wf0 = results[("anchored", "annual")]
    mc = bootstrap_sharpe(wf0, PPY, 1000, bo.SEED)
    wf0.rename("ret").to_frame().to_parquet(bo.BREAKOUT / f"bo_walk_forward_{CFG_ID}.parquet")
    print(f"  PRIMARY (anchored, annual) OOS: Sharpe {s0['sharpe_ann']:+.2f}  "
          f"MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]  "
          f"total {s0['total_return']:+.0%}")
    return results


def parameter_wf():
    """WF over the CONSTRUCTION: pick the best config on each train block, apply OOS on the next."""
    print("\n[2] PARAMETER WF  (best-of-grid on train -> apply OOS; BTC/ETH/SOL/AVAX 1d+4h book)")
    grid = [c for c in dict(SLOW_CFGS)]                    # the 9 a-priori constructions
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT", "ADAUSDT"]
    # per (sym,tf,config) daily returns, once
    daily = {}
    for tf in ["1d", "4h"]:
        for sym in syms:
            px = bo.load_crypto(sym, tf)
            if px is None:
                continue
            fund, adv = bo.safe_funding(sym), px["quote_volume"].rolling(20).median().shift(1)
            for cid in grid:
                pos = build_pos(px, tf, dict(SLOW_CFGS)[cid])
                if pos is None or pos.abs().sum() == 0:
                    continue
                _, ret = bo.evaluate(px["close"], pos, bo.CRYPTO_TF[tf], bo.CC, fund=fund,
                                     adv=adv, ppy_daily=365, with_mc=False)
                daily[(sym, tf, cid)] = ret
    # book return of a config = equal-risk mean across (sym,tf) for that config
    def book_ret(cid, lo, hi):
        cols = [daily[k] for k in daily if k[2] == cid]
        if not cols:
            return pd.Series(dtype=float)
        m = pd.concat(cols, axis=1).loc[lo:hi]
        return m.mean(axis=1)
    idx = pd.concat(list(daily.values()), axis=1).sort_index().index
    bounds = pd.date_range("2021-01-01", "2026-07-01", freq="YS", tz=idx.tz)
    oos, chosen = [], []
    for k in range(len(bounds) - 1):
        tr_hi, te0, te1 = bounds[k], bounds[k], bounds[k + 1]
        best = max(grid, key=lambda c: summarise(book_ret(c, idx.min(), tr_hi).dropna(), PPY)["sharpe_ann"])
        seg = book_ret(best, te0, te1)
        seg = seg[(seg.index >= te0) & (seg.index < te1)]
        oos.append(seg); chosen.append((te0.year, best))
    oos = pd.concat(oos).dropna()
    s = summarise(oos, PPY)
    fs = summarise(book_ret("d55_atr3", idx.min(), idx.max()).dropna(), PPY)  # full-sample peak of default
    print("    chosen config per OOS year: " + ", ".join(f"{y}:{c}" for y, c in chosen))
    print(f"    WALK-FORWARD config-selected OOS Sharpe: {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}")
    print(f"    (reference: fixed d55_atr3 full-sample Sharpe {fs['sharpe_ann']:+.2f} on the same 6-name book)")


def main():
    selection_wf()
    parameter_wf()
    print("\nBO WALK-FORWARD OK")


if __name__ == "__main__":
    main()
