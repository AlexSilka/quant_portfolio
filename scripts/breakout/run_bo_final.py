"""The final breakout book + full robustness (Task A §7-10, §12). Combines the honest pieces:

  - PIT top-10 x 1d : raw Donchian-55 chandelier (too few trades to meta-label — the non-ML
                   trend-capture leg)
  - PIT top-10 x 4h+1h : the SAME primary gated by a LightGBM meta-label confidence model fitted
                   under an expanding WALK-FORWARD (threshold 0.55), which filters false breakouts

Everything is on the POINT-IN-TIME liquid universe — the ten most liquid perps by trailing 30-day
median dollar volume, lagged, over every name on disk including the delisted ones. It used to be a
frozen `CORE10` typed from the 2026 mega-caps, and the gate used to be purged k-fold, which trains
each fold on its whole complement. Both are removed.
Reports the
portfolio, Monte-Carlo, per-year and per-quarter metrics, the isolated crisis windows, the strict
2024-07 held-out block, three cost levels + break-even, and the sleeve correlation matrix. Persists
the series the report/figures consume.

    python scripts/breakout/run_bo_final.py
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_ml import OOS_START, models, precompute, proba_cache  # noqa: E402
from src.backtest.engine import positions_from_events, vol_target  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

THR = 0.55
CRISES = {"covid_2020H1": ("2020-01-01", "2020-06-30"),
          "bull_2021": ("2021-01-01", "2021-12-31"),
          "bear_2022": ("2022-01-01", "2022-12-31"),
          "chop_2023_25": ("2023-01-01", "2025-12-31")}


def daily_ret_cost(sym, px, pos, tf, fund):
    """Vol-target the signal, then fill it across venues: long on spot, short on perps (§12).

    The signal, universe and sizing are unchanged — only the fill moves, so the difference against
    the all-perp book is the funding bill and the extra spot commission, nothing else.
    """
    posv = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bt = bo.backtest_split(sym, tf, px, posv, fund)
    ret = (1 + bt["net_ret"]).resample("D").prod() - 1
    cost = bt["cost"].resample("D").sum()
    return ret.rename("ret"), cost.rename("cost")


def build_final():
    """Return {sleeve: (ret, cost)} for the combined book: 1d raw + 4h/1h ML-gated.

    Two things this used to do that a desk could not have done, both now removed, and the cost of
    removing each is measured in the arm this builder scores against:

      * it traded `CORE10` — the 2026 mega-cap list, typed once and applied from 2020-01. The
        universe is now point-in-time: the ten most liquid perps by trailing 30-day median dollar
        volume, lagged, over every name on disk including the delisted ones. Worth more to the old
        Sharpe than the ML gate was (1.12 -> 0.69 on the universe alone).
      * it gated with `purged_kfold`, whose folds are contiguous in time but whose TRAINING set is
        the whole complement — a 2021 trade filtered by a model fitted on 2022-2026. The gate is now
        an expanding walk-forward that only ever sees resolved trades (1.12 -> 0.88 on the gate
        alone; 1.12 -> 0.52 with both, and OOS +0.20 -> -0.01).

    What is left is the honest leg. It is not a good one, and the report says so.
    """
    out = {}
    universe = bo.pit_universe(10)
    names = sorted(universe.columns[universe.any()])
    # 1d raw chandelier leg — no ML (too few Donchian-55 trades per sleeve to meta-label)
    for sym in names:
        px = bo.load_crypto(sym, "1d")
        if px is None:
            continue
        side = bl.donchian_side(px["close"], px["high"], px["low"], 55)
        side = side.where(bo.pit_mask(sym, side.index, universe), 0.0)
        pos = bl.hold_atr_trailing(px["close"], px["high"], px["low"], side, 3.0, 14)
        if (pos != 0).sum() < 30:                  # a name that never held long enough to be a sleeve
            continue
        out[f"{sym}_1d"] = daily_ret_cost(sym, px, pos, "1d", bo.safe_funding(sym))
    # 4h + 1h walk-forward-gated leg
    sleeves = precompute(symbols=names, universe=universe)
    pc = proba_cache(sleeves, models()["lightgbm"], weighted=False, walk_forward=True)
    for key, s in sleeves.items():
        p = pc[key]
        if not len(p):                             # the gate never had enough resolved history here
            continue
        kept = p.index[p.values >= THR]
        pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], kept)
        out[key] = daily_ret_cost(key.rsplit("_", 1)[0], s["px"], pos, s["tf"], s["fund"])
    return out


def main():
    book = build_final()
    rets = pd.DataFrame({k: v[0] for k, v in book.items()}).sort_index()
    costs = pd.DataFrame({k: v[1] for k, v in book.items()}).reindex(index=rets.index, columns=rets.columns).fillna(0.0)

    def equal_weight(R):
        """Equal weight over the sleeves that have STARTED — a sleeve whose perp has not listed yet is
        not in the denominator. `fillna(0).mean()` divided by the full column count on every bar, so the
        early years were the same book scaled down by however many sleeves were still to come, which is
        a leg diluted by its own future. A started sleeve that stops printing is held at zero, the same
        rule `run_master_book.hold_started` applies one layer up."""
        started = R.notna().cummax()
        return R.where(R.notna(), 0.0).where(started).mean(axis=1)
    port = equal_weight(rets)
    s = summarise(port, 365)
    mc = bootstrap_sharpe(port, 365, 2000, bo.SEED)
    print(f"=== FINAL BREAKOUT BOOK (PIT top-10; 1d raw + 4h/1h walk-forward-gated; {rets.shape[1]} sleeves) ===")
    print(f"Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"total {s['total_return']:+.0%}  MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} "
          f"P50 {mc.get('sharpe_p50', float('nan')):+.2f} P95 {mc.get('sharpe_p95', float('nan')):+.2f}]")

    # per-year + per-quarter
    def sh(g):
        g = g.dropna()
        return round(float(np.sqrt(365) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    per_year = {int(y): sh(g) for y, g in port.groupby(port.index.year)}
    per_q = {str(q): sh(g) for q, g in port.groupby(port.index.to_period("Q"))}
    print(f"\nper-year Sharpe: {per_year}")

    # isolated crisis/regime windows
    print("\nisolated regime windows (Sharpe / maxDD / total):")
    regimes = {}
    for name, (a, b) in CRISES.items():
        g = port.loc[a:b]
        ss = summarise(g, 365)
        regimes[name] = {"sharpe": ss["sharpe_ann"], "max_dd": ss["max_dd"], "total": ss["total_return"]}
        print(f"    {name:14s}: Sh {ss['sharpe_ann']:+.2f}  DD {ss['max_dd']:+.1%}  tot {ss['total_return']:+.0%}")

    # strict held-out OOS block
    is_, oos = port[port.index < OOS_START], port[port.index >= OOS_START]
    print(f"\nstrict OOS split @ {OOS_START.date()}: IS Sharpe {summarise(is_,365)['sharpe_ann']:+.2f}  "
          f"OOS Sharpe {summarise(oos,365)['sharpe_ann']:+.2f}")

    # cost sensitivity + break-even (§9): net_m = ret - (m-1)*cost (cost already charged once at 1x)
    def port_at(m):
        return equal_weight(rets - (m - 1.0) * costs.where(rets.notna()))
    levels = []
    for m, lab in [(1.0, "1x base"), (2.0, "2x base"), (3.0, "3x base")]:
        pm = port_at(m); sm = summarise(pm, 365)
        levels.append({"label": lab, "sharpe": sm["sharpe_ann"], "max_dd": sm["max_dd"]})
    breakeven = None
    for m in np.linspace(1.0, 30.0, 291):
        if (1 + port_at(m)).prod() - 1 <= 0:
            breakeven = float(m); break
    print("\ncost sensitivity: " + "  ".join(f"{l['label']} Sh{l['sharpe']:+.2f}" for l in levels)
          + (f"  | break-even {breakeven:.1f}x base cost" if breakeven else "  | break-even >30x"))

    # correlation + deflated Sharpe of the best sleeve at the trial count
    # On a point-in-time universe most sleeve pairs never overlap in time, so their pairwise
    # correlation is undefined rather than zero — take the mean over the pairs that DO overlap
    # instead of letting a single NaN turn the reported figure into "nan".
    corr = rets.corr()
    iu = np.triu_indices_from(corr, k=1)
    best_key = max(book, key=lambda k: summarise(book[k][0].dropna(), 365)["sharpe_ann"])
    b = book[best_key][0].dropna()
    n_trials = 635 + 405 + 20 * 6           # sweep + book + ML variants — the honest trial count
    dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0, n_trials, 0.25 / 365)
    print(f"\nsleeve correlation: mean {np.nanmean(corr.values[iu]):+.2f}  max {np.nanmax(corr.values[iu]):+.2f}"
          f"  ({np.isfinite(corr.values[iu]).sum()} of {len(iu[0])} pairs overlap)")
    print(f"best sleeve ({best_key}) deflated Sharpe @ N={n_trials}: {dsr:.2f}")

    rets.to_parquet(bo.BREAKOUT / "bo_final_sleeve_returns.parquet")
    costs.to_parquet(bo.BREAKOUT / "bo_final_costs.parquet")
    port.rename("ret").to_frame().to_parquet(bo.BREAKOUT / "bo_final_portfolio.parquet")
    (bo.BREAKOUT / "bo_final_summary.json").write_text(json.dumps({
        "portfolio": s, "mc": mc, "per_year": per_year, "per_quarter": per_q, "regimes": regimes,
        "oos_split": {"is": summarise(is_, 365), "oos": summarise(oos, 365)},
        "cost_levels": levels, "breakeven_mult": breakeven,
        "corr_mean": float(np.nanmean(corr.values[iu])), "corr_max": float(np.nanmax(corr.values[iu])),
        "corr_pairs_overlapping": int(np.isfinite(corr.values[iu]).sum()),
        "best_sleeve": best_key, "best_sleeve_dsr": dsr, "n_sleeves": int(rets.shape[1]),
    }, indent=2, default=float))
    print("\nBO FINAL OK")


if __name__ == "__main__":
    main()
