"""The final breakout book + full robustness (Task A §7-10, §12). Combines the honest pieces:

  - core-10 x 1d : raw Donchian-55 chandelier (too few trades to meta-label — kept as the non-ML
                   trend-capture leg)
  - core-10 x 4h+1h : the SAME primary gated by the LightGBM meta-label confidence model
                   (uniqueness-weighted, threshold 0.55), which filters false breakouts

Everything is on the FROZEN core-10 universe (no per-sleeve survivor selection). Reports the
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
from scripts.breakout.run_bo_ml import CORE10, OOS_START, models, precompute, proba_cache  # noqa: E402
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
    """Return {sleeve: (ret, cost)} for the combined book: 1d raw + 4h/1h ML-gated."""
    out = {}
    # 1d raw chandelier leg
    for sym in CORE10:
        px = bo.load_crypto(sym, "1d")
        if px is None:
            continue
        side = bl.donchian_side(px["close"], px["high"], px["low"], 55)
        pos = bl.hold_atr_trailing(px["close"], px["high"], px["low"], side, 3.0, 14)
        out[f"{sym}_1d"] = daily_ret_cost(sym, px, pos, "1d", bo.safe_funding(sym))
    # 4h + 1h ML-gated leg (LightGBM + uniqueness weights)
    sleeves = precompute()
    pc = proba_cache(sleeves, models()["lightgbm"], weighted=True)
    for key, s in sleeves.items():
        kept = pc[key].index[pc[key].values >= THR]
        pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], kept)
        out[key] = daily_ret_cost(key.rsplit("_", 1)[0], s["px"], pos, s["tf"], s["fund"])
    return out


def main():
    book = build_final()
    rets = pd.DataFrame({k: v[0] for k, v in book.items()}).sort_index()
    costs = pd.DataFrame({k: v[1] for k, v in book.items()}).reindex(index=rets.index, columns=rets.columns).fillna(0.0)
    port = rets.fillna(0.0).mean(axis=1)
    s = summarise(port, 365)
    mc = bootstrap_sharpe(port, 365, 2000, bo.SEED)
    print(f"=== FINAL BREAKOUT BOOK (frozen core-10; 1d raw + 4h/1h ML-gated; {rets.shape[1]} sleeves) ===")
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
        return (rets.fillna(0.0) - (m - 1.0) * costs).mean(axis=1)
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
    corr = rets.corr()
    iu = np.triu_indices_from(corr, k=1)
    best_key = max(book, key=lambda k: summarise(book[k][0].dropna(), 365)["sharpe_ann"])
    b = book[best_key][0].dropna()
    n_trials = 635 + 405 + 20 * 6           # sweep + book + ML variants — the honest trial count
    dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0, n_trials, 0.25 / 365)
    print(f"\nsleeve correlation: mean {corr.values[iu].mean():+.2f}  max {corr.values[iu].max():+.2f}")
    print(f"best sleeve ({best_key}) deflated Sharpe @ N={n_trials}: {dsr:.2f}")

    rets.to_parquet(bo.BREAKOUT / "bo_final_sleeve_returns.parquet")
    costs.to_parquet(bo.BREAKOUT / "bo_final_costs.parquet")
    port.rename("ret").to_frame().to_parquet(bo.BREAKOUT / "bo_final_portfolio.parquet")
    (bo.BREAKOUT / "bo_final_summary.json").write_text(json.dumps({
        "portfolio": s, "mc": mc, "per_year": per_year, "per_quarter": per_q, "regimes": regimes,
        "oos_split": {"is": summarise(is_, 365), "oos": summarise(oos, 365)},
        "cost_levels": levels, "breakeven_mult": breakeven,
        "corr_mean": float(corr.values[iu].mean()), "corr_max": float(corr.values[iu].max()),
        "best_sleeve": best_key, "best_sleeve_dsr": dsr, "n_sleeves": int(rets.shape[1]),
    }, indent=2, default=float))
    print("\nBO FINAL OK")


if __name__ == "__main__":
    main()
