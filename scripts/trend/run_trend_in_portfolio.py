"""Integrate the trend block into the master multi-strategy portfolio.

Builds the recommended trend block (point-in-time crypto @ 1d+4h + index ETFs and point-in-time
single names @ 1d, EMA long-biased, equal-risk) as a clean published return stream, then combines it with the OTHER
strategy families at risk parity — READ-ONLY on their series, exactly the convention of
`run_master_book.py` — and reports the master book with this trend block as the trend leg. Writes only
new `reports/trend/*` artifacts; never touches other families' files.

    python scripts/trend/run_trend_in_portfolio.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import scripts.trend.run_trend_pit_universe as P  # noqa: E402
import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import sh  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

PPY = 365
R = T.bo.REPORTS
# the universe rule (and the hindsight arm it is scored against) lives in run_trend_pit_universe
BLOCK_TFS = P.BLOCK_TFS

# the OTHER families' honest published headlines (same as run_master_book.FAMILIES, minus trend)
OTHERS = [
    ("carry", "carry_refined.parquet", "ret"),
    ("volprem", "volprem_returns.parquet", "VRP_baseline_alwaysshort"),
    ("xs_momentum", "all_returns.parquet", "CRYPTO50_1d_cross_sectional"),
    ("breakout", "bo_combined_portfolio.parquet", "ret"),
]


def build_trend_block() -> pd.Series:
    """Point-in-time crypto (1d+4h) + index ETFs & point-in-time single names (1d), EMA long-only.

    The crypto universe is chosen by TRAILING dollar volume at each bar, not by a hard-coded list of
    today's majors. A fixed CORE10 is picked with hindsight, and hindsight is the one bias this project
    corrects everywhere else — the x-sect leg's headline finding is that a curated list scores +1.06
    against +0.70 honest, and the carry leg deliberately ships the weaker survivorship-free build. Trend
    was the last exception. Measured cost of removing it (`run_trend_pit_universe.py`): leg Sharpe
    +1.31 -> +1.13, book 3.72 -> 3.67 on the selection window, still 5/5 on both windows, and the worst
    month actually improves. 78 distinct names are ever in the honest top-10; today's CORE10 hold only
    ~63% of member-days, so the two universes are materially different."""
    spec = {"entry": "ema", "direction": "long_only", "exit": "reversal"}
    cols = {}
    for tf in BLOCK_TFS:
        rets, vol = P.pool(tf)          # not `R` — that name is the reports Path at module scope
        mem = P.pit_members(vol, P.TOP_N, P.LOOKBACK_D)
        for sym in rets.columns:
            r = rets[sym].where(mem[sym].reindex(rets.index).fillna(False))
            if r.notna().sum() > 60 and r.std(ddof=1) > 0:
                cols[f"{sym}_{tf}"] = r
    cols |= P.equity_legs(pit=True)      # index ETFs (a-priori) + a point-in-time top-7 by liquidity
    block = pd.DataFrame(cols).mean(axis=1).dropna().rename("ret")
    block.to_frame().to_parquet(R / "trend" / "trend_block_returns.parquet")
    return block


def load(file, col, label):
    p = R / file
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = df[col] if col in df.columns else df.iloc[:, 0]
    return s.dropna().rename(label)


def rescale(net, target=0.15):
    scale = (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return net * scale


def describe(s):
    ss = summarise(s, PPY)
    mc = bootstrap_sharpe(s, PPY, 2000, T.SEED)
    return {"sharpe": round(ss["sharpe_ann"], 3), "max_dd": round(ss["max_dd"], 4),
            "months_in_profit": round(ss["months_in_profit"], 3), "total_return": round(ss["total_return"], 3),
            "mc_p5": mc.get("sharpe_p5"), "mc_p50": mc.get("sharpe_p50")}


def per_year(s):
    return {int(y): sh(g) for y, g in s.groupby(s.index.year)}


def main():
    print("=== Integrate the improved trend block into the master multi-strategy portfolio ===\n")
    block = build_trend_block()
    print(f"trend block built (point-in-time crypto 1d+4h + equity 1d): standalone Sharpe "
          f"{summarise(block, PPY)['sharpe_ann']:+.2f}, {block.index.min().date()}..{block.index.max().date()}")
    print("  -> wrote reports/trend/trend_block_returns.parquet\n")

    others = {lab: load(f, c, lab) for lab, f, c in OTHERS}
    others = {k: v for k, v in others.items() if v is not None}

    # risk-parity master with the trend block as the trend leg, on the common window
    legs_new = {"trend": rescale(block), **{k: rescale(v) for k, v in others.items()}}
    df_new = pd.DataFrame(legs_new).dropna().sort_index()
    print(f"families: {list(df_new.columns)}\ncommon window: {df_new.index.min().date()}..{df_new.index.max().date()} "
          f"({len(df_new)} days)\n")

    master_new = df_new.mean(axis=1)
    mn = describe(master_new)
    print("=== MASTER BOOK (risk parity across families) ===")
    print(f"  trend leg standalone Sharpe {summarise(block,PPY)['sharpe_ann']:+.2f}  ->  "
          f"master Sharpe {mn['sharpe']:+.2f}  maxDD {mn['max_dd']:+.1%}  MC-P5 {mn['mc_p5']:+.2f}")

    # trend's decorrelation + marginal contribution (add families by standalone Sharpe)
    corr = df_new.corr()
    tr_corr = {c: round(float(corr.loc["trend", c]), 2) for c in corr.columns if c != "trend"}
    solo = {c: summarise(df_new[c], PPY)["sharpe_ann"] for c in df_new.columns}
    order = sorted(df_new.columns, key=lambda c: -solo[c])
    marg = [{"n": k, "added": order[k-1], "sharpe": round(summarise(df_new[order[:k]].mean(axis=1), PPY)["sharpe_ann"], 3)}
            for k in range(1, len(order)+1)]
    without_trend = summarise(df_new[[c for c in df_new.columns if c != "trend"]].mean(axis=1), PPY)["sharpe_ann"]
    print(f"\ntrend's correlation to other families: {tr_corr}  (mean {np.mean(list(tr_corr.values())):+.2f})")
    print(f"master WITHOUT trend: {without_trend:+.2f}  ->  WITH trend: {mn['sharpe']:+.2f} "
          f"({mn['sharpe']-without_trend:+.2f})")
    print("marginal curve: " + " -> ".join(f"{r['added'][:4]} {r['sharpe']:+.2f}" for r in marg))
    print(f"\nper-year (master with new trend): {per_year(master_new)}")

    master_new.rename("ret").to_frame().to_parquet(R / "trend" / "master_with_trend.parquet")
    (R / "trend" / "master_with_trend_summary.json").write_text(json.dumps({
        "trend_block_sharpe": round(summarise(block, PPY)["sharpe_ann"], 3),
        "families": list(df_new.columns), "window": [str(df_new.index.min().date()), str(df_new.index.max().date())],
        "master_trend": mn,
        "trend_corr_to_others": tr_corr, "trend_mean_corr": round(float(np.mean(list(tr_corr.values()))), 3),
        "master_without_trend_sharpe": round(without_trend, 3), "marginal": marg,
        "per_year": per_year(master_new),
    }, indent=2, default=float))
    print("\nwrote reports/trend/master_with_trend.parquet + master_with_trend_summary.json")


if __name__ == "__main__":
    main()
