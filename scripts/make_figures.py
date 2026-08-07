"""Regenerate every required chart (§13) as a standalone PNG in reports/figures/, from the current
master-book artifacts. The interactive dashboard renders these inline as SVG; this writes the static
PNG deliverables the brief asks for so they never go stale relative to the book.

Charts: portfolio equity, per-sleeve equity curves, drawdown, monthly-return heatmap, rolling 12-month
Sharpe, book gross exposure over time, book turnover over time, sleeve correlation matrix, edge map.

    python scripts/make_figures.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import OOS_START  # noqa: E402

R = Path("reports")
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PPY = 365
OOS = pd.Timestamp(OOS_START).tz_localize(None)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.bbox": "tight"})


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  reports/figures/{name}")


def main():
    master = pd.read_parquet(R / "master_book.parquet")["ret"].dropna()
    legs = pd.read_parquet(R / "master_book_legs.parquet")
    corr = pd.read_csv(R / "master_book_correlation.csv", index_col=0)
    eq = (1.0 + master).cumprod()

    # (a) portfolio equity curve (log), OOS block shaded
    fig, ax = plt.subplots(figsize=(11, 4))
    eq.plot(ax=ax, color="#1f77b4", lw=1.6)
    ax.axvspan(OOS, eq.index[-1], color="#f0c000", alpha=0.12, label="final OOS block")
    ax.set_yscale("log"); ax.set_title("Master book equity (net, log) — risk-managed deliverable")
    ax.set_ylabel("growth of $1"); ax.legend()
    _save(fig, "equity.png")

    # (b) per-sleeve equity curves + book
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for c in legs.columns:
        (1.0 + legs[c].fillna(0.0)).cumprod().plot(ax=ax, lw=1.0, alpha=0.75, label=c)
    eq.plot(ax=ax, color="k", lw=2.0, label="master book")
    ax.set_yscale("log"); ax.set_title("Per-sleeve equity curves (each flat until its family lists)")
    ax.legend(fontsize=7, ncol=4); ax.set_ylabel("growth of $1")
    _save(fig, "equity_curves.png")

    # (c) drawdown
    dd = eq / eq.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.5)
    ax.set_title(f"Drawdown (max {dd.min():+.1%})"); ax.set_ylabel("%")
    _save(fig, "drawdown.png")

    # (d) monthly-return heatmap
    mo = (1.0 + master).resample("ME").prod() - 1.0
    yrs = sorted({d.year for d in mo.index})
    mat = np.full((len(yrs), 12), np.nan)
    yi = {y: i for i, y in enumerate(yrs)}
    for d, v in mo.items():
        mat[yi[d.year], d.month - 1] = v * 100
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(yrs) + 1.2))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(yrs))); ax.set_yticklabels(yrs)
    for i in range(len(yrs)):
        for j in range(12):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6)
    ax.set_title("Monthly return heatmap (%)"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.02)
    _save(fig, "monthly_heatmap.png")

    # (e) rolling 12-month Sharpe
    rmu = master.rolling(365, min_periods=180).mean()
    rsd = master.rolling(365, min_periods=180).std(ddof=1)
    roll = (np.sqrt(PPY) * rmu / rsd).dropna()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(roll.index, roll.values, color="#2ca02c", lw=1.3)
    ax.axhline(0, color="k", lw=0.5); ax.axhline(2.5, color="#888", ls="--", lw=0.8, label="target floor 2.5")
    ax.set_title("Rolling 12-month Sharpe"); ax.legend()
    _save(fig, "rolling_sharpe.png")

    # (f)/(g) book gross exposure + turnover over time
    ep = R / "master_book_exposure.parquet"
    if ep.exists():
        ex = pd.read_parquet(ep)
        fig, ax = plt.subplots(figsize=(11, 3.0))
        ax.plot(ex.index, ex["gross"], color="#1f77b4", lw=1.2)
        ax.set_title("Book gross exposure over time (§8 drawdown-ladder de-risking)")
        ax.set_ylabel("gross"); ax.axhline(1.0, color="#888", ls="--", lw=0.7)
        _save(fig, "exposure.png")

        fig, ax = plt.subplots(figsize=(11, 3.0))
        ax.plot(ex.index, ex["turnover"], color="#9467bd", lw=1.2)
        ax.set_title("Book rebalancing turnover over time (annualised; intra-sleeve reported per-family)")
        ax.set_ylabel("×/yr")
        _save(fig, "turnover.png")

    # (h) sleeve correlation matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Cross-family correlation (mean {corr.values[np.triu_indices_from(corr.values,1)].mean():+.2f})")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "correlation.png")

    # (i) edge map — timeframe × family Sharpe (where edge is and is not)
    em = R / "book" / "zoo_edge_map.csv"
    if em.exists():
        e = pd.read_csv(em, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(e) + 1.5))
        vmax = np.nanmax(np.abs(e.values))
        im = ax.imshow(e.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(e.shape[1])); ax.set_xticklabels(e.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(e))); ax.set_yticklabels(e.index)
        for i in range(len(e)):
            for j in range(e.shape[1]):
                if np.isfinite(e.values[i, j]):
                    ax.text(j, i, f"{e.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Edge map — Sharpe by timeframe × strategy family (discovery zoo)")
        ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.03)
        _save(fig, "edge_map.png")

    print("FIGURES OK")


if __name__ == "__main__":
    main()
