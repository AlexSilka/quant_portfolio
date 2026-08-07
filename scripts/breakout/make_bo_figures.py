"""Breakout story in figures. Reads the saved bo_* artifacts (no recompute) and writes
reports/figures/breakout.png (the 6-panel story) and reports/figures/breakout_book.png
(monthly heatmap, rolling Sharpe, sleeve correlation).

    python scripts/breakout/make_bo_figures.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.metrics import summarise  # noqa: E402

R = Path("reports")
FIG = R / "figures"
PPY = 365
GREEN, RED, BLUE, GREY, PURPLE, ORANGE = "#2ca02c", "#d62728", "#1f77b4", "#b0b0b0", "#9467bd", "#ff7f0e"
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def story():
    exits = pd.read_csv(R / "breakout" / "bo_exits.csv")
    sweep = pd.read_csv(R / "breakout" / "bo_sweep.csv")
    port = pd.read_parquet(R / "breakout" / "bo_final_portfolio.parquet")["ret"]
    rets = pd.read_parquet(R / "breakout" / "bo_final_sleeve_returns.parquet")
    costs = pd.read_parquet(R / "breakout" / "bo_final_costs.parquet")
    ml = json.loads((R / "breakout" / "bo_ml.json").read_text())
    summ = json.loads((R / "breakout" / "bo_final_summary.json").read_text())

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # (1) exit-style comparison — the fat-tail thesis
    a = ax[0, 0]
    order = ["triple_barrier", "time", "channel", "atr_trailing", "reversal"]
    m = exits.groupby("exit")["sharpe"].mean().reindex(order)
    cols = [GREY if e == "triple_barrier" else GREEN for e in order]
    a.bar(range(len(order)), m.values, color=cols)
    a.set_xticks(range(len(order)))
    a.set_xticklabels(["triple\nbarrier", "time", "channel", "ATR\ntrail", "reversal"], fontsize=8)
    a.axhline(0, color="k", lw=0.5)
    a.set_title("1) Exit style: trend-riding beats the bounded\ntriple-barrier (fat-tail thesis)")
    a.set_ylabel("mean Sharpe (all sym x 1d/4h)")

    # (2) edge map heatmap — config x timeframe
    a = ax[0, 1]
    piv = sweep.pivot_table(index="config", columns="tf", values="sharpe", aggfunc="mean")
    piv = piv.reindex(columns=["1d", "4h", "1h", "15m", "5m"])
    order_c = ["base_d55_tb", "d55_atr3", "d55_atr3_tr", "kelt_atr3", "d55_chan20", "d20_chan10",
               "boll_atr3", "d55_rev"]
    piv = piv.reindex([c for c in order_c if c in piv.index])
    a.imshow(piv.values, cmap="RdYlGn", vmin=-1.5, vmax=1.0, aspect="auto")
    a.set_xticks(range(piv.shape[1])); a.set_xticklabels(piv.columns)
    a.set_yticks(range(piv.shape[0])); a.set_yticklabels(piv.index, fontsize=7)
    for (i, j), v in np.ndenumerate(piv.values):
        if np.isfinite(v):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    a.set_title("2) Edge map: mean Sharpe by construction x TF\n(crypto trend, dies intraday)")
    a.grid(False)

    # (3) final book equity + drawdown
    a = ax[0, 2]
    eq = (1 + port).cumprod()
    eq.plot(ax=a, color=BLUE, lw=1.5, label=f"equity (Sh {summ['portfolio']['sharpe_ann']:+.2f})")
    a.set_ylabel("equity", color=BLUE); a.legend(loc="upper left", fontsize=8)
    a2 = a.twinx()
    dd = eq / eq.cummax() - 1.0
    a2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.25)
    a2.set_ylabel("drawdown", color=RED); a2.grid(False); a2.set_ylim(-0.4, 0)
    a.set_title(f"3) Final book equity + drawdown\n(maxDD {summ['portfolio']['max_dd']:+.1%}, MC-P5 {summ['mc']['sharpe_p5']:+.2f})")

    # (4) per-year Sharpe (regime profile) + OOS marker
    a = ax[1, 0]
    py = summ["per_year"]
    yrs = list(py.keys())
    a.bar(range(len(yrs)), [py[y] for y in yrs],
          color=[GREEN if py[y] > 0 else RED for y in yrs])
    a.set_xticks(range(len(yrs))); a.set_xticklabels([y[2:] for y in yrs])
    a.axhline(0, color="k", lw=0.5)
    a.axvline(len(yrs) - 2.5, color=PURPLE, ls="--", lw=1)
    a.text(len(yrs) - 2.4, a.get_ylim()[1] * 0.85, "held-out\nOOS", color=PURPLE, fontsize=7)
    a.set_title("4) Per-year Sharpe: strong in trending yrs,\nflat in trendless 2025-26 (market-wide)")
    a.set_ylabel("Sharpe")

    # (5) ML incremental value — baseline vs gated Sharpe & DD
    a = ax[1, 1]
    base = ml["baseline_ungated"]
    best = max((k for k in ml if k != "baseline_ungated"), key=lambda k: ml[k]["sharpe"])
    labels = ["ungated", "ML gated"]
    shs = [base["sharpe"], ml[best]["sharpe"]]
    dds = [-base["max_dd"], -ml[best]["max_dd"]]
    x = np.arange(2)
    a.bar(x - 0.2, shs, 0.4, color=BLUE, label="Sharpe")
    a.set_ylabel("Sharpe", color=BLUE); a.set_xticks(x); a.set_xticklabels(labels)
    a2 = a.twinx()
    a2.bar(x + 0.2, dds, 0.4, color=RED, label="max DD")
    a2.set_ylabel("max drawdown", color=RED); a2.grid(False)
    a.set_title(f"5) ML meta-label value (4h+1h book):\nSharpe {base['sharpe']:+.2f}->{ml[best]['sharpe']:+.2f}, "
                f"DD {base['max_dd']:+.0%}->{ml[best]['max_dd']:+.0%}")

    # (6) cost sensitivity + break-even
    a = ax[1, 2]
    mults = np.linspace(1, 14, 27)

    def at(m):
        return (rets.fillna(0.0) - (m - 1.0) * costs.reindex_like(rets).fillna(0.0)).mean(axis=1)
    shs = [summarise(at(m), PPY)["sharpe_ann"] for m in mults]
    a.plot(mults, shs, "o-", color=PURPLE, ms=3)
    a.axhline(0, color="k", lw=0.5)
    be = summ.get("breakeven_mult")
    if be:
        a.axvline(be, color=RED, ls="--", lw=1); a.text(be - 0.3, max(shs) * 0.5, f"break-even\n{be:.1f}x", color=RED, ha="right", fontsize=7)
    a.set_title("6) Cost sensitivity (survives ~10x base cost)")
    a.set_xlabel("cost multiple x base"); a.set_ylabel("Sharpe")

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "breakout.png", bbox_inches="tight")
    print("wrote reports/figures/breakout.png")


def book_panels():
    port = pd.read_parquet(R / "breakout" / "bo_final_portfolio.parquet")["ret"]
    rets = pd.read_parquet(R / "breakout" / "bo_final_sleeve_returns.parquet")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # monthly return heatmap
    a = ax[0]
    m = (1 + port).resample("ME").prod() - 1
    tab = m.groupby([m.index.year, m.index.month]).first().unstack()
    im = a.imshow(tab.values * 100, cmap="RdYlGn", vmin=-15, vmax=15, aspect="auto")
    a.set_yticks(range(len(tab.index))); a.set_yticklabels(tab.index)
    a.set_xticks(range(12)); a.set_xticklabels(list("JFMAMJJASOND"))
    for (i, j), v in np.ndenumerate(tab.values):
        if np.isfinite(v):
            a.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6)
    a.set_title("Monthly returns, % (final breakout book)"); a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046)

    # rolling 12-month Sharpe
    a = ax[1]
    roll = port.rolling(365).apply(lambda x: np.sqrt(PPY) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0)
    roll.plot(ax=a, color=BLUE)
    a.axhline(0, color="k", lw=0.5); a.axhline(1, color=GREEN, ls=":", lw=1)
    a.set_title("Rolling 12-month Sharpe"); a.set_ylabel("Sharpe")

    # sleeve correlation heatmap
    a = ax[2]
    corr = rets.corr()
    im = a.imshow(corr.values, cmap="coolwarm", vmin=-0.5, vmax=1.0, aspect="auto")
    a.set_title(f"Sleeve correlation (mean {corr.values[np.triu_indices_from(corr,1)].mean():+.2f})")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046)

    fig.tight_layout()
    fig.savefig(FIG / "breakout_book.png", bbox_inches="tight")
    print("wrote reports/figures/breakout_book.png")


if __name__ == "__main__":
    story()
    book_panels()
