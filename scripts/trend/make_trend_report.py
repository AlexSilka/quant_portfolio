"""Phase 5 — assemble the trend deep-dive report + self-contained dashboard from the result files.

Reads reports/trend/{trend_sweep.csv, trend_book_summary.json, trend_book_*.parquet,
trend_headline_sleeves.parquet, trend_wfo_summary.json, trend_ml.json} and emits:
  docs/strategies/TREND.md              — the prose write-up (mirrors the xsect/carry deep-dives)
  reports/trend/trend_dashboard.html  — one self-contained page (figures inlined as base64 PNG)

    python scripts/trend/make_trend_report.py
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = T.REPORTS
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
C = {"ls": "#4C78A8", "long_only": "#59A14F", "asym": "#E15759", "grid": "#BAB0AC"}


def _load(name, kind="json"):
    p = R / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if kind == "json" else (pd.read_parquet(p) if kind == "parquet"
                                                             else pd.read_csv(p))


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_equity():
    figs = []
    curves = {}
    for d in ("ls", "long_only", "asym"):
        df = _load(f"trend_book_{d}.parquet", "parquet")
        if df is not None:
            curves[d] = df["ret"]
    if not curves:
        return None
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), height_ratios=[2, 1], sharex=True)
    for d, r in curves.items():
        eq = (1 + r.fillna(0)).cumprod()
        a1.plot(eq.index, eq.values, label=d, color=C[d], lw=1.3)
        dd = eq / eq.cummax() - 1
        a2.fill_between(dd.index, dd.values, 0, color=C[d], alpha=0.25)
    a1.set_yscale("log"); a1.set_ylabel("growth of $1 (log)"); a1.legend(loc="upper left", frameon=False)
    a1.set_title("Trend book — equity curve by direction mode")
    a2.set_ylabel("drawdown")
    return _png(fig)


def fig_peryear(bk):
    hy = bk.get("headline", {}).get("per_year", {})
    if not hy:
        return None
    ys = sorted(int(y) for y in hy)
    vals = [hy[str(y)] if str(y) in hy else hy[y] for y in ys]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar([str(y) for y in ys], vals, color=[C["long_only"] if v >= 0 else C["asym"] for v in vals])
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("Sharpe")
    ax.set_title(f"Headline book ({bk.get('headline_direction','')}) — per-year Sharpe")
    return _png(fig)


def fig_edgemap(sweep):
    if sweep is None:
        return None
    ls = sweep[(sweep.direction == "ls") & (sweep.group == "entry")]
    piv = ls.pivot_table(index="entry", columns=["asset_class", "tf"], values="sharpe", aggfunc="median")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(["/".join(map(str, c)) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Edge map — median single-instrument trend Sharpe (entry × asset/TF, dir=ls)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    return _png(fig)


def fig_direction(bk):
    dirs = bk.get("directions", {})
    if not dirs:
        return None
    names = list(dirs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.8))
    a1.bar(names, [dirs[d]["sharpe"] for d in names], color=[C[d] for d in names])
    a1.set_title("Book Sharpe"); a1.axhline(0, color="k", lw=0.5)
    a2.bar(names, [abs(dirs[d]["max_dd"]) for d in names], color=[C[d] for d in names])
    a2.set_title("Book max drawdown (abs)")
    return _png(fig)


def fig_cost(bk):
    cl = bk.get("cost_levels", {})
    if not cl:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ks = [k for k in ("1x", "2x", "3x") if k in cl]
    ax.plot(ks, [cl[k] for k in ks], "o-", color=C["ls"])
    ax.set_ylabel("Sharpe"); ax.set_title("Cost sensitivity (headline book)")
    return _png(fig)


def fig_corr():
    df = _load("trend_headline_sleeves.parquet", "parquet")
    if df is None:
        return None
    corr = df.corr().values
    iu = np.triu_indices_from(corr, k=1)
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.hist(corr[iu], bins=40, color=C["ls"], alpha=0.85)
    ax.axvline(float(np.nanmean(corr[iu])), color=C["asym"], lw=1.2,
               label=f"mean {np.nanmean(corr[iu]):+.2f}")
    ax.set_title("Sleeve pairwise correlation"); ax.legend(frameon=False)
    return _png(fig)


def fig_marginal(bk):
    mc = bk.get("headline", {}).get("marginal", [])
    if not mc:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot([m["n"] for m in mc], [m["sharpe"] for m in mc], color=C["long_only"])
    ax.set_xlabel("# sleeves added (by contribution)"); ax.set_ylabel("book Sharpe")
    ax.set_title("Marginal contribution")
    return _png(fig)


def fig_breadth():
    """Book Sharpe & OOS vs crypto universe size — fewer liquid names win."""
    ff = _load("trend_breadth_ema_long_only.json")
    if not ff or not ff.get("scaling"):
        return None
    sc = ff["scaling"]
    ns = sorted(int(k) for k in sc)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(ns, [sc[str(n)]["sharpe"] for n in ns], "o-", color=C["ls"], label="full-sample Sharpe")
    ax.plot(ns, [sc[str(n)]["oos"] for n in ns], "s--", color=C["asym"], label="held-out OOS Sharpe")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("crypto instruments (ranked by liquidity)"); ax.set_ylabel("Sharpe")
    ax.set_title("Crypto breadth — more instruments HURT (one correlated cluster)")
    ax.legend(frameon=False, fontsize=8)
    return _png(fig)


def fig_composition():
    """P&L share vs risk share by asset class and timeframe — the trend block's make-up."""
    p = R / "trend_composition_asym.csv"
    if not p.exists():
        return None
    comp = pd.read_csv(p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, key, order in [(a1, "asset_class", ["crypto", "equity"]), (a2, "tf", ["1d", "4h", "1h"])]:
        g = comp.groupby(key)[["pnl_share", "risk_contrib"]].sum()
        g = g.reindex([o for o in order if o in g.index])
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["pnl_share"] * 100, 0.4, label="P&L share", color=C["long_only"])
        ax.bar(x + 0.2, g["risk_contrib"] * 100, 0.4, label="risk share", color=C["ls"])
        ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_ylabel("%")
        ax.set_title(f"by {key.replace('_',' ')}")
    a1.legend(frameon=False, fontsize=7)
    fig.suptitle("Trend block composition — P&L vs risk contribution", y=1.02)
    return _png(fig)


def fig_features():
    """Feature-family gain-importance of the trend meta-model (in-sample) + the OOS-AUC caption."""
    ff = _load("trend_features.json")
    if not ff or not ff.get("family_importance"):
        return None
    fam = ff["family_importance"]
    items = sorted(fam.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh([k for k, _ in items], [v * 100 for _, v in items], color=C["ls"])
    ax.set_xlabel("share of model gain (%)")
    ax.set_title(f"Trend meta-model — feature-family importance  "
                 f"(AUC in-sample {ff.get('auc_is',0):.2f} → OOS {ff.get('auc_oos',0):.2f}: no OOS edge)")
    return _png(fig)


def fig_risk():
    """Gross exposure + drawdown of the vol-target + drawdown-ladder overlay on the headline book —
    visualises the §8 risk logic cutting exposure through the crises."""
    p = R / "trend_book_asym.parquet"
    if not p.exists():
        return None
    try:
        from src.risk.overlay import apply_overlay
    except Exception:
        return None
    ret = pd.read_parquet(p)["ret"]
    fin, expos = apply_overlay(ret, target_vol=0.12, cap=3.0)
    eq = (1 + fin.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 3.6), height_ratios=[1, 1], sharex=True)
    a1.plot(expos.index, expos["gross"].values, color=C["ls"], lw=0.8)
    a1.axhline(1.0, color=C["grid"], lw=0.6, ls="--")
    a1.set_ylabel("gross exposure"); a1.set_title("Risk overlay — vol-target + drawdown ladder (headline book)")
    a2.fill_between(dd.index, dd.values, 0, color=C["asym"], alpha=0.3)
    a2.set_ylabel("drawdown")
    return _png(fig)


def fig_ml(ml):
    if not ml:
        return None
    base = ml.get("baseline_ungated", {}).get("sharpe")
    variants = {k: v for k, v in ml.items() if k != "baseline_ungated" and isinstance(v, dict) and "sharpe" in v}
    pick = ["baseline_ungated"] + [k for k in variants if "gate" in k or "sized" in k][:6]
    labels, sh_is, sh_oos = [], [], []
    for k in pick:
        v = ml[k]
        labels.append(k.replace("lightgbm", "lgb").replace("_gate", "").replace("randomforest", "rf"))
        sh_is.append(v.get("sharpe_is", v.get("sharpe")))
        sh_oos.append(v.get("sharpe_oos"))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(x - 0.2, sh_is, 0.4, label="IS", color=C["ls"])
    ax.bar(x + 0.2, sh_oos, 0.4, label="OOS", color=C["asym"])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Sharpe"); ax.legend(frameon=False); ax.set_title("ML incremental value (trend book)")
    return _png(fig)


def build_html(figs, bk, wfo, ml):
    parts = ["<h1>Trend-Following Deep-Dive — Dashboard</h1>"]
    hb = bk.get("headline", {}) if bk else {}
    hd = bk.get("headline_direction", "") if bk else ""
    if hb:
        mc = hb.get("mc", {})
        _fdr = hb.get("placebo_beta_neutral", {}).get("exceedance_rate")
        fdr = f"{_fdr*100:.0f}%" if isinstance(_fdr, (int, float)) else "–"
        _p5 = mc.get("sharpe_p5")
        p5 = f"{_p5:+.2f}" if isinstance(_p5, (int, float)) else "–"
        parts.append(f"""<div class="kpis">
        <div class="kpi"><b>{hb.get('sharpe','–')}</b><span>Sharpe ({hd})</span></div>
        <div class="kpi"><b>{hb.get('max_dd',0)*100:.1f}%</b><span>max drawdown</span></div>
        <div class="kpi"><b>{hb.get('months_in_profit',0)*100:.0f}%</b><span>months in profit</span></div>
        <div class="kpi"><b>{p5}</b><span>MC P5 Sharpe</span></div>
        <div class="kpi"><b>{fdr}</b><span>β-neutral placebo FDR</span></div>
        <div class="kpi"><b>{hb.get('n_sleeves','–')}</b><span>sleeves</span></div></div>""")
    order = [("Equity curve by direction", "equity"), ("Per-year Sharpe", "peryear"),
             ("Edge map", "edgemap"), ("Direction comparison", "direction"),
             ("Crypto breadth scaling", "breadth"), ("Block composition (P&L vs risk)", "composition"),
             ("ML incremental value", "ml"), ("Meta-model feature families", "features"),
             ("Risk overlay (vol-target + drawdown ladder)", "risk"), ("Cost sensitivity", "cost"),
             ("Sleeve correlation", "corr"), ("Marginal contribution", "marginal")]
    for title, key in order:
        if figs.get(key):
            parts.append(f'<section><h2>{title}</h2><img src="data:image/png;base64,{figs[key]}"/></section>')
    css = """<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
    padding:0 1rem;color:#1a1a1a}h1{font-size:1.5rem}h2{font-size:1.05rem;margin-top:1.6rem;color:#333}
    img{max-width:100%;border:1px solid #eee;border-radius:6px}.kpis{display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0}
    .kpi{flex:1;min-width:120px;background:#f7f7f8;border:1px solid #ececf0;border-radius:8px;padding:.7rem 1rem;text-align:center}
    .kpi b{display:block;font-size:1.35rem;color:#111}.kpi span{font-size:.72rem;color:#666}
    section{margin:1rem 0}@media(prefers-color-scheme:dark){body{background:#16161a;color:#e6e6e6}
    .kpi{background:#1f1f26;border-color:#2a2a33}.kpi b{color:#fff}h2{color:#ccc}img{border-color:#2a2a33}}</style>"""
    (R / "trend_dashboard.html").write_text(f"<!doctype html><meta charset=utf-8><title>Trend Deep-Dive</title>{css}" + "\n".join(parts))


def main():
    sweep = _load("trend_sweep.csv", "csv")
    bk = _load("trend_book_summary.json")
    wfo = _load("trend_wfo_summary.json")
    ml = _load("trend_ml.json")
    figs = {"equity": fig_equity(), "peryear": fig_peryear(bk) if bk else None,
            "edgemap": fig_edgemap(sweep), "direction": fig_direction(bk) if bk else None,
            "cost": fig_cost(bk) if bk else None, "corr": fig_corr(),
            "marginal": fig_marginal(bk) if bk else None, "ml": fig_ml(ml),
            "breadth": fig_breadth(), "composition": fig_composition(),
            "features": fig_features(), "risk": fig_risk()}
    build_html(figs, bk, wfo, ml)
    print(f"wrote {R/'trend_dashboard.html'}  (figures: {[k for k,v in figs.items() if v]})")
    print("(TREND.md is authored separately from these same result files)")


if __name__ == "__main__":
    main()
