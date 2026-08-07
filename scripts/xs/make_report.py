"""Render the cross-sectional study to figures + a self-contained HTML dashboard.

Reads only saved artifacts (sweeps, walk-forwards, ML runs, the assembled book) — no re-fitting —
and emits PNG charts plus one standalone HTML (charts embedded as base64) for screen-share.

    python scripts/xs/make_report.py
"""
import base64
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import XS_DIR  # noqa: E402
from src.metrics import max_drawdown, summarise  # noqa: E402

XS = XS_DIR
FIG = XS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, RED, GREY = "#2563eb", "#0a8f5b", "#dc2626", "#94a3b8"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
                     "axes.edgecolor": "#cbd5e1", "axes.grid": True, "grid.color": "#eef2f7",
                     "axes.spines.top": False, "axes.spines.right": False})


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_equity(book):
    eq = (1 + book).cumprod()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), height_ratios=[3, 1], sharex=True)
    a1.plot(eq.index, eq.values, color=BLUE, lw=1.6)
    a1.set_yscale("log"); a1.set_ylabel("growth of $1 (log)")
    a1.set_title("Cross-asset x-sect book — equity curve (net, vol-targeted)", loc="left", fontweight="bold")
    dd = eq / eq.cummax() - 1
    a2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.35)
    a2.set_ylabel("drawdown"); a2.set_ylim(dd.min() * 1.1, 0.01)
    return _b64(fig)


def fig_sleeves(R):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for c in R.columns:
        eq = (1 + R[c].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3, label=c)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)"); ax.legend(ncol=4, fontsize=9)
    ax.set_title("Per-sleeve equity (each vol-targeted to 15%)", loc="left", fontweight="bold")
    return _b64(fig)


def fig_peryear(per_year):
    yrs = [y for y in per_year if per_year[y] != 0.0]
    vals = [per_year[y] for y in yrs]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(range(len(yrs)), vals, color=[GREEN if v > 0 else RED for v in vals])
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs, rotation=0, fontsize=9)
    ax.axhline(0, color="#475569", lw=0.8); ax.set_ylabel("annualised Sharpe")
    ax.set_title("Per-year Sharpe — cross-asset x-sect book", loc="left", fontweight="bold")
    return _b64(fig)


def fig_corr(corr):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "#0f172a")
    ax.grid(False); fig.colorbar(im, shrink=0.8)
    ax.set_title("Sleeve correlation", loc="left", fontweight="bold")
    return _b64(fig)


def fig_edgemap():
    """Walk-forward OOS Sharpe (top-K ensemble, best scheme) per panel — the honest edge map."""
    panels = ["crypto_1d", "crypto_4h", "crypto_1h", "stocks_1d", "fx_1d"]
    wf, isb, plac = [], [], []
    for tag in panels:
        w = pd.read_csv(XS / f"wf_{tag}.csv")
        wf.append(w[w.scheme.str.startswith("WF")]["sharpe"].max())      # best-scheme ensemble OOS
        isb.append(w[w.scheme.str.contains("in-sample")]["sharpe"].iloc[0])
        s = pd.read_csv(XS / f"sweep_{tag}.csv")
        plac.append(s[s.signal == "PLACEBO"]["sharpe"].max())
    x = np.arange(len(panels)); fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.27, isb, 0.26, label="in-sample best (overfit)", color=GREY)
    ax.bar(x, wf, 0.26, label="walk-forward OOS (honest)", color=BLUE)
    ax.bar(x + 0.27, plac, 0.26, label="placebo max (noise)", color=RED, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(panels, fontsize=9); ax.axhline(0, color="#475569", lw=0.8)
    ax.set_ylabel("annualised Sharpe"); ax.legend(fontsize=9)
    ax.set_title("Edge map — in-sample vs walk-forward vs placebo, by panel", loc="left", fontweight="bold")
    return _b64(fig)


def fig_ml():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, tag, title in [(axes[0], "crypto_1d", "crypto 1d"), (axes[1], "stocks_1d", "stocks 1d")]:
        m = pd.read_csv(XS / f"ml_{tag}.csv")
        m = m[~m.model.str.startswith("meta")]
        colors = [GREEN if "rule" in x else BLUE for x in m.model]
        ax.bar(range(len(m)), m.sharpe, color=colors)
        ax.set_xticks(range(len(m))); ax.set_xticklabels(m.model, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="#475569", lw=0.8); ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("OOS Sharpe")
    fig.suptitle("Learning-to-rank vs rule baseline — ML helps equities, not crypto", x=0.12,
                 ha="left", fontweight="bold", fontsize=12)
    return _b64(fig)


def fig_breadth():
    """Equity breadth test: WF-OOS Sharpe as the universe widens narrow→large→mid/small."""
    import json as _json
    rows = [("narrow mixed\n(78, stocks+ETFs)", 0.55, 0.90)]
    for tag, lab in [("stocks_broad", "broad large-cap\n(692, survivorship-free)"),
                     ("stocks_midsmall", "mid/small-cap\n(893, S&P 400+600)")]:
        p = XS / f"{tag}_summary.json"
        if p.exists():
            b = _json.load(open(p))
            rows.append((lab, b["walk_forward"]["sharpe_ann"], b["sweep_max"]))
    labels, wf, isb = zip(*rows)
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - 0.2, isb, 0.38, label="in-sample best", color=GREY)
    ax.bar(x + 0.2, wf, 0.38, label="walk-forward OOS (honest)", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.axhline(0, color="#475569", lw=0.8)
    ax.set_ylabel("annualised Sharpe"); ax.legend(fontsize=9)
    ax.set_title("Equity breadth does NOT strengthen it — pure-stock momentum stays modest",
                 loc="left", fontweight="bold")
    return _b64(fig)


def fig_cost(cost_levels, be):
    ms = [1, 2, 3]; vals = [cost_levels[f"{m}x"] for m in ms]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ms, vals, "o-", color=BLUE, lw=2, ms=8)
    for m, v in zip(ms, vals):
        ax.annotate(f"{v:.2f}", (m, v), textcoords="offset points", xytext=(0, 8), fontsize=9)
    ax.set_xticks(ms); ax.set_xticklabels(["1× base", "2× base", "3× base"])
    ax.set_ylabel("Sharpe"); ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title(f"Cost sensitivity (break-even {be:.0f}× base)", loc="left", fontweight="bold")
    return _b64(fig)


KPI = """<div class="kpi"><div class="label">{l}</div><div class="val">{v}</div><div class="d">{d}</div></div>"""
CARD = """<div class="card"><h3>{t}</h3><img src="data:image/png;base64,{b}"/></div>"""


def main():
    s = json.load(open(XS / "xs_summary.json"))
    book = pd.read_parquet(XS / "xs_book.parquet")["ret"]
    R = pd.read_parquet(XS / "xs_sleeve_returns.parquet")
    corr = pd.read_csv(XS / "xs_correlation.csv", index_col=0)
    cab = s["cross_asset_book"]

    figs = {
        "Equity & drawdown": fig_equity(book),
        "Per-sleeve equity": fig_sleeves(R),
        "Edge map (honest)": fig_edgemap(),
        "Equity breadth test": fig_breadth(),
        "Per-year Sharpe": fig_peryear({int(k): v for k, v in s["per_year"].items()}),
        "Sleeve correlation": fig_corr(corr),
        "ML: rank vs rule": fig_ml(),
        "Cost sensitivity": fig_cost(s["cost_levels"], s["breakeven_cost_mult"]),
    }
    cb = s["crypto_book"]
    kpis = "".join(KPI.format(l=l, v=v, d=d) for l, v, d in [
        ("Crypto book Sharpe", f"{cb['sharpe_ann']:+.2f}", "market-neutral, 1d→15m"),
        ("Cross-asset Sharpe", f"{cab['sharpe_ann']:+.2f}", f"survivorship-free equity leg"),
        ("Max drawdown", f"{cab['max_dd']:.1%}", "cross-asset book"),
        ("Monte-Carlo P5", f"{s['mc']['sharpe_p5']:+.2f}", "block-bootstrap 5th pct"),
        ("Deflated Sharpe", f"{s['deflated_sharpe']:.2f}", f"N={s['n_grid_trials']} grid trials"),
        ("Corr to trend book", f"{s['corr_to_trend']:+.2f}", f"combo DD {s['combo']['max_dd']:.1%}"),
    ])
    cards = "".join(CARD.format(t=t, b=b) for t, b in figs.items())
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cross-Sectional Momentum — Results</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{color-scheme:light dark;--bg:#eaeef4;--panel:#fff;--text:#0f172a;--muted:#516080;--border:#d5dde8;--accent:#2563eb;}}
@media(prefers-color-scheme:dark){{:root{{--bg:#070c15;--panel:#0f1929;--text:#eaf1fb;--muted:#93a7c4;--border:#233350;--accent:#69a6ff;}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:34px clamp(16px,4vw,44px);background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}}
.wrap{{max-width:1180px;margin:0 auto}}.eyebrow{{font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.2em;text-transform:uppercase;color:var(--accent)}}
h1{{font-size:clamp(23px,3vw,32px);margin:10px 0 6px;letter-spacing:-.015em}}.sub{{color:var(--muted);margin:0 0 22px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:0 0 24px}}
.kpi{{background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 9%,var(--panel)),var(--panel));border:1px solid color-mix(in srgb,var(--accent) 30%,var(--border));border-radius:14px;padding:15px 18px}}
.kpi .label{{font:600 11px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
.kpi .val{{font:700 clamp(26px,3.2vw,33px)/1.04 ui-monospace,Menlo,monospace;margin:8px 0 4px;color:var(--accent);font-variant-numeric:tabular-nums}}
.kpi .d{{font-size:12px;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px}}
.card:first-child,.card:nth-child(2),.card:nth-child(3){{grid-column:1/-1}}.card h3{{margin:0 0 10px;font-size:15px}}
.card img{{width:100%;height:auto;border-radius:6px}}@media(max-width:820px){{.grid{{grid-template-columns:1fr}}.card{{grid-column:1/-1}}}}
footer{{color:var(--muted);font-size:12px;margin-top:24px}}
</style></head><body><div class="wrap">
<div class="eyebrow">Task A · Cross-sectional relative-value</div>
<h1>Cross-Sectional Momentum — a market-neutral book</h1>
<p class="sub">Long recent winners / short recent losers, dollar-neutral, across crypto (68 perps ·
1d→15m, survivorship-free 830-perp universe) and a survivorship-free US-equity universe (S&P 500
PIT · S&P 400+600). Net of liquidity-aware costs, t+2 execution, walk-forward-validated. The edge
is <b>modest on crypto (~0.6 survivorship-free — a hand-picked coin list had inflated it to ~1.2)</b>,
a <b>~0.4 large-cap-only effect on equities that breadth does not help</b>, and <b>dead on FX and
all intraday equity/FX</b> — an honest edge map, and a decorrelated overlay more than a standalone book.</p>
<div class="kpis">{kpis}</div>
<div class="grid">{cards}</div>
<footer>All charts render saved artifacts (no re-fit). Reproduce: <code>python scripts/xs/build_panels.py
&& scripts/xs/sweep.py && walk_forward.py && ml.py && portfolio.py && make_report.py</code>.
Leakage-audited (shift audit = 0, panel-placebo beaten, exec-lag flat).</footer>
</div></body></html>"""
    (XS / "xsect_dashboard.html").write_text(html)
    print(f"wrote {XS/'xsect_dashboard.html'} ({len(html)//1024} KB) + {len(figs)} figures")


if __name__ == "__main__":
    main()
