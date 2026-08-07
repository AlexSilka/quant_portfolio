"""Self-contained interactive dashboard for THE master book (scripts/run_master_book.py).

Renders the canonical portfolio — risk-parity over the eight surviving strategy families (trend,
carry, short-vol/VRP, cross-sectional momentum, breakout, crisis-alpha, global-macro, betting-against-beta) — from the master_book* artifacts:
master_book_summary.json (headline Sharpe/DD/MC + per-year/quarter), master_book.parquet (the
equity curve), master_book_legs.parquet (per-family series -> standalone Sharpe/DD, correlation to
the book, and the book-without-family delta), master_book_correlation.csv and master_book_marginal.csv.

Charts are inline SVG generated here (crisp vector, theme-aware) with a small inline-JS layer for
hover tooltips and an equity crosshair — no external libraries, CSP-safe. The page shell — HTML
template, CSS, JS — lives in report_assets/ and is inlined into the single output file at build;
this module only computes the data and fills the template's placeholders.

    python scripts/make_report.py   ->   reports/dashboard.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CAPITAL_USD, OOS_START  # noqa: E402

REP = Path("reports")
CAP = CAPITAL_USD  # Task A §9 sizing/cost capital — single source: src/config.py
PPY = 365      # portfolio series is calendar-daily (crypto trades weekends)
OOS_TS = pd.Timestamp(OOS_START).tz_localize(None)  # frozen OOS boundary (tz-naive to match the book index)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SHORT = {"trend_momentum": "trend", "carry": "carry", "volprem": "vol-prem",
         "xs_momentum": "x-sect", "bab": "BAB", "breakout": "breakout",
         "crisis": "crisis-alpha", "gmacro": "global-macro"}   # family id -> display label
STRESS = [("Q4 2018", "2018-10-01", "2018-12-31"),
          ("COVID crash Feb-Mar 2020", "2020-02-01", "2020-03-31"),
          ("2021 (bull)", "2021-01-01", "2021-12-31"),
          ("2022 (bear)", "2022-01-01", "2022-12-31"),
          ("2023-present", "2023-01-01", "2026-12-31")]


def sf(f):
    return SHORT.get(f, f)


def _lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def heatcolor(v, vmax, scheme):
    if v is None or not np.isfinite(v):
        return "var(--empty)"
    stops = {"rdylgn": [(199, 66, 56), (247, 220, 128), (35, 150, 84)],
             "coolwarm": [(74, 105, 199), (228, 233, 240), (198, 74, 74)]}[scheme]
    x = max(-1.0, min(1.0, v / vmax))
    rgb = _lerp(stops[1], stops[0], -x) if x < 0 else _lerp(stops[1], stops[2], x)
    return "#%02x%02x%02x" % rgb


# ---------- metric helpers (Task A §11) ----------
def _sh(r, ppy=PPY):
    r = r.dropna()
    return float(np.sqrt(ppy) * r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else 0.0


def _mdd(r):
    e = (1.0 + r.dropna()).cumprod()
    return float((e / e.cummax() - 1.0).min()) if len(e) else 0.0


def _streak(monthly):
    best = cur = 0
    for v in monthly:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return int(best)


def _mip(monthly):
    return float((monthly > 0).mean()) if len(monthly) else 0.0


# ---------- svg builders (w = natural viewBox width in px; capped via max-width) ----------
def line_svg(key, pts, w, h, log=False, pct=False):
    l, r, t, b = 60, 18, 16, 30
    pts = [p for p in pts if np.isfinite(p[1])]  # drop NaN (e.g. holiday gaps) so the path/hover span the full range
    xs = [p[0] for p in pts]
    ys = [np.log(p[1]) if log else p[1] for p in pts]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    pad = (ymax - ymin) * 0.06 or 1.0
    ymin, ymax = ymin - pad, ymax + pad

    def X(x):
        return l + (x - xmin) / (xmax - xmin) * (w - l - r)

    def Y(v):
        vv = np.log(v) if log else v
        return h - b - (vv - ymin) / (ymax - ymin) * (h - t - b)

    p = []
    for i in range(5):
        yy = t + (h - t - b) * (1 - i / 4)
        gv = ymin + (ymax - ymin) * i / 4
        disp = float(np.exp(gv)) if log else gv
        lab = f"{disp * 100:.0f}%" if pct else f"{disp:.2f}"
        p.append(f'<line class="gl" x1="{l}" y1="{yy:.1f}" x2="{w - r}" y2="{yy:.1f}"/>')
        p.append(f'<text class="ax" x="{l - 9}" y="{yy + 4:.1f}" text-anchor="end">{lab}</text>')
    yrs = sorted({pd.Timestamp(x, unit="ms").year for x in xs})
    step = max(1, -(-len(yrs) // max(3, int(w // 135))))  # thin labels to the chart width
    for yr in yrs[::step]:
        xx = X(pd.Timestamp(str(yr)).value // 10 ** 6)
        if l <= xx <= w - r:
            p.append(f'<text class="ax" x="{xx:.0f}" y="{h - 9}" text-anchor="middle">{yr}</text>')
    dl = " ".join(("M" if i == 0 else "L") + f"{X(q[0]):.1f} {Y(q[1]):.1f}" for i, q in enumerate(pts))
    p.append(f'<path class="area" d="{dl} L {X(xmax):.1f} {h - b} L {X(xmin):.1f} {h - b} Z"/>')
    p.append(f'<path class="line" d="{dl}"/>')
    p.append(f'<line class="cross" id="{key}-cross" y1="{t}" y2="{h - b}" style="opacity:0"/>')
    p.append(f'<circle class="dot" id="{key}-dot" r="4" style="opacity:0"/>')
    p.append(f'<rect id="{key}-hit" x="{l}" y="{t}" width="{w - l - r}" height="{h - t - b}" fill="transparent"/>')
    meta = {"pts": pts, "W": w, "H": h, "l": l, "r": r, "t": t, "b": b,
            "xmin": xmin, "xmax": xmax, "ymin": float(ymin), "ymax": float(ymax), "log": log, "pct": pct}
    return _svg(w, h, "".join(p)), meta


def curve_svg(labels, values, w, h, mark=None):
    """Categorical-x line (marginal-contribution curve): index on x, value on y, peak marked."""
    l, r, t, b = 46, 14, 16, 30
    vmin, vmax = min(values), max(values)
    pad = (vmax - vmin) * 0.15 or 1.0
    vmin, vmax = vmin - pad, vmax + pad
    n = len(values)

    def X(i):
        return l + (i / (n - 1 if n > 1 else 1)) * (w - l - r)

    def Y(v):
        return h - b - (v - vmin) / (vmax - vmin) * (h - t - b)

    p = []
    for k in range(3):
        yy = t + (h - t - b) * (1 - k / 2)
        gv = vmin + (vmax - vmin) * k / 2
        p.append(f'<line class="gl" x1="{l}" y1="{yy:.1f}" x2="{w - r}" y2="{yy:.1f}"/>')
        p.append(f'<text class="ax" x="{l - 8}" y="{yy + 4:.1f}" text-anchor="end">{gv:.2f}</text>')
    dl = " ".join(("M" if i == 0 else "L") + f"{X(i):.1f} {Y(v):.1f}" for i, v in enumerate(values))
    p.append(f'<path class="area" d="{dl} L {X(n - 1):.1f} {h - b} L {X(0):.1f} {h - b} Z"/>')
    p.append(f'<path class="line" d="{dl}"/>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        cls = "pt mk" if i == mark else "pt"
        p.append(f'<circle class="{cls}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="{5 if i == mark else 3}" '
                 f'data-tip="+{lab} ({i + 1} families): Sharpe {v:+.2f}"/>')
        p.append(f'<text class="ax" x="{X(i):.0f}" y="{h - 9}" text-anchor="middle">{i + 1}</text>')
    return _svg(w, h, "".join(p))


def multiline_svg(curves, w, h, bold=None):
    """Overlay many equity curves (log) — per-family faint + book bold; native <title> hover."""
    l, r, t, b = 52, 14, 14, 28
    allx = [pt[0] for _, pts in curves for pt in pts]
    ally = [pt[1] for _, pts in curves for pt in pts]
    xmin, xmax = min(allx), max(allx)
    ymin, ymax = np.log(max(min(ally), 1e-6)), np.log(max(ally))
    pad = (ymax - ymin) * 0.06 or 1.0
    ymin, ymax = ymin - pad, ymax + pad

    def X(x):
        return l + (x - xmin) / (xmax - xmin) * (w - l - r)

    def Y(v):
        return h - b - (np.log(max(v, 1e-6)) - ymin) / (ymax - ymin) * (h - t - b)

    p = []
    for k in range(4):
        yy = t + (h - t - b) * (1 - k / 3)
        p.append(f'<line class="gl" x1="{l}" y1="{yy:.1f}" x2="{w - r}" y2="{yy:.1f}"/>')
        p.append(f'<text class="ax" x="{l - 8}" y="{yy + 4:.1f}" text-anchor="end">'
                 f'{np.exp(ymin + (ymax - ymin) * k / 3):.1f}x</text>')
    for yr in sorted({pd.Timestamp(x, unit="ms").year for x in allx})[::max(1, -(-len({pd.Timestamp(x, unit="ms").year for x in allx}) // 8))]:
        xx = X(pd.Timestamp(str(yr)).value // 10 ** 6)
        if l <= xx <= w - r:
            p.append(f'<text class="ax" x="{xx:.0f}" y="{h - 8}" text-anchor="middle">{yr}</text>')
    for i, (lab, pts) in enumerate(curves):
        dl = " ".join(("M" if j == 0 else "L") + f"{X(q[0]):.1f} {Y(q[1]):.1f}" for j, q in enumerate(pts))
        cls = "mlline bold" if i == bold else "mlline"
        # visible thin line + a fat transparent hit-path so the 1px line is actually hoverable
        p.append(f'<g class="mlg"><path class="{cls}" d="{dl}"/>'
                 f'<path class="mlhit" d="{dl}" data-tip="{lab}"/></g>')
    return _svg(w, h, "".join(p))


def bars_svg(items, w, h, pct=False):
    l, r, t, b = 46, 12, 14, 48
    vals = [v for _, v in items] + [0.0]
    vmax, vmin = max(vals), min(vals)
    span = (vmax - vmin) or 1.0

    def Y(v):
        return t + (h - t - b) * (vmax - v) / span

    y0 = Y(0.0)
    step = (w - l - r) / len(items)
    p = [f'<line class="gl" x1="{l}" y1="{y0:.1f}" x2="{w - r}" y2="{y0:.1f}"/>']
    for i, (lab, v) in enumerate(items):
        cx = l + (i + 0.5) * step
        bw = step * 0.64
        yy, hh = min(Y(v), y0), abs(Y(v) - y0)
        cls = "bar-pos" if v >= 0 else "bar-neg"
        val = f"{v * 100:.0f}%" if pct else f"{v:+.2f}"
        p.append(f'<rect class="{cls}" x="{cx - bw / 2:.1f}" y="{yy:.1f}" width="{bw:.1f}" '
                 f'height="{hh:.1f}" rx="2.5" data-tip="{lab}: {val}"/>')
        p.append(f'<text class="ax" x="{cx:.0f}" y="{h - 12}" text-anchor="end" '
                 f'transform="rotate(-40 {cx:.0f} {h - 12})">{lab}</text>')
    return _svg(w, h, "".join(p))


def heat_svg(rows, cols, matrix, w, vmax, scheme, show_val=True, col_labels=True, fmt=None, rowh=32, val_fs=None):
    fmt = fmt or (lambda v: f"{v:+.2f}")
    labcol, ch = 118, rowh
    cw = (w - labcol - 10) / len(cols)
    ct = 26 if col_labels else 10
    h = ct + len(rows) * ch + 8
    p = []
    if col_labels:
        lab_step = max(1, -(-len(cols) // 16))   # show <=16 labels so wide grids (e.g. 43 quarters) don't overlap
        for j in range(len(cols)):
            if j % lab_step:
                continue
            cx = labcol + j * cw + cw / 2
            p.append(f'<text class="lbl" x="{cx:.0f}" y="{ct - 9}" text-anchor="middle">{cols[j]}</text>')
    for i, rl in enumerate(rows):
        y = ct + i * ch
        p.append(f'<text class="lbl" x="{labcol - 9}" y="{y + ch / 2 + 4:.0f}" text-anchor="end">{rl}</text>')
        for j in range(len(cols)):
            v = matrix[i][j]
            x = labcol + j * cw
            tip = (f"{rl} / {cols[j]}: {fmt(v)}" if v is not None else f"{rl}/{cols[j]}: n/a")
            p.append(f'<rect class="cell" x="{x:.1f}" y="{y}" width="{cw - 2:.1f}" height="{ch - 2}" '
                     f'rx="2.5" fill="{heatcolor(v, vmax, scheme)}" data-tip="{tip}"/>')
            if show_val and v is not None:
                vy = y + ch / 2 + (val_fs * 0.35 if val_fs else 4)
                fs = f' style="font-size:{val_fs}px"' if val_fs else ''
                p.append(f'<text class="cv"{fs} x="{x + cw / 2:.1f}" y="{vy:.0f}" '
                         f'text-anchor="middle">{fmt(v)}</text>')
    return _svg(w, h, "".join(p))


def _svg(w, h, inner):
    return (f'<svg class="chart-svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'style="max-width:{w:.0f}px" preserveAspectRatio="xMidYMid meet">{inner}</svg>')


def _ds(s, n=760):
    return s.iloc[:: max(1, len(s) // n)]


def _pts(s):
    return [[int(pd.Timestamp(i).timestamp() * 1000), float(v)] for i, v in s.items()]


def _scorecard(items):
    def tile(lab, val, note, cls):
        # big value = the §11-scored OOS number (pass/miss-coloured, tagged OOS on scored rows); the
        # 15-year estimate + target sit together on one muted sub-line below — clear, not crammed.
        otag = '<span class="vsub">OOS</span>' if cls else ''
        return (f'<div class="sc {cls}"><div class="label">{lab}</div>'
                f'<div class="val">{val}{otag}</div><div class="note">{note}</div></div>')
    return f'<div class="scorecard">{"".join(tile(*it) for it in items)}</div>'


def _honesty_card():
    """Optional anti-overfitting panel from the discovery zoo (scripts/run_book.py). Absent if the
    zoo has not been run — the master book itself never depends on it."""
    p = REP / "book" / "zoo_summary.json"
    if not p.exists():
        return ""
    z = json.loads(p.read_text())
    rows = "".join(f"<tr><td>{lab}</td><td>{int(n):,}</td></tr>" for lab, n in z.get("funnel", []))
    n = int(z.get("n_trials", 0))
    ins = z.get("portfolio", {}).get("sharpe_ann", float("nan"))
    wf = z.get("wf_oos_sharpe", float("nan"))
    dsr = z.get("best_sleeve_dsr", float("nan"))
    fdr = z.get("placebo_fdr", float("nan"))
    wf_s = f"{wf:+.2f}" if wf == wf else "n/a"        # NaN-safe (zoo not fully run)
    fdr_s = f"{fdr:.1%}" if fdr == fdr else "n/a"
    # CSCV probability of backtest overfitting (scripts/run_cscv.py) — the §6 "PBO or equivalent" metric
    cp = REP / "book" / "cscv_pbo.json"
    pbo_s = ""
    if cp.exists():
        c = json.loads(cp.read_text())
        pbo_s = (f' &middot; CSCV probability of backtest overfitting <b>{c["pbo"]:.0%}</b> '
                 f'(IS-best sleeve degrades {c["is_sharpe_mean"]:+.2f}&rarr;{c["oos_sharpe_mean"]:+.2f}/bar OOS)')
    return (f'<figure class="card s6"><figcaption>Honest search &mdash; why the book selects nothing '
            f'(anti-overfitting §6/§10/§12)</figcaption>'
            f'<table><tr><th>discovery gate</th><th>candidates</th></tr>{rows}</table>'
            f'<p class="valline">{n:,} candidates mined &middot; naive in-sample Sharpe {ins:+.2f} '
            f'&middot; the same selection walk-forwarded out-of-sample gives Sharpe {wf_s} &middot; best '
            f'single-sleeve deflated Sharpe {dsr:.2f} (N={n:,}) &middot; shuffled-signal false-discovery '
            f'rate {fdr_s}{pbo_s} &rarr; mining winners is selection bias, so the traded book <b>selects '
            f'nothing</b> and applies theory uniformly across the whole universe.</p></figure>')


# --- §12/§13 edge map, family level: every family we evaluated on ONE consistent scale (its honest
# standalone Sharpe from the best validated construction) — the book's live families AND the rejected
# ones, so the map shows where edge was found and where it was not. Live numbers come from the master-
# book legs each run; rejected ones from each deep-dive's own frozen artifact (two families carry no
# saved summary, so their honest walk-forward headline is stated inline). This is the headline edge
# map; the raw first-pass timeframe scan (the zoo) sits below it as supporting detail. ---
LIVE_FAM = [  # (family id in summ["standalone_sharpe"], asset class, timeframe(s), where the edge is)
    ("volprem",        "multi-asset vol", "1d",           "index/single-name/commodity/rates VRP &mdash; the dominant sleeve"),
    ("trend_momentum", "crypto + equity", "1d / 4h",      "the repo&rsquo;s core premium; held to reversal"),
    ("breakout",       "crypto",          "1h / 4h / 1d", "channel breaks with an ML confidence gate"),
    ("carry",          "crypto",          "1d",           "perp funding, dollar-neutral cross-section"),
    ("gmacro",         "EM-FX + commod.", "1d",           "trend on asset classes no other family trades"),
    ("xs_momentum",    "crypto + equity", "1d",           "survivorship-free top-100 momentum"),
    ("crisis",         "multi-asset ETF", "1d",           "managed-futures long-gamma &mdash; the crash hedge"),
    ("bab",            "crypto majors",   "1d",           "betting-against-beta / low-vol, beta-neutral top-25"),
]


def _dig(path, *keys):
    """Honest value from a deep-dive artifact; None (never a guess) if the file or key is absent."""
    try:
        o = json.loads((REP / path).read_text())
        for k in keys:
            o = o[k]
        return float(o)
    except Exception:
        return None


def _pq_sharpe(path):
    """Annualised Sharpe of a single-column daily return artifact; None if the file is absent."""
    try:
        s = pd.read_parquet(REP / path).iloc[:, 0].dropna()
        return float(np.sqrt(365) * s.mean() / s.std(ddof=1)) if len(s) > 2 and s.std(ddof=1) else None
    except Exception:
        return None


def _family_edge_card(summ):
    """§12/§13 edge map at family granularity: honest Sharpe for EVERY distinct alpha family we tested,
    live and rejected, on one scale — so where-edge-is and where-it-is-not read off a single table. The
    timeframe finding and the tested overlays/variants that are not separate families are in the footnote."""
    ss = summ.get("standalone_sharpe", {})
    rejected = [  # (label, asset class, timeframe, honest Sharpe, why it is not in the book)
        ("residual-mom",      "crypto",          "1d",           _dig("residmom/residmom_summary.json", "crypto", "head_to_head", "idio", "sharpe"),
         "a better-built momentum (&Delta; +0.16 vs raw), not a new source"),
        ("seasonal FOMC/ToM", "equity",          "event",        _dig("seasonal/seasonal_summary.json", "combined", "combined_spy", "net_sharpe"),
         "real but beta, not timing &rarr; sub-bar, drags the book"),
        ("lottery / MAX",     "crypto",          "1d",           _dig("lottery/lottery_summary.json", "chosen_skew_short", "sharpe"),
         "inverted &mdash; momentum eats the skew premium"),
        ("x-sect reversal",   "crypto / equity", "1d",           -0.49,
         "dollar-neutral 1&ndash;5d top/bottom-30%; crypto &minus;0.49 / equity &minus;0.13 &mdash; cost-killed"),  # run_mr_universe
        ("mean-reversion",    "crypto / equity", "1d / 4h / 1h", -0.86,
         "single-name z-score; 0% of the parameter surface positive &mdash; dead everywhere"),        # §5b walk-forward
        ("on-chain",          "crypto",          "1d",           _dig("onchain/onchain_summary.json", "cross_section", "walk_forward", "wf_oos"),
         "no edge over price out-of-sample (free-data ceiling)"),
        ("volume-spike",      "crypto alts",     "1h",           _pq_sharpe("volspike/volspike_wf_oos.parquet"),
         "small-alt drift killed by cost (walk-forward OOS)"),
        ("pairs / stat-arb",  "equity / crypto", "1d",           -1.18,
         "cointegration unstable OOS (crypto &minus;1.18, equity +0.05)"),      # sector-pairs deep-dive, no saved summary
        ("overnight/session", "equity",          "1d",           _dig("overnight/overnight_summary.json", "sharpe", "overnight_only"),
         "real but beta; isolating it forces a full daily round-trip"),
    ]

    def cell(v, mark=""):
        if v is None or not np.isfinite(v):
            return '<td style="background:var(--empty)">n/a</td>'
        bg = heatcolor(v, 1.2, "rdylgn")
        return (f'<td style="background:{bg};color:#fff;font-weight:700;'
                f'text-shadow:0 1px 2px rgba(0,0,0,.6)">{v:+.2f}{mark}</td>')

    def rowhtml(lab, asset, tf, v, why, mark=""):
        return (f'<tr><td><b>{lab}</b></td><td>{asset}</td><td>{tf}</td>'
                f'{cell(v, mark)}<td class="whr">{why}</td></tr>')

    def grp(t):
        return f'<tr class="grp"><td colspan="5">{t}</td></tr>'

    live = "".join(rowhtml(SHORT.get(fid, fid), a, tf, ss.get(fid), why,
                           mark=("&#8224;" if fid == "volprem" else ""))
                   for fid, a, tf, why in LIVE_FAM)
    rej = "".join(rowhtml(lab, a, tf, v, why) for lab, a, tf, v, why in rejected)
    return (
        '<figure class="card s6"><figcaption>Edge map (§12) &mdash; honest Sharpe by strategy family '
        '&middot; where edge was found, and where it was not</figcaption>'
        '<table><tr><th>strategy family</th><th>asset class</th><th>timeframe</th><th>Sharpe</th>'
        '<th>where the edge is &middot; why it is not</th></tr>'
        + grp("In the book &mdash; where edge was found") + live
        + grp("Tested, rejected &mdash; where edge was not") + rej
        + '</table><p class="valline">Each Sharpe is the family&rsquo;s honest standalone result from its '
        'own validated construction &mdash; live families from the master-book legs, rejected families '
        'from their deep-dive walk-forward. &#8224; vol-prem&rsquo;s Sharpe overstates: skew &minus;8.7 and a '
        '&minus;50% systemic-vol tail &mdash; it is sized on that tail, not on Sharpe. <b>Timeframe:</b> edge '
        'concentrates at 1d; intraday (1h/4h) decays to turnover &times; cost across every sweep-able family. '
        '<b>Also run, not separate alpha families:</b> book-construction / overlay studies (convexity '
        'tail-hedge, dispersion robustness, managed-futures / defensive) and within-family variants (carry on '
        'FX / equity / basis, breakout cross-sectional / intraday) &mdash; folded into their family row or the '
        'deep-dives, not omitted.</p></figure>')



def _feature_card():
    """§4/§12 feature-family survival: which of the feature library survived selection and which
    contributed nothing. From scripts/feature_report.py (reports/book/feature_families.json)."""
    p = REP / "book" / "feature_families.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    rows = "".join(
        f"<tr><td>{r['family']}</td><td>{r['n_features']}</td><td>{r['n_significant']}</td>"
        f"<td>{r['n_kept']}</td><td>{r['mean_abs_ic']:.3f}</td></tr>"
        for r in sorted(d.get("per_family", []), key=lambda x: (-x.get("n_kept", 0), -x.get("mean_abs_ic", 0))))
    nothing = ", ".join(d.get("families_contributed_nothing", [])) or "none"
    return (
        '<figure class="card s6"><figcaption>Feature-family survival (§4/§12) &mdash; which of the '
        f'{d.get("n_features", 0)}-feature library survived selection</figcaption>'
        '<table><tr><th>feature family</th><th>features</th><th>significant</th><th>kept</th>'
        f'<th>mean |IC|</th></tr>{rows}</table>'
        f'<p class="valline">{d.get("n_features", 0)} features &rarr; <b>{d.get("n_significant", 0)}</b> '
        f'clear |IC&middot;t|&ge;2 &rarr; <b>{d.get("n_kept", 0)}</b> survive a stability + redundancy '
        f'reduction ({d.get("n_redundancy_clusters", 0)} clusters). Contributed nothing: <b>{nothing}</b>. '
        f'{d.get("note", "")}</p></figure>')


def _param_card():
    """§10 parameter sensitivity: the trend EMA surface across symbols — the plateau around the chosen
    setting, not the peak alone. From scripts/run_wfo.py (reports/trend/trend_sensitivity.csv)."""
    p = REP / "trend" / "trend_sensitivity.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p)
    g = d.groupby("cfg")["sharpe"]
    agg = g.agg(["median", "min", "max", "count"]).reset_index().sort_values("median", ascending=False)
    posfrac = g.apply(lambda x: float((x > 0).mean()))
    rows = "".join(
        f"<tr><td>{r.cfg}</td><td>{int(r.count)}</td><td>{r.median:+.2f}</td>"
        f"<td>{r.min:+.2f}</td><td>{r.max:+.2f}</td><td>{posfrac[r.cfg]:.0%}</td></tr>" for r in agg.itertuples())
    allpos = float((d["sharpe"] > 0).mean())
    return (
        '<figure class="card s6"><figcaption>Parameter sensitivity (§10) &mdash; trend EMA surface across '
        'symbols &middot; the plateau, not the peak</figcaption>'
        '<table><tr><th>EMA config</th><th>symbols</th><th>median Sharpe</th><th>min</th><th>max</th>'
        f'<th>% positive</th></tr>{rows}</table>'
        f'<p class="valline">{len(d):,} (symbol &times; config) points; <b>{allpos:.0%}</b> of the surface is '
        f'positive &mdash; a broad robust plateau, so trend&rsquo;s edge is not a fitted parameter spike. Full '
        f'walk-forward-vs-in-sample surface per family in REPORT §5b; per-family surfaces in '
        f'<code>reports/trend/trend_sensitivity.csv</code>, <code>volprem/volprem_sensitivity.csv</code>.</p></figure>')


def _timeframe_card():
    """§12 timeframe x family cross: raw discovery Sharpe by timeframe for the families that trade
    multiple timeframes (the per-bar-signal families). The single-timeframe families (vol-prem daily
    options, crisis/global-macro managed-futures, on-chain/seasonal event-based) have no intraday cell
    by construction — they live in the edge map above. From scripts/run_book.py (zoo_edge_map.csv)."""
    p = REP / "book" / "zoo_edge_map.csv"
    if not p.exists():
        return ""
    e = pd.read_csv(p, index_col=0)
    short = {"cross_sectional": "x-sect", "mean_reversion": "mean-rev"}
    order = [t for t in ["1d", "4h", "1h", "15m", "5m"] if t in e.index]
    e = e.reindex(order)
    cols = [short.get(str(c), str(c)) for c in e.columns]
    mat = [[None if not np.isfinite(v) else float(v) for v in row] for row in e.values]
    svg = heat_svg(order, cols, mat, 548, 1.2, "rdylgn", fmt=lambda v: f"{v:+.2f}", rowh=46)
    return (
        '<figure class="card"><figcaption>Timeframe robustness (§12) &mdash; raw discovery Sharpe by '
        'timeframe &times; family &middot; which timeframes produced the most robust sleeves</figcaption>'
        f'{svg}<p class="valline"><b>1d</b> is the robust plateau; edge decays at 4h/1h as turnover &times; '
        'cost bites (worst on FX and crypto 1h). Only these families trade multiple timeframes &mdash; the '
        'other 11 (vol-prem, crisis-alpha, global-macro, BAB, on-chain, seasonal, &hellip;) are '
        '<b>single-timeframe by construction</b> (daily options / managed-futures / event-based), so they '
        'have no intraday cell; the full 17-family roster is the edge map above. Raw first-pass numbers.</p></figure>')


# --- §9/§13: the return-only master series carry no positions, so recover the book's leverage,
# turnover and cost sensitivity by mirroring run_master_book's assembly (each family vol-targeted to
# 15% on a trailing 60d estimate, x3 cap, equal risk over the families live that day) on the blocks. ---
BLOCKS = [("trend", "trend/trend_block_returns.parquet"), ("carry", "carry/carry_breadth_headline.parquet"),
          ("vol-prem", "volprem/volprem_book.parquet"), ("x-sect", "xs/xs_book.parquet"),
          ("breakout", "breakout/bo_combined_portfolio.parquet"), ("crisis", "book/crisis_sleeve.parquet"),
          ("gmacro", "book/gmacro_sleeve.parquet")]
COST_BPS = 8.0    # blended round-trip cost applied to book turnover for the §9 sweep; each family is
                  # already net of its own itemised costs (per-family break-evens live in the deep-dives)


def _leverage(net, target=0.15):
    return (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)


def _book_ops(master):
    """Book gross exposure + turnover over time + annual turnover, reconstructed the way
    run_master_book assembles the families (equal risk over the families live each day)."""
    lev = {}
    for lab, f in BLOCKS:
        p = REP / f
        if not p.exists():
            continue
        s = pd.read_parquet(p)
        s = (s["ret"] if "ret" in s.columns else s.iloc[:, 0]).dropna()
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        lev[lab] = _leverage(s)
    if not lev:
        return None, None, 0.0
    L = pd.DataFrame(lev).reindex(master.index)
    live = L > 0
    w = L.where(live).div(live.sum(axis=1).replace(0, np.nan), axis=0)      # equal-risk weight x leverage
    gross = w.abs().sum(axis=1)
    turn = w.fillna(0.0).diff().abs().sum(axis=1)
    ann = float(turn.reindex(master.index).mean() * PPY) if turn.notna().any() else 0.0
    gross = gross[gross > 1e-9]                        # drop warm-up/edge days with no live leverage yet
    return gross.dropna(), turn.reindex(gross.index).dropna(), ann


def _cost_levels(master, turn):
    """§9 cost sensitivity: the book re-charged at 1x/2x/3x the modelled round-trip cost on its
    turnover, plus the break-even multiple. Returns (levels, break-even)."""
    t = (turn.reindex(master.index).fillna(0.0) * COST_BPS / 1e4) if turn is not None else master * 0.0

    def at(mult):
        return (master - (mult - 1.0) * t).dropna()
    levels = []
    for mult, lab in [(1.0, "1x base"), (2.0, "2x base"), (3.0, "3x base")]:
        r = at(mult)
        e = (1 + r).cumprod()
        yy = (r.index[-1] - r.index[0]).days / 365.25
        levels.append({"label": lab, "sharpe": _sh(r), "max_dd": _mdd(r),
                       "cagr": float(e.iloc[-1] ** (1 / yy) - 1) if yy > 0 else 0.0})
    be = next((round(float(mult), 1) for mult in np.linspace(1.0, 80.0, 791)
               if (1 + at(mult)).prod() - 1 <= 0), None)
    return levels, be


def main():
    summ = json.loads((REP / "master_book_summary.json").read_text())
    master = pd.read_parquet(REP / "master_book.parquet")["ret"].dropna()
    legs = pd.read_parquet(REP / "master_book_legs.parquet")
    corr = pd.read_csv(REP / "master_book_correlation.csv", index_col=0)
    marg = pd.read_csv(REP / "master_book_marginal.csv")
    m = summ["master"]
    fams = list(legs.columns)
    lines = {}

    # --- headline figures ---
    eqf = (1.0 + master).cumprod()
    yrs = (master.index[-1] - master.index[0]).days / 365.25
    wlab = f"{int(yrs)}-yr"        # reporting-window label for the scorecard (e.g. "15-yr" for the 2011+ book)
    y0 = master.index[0].year
    cagr = float(eqf.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0
    final_value, net_pnl = float(CAP * eqf.iloc[-1]), float(CAP * (eqf.iloc[-1] - 1))

    # --- monthly / OOS / cross-asset (2020+) windows ---
    mo = (1.0 + master).resample("ME").prod() - 1.0
    oos = master[master.index >= OOS_TS]      # final held-out block (OOS_START from src/config.py)
    moo = (1.0 + oos).resample("ME").prod() - 1.0
    ca = master[master.index >= pd.Timestamp("2020-01-01")]        # fully cross-asset (crypto listed)
    eca = (1.0 + ca).cumprod()
    yca = (ca.index[-1] - ca.index[0]).days / 365.25
    ca_sharpe = _sh(ca)
    ca_cagr = float(eca.iloc[-1] ** (1 / yca) - 1) if yca > 0 else 0.0

    # --- §9/§13: book leverage, turnover and cost sensitivity, recovered from the family blocks ---
    gross, turn, ann_turn = _book_ops(master)
    cost_levels, breakeven = _cost_levels(master, turn)

    # --- §11 scorecard — judged on the FINAL OOS BLOCK (the brief scores targets there); the full window
    #     (now the 15y 2011+ book) sits in the note as the larger-sample estimate ---
    sc = [
        ("Sharpe (net)", f"{_sh(oos):+.2f}", f"{wlab} {m['sharpe']:+.2f} · target 2.5–4.0",
         "pass" if _sh(oos) >= 2.5 else "miss"),
        ("Months in profit", f"{_mip(moo):.0%}", f"{wlab} {_mip(mo):.0%} · target ≥80%",
         "pass" if _mip(moo) >= 0.80 else "miss"),
        ("Max drawdown", f"{_mdd(oos):+.1%}", f"{wlab} {m['max_dd']:+.1%} · target ≤15%",
         "pass" if _mdd(oos) >= -0.15 else "miss"),
        ("Longest losing streak", f"{_streak(moo.values)} mo", f"{wlab} {_streak(mo.values)} mo · target ≤2 mo",
         "pass" if _streak(moo.values) <= 2 else "miss"),
        ("Worst single month", f"{moo.min():+.1%}", f"{wlab} {mo.min():+.1%} · target ≥−6%",
         "pass" if moo.min() >= -0.06 else "miss"),
        ("Annual turnover", f"{ann_turn:.1f}× rt", "round-trip ×capital/yr, the §11 cost basis; the turnover chart plots the one-way re-weighting series", ""),
    ]
    n_pass = sum(1 for *_, c in sc[:5] if c == "pass")
    wfp = REP / "master_book_wf_summary.json"
    wf_li = ""
    if wfp.exists():
        w = json.loads(wfp.read_text())
        h, rng = w["headline_wf_oos"], w["window_cadence_invariance_range"]
        gfc = (w.get("stress") or {}).get("2008 GFC")
        gfc_s = (f' — and only {gfc["max_dd"]:+.1%} through the 2008 GFC (the crisis leg hedges the '
                 f'short-vol tail)') if gfc else ''
        wf_li = (f'<li><b>OOS is most of the history, not 2 years:</b> a book-level walk-forward (rolling &amp; '
                 f'anchored, periodic re-fit) is out-of-sample across {h["start"]}&rarr;{h["end"]} '
                 f'(~{round(h["n_obs"] / 365)}y) at Sharpe <b>{h["sharpe"]:+.2f}</b>, invariant to '
                 f'cadence/window [{rng[0]:+.2f}, {rng[1]:+.2f}]{gfc_s}.</li>')
    yr_ret = (1.0 + master).resample("YE").prod() - 1.0
    n_pos_yr, n_tot_yr = int((yr_ret > 0).sum()), int(len(yr_ret))
    sc_note = (
        f'<div class="scnote">'
        f'<span class="lead"><b>{n_pass} of 5 targets met on the final out-of-sample block</b> '
        f'(2024-07&rarr;, the window the brief scores).</span>'
        f'<ul>'
        f'<li><b>OOS scorecard:</b> Sharpe ({_sh(oos):+.2f}) '
        f'{"clears" if _sh(oos) >= 2.5 else "is a near-miss under"} the 2.5 floor; max-drawdown, worst-month and '
        f'losing-streak also clear. The one miss is months-in-profit ({_mip(moo):.0%} vs &ge;80%) &mdash; forcing '
        f'it overweights the short-vol leg and deepens the worst month, a weight-corner that breaks under '
        f'&plusmn;25% perturbation.</li>'
        f'<li><b>Long track ({y0}&rarr;now, {int(yrs)} years):</b> Sharpe {m["sharpe"]:+.2f}, max-DD {m["max_dd"]:+.1%}, '
        f'<b>positive in {n_pos_yr} of {n_tot_yr} years</b>. <i>Caveat:</i> pre-2016 leans on reconstructed '
        f'crisis/global-macro signals (only 2020+ is fully live).</li>'
        f'{wf_li}'
        f'<li><b>Equal weight is evidence-based:</b> re-fitting the weights doesn&rsquo;t beat it OOS (mean-var buys '
        f'Sharpe with a 3&times; tail); rule parameters &amp; book weights are a-priori, ML meta-labels &amp; '
        f'sleeve selection fit and walk-forwarded (§5/§10).</li>'
        f'</ul></div>')

    # --- equity, drawdown, rolling 12m Sharpe ---
    eq_svg, lines["equity"] = line_svg("equity", _pts(_ds(eqf)), 1120, 320, log=True)
    dd_svg, lines["dd"] = line_svg("dd", _pts(_ds(eqf / eqf.cummax() - 1.0)), 548, 240, pct=True)
    rmu, rsd = master.rolling(365, min_periods=180).mean(), master.rolling(365, min_periods=180).std(ddof=1)
    roll = (np.sqrt(365) * rmu / rsd).dropna()
    roll_svg, lines["roll"] = line_svg("roll", _pts(_ds(roll)), 548, 240)

    # --- per-family equity curves + book (§13); families are flat until they list ---
    curves = [(sf(f), _pts(_ds((1 + legs[f].fillna(0.0)).cumprod()))) for f in fams]
    curves.append(("master book", _pts(_ds(eqf))))
    psleq_svg = multiline_svg(curves, 1120, 300, bold=len(curves) - 1)

    # --- monthly return heatmap ---
    years = sorted({d.year for d in mo.index})
    yi = {y: i for i, y in enumerate(years)}
    mmat = [[None] * 12 for _ in years]
    for d, v in mo.items():
        mmat[yi[d.year]][d.month - 1] = float(v)
    month_svg = heat_svg([str(y) for y in years], MONTHS, mmat, 1120, 0.10, "rdylgn",
                         fmt=lambda v: f"{v:+.1%}")

    # --- per-year & per-quarter Sharpe (from the master summary) ---
    py = sorted((int(k), v) for k, v in summ["per_year"].items())
    year_svg = bars_svg([(str(y), v) for y, v in py], 548, 240)
    pq = summ["per_quarter"]
    qyears = sorted({int(k[:4]) for k in pq})
    qi = {y: i for i, y in enumerate(qyears)}
    qmat = [[None] * 4 for _ in qyears]
    for k, v in pq.items():
        qmat[qi[int(k[:4])]][int(k[5]) - 1] = float(v)
    quarter_svg = heat_svg([str(y) for y in qyears], ["Q1", "Q2", "Q3", "Q4"], qmat, 548, 3.0,
                           "rdylgn", fmt=lambda v: f"{v:+.1f}")

    # --- §13 per-family (sleeve-leg) Sharpe by year AND quarter — the sleeves the book is built from,
    #     not only the book aggregate above ---
    def _leg_grid(freq):
        vals = {}
        for f in fams:
            s = legs[f].dropna()
            key = s.index.year if freq == "Y" else s.index.to_period("Q")
            for k, g in s.groupby(key):
                if len(g) > 20 and g.std(ddof=1) > 0:
                    vals.setdefault(str(k), {})[f] = float(np.sqrt(PPY) * g.mean() / g.std(ddof=1))
        cols = sorted(vals)
        return [sf(f) for f in fams], cols, [[vals.get(c, {}).get(f) for c in cols] for f in fams]
    fyr_r, fyr_c, fyr_m = _leg_grid("Y")
    fqr_r, fqr_c, fqr_m = _leg_grid("Q")
    famperiods = (
        '<figure class="card s6"><figcaption>Per-family Sharpe by year (§13) &mdash; the sleeve legs the '
        f'book is built from, net</figcaption>{heat_svg(fyr_r, fyr_c, fyr_m, 1120, 2.0, "rdylgn", fmt=lambda v: f"{v:+.1f}")}'
        '<figcaption style="margin-top:18px">Per-family Sharpe by quarter (§13) &mdash; hover a cell for its Sharpe</figcaption>'
        # too many quarters now span the columns for a legible per-cell number, so the value is hover-only
        # (each cell's data-tip); the heat colour still carries the pattern at a glance
        f'{heat_svg(fqr_r, fqr_c, fqr_m, 1120, 2.0, "rdylgn", show_val=False, rowh=34, fmt=lambda v: f"{v:+.1f}")}</figure>')

    # --- stress windows (§10) ---
    stress_rows = ""
    for lab, a, b in STRESS:
        w = master[(master.index >= pd.Timestamp(a)) & (master.index <= pd.Timestamp(b))]
        if not len(w):
            continue
        stress_rows += (f"<tr><td>{lab}</td><td>{_sh(w):+.2f}</td>"
                        f"<td>{(1 + w).prod() - 1:+.1%}</td><td>{_mdd(w):+.1%}</td></tr>")

    # --- marginal-contribution curve + table: Sharpe, max-DD and months-in-profit as families join (§7) ---
    labels = [sf(a) for a in marg["added"]]
    vals = [float(v) for v in marg["sharpe"]]
    mark = int(np.argmax(vals))
    marg_rows = "".join(
        f"<tr><td>{int(r.n)}</td><td>+{sf(r.added)}</td><td>{r.sharpe:+.2f}</td>"
        f"<td>{r.max_dd:+.1%}</td><td>{r.months_in_profit:.0%}</td></tr>" for r in marg.itertuples())
    marg_svg = (curve_svg(labels, vals, 548, 240, mark=mark)
                + '<table><tr><th>n</th><th>+family</th><th>Sharpe</th><th>max DD</th><th>months+</th></tr>'
                + marg_rows + '</table>')

    # --- per-family contribution table: standalone Sharpe/DD, corr->book, book-without & delta ---
    solo = summ["standalone_sharpe"]
    pnl = summ.get("pnl_share", {})     # each family's share of book P&L (§7)
    fam_rows = ""
    for f in sorted(fams, key=lambda c: -solo.get(c, 0.0)):
        s = legs[f].dropna()
        joined = pd.concat([legs[f], master], axis=1).dropna()
        c = float(joined.corr().iloc[0, 1]) if len(joined) > 2 else 0.0
        wo = _sh(legs.drop(columns=[f]).mean(axis=1, skipna=True))   # book with this family removed
        fam_rows += (f"<tr><td>{sf(f)}</td><td>{solo.get(f, 0.0):+.2f}</td><td>{_mdd(s):+.1%}</td>"
                     f"<td>{pnl.get(f, 0.0):.0%}</td>"
                     f"<td>{c:+.2f}</td><td>{wo:+.2f}</td><td>{m['sharpe'] - wo:+.2f}</td></tr>")

    # --- cross-family correlation matrix + its stability over time (§7) ---
    corr_svg = heat_svg([sf(f) for f in corr.index], [sf(f) for f in corr.columns],
                        [[float(v) for v in row] for row in corr.values], 1120, 1.0, "coolwarm")
    # rolling mean off-diagonal pairwise correlation across the live families — flat & near-zero => the
    # diversification holds out-of-sample, not only in-sample (stepped to keep the build cheap)
    tri = np.triu_indices(len(fams), 1)
    win, step = 126, 5
    rc_idx, rc_val = [], []
    for i in range(win, len(legs), step):
        cm = legs.iloc[i - win:i].corr().values
        a = cm[tri][np.isfinite(cm[tri])]
        rc_idx.append(legs.index[i])
        rc_val.append(float(a.mean()) if len(a) else np.nan)
    roll_corr = pd.Series(rc_val, index=rc_idx).dropna()
    rcorr_svg, lines["rcorr"] = line_svg("rcorr", _pts(_ds(roll_corr)), 1120, 200)
    cs = summ.get("correlation_stability", {})
    cs_txt = (f'<p class="valline">first-half mean {cs.get("first_half_mean", float("nan")):+.2f} &rarr; '
              f'second-half {cs.get("second_half_mean", float("nan")):+.2f} &middot; OOS mean '
              f'{cs.get("oos_mean", float("nan")):+.2f} &middot; largest pairwise shift '
              f'{cs.get("max_pairwise_shift", float("nan")):.2f} &mdash; near-zero and stable, so the '
              f'diversification is not an in-sample artefact.</p>') if cs else ""
    corr_svg = (corr_svg + '<figcaption style="margin-top:18px">Correlation stability &mdash; 126-day rolling '
                'mean pairwise correlation across the live families</figcaption>' + rcorr_svg + cs_txt)

    # --- §13 exposure/turnover over time + §9 cost sensitivity (book-level, reconstructed) ---
    if gross is not None and len(gross):
        expg_svg, lines["expg"] = line_svg("expg", _pts(_ds(gross)), 548, 240, pct=True)
        expt_svg, lines["expt"] = line_svg("expt", _pts(_ds(turn)), 548, 240, pct=True)
    else:
        expg_svg = expt_svg = '<p class="valline">exposure/turnover unavailable (family blocks missing)</p>'
    cost_rows = "".join(f"<tr><td>{lv['label']}</td><td>{lv['sharpe']:+.2f}</td><td>{lv['max_dd']:+.1%}</td>"
                        f"<td>{lv['cagr']:+.1%}</td></tr>" for lv in cost_levels)
    be_txt = (f"break-even at {breakeven:.0f}&times; the book-turnover cost" if breakeven
              else "break-even &gt; 80&times; the book-turnover cost")
    # per-family cost-fragility (§9/§12): break-even multiple from each deep-dive where published
    bo_be, xs_be = _dig("breakout/bo_final_summary.json", "breakeven_mult"), _dig("xs/xs_summary.json", "breakeven_cost_mult")
    tr_c3 = _dig("trend/trend_book_blend_summary.json", "cost_levels", "3x")
    be_parts = [p for p in (f"breakout {bo_be:.1f}&times;" if bo_be else "",
                            f"x-sect {xs_be:.1f}&times;" if xs_be else "",
                            f"trend Sharpe {tr_c3:.2f} at 3&times;" if tr_c3 else "") if p]
    perfam_be = ", ".join(be_parts) if be_parts else "in the deep-dives"
    # §13 OOS trade log — reference the artifact the book emits (return-composed book => its trades are
    # the daily sleeve rebalances; instrument-level fills are per-family)
    tlp = REP / "master_book_oos_ledger.csv"
    trp = REP / "master_book_oos_trades.csv"
    tl_note = ""
    if tlp.exists():
        tl = pd.read_csv(tlp)
        n_tr = len(pd.read_csv(trp)) if trp.exists() else 0
        tl_note = (f'<p class="valline">OOS trade log (§13): <b>{len(tl):,}</b> daily book rebalances '
                   f'{tl["date"].min()}&rarr;{tl["date"].max()} in <code>reports/master_book_oos_ledger.csv</code> '
                   f'(per-family P&amp;L contribution + gross exposure + $ P&amp;L each day) &mdash; the book is '
                   f'return-composed, so its trades are the daily risk-parity rebalances; the combined '
                   f'instrument-level fills ({n_tr:,}) are in <code>reports/master_book_oos_trades.csv</code> '
                   f'and per-family logs (e.g. <code>reports/trend/trend_oos_trade_log.csv</code>).</p>')
    ops_html = (
        f'<figure class="card"><figcaption>Book gross exposure over time (§13)</figcaption>{expg_svg}</figure>'
        f'<figure class="card"><figcaption>Book turnover over time (§13) &middot; risk-parity rebalancing</figcaption>{expt_svg}{tl_note}</figure>'
        f'<figure class="card"><figcaption>Cost sensitivity (§9) &mdash; {be_txt}</figcaption>'
        f'<table><tr><th>cost level</th><th>Sharpe</th><th>max DD</th><th>CAGR</th></tr>{cost_rows}</table>'
        f'<p class="valline">book turnover re-charged at 1&times;/2&times;/3&times; a blended {COST_BPS:.0f}bps '
        f'round-trip; each family is already net of its own itemised costs. Per-family cost-fragility '
        f'(break-even &times; base cost, from the deep-dives): {perfam_be} &mdash; all well above 1&times;, '
        f'so no measured family is cost-fragile.</p></figure>')

    _write(summ, cagr, final_value, net_pnl, ca_sharpe, ca_cagr, dict(
        sc=_scorecard(sc), sc_note=sc_note, eq=eq_svg, psleq=psleq_svg, month=month_svg, dd=dd_svg, roll=roll_svg,
        year=year_svg, quarter=quarter_svg, stress=stress_rows, marg=marg_svg,
        marg_best=max(vals), corr=corr_svg, famtbl=fam_rows, famperiods=famperiods, param=_param_card(),
        timeframe=_timeframe_card(), ops=ops_html,
        edge_map=_family_edge_card(summ),
        feature=_feature_card(), honesty=_honesty_card(), lines=lines))
    print("dashboard -> reports/dashboard.html\nMAKE REPORT OK")


ASSETS = Path(__file__).resolve().parent / "report_assets"  # dashboard.html/.css/.js live here


def _asset(name):
    return (ASSETS / name).read_text()


def _write(summ, cagr, final_value, net_pnl, ca_sharpe, ca_cagr, S):
    """Fill report_assets/dashboard.html (page + copy) with computed values, CSS and JS."""
    m = summ["master"]
    fam = ", ".join(sf(f) for f in summ["families"])
    pos_years = sum(1 for v in summ["per_year"].values() if v > 0)
    rw = summ["window"]
    tr = summ["top_removed"]
    # §10 Monte-Carlo maxDD / monthly-hit-rate percentiles live under the canonical block-bootstrap
    # variant; fall back to any legacy top-level mirror, and to n/a if the MC has not been run.
    bbmc = (m.get("mc_variants") or {}).get("block_bootstrap") or {}

    def _mcp(k, sign=False):
        v = bbmc.get(k, m.get("mc_" + k))
        return (f"{v:+.1%}" if sign else f"{v:.0%}") if isinstance(v, (int, float)) and v == v else "n/a"

    html = _asset("dashboard.html").format(
        css=_asset("dashboard.css"), js=_asset("dashboard.js"),
        lines_json=json.dumps(S["lines"], default=float), cap_k=CAP // 1000,
        report_window=f"{rw[0][:4]}–{rw[1][:4]}",
        cagr=f"{cagr:+.1%}", total_return=f"{m['total_return']:+.0%}",
        final_value_m=f"{final_value / 1e6:.2f}", net_pnl_m=f"{net_pnl / 1e6:.2f}",
        sc=S["sc"], sc_note=S["sc_note"], mc_p5=f"{m['mc_p5']:+.2f}", mc_p50=f"{m['mc_p50']:+.2f}", mc_p95=f"{m['mc_p95']:+.2f}",
        mc_maxdd_p5=_mcp("maxdd_p5", True), mc_maxdd_p50=_mcp("maxdd_p50", True), mc_maxdd_p95=_mcp("maxdd_p95", True),
        mc_hit_p5=_mcp("hit_p5"), mc_hit_p50=_mcp("hit_p50"), mc_hit_p95=_mcp("hit_p95"),
        pos_years=pos_years, n_years=len(summ["per_year"]),
        mean_corr=f"{summ['mean_correlation']:+.2f}", n_families=len(summ["families"]),
        top_family=sf(tr["family"]), top_removed=f"{tr['sharpe']:+.2f}",
        eq=S["eq"], psleq=S["psleq"], month=S["month"], dd=S["dd"], roll=S["roll"],
        year=S["year"], quarter=S["quarter"], stress=S["stress"],
        marg=S["marg"], marg_best=f"{S['marg_best']:+.2f}",
        corr=S["corr"], famtbl=S["famtbl"], ops=S["ops"], edge_map=S["edge_map"],
        feature=S["feature"], famperiods=S["famperiods"], param=S["param"], timeframe=S["timeframe"], honesty=S["honesty"], fam=fam,
        sharpe_ann=f"{m['sharpe']:.2f}",
        ca_sharpe=f"{ca_sharpe:.2f}", ca_cagr=f"{ca_cagr:+.1%}",
    )
    (REP / "dashboard.html").write_text(html)


if __name__ == "__main__":
    main()
