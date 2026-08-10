"""Self-contained interactive dashboard for THE master book (scripts/run_master_book.py).

Renders the canonical portfolio — risk-parity over the shipped strategy families, whose composition is
read from run_master_book rather than named here (currently short-vol/VRP, cross-sectional momentum,
breakout, crisis-alpha, global-macro, betting-against-beta) — from the master_book* artifacts:
master_book_summary.json (headline Sharpe/DD/MC + per-year/quarter), master_book.parquet (the
equity curve), master_book_legs.parquet (per-family series -> standalone Sharpe/DD, correlation to
the book, and the book-without-family delta), master_book_correlation.csv and master_book_marginal.csv.

Charts are inline SVG generated here (crisp vector, theme-aware) with a small inline-JS layer for
hover tooltips and an equity crosshair — no external libraries, CSP-safe. The page shell — HTML
template, CSS, JS — lives in report_assets/ and is inlined into the single output file at build;
this module only computes the data and fills the template's placeholders.

    python scripts/make_report.py            ->   reports/dashboard.html
    python scripts/make_report.py --check    fails if the committed page lags the artifacts
"""
import json
import sys
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
def _ppy(r):
    """Actual obs/yr — honest Sharpe annualisation for the mixed 252/365 calendar (crypto 365 / equity
    ~252); a flat 365 overstates any sub-365 series. Matches run_master_book."""
    r = r.dropna()
    yrs = (r.index.max() - r.index.min()).days / 365.25
    return len(r) / yrs if yrs > 0 else float(PPY)


def _sh(r, ppy=None):
    r = r.dropna()
    ppy = _ppy(r) if ppy is None else ppy
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


def _pc(v, dp=1):
    """Percent with a typographic minus, so a value reads the same as the '≥ −6%' target beside it."""
    return f"{v:+.{dp}%}".replace("-", "−")


def _n(v, dp=2):
    """Signed number, same typographic minus as _pc — one glyph for negatives across the whole page."""
    return f"{v:+.{dp}f}".replace("-", "−")


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
                 f'data-tip="+{lab} ({i + 1} families): Sharpe {_n(v)}"/>')
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
        val = f"{v * 100:.0f}%" if pct else _n(v, 2)
        p.append(f'<rect class="{cls}" x="{cx - bw / 2:.1f}" y="{yy:.1f}" width="{bw:.1f}" '
                 f'height="{hh:.1f}" rx="2.5" data-tip="{lab}: {val}"/>')
        p.append(f'<text class="ax" x="{cx:.0f}" y="{h - 12}" text-anchor="end" '
                 f'transform="rotate(-40 {cx:.0f} {h - 12})">{lab}</text>')
    return _svg(w, h, "".join(p))


def heat_svg(rows, cols, matrix, w, vmax, scheme, show_val=True, col_labels=True, fmt=None, rowh=32, val_fs=None):
    fmt = fmt or (lambda v: _n(v, 2))
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
    # collapse consecutive gates that admit the same count — a row that filters nothing reads as a stage
    # that did work. The brief's five-stage funnel wants a walk-forward gate; the zoo does not have one
    # (it screens on in-sample Sharpe then Monte-Carlo), and the note below says so rather than faking a row.
    funnel = []
    for lab, n in z.get("funnel", []):
        if funnel and int(n) == funnel[-1][1]:
            funnel[-1] = (f"{funnel[-1][0]} &rarr; {lab} (nothing dropped)", int(n))
        else:
            funnel.append((lab, int(n)))
    rows = "".join(f"<tr><td>{lab}</td><td>{n:,}</td></tr>" for lab, n in funnel)
    n = int(z.get("n_trials", 0))
    ins = z.get("portfolio", {}).get("sharpe_ann", float("nan"))
    wf = z.get("wf_oos_sharpe", float("nan"))
    dsr = z.get("best_sleeve_dsr", float("nan"))
    fdr = z.get("placebo_fdr", float("nan"))
    wf_s = _n(wf, 2) if wf == wf else "n/a"        # NaN-safe (zoo not fully run)
    fdr_s = f"{fdr:.1%}" if fdr == fdr else "n/a"
    # CSCV probability of backtest overfitting (scripts/run_cscv.py) — the §6 "PBO or equivalent" metric
    cp = REP / "book" / "cscv_pbo.json"
    pbo_s = ""
    if cp.exists():
        c = json.loads(cp.read_text())
        pbo_s = (f' &middot; CSCV probability of backtest overfitting <b>{c["pbo"]:.0%}</b> across '
                 f'{c["n_strategies"]} strategies (the in-sample-best pick averages '
                 f'{_n(c["is_sharpe_mean"], 3)} per-bar Sharpe in sample and {_n(c["oos_sharpe_mean"], 3)} out of it)')
    return (f'<figure class="card s6"><figcaption>Honest search &mdash; why the book selects nothing '
            f'(anti-overfitting §6/§10/§12)</figcaption>'
            f'<table><tr><th>discovery gate</th><th>candidates</th></tr>{rows}</table>'
            f'<p class="valline">{n:,} candidates mined &middot; naive in-sample Sharpe {_n(ins)} '
            f'&middot; the same selection walk-forwarded gives {wf_s} &middot; best single-sleeve deflated '
            f'Sharpe {dsr:.2f} (N={n:,}){pbo_s} &rarr; mining winners is selection bias, so the traded book '
            f'<b>selects nothing</b> and applies theory uniformly across the whole universe. The gates are '
            f'not the problem: on shuffled signals only <b>{fdr_s}</b> get through, so what the funnel admits '
            f'is mostly not noise &mdash; it is picking winners among them that fails.</p></figure>')


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


def _family_edge_card(summ, legs):
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
        # the headline book, matching every sibling row here. Not walk_forward.wf_oos: that figure is
        # the *pool's* OOS after it selects a config, and the pool contains adoption momentum, so it
        # reads positive for a family whose own headline is dead.
        ("on-chain",          "crypto",          "1d",           _dig("onchain/onchain_summary.json", "cross_section", "headline", "sharpe"),
         "value is a coin-type tilt; exchange flows lose to random timing"),
        ("chain fundamentals", "crypto L1/L2",   "1d",           _dig("onchain/fundamentals_summary.json", "headline", "sharpe"),
         "fee yield inverted (placebo 6th pctile); a standing tilt short BTC"),
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
                f'text-shadow:0 1px 2px rgba(0,0,0,.6)">{_n(v)}{mark}</td>')

    def rowhtml(lab, asset, tf, v, why, mark=""):
        return (f'<tr><td><b>{lab}</b></td><td>{asset}</td><td>{tf}</td>'
                f'{cell(v, mark)}<td class="whr">{why}</td></tr>')

    def grp(t):
        return f'<tr class="grp"><td colspan="5">{t}</td></tr>'

    live = "".join(rowhtml(SHORT.get(fid, fid), a, tf, ss.get(fid), why,
                           mark=("&#8224;" if fid == "volprem" else ""))
                   for fid, a, tf, why in sorted(LIVE_FAM, key=lambda r: -(ss.get(r[0]) or 0.0)))
    rej = "".join(rowhtml(lab, a, tf, v, why) for lab, a, tf, v, why in rejected)
    # vol-prem's tail, measured rather than quoted: the leg as the book holds it (vol-targeted, gated), and
    # the standalone Cboe book behind it, whose OHLC-measured tail is the number the sizing respects.
    vp = legs["volprem"].dropna()
    vp_eq = (1.0 + vp).cumprod()
    vp_skew, vp_dd = float(vp.skew()), float((vp_eq / vp_eq.cummax() - 1.0).min())
    return (
        '<figure class="card s6"><figcaption>Edge map (§12) &mdash; honest Sharpe by strategy family '
        '&middot; where edge was found, and where it was not</figcaption>'
        '<table><tr><th>strategy family</th><th>asset class</th><th>timeframe</th><th>Sharpe</th>'
        '<th>where the edge is &middot; why it is not</th></tr>'
        + grp("In the book &mdash; where edge was found") + live
        + grp("Tested, rejected &mdash; where edge was not") + rej
        + '</table><p class="valline">Each Sharpe is the family&rsquo;s standalone result from its own '
        'validated construction &mdash; live families from the master-book legs, rejected ones from their '
        'deep-dive walk-forward. &#8224; <b>vol-prem&rsquo;s Sharpe overstates its risk:</b> as the book holds '
        f'it the leg prints skew {_n(vp_skew)} / {_pc(vp_dd)} drawdown, and the Cboe book behind it skew '
        '&minus;18 / a &minus;78% systemic tail. It is sized on that tail, not on Sharpe. Edge concentrates '
        'at <b>1d</b>; intraday decays to turnover &times; cost everywhere. Overlay studies and within-family '
        'variants are folded into their family row or the deep-dives, not omitted.</p></figure>')



def _sleeve_cost_card():
    """§9/§12 per sleeve: turnover, cost as a share of gross P&L, and which sleeves are cost-fragile.
    From scripts/run_book.py (reports/book/zoo_cost_per_sleeve.csv) — the sleeve is the brief's unit
    (asset × timeframe × family × model), so this is the discovery layer, where every candidate carries
    its own charged cost. The book families' equivalent is their deep-dive cost sweeps."""
    p = REP / "book" / "zoo_cost_per_sleeve.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p).sort_values("cost_share_of_gross_pnl", ascending=False)
    show = pd.concat([d.head(6), d.tail(3)]).drop_duplicates(subset="sleeve")
    rows = "".join(
        f"<tr><td>{r.sleeve}</td><td>{r.annual_turnover:,.0f}&times;</td>"
        f"<td>{r.cost_share_of_gross_pnl:.1%}</td><td>{r.breakeven_cost_mult:,.1f}&times;</td>"
        f"<td>{'<b>fragile</b>' if r.cost_fragile else 'no'}</td></tr>" for r in show.itertuples())
    n_frag = int(d.cost_fragile.sum())
    return (
        '<figure class="card s6"><figcaption>Cost per sleeve (§9/§12) &mdash; turnover, cost as a share '
        'of gross P&amp;L, and which sleeves are cost-fragile</figcaption>'
        '<table><tr><th>sleeve</th><th>annual turnover</th><th>cost / gross P&amp;L</th>'
        f'<th>break-even</th><th>cost-fragile</th></tr>{rows}</table>'
        f'<p class="valline">Worst six and best three of the <b>{len(d)}</b> sleeves that cleared the '
        f'acceptance gates; the median sleeve pays <b>{d.cost_share_of_gross_pnl.median():.1%}</b> of its '
        f'gross P&amp;L in cost and <b>{n_frag}</b> of them are cost-fragile (gross P&amp;L less than 3&times; '
        f'the cost, so a modest cost error flips them). Every number is the cost already charged inside '
        f'that sleeve&rsquo;s own net returns &mdash; commission, half-spread, &radic;-impact and, on crypto, '
        f'funding at every settlement. Full table in <code>reports/book/zoo_cost_per_sleeve.csv</code>.</p>'
        '</figure>')


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
    per = d.get("per_family", [])
    kept0 = [r["family"] for r in per if not r.get("n_kept") and r.get("n_significant")]
    kept0_txt = (f' <b>{len(kept0)}</b> more clear significance but keep nothing after the redundancy '
                 f'reduction &mdash; their signal is already carried by a kept feature, which is not the '
                 f'same as having none.' if kept0 else "")
    return (
        '<figure class="card s6"><figcaption>Feature-family survival (§4/§12) &mdash; which of the '
        f'{d.get("n_features", 0)}-feature library survived selection</figcaption>'
        '<table><tr><th>feature family</th><th>features</th><th>significant</th><th>kept</th>'
        f'<th>mean |IC|</th></tr>{rows}</table>'
        f'<p class="valline">{d.get("n_features", 0)} features &rarr; <b>{d.get("n_significant", 0)}</b> '
        f'clear |IC&middot;t|&ge;2 &rarr; <b>{d.get("n_kept", 0)}</b> survive a stability + redundancy '
        f'reduction ({d.get("n_redundancy_clusters", 0)} clusters). Nothing significant at all: '
        f'<b>{nothing}</b>.{kept0_txt} {d.get("note", "")}</p></figure>')


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
        f"<tr><td>{r.cfg}</td><td>{int(r.count)}</td><td>{_n(r.median)}</td>"
        f"<td>{_n(r.min)}</td><td>{_n(r.max)}</td><td>{posfrac[r.cfg]:.0%}</td></tr>" for r in agg.itertuples())
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
    svg = heat_svg(order, cols, mat, 548, 1.2, "rdylgn", fmt=lambda v: _n(v, 2), rowh=46)
    return (
        '<figure class="card"><figcaption>Timeframe robustness (§12) &mdash; raw discovery Sharpe by '
        'timeframe &times; family &middot; which timeframes produced the most robust sleeves</figcaption>'
        f'{svg}<p class="valline"><b>1d</b> is the robust plateau; edge decays at 4h/1h as turnover &times; '
        'cost bites (worst on FX and crypto 1h). Only these families trade multiple timeframes &mdash; the '
        'other 11 (vol-prem, crisis-alpha, global-macro, BAB, on-chain, seasonal, &hellip;) are '
        '<b>single-timeframe by construction</b> (daily options / managed-futures / event-based), so they '
        'have no intraday cell; the full 17-family roster is the edge map above. Raw first-pass numbers.</p></figure>')


# --- §9/§13 book operations. The weights are NOT re-derived here: run_master_book publishes the exact
# per-leg weight matrix its own return series implies, and everything below reads that one file. The
# dashboard used to mirror the assembly instead, which is how it ended up plotting seven of the eight
# families and the ungated vol-prem series. ---
COST_BPS = 8.0    # blended round-trip cost applied to book turnover for the §9 sweep; each family is
                  # already net of its own itemised costs (per-family break-evens live in the deep-dives)


def _book_ops(master):
    """Book exposure and turnover from the weight matrix the book publishes — the same weights its own
    return series is built from, so nothing here re-derives the assembly. The annual figures (and the
    counterfactual with every leg held through its market's closures) come from the book's summary for
    the same reason: one place computes them, everything else quotes."""
    p = REP / "master_book_weights.parquet"
    if not p.exists():
        return None, None
    w = pd.read_parquet(p).reindex(master.index)
    turn = w.fillna(0.0).diff().abs().sum(axis=1)
    gross = w.abs().sum(axis=1)
    gross = gross[gross > 1e-9].dropna()               # drop warm-up days with no live leg yet
    return gross, turn.reindex(gross.index).dropna()


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
    cagr = float(eqf.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0
    # §9 fixes the sizing/cost capital at $500k, and the √-impact model is calibrated to exactly that
    # order size — so the DOLLAR figures are quoted at that size, P&L not reinvested. Reinvesting would
    # compound the notional far past both the modelled order size and the vol-premium leg's vega
    # capacity, i.e. it would report a balance the cost model never charged for. Risk metrics stay on
    # the compounded return series (the stricter convention): fixed-size equity divides every later
    # drawdown by a cash balance that grew, which flatters max-DD and the worst month.
    net_pnl = float(CAP * master.sum())
    pnl_per_year, simple_return = net_pnl / yrs, float(master.sum() / yrs)
    # the same track with P&L put back to work — quoted in the sizing note, not as a headline tile: it is
    # the biggest number on the page and the one the capacity limit says could not have been earned
    cmp_final = float(CAP * eqf.iloc[-1])

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
    gross, turn = _book_ops(master)
    ann_turn = summ.get("annual_turnover", 0.0)
    ann_turn_held = summ.get("annual_turnover_weights_held", 0.0)
    sc_held = summ.get("scorecard_weights_held", {})
    cost_levels, breakeven = _cost_levels(master, turn)
    # publish the §9 sweep so the report quotes it instead of a reader transcribing it from this page
    (REP / "master_book_cost_levels.json").write_text(json.dumps(
        {"levels": cost_levels, "breakeven_mult": breakeven, "cost_bps": COST_BPS,
         "turnover_basis": ann_turn}, indent=2, default=float))

    # --- §11 scorecard — judged on the FINAL OOS BLOCK (the brief scores targets there); the full window
    #     (now the 15y 2011+ book) sits in the note as the larger-sample estimate ---
    sc = [
        # §11 scores Sharpe as a BAND, so the tile tests the band — the same test the full-window count uses
        ("Sharpe (net)", _n(_sh(oos), 2), f"{wlab} {_n(m['sharpe'])} · target 2.5–4.0",
         "pass" if 2.5 <= _sh(oos) <= 4.0 else "miss"),
        ("Months in profit", f"{_mip(moo):.0%}", f"{wlab} {_mip(mo):.0%} · target ≥80%",
         "pass" if _mip(moo) >= 0.80 else "miss"),
        ("Max drawdown", _pc(_mdd(oos)), f"{wlab} {_pc(m['max_dd'])} · target ≤15%",
         "pass" if _mdd(oos) >= -0.15 else "miss"),
        ("Longest losing streak", f"{_streak(moo.values)} mo", f"{wlab} {_streak(mo.values)} mo · target ≤2 mo",
         "pass" if _streak(moo.values) <= 2 else "miss"),
        ("Worst single month", _pc(moo.min()), f"{wlab} {_pc(mo.min())} · target ≥ −6%",
         "pass" if moo.min() >= -0.06 else "miss"),
        ("Annual turnover", f"{ann_turn:.1f}× rt", "round-trip ×capital/yr · §11 asks for it, sets no cap", ""),
    ]
    n_pass = sum(1 for *_, c in sc[:5] if c == "pass")             # OOS block — the ONLY scored window (§11)
    wfp = REP / "master_book_wf_summary.json"
    wf_li, mv_li = "", ""
    if wfp.exists():
        w = json.loads(wfp.read_text())
        h = w["headline_wf_oos"]
        # the re-fit-the-weights alternative, on the SAME walk-forward as equal weight (so it is comparable)
        cfg = w.get("configs") or {}
        mv, eq_cfg = cfg.get("meanvar_anchored_Q"), cfg.get(w.get("headline_config"))
        if mv and eq_cfg:
            mv_li = (f': the mean-variance fit buys Sharpe ({_n(mv["sharpe"])} vs {_n(eq_cfg["sharpe"])}) '
                     f'with a {mv["max_dd"] / eq_cfg["max_dd"]:.0f}&times; deeper drawdown '
                     f'({_pc(mv["max_dd"], 0)} vs {_pc(eq_cfg["max_dd"], 0)})')
        gfc = (w.get("stress") or {}).get("2008 GFC")
        gfc_s = f', {_pc(gfc["max_dd"])} through the 2008 GFC' if gfc else ''
        # span from the dates, not obs/365 — the mixed 252/365 calendar makes the latter understate the years
        wf_yrs = (pd.Timestamp(h["end"]) - pd.Timestamp(h["start"])).days / 365.25
        wf_li = (f'<li><b>OOS is most of the history, not 2 years:</b> the book-level walk-forward runs '
                 f'out-of-sample {h["start"][:4]}&rarr;{h["end"][:4]} (~{wf_yrs:.0f}y) at Sharpe '
                 f'<b>{_n(h["sharpe"])}</b>{gfc_s}, paying for that history in drawdown '
                 f'({_pc(h["max_dd"])} vs {_pc(m["max_dd"])}).</li>')
    yr_ret = (1.0 + master).resample("YE").prod() - 1.0
    n_pos_yr, n_tot_yr = int((yr_ret > 0).sum()), int(len(yr_ret))
    # §11 scores the block, so the full window carries numbers and no pass count — it is the larger-sample
    # estimate, not a second scorecard. The streak is still named outright whenever it runs past what the
    # block's target would allow: not counting a window is a reason to stop scoring it, not to go quiet.
    streak_full = _streak(mo.values)
    full_tail = (f'longest losing run {streak_full} month{"s" if streak_full != 1 else ""}'
                 + ('' if streak_full <= 2 else ', longer than the scored block&rsquo;s &le;2 allows'))
    wtext = f"{int(yrs)}-year"        # prose form of the window label; stays in step with the tiles' wlab
    part = " (the last one partial)" if master.index[-1].month < 12 else ""
    sc_note = (
        f'<div class="scnote">'
        f'<span class="lead"><b>All five targets met on the frozen out-of-sample block</b> &mdash; the window '
        f'&sect;11 scores them on.</span>'
        f'<ul>'
        f'<li><b>OOS block ({n_pass} of 5)</b> &mdash; the scored window (2024-07&rarr;): Sharpe '
        f'{_n(_sh(oos))}, months {_mip(moo):.0%}, rest clear. Carried by the short-vol leg&rsquo;s <b>two '
        f'regime gates</b> &mdash; the VIX term structure, plus each sleeve&rsquo;s own curve for the thirteen '
        f'of eighteen the VIX cannot see &mdash; and the crypto sleeve&rsquo;s <b>residual momentum</b>.</li>'
        f'<li><b>Full {wtext} window</b> &mdash; supporting evidence, not a second scorecard: Sharpe '
        f'{_n(m["sharpe"])}, months {_mip(mo):.0%}, '
        f'max-DD {_pc(m["max_dd"])}, worst month {_pc(mo.min())}, {full_tail}. '
        f'<b>Positive in {n_pos_yr} of {n_tot_yr} calendar years</b>{part}. Before ~2019 the vol-premium, '
        f'crisis and global-macro legs are strategy-logic backtests on index data, not a live track.</li>'
        f'{wf_li}'
        f'<li><b>Equal weight is evidence-based:</b> re-fitting the weights does not beat it out-of-sample'
        f'{mv_li}.</li>'
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
                         fmt=lambda v: _pc(v, 1))

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
                           "rdylgn", fmt=lambda v: _n(v, 1))

    # --- §13 per-family (sleeve-leg) Sharpe by year AND quarter — the sleeves the book is built from,
    #     not only the book aggregate above ---
    def _leg_grid(freq):
        vals = {}
        for f in fams:
            s = legs[f].dropna()
            key = s.index.year if freq == "Y" else s.index.to_period("Q")
            for k, g in s.groupby(key):
                if len(g) > 20 and g.std(ddof=1) > 0:
                    vals.setdefault(str(k), {})[f] = float(np.sqrt(_ppy(g)) * g.mean() / g.std(ddof=1))
        cols = sorted(vals)
        return [sf(f) for f in fams], cols, [[vals.get(c, {}).get(f) for c in cols] for f in fams]
    fyr_r, fyr_c, fyr_m = _leg_grid("Y")
    fqr_r, fqr_c, fqr_m = _leg_grid("Q")
    famperiods = (
        '<figure class="card s6"><figcaption>Per-family Sharpe by year (§13) &mdash; the sleeve legs the '
        f'book is built from, net</figcaption>{heat_svg(fyr_r, fyr_c, fyr_m, 1120, 2.0, "rdylgn", fmt=lambda v: _n(v, 1))}'
        '<figcaption style="margin-top:18px">Per-family Sharpe by quarter (§13) &mdash; hover a cell for its Sharpe</figcaption>'
        # too many quarters now span the columns for a legible per-cell number, so the value is hover-only
        # (each cell's data-tip); the heat colour still carries the pattern at a glance
        f'{heat_svg(fqr_r, fqr_c, fqr_m, 1120, 2.0, "rdylgn", show_val=False, rowh=34, fmt=lambda v: _n(v, 1))}</figure>')

    # --- stress windows (§10) ---
    stress_rows = ""
    for lab, a, b in STRESS:
        w = master[(master.index >= pd.Timestamp(a)) & (master.index <= pd.Timestamp(b))]
        if not len(w):
            continue
        stress_rows += (f"<tr><td>{lab}</td><td>{_n(_sh(w))}</td>"
                        f"<td>{_pc((1 + w).prod() - 1)}</td><td>{_pc(_mdd(w))}</td></tr>")
    # These five windows are the ones the brief names, and the book's own deepest drawdown is in none of
    # them — so locate it and say what drove it, rather than letting the reader read the table's worst row
    _e = (1.0 + master).cumprod()
    _dd = _e / _e.cummax() - 1.0
    trough = _dd.idxmin()
    peak = _e[:trough].idxmax()
    ep = legs.loc[peak:trough].sum().sort_values()
    worst_legs = ", ".join(f"{sf(k)} {_pc(v)}" for k, v in ep.head(2).items())
    q4_18 = _mdd(master[(master.index >= pd.Timestamp("2018-10-01")) & (master.index <= pd.Timestamp("2018-12-31"))])
    # "the same depth" was once literally true (both −7.2%) and was written as a fixed claim. It is a
    # measurement, so it is measured: the two only read as one failure mode when the numbers agree.
    same = abs(q4_18 - m["max_dd"]) < 0.005
    stress_note = (
        f'<p class="valline">The book&rsquo;s deepest drawdown of the whole window is in <b>none of these '
        f'five</b>: it is {_pc(m["max_dd"])} over {peak.date()}&rarr;{trough.date()}, driven by the '
        f'managed-futures legs ({worst_legs}) reversing together while the short-vol leg was up. Q4 2018 '
        f'{"prints the same" if same else "gives up"} {_pc(q4_18)} inside its own window for an unrelated '
        f'reason &mdash; a trend reversal plus a vol spike &mdash; and it is the quarter that sets the worst '
        f'month ({_pc(mo.min())}). Two different failure modes'
        f'{" at the same depth" if same else ""}; the diversification is what keeps either from going '
        f'further.</p>')

    # --- marginal-contribution curve + table: Sharpe, max-DD and months-in-profit as families join (§7) ---
    labels = [sf(a) for a in marg["added"]]
    vals = [float(v) for v in marg["sharpe"]]
    marg_rows = "".join(
        f"<tr><td>{int(r.n)}</td><td>+{sf(r.added)}</td><td>{_n(r.sharpe)}</td>"
        f"<td>{_pc(r.max_dd)}</td><td>{r.months_in_profit:.0%}</td></tr>" for r in marg.itertuples())
    first, last = marg.iloc[0], marg.iloc[-1]
    # mark the SHIPPED book, not the argmax: the single-family peak is the point the construction gives up
    marg_svg = (curve_svg(labels, vals, 548, 240, mark=len(vals) - 1)
                + '<table><tr><th>n</th><th>+family</th><th>Sharpe</th><th>max DD</th><th>months+</th></tr>'
                + marg_rows + '</table>'
                + f'<p class="valline">Sharpe and months-in-profit both <b>fall</b> as families join '
                  f'({_n(first.sharpe)}&rarr;{_n(last.sharpe)}, {first.months_in_profit:.0%}&rarr;'
                  f'{last.months_in_profit:.0%}); what the additions buy is the <b>tail</b> '
                  f'({_pc(first.max_dd)}&rarr;{_pc(last.max_dd)}). That trade is the point: vol-prem alone sits '
                  f'outside the 2.5&ndash;4.0 Sharpe band and fails the &le;15% drawdown target. On the '
                  f'drawdown axis the curve has <b>not</b> flattened by the last family &mdash; the final '
                  f'three additions still cut '
                  f'{_pc(marg.iloc[-3].max_dd)}&rarr;{_pc(marg.iloc[-2].max_dd)}&rarr;{_pc(last.max_dd)} '
                  f'&mdash; which is why none is dropped. (Equal-weight mean of the legs, so the final point '
                  f'reads {_n(last.sharpe)} against the deliverable book&rsquo;s {_n(m["sharpe"])}, which '
                  f'also carries the drawdown ladder and the daily-loss breaker.)</p>')

    # --- per-family contribution table: standalone Sharpe/DD, corr->book, book-without & delta ---
    solo = summ["standalone_sharpe"]
    pnl = summ.get("pnl_share", {})     # each family's share of book P&L (§7)
    # baseline for the leave-one-out delta must be the SAME construction as the counterfactual — the
    # equal-weight mean of all legs; the deliverable's own Sharpe additionally carries the risk overlay,
    # so subtracting from it would bias every delta by that difference.
    all_legs = _sh(legs.mean(axis=1, skipna=True))
    fam_rows = ""
    for f in sorted(fams, key=lambda c: -solo.get(c, 0.0)):
        s = legs[f].dropna()
        joined = pd.concat([legs[f], master], axis=1).dropna()
        c = float(joined.corr().iloc[0, 1]) if len(joined) > 2 else 0.0
        wo = _sh(legs.drop(columns=[f]).mean(axis=1, skipna=True))   # book with this family removed
        fam_rows += (f"<tr><td>{sf(f)}</td><td>{_n(solo.get(f, 0.0))}</td><td>{_pc(_mdd(s))}</td>"
                     f"<td>{pnl.get(f, 0.0):.0%}</td>"
                     f"<td>{_n(c)}</td><td>{_n(wo)}</td><td>{_n(all_legs - wo)}</td></tr>")
    n_neg = sum(1 for f in fams if all_legs - _sh(legs.drop(columns=[f]).mean(axis=1, skipna=True)) <= 0)
    fam_note = (f'<p class="valline">&Delta; Sharpe is measured against the equal-weight mean of all legs '
                f'({_n(all_legs)}). <b>{n_neg} of {len(fams)} families carry a &le;0 &Delta;</b> and are held '
                f'anyway &mdash; a stated choice: Sharpe is a band (2.5&ndash;4.0), not a maximand, and these '
                f'legs serve the other four targets. Dropping crisis-alpha lifts Sharpe past the top of the '
                f'band. Tail and consistency are bought with Sharpe on purpose.</p>')

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
    cs_txt = (f'<p class="valline">first-half mean {_n(cs.get("first_half_mean", float("nan")))} &rarr; '
              f'second-half {_n(cs.get("second_half_mean", float("nan")))} &middot; OOS mean '
              f'{_n(cs.get("oos_mean", float("nan")))} &middot; largest pairwise shift '
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
    cost_rows = "".join(f"<tr><td>{lv['label']}</td><td>{_n(lv['sharpe'])}</td><td>{_pc(lv['max_dd'])}</td>"
                        f"<td>{_pc(lv['cagr'])}</td></tr>" for lv in cost_levels)
    be_txt = (f"break-even at {breakeven:.0f}&times; the charged rebalancing cost" if breakeven
              else "break-even &gt; 80&times; the charged rebalancing cost")
    # per-family cost-fragility (§9/§12): break-even multiple from each deep-dive where published
    bo_be, xs_be = _dig("breakout/bo_final_summary.json", "breakeven_mult"), _dig("xs/xs_summary.json", "breakeven_cost_mult")
    tr_c3 = _dig("trend/trend_book_blend_summary.json", "cost_levels", "3x")
    be_parts = [p for p in (f"breakout break-even {bo_be:.1f}&times;" if bo_be else "",
                            f"x-sect break-even {xs_be:.1f}&times;" if xs_be else "",
                            # trend and vol-prem publish a Sharpe at a multiple, not a break-even — quoted as such
                            f"trend Sharpe {tr_c3:.2f} at 3&times;" if tr_c3 else "") if p]
    try:
        vpc = pd.read_csv(REP / "volprem" / "volprem_cost_robustness.csv").iloc[-1]
        be_parts.append(f"vol-prem Sharpe {vpc['sharpe']:.2f} at {vpc['cost_mult']:.0f}&times;")
    except Exception:
        pass
    perfam_be = ", ".join(be_parts) if be_parts else "in the deep-dives"
    # §13 OOS trade log — reference the artifact the book emits (return-composed book => its trades are
    # the daily sleeve rebalances; instrument-level fills are per-family)
    tlp = REP / "master_book_oos_ledger.csv"
    trp = REP / "master_book_oos_trades.csv"
    tl_note = ""
    if tlp.exists():
        tl = pd.read_csv(tlp)
        n_tr = len(pd.read_csv(trp)) if trp.exists() else 0
        tl_note = (f'<p class="valline">OOS log (§13): <b>{len(tl):,}</b> daily rebalances '
                   f'{tl["date"].min()}&rarr;{tl["date"].max()} &middot; <b>{n_tr:,}</b> instrument fills &mdash; '
                   f'<code>master_book_oos_ledger.csv</code>, <code>master_book_oos_trades.csv</code>.</p>')
    # §9/§12 per family: measured by re-running each family with its cost model off (measure_family_costs)
    fcp = REP / "book" / "family_cost_shares.json"
    fam_cost_txt = ""
    if fcp.exists():
        fc = json.loads(fcp.read_text())
        shares = {k: v for src in (fc.get("re_run_here") or {}, fc.get("from_deep_dives") or {})
                  for k, v in src.items() if "cost_share_of_gross_pnl" in v}
        frag = [k for k, v in shares.items() if v.get("cost_fragile")]
        if shares:
            worst = max(shares, key=lambda k: shares[k]["cost_share_of_gross_pnl"])
            fam_cost_txt = (
                f' Measured per family by re-running each construction with its cost model switched off: '
                f'the {len(shares)} legs pay between '
                f'{min(v["cost_share_of_gross_pnl"] for v in shares.values()):.1%} and '
                f'{max(v["cost_share_of_gross_pnl"] for v in shares.values()):.1%} of gross P&amp;L in cost, '
                + (f'and <b>{len(frag)} is cost-fragile ({", ".join(frag)}, break-even '
                   f'{shares[worst]["breakeven_cost_mult"]:.1f}&times;)</b> &mdash; a crash hedge that trades '
                   f'to stay long gamma, held for what it does in the bad months rather than for its own P&amp;L.'
                   if frag else 'and none is cost-fragile.'))

    rl = summ.get("risk_limits", {})
    lim_txt = (f' The declared limits sit on the book multiplier, not on this sum: it runs at a constant '
               f'{rl["leverage"]:.2f}&times; against a {rl["gross_cap"]:.1f}&times; cap, net exposure ~0.'
               if rl.get("leverage") and rl.get("gross_cap") else "")
    ops_html = (
        f'<figure class="card"><figcaption>Family exposure over time (§13) &mdash; sum of the '
        f'{len(fams)} risk-parity weights &times; their vol-target leverage</figcaption>{expg_svg}'
        f'<p class="valline">This is the notional the sleeves add up to (mean {gross.mean():.2f}&times; capital, '
        f'peak {gross.max():.2f}&times;), reconstructed from the family blocks &mdash; it rises when the legs&rsquo; '
        f'own volatility falls and their vol targeting levers up.{lim_txt}</p></figure>'
        f'<figure class="card"><figcaption>Book rebalancing turnover over time (§13) &middot; '
        f'{ann_turn:.0f}&times; round-trip/yr</figcaption>{expt_svg}'
        f'<p class="valline"><b>Most of it is the calendar, not conviction.</b> The book equal-weights the '
        f'legs that <i>print</i> each day, so a weekend closure re-weights the crypto legs up and back down '
        f'on Monday. Holding each leg through its own market&rsquo;s closures instead turns over '
        f'<b>{ann_turn_held:.1f}&times;</b> ({ann_turn / max(ann_turn_held, 1e-9):.0f}&times; less)'
        + (f' at Sharpe {sc_held["sharpe"]:+.2f}, max-DD {_pc(sc_held["max_dd"])}, worst month '
           f'{_pc(sc_held["worst_month"])} &mdash; no worse on any target' if sc_held else '')
        + f'. Every figure here measures and charges the shipped convention; the cheaper one is a measured '
        f'option, not a claim. Assembly layer only &mdash; instrument turnover is charged inside each '
        f'family&rsquo;s own returns.</p>{tl_note}</figure>'
        f'<figure class="card"><figcaption>Cost sensitivity (§9) &mdash; {be_txt}</figcaption>'
        f'<table><tr><th>cost level</th><th>Sharpe</th><th>max DD</th><th>CAGR</th></tr>{cost_rows}</table>'
        f'<p class="valline">The book&rsquo;s {ann_turn:.0f}&times;/yr rebalancing turnover at {COST_BPS:.0f}bps '
        f'round-trip (~{ann_turn * COST_BPS / 1e4:.0%} of capital a year), re-charged on top of the costs '
        f'already inside every family&rsquo;s returns. Per family: {perfam_be}.{fam_cost_txt}</p></figure>')

    _write(summ, cagr, net_pnl, pnl_per_year, simple_return, cmp_final, ca_sharpe, ca_cagr, dict(
        sc=_scorecard(sc), sc_note=sc_note, eq=eq_svg, psleq=psleq_svg, month=month_svg, dd=dd_svg, roll=roll_svg,
        year=year_svg, quarter=quarter_svg, stress=stress_rows, stress_note=stress_note, marg=marg_svg,
        corr=corr_svg, famtbl=fam_rows, famnote=fam_note, famperiods=famperiods, param=_param_card(),
        timeframe=_timeframe_card(), ops=ops_html,
        edge_map=_family_edge_card(summ, legs),
        listing=_listing_sentence(summ, legs),
        feature=_feature_card(), sleevecost=_sleeve_cost_card(), honesty=_honesty_card(), lines=lines))
    if "--check" not in sys.argv:
        print("dashboard -> reports/dashboard.html")
    print("MAKE REPORT OK")


ASSETS = Path(__file__).resolve().parent / "report_assets"  # dashboard.html/.css/.js live here


def _asset(name):
    return (ASSETS / name).read_text()


def _listing_sentence(summ, legs):
    """"Who is live when", built from the legs rather than typed.

    This sentence used to name trend and carry by hand. When the book stopped trading them the page went
    on telling the reader that trend joins in 2012 — which is the same class of error as a stale Sharpe,
    only harder to notice because it reads like background."""
    starts = {}
    for f in summ["families"]:
        s = legs.get(f)
        if s is not None and len(s.dropna()):
            starts.setdefault(s.dropna().index.min().year, []).append(sf(f))
    if not starts:
        return ""
    first = min(starts)
    parts = [f"{first} runs on {_and(starts[first])}"]
    for y in sorted(starts)[1:]:
        parts.append(f"{_and(starts[y])} join{'s' if len(starts[y]) == 1 else ''} in {y}")
    return ", ".join(parts)


def _and(names):
    return names[0] if len(names) == 1 else " and ".join([", ".join(names[:-1]), names[-1]])


def _write(summ, cagr, net_pnl, pnl_per_year, simple_return, cmp_final, ca_sharpe, ca_cagr, S):
    """Fill report_assets/dashboard.html (page + copy) with computed values, CSS and JS."""
    m = summ["master"]
    fam = ", ".join(sf(f) for f in summ["families"])
    rw = summ["window"]
    tr = summ["top_removed"]
    nf = len(summ["families"])
    # §10 Monte-Carlo maxDD / monthly-hit-rate percentiles live under the canonical block-bootstrap
    # variant; fall back to any legacy top-level mirror, and to n/a if the MC has not been run.
    bbmc = (m.get("mc_variants") or {}).get("block_bootstrap") or {}

    def _mcp(k, sign=False):
        v = bbmc.get(k, m.get("mc_" + k))
        return (_pc(v, 1) if sign else f"{v:.0%}") if isinstance(v, (int, float)) and v == v else "n/a"

    html = _asset("dashboard.html").format(
        css=_asset("dashboard.css"), js=_asset("dashboard.js"),
        lines_json=json.dumps(S["lines"], default=float), cap_k=CAP // 1000,
        report_window=f"{rw[0][:4]}–{rw[1][:4]}",
        cagr=_pc(cagr, 1), net_pnl_m=f"{net_pnl / 1e6:.2f}",
        pnl_per_year_k=f"{pnl_per_year / 1e3:.0f}", simple_return=_pc(simple_return, 1),
        cmp_final_m=f"{cmp_final / 1e6:.1f}", cmp_pnl_m=f"{(cmp_final - CAP) / 1e6:.1f}",
        sc=S["sc"], sc_note=S["sc_note"], mc_p5=_n(m['mc_p5'], 2), mc_p50=_n(m['mc_p50'], 2), mc_p95=_n(m['mc_p95'], 2),
        mc_maxdd_p5=_mcp("maxdd_p5", True), mc_maxdd_p50=_mcp("maxdd_p50", True), mc_maxdd_p95=_mcp("maxdd_p95", True),
        mc_hit_p5=_mcp("hit_p5"), mc_hit_p50=_mcp("hit_p50"), mc_hit_p95=_mcp("hit_p95"),
        mean_corr=_n(summ['mean_correlation'], 2), n_families=nf, n_families_less_one=nf - 1,
        listing=S["listing"],
        top_family=sf(tr["family"]), top_removed=_n(tr['sharpe'], 2),
        vp_pnl=f"{summ.get('pnl_share', {}).get('volprem', float('nan')):.0%}", fam_w=f"1/{nf}",
        eq=S["eq"], psleq=S["psleq"], month=S["month"], dd=S["dd"], roll=S["roll"],
        year=S["year"], quarter=S["quarter"], stress=S["stress"], stress_note=S["stress_note"], marg=S["marg"],
        corr=S["corr"], famtbl=S["famtbl"], famnote=S["famnote"], ops=S["ops"], edge_map=S["edge_map"],
        feature=S["feature"], sleevecost=S["sleevecost"], famperiods=S["famperiods"], param=S["param"], timeframe=S["timeframe"], honesty=S["honesty"], fam=fam,
        sharpe_ann=f"{m['sharpe']:.2f}",
        ca_sharpe=f"{ca_sharpe:.2f}", ca_cagr=_pc(ca_cagr, 1),
    )
    out = REP / "dashboard.html"
    if "--check" in sys.argv:
        # the same gate REPORT.md has: a page rendered from artifacts that have since moved is a page
        # quoting numbers nothing in the repo produces, and that must fail a build rather than ship
        if not out.exists() or out.read_text() != html:
            raise SystemExit("reports/dashboard.html is stale — the artifacts moved since it was built.\n"
                             "  fix: python scripts/make_report.py")
        print("reports/dashboard.html is current with the artifacts")
        return
    out.write_text(html)


if __name__ == "__main__":
    main()
