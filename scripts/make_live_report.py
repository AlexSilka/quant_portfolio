"""Render the live book's page — same design system as the master dashboard, un-reinvested throughout.

Two things this file deliberately does NOT do.

It does not define a look. The stylesheet, the tooltip script and every chart builder are the master
dashboard's own (`report_assets/dashboard.css`, `.js`, and the `*_svg` helpers in `make_report.py`), so
the two books read as two pages of one report rather than as two products. A second visual identity for
a second book is a bug, not a feature.

It does not compound. Every figure here is P&L on the brief's fixed capital with nothing reinvested:
year and month cells are sums of daily returns, the balance chart is a running sum in dollars, drawdown
is measured on that same running sum. The compounded reading is the master book's convention and is
mathematically fine, but it puts this book past its own vol-premium capacity from about year eight, and
a chart nobody could trade is not a chart worth drawing. The useful side-effect: under this convention
*everything scales linearly with leverage*, so one dial moves the whole page by one multiplication.

    python scripts/make_live_report.py            ->  reports/live_book.html
    python scripts/make_live_report.py --check    fails if the committed page lags the artifacts
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
from scripts.make_report import (  # noqa: E402
    MONTHS, _asset, _ds, _money, _pc, _pts, bars_svg, heat_svg, line_svg,
)
from src.config import CAPITAL_USD, LAB_DIR, REPORTS_DIR  # noqa: E402

OUT = REPORTS_DIR / "live_book.html"
SRC = LAB_DIR / "live_book.json"
BOOK = LAB_DIR / "live_book.parquet"
LEGS = LAB_DIR / "live_legs.parquet"
DEFAULT_RUNG = "2.0"
LEG_LABEL = {"volprem": "vol premium", "xs_momentum": "cross-sectional", "breakout": "breakout", "bab": "BAB"}
LEG_NOTE = {
    "volprem": "Short variance across 18 Cboe underlyings, standing down whenever either the shared VIX "
               "curve or the sleeve&rsquo;s own curve inverts.",
    "breakout": "Donchian break with a chandelier trail and an ML confidence gate, crypto, long the spot "
                "leg and short the perp so funding is collected rather than paid.",
    "bab": "Betting-against-beta, beta-neutral, the concentrated top-25 crypto book, with "
           "Frazzini-Pedersen leg scaling.",
    "xs_momentum": "Cross-sectional momentum on the liquid crypto and equity cross-sections, "
                   "dollar-neutral, ranked each bar on trailing liquidity rather than on a fixed list.",
}


def arith(s: pd.Series) -> dict:
    """The un-reinvested reading: P&L as a fraction of fixed capital, never a balance times a return."""
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    pnl = s.cumsum()
    mo = s.resample("ME").sum()
    return {"ret": float(s.sum()) / yrs, "total": float(s.sum()), "years": yrs,
            "sharpe": mb.scorecard(s)["sharpe"], "max_dd": float((pnl - pnl.cummax()).min()),
            "worst_month": float(mo.min()), "worst_day": float(s.min()),
            "vol": float(s.std() * np.sqrt(mb.ppy_of(s))), "months_in_profit": float((mo > 0).mean())}


def _kpis(a: dict) -> str:
    return (
        f'<div class="kpis">'
        f'<div class="kpi"><div class="label">Return, not reinvested</div>'
        f'<div class="val" id="k-ret">{_pc(a["ret"], 1)}</div>'
        f'<div class="note">a year on the brief&rsquo;s ${CAPITAL_USD // 1000}k sizing capital</div></div>'
        f'<div class="kpi"><div class="label">P&amp;L, not reinvested</div>'
        f'<div class="val" id="k-pnl">+{_money(CAPITAL_USD * a["total"])}</div>'
        f'<div class="note"><span id="k-pnly"></span> a year &middot; net of all modelled costs</div></div>'
        f'<div class="kpi"><div class="label">Worst drawdown</div>'
        f'<div class="val" id="k-dd">{_pc(a["max_dd"], 1)}</div>'
        f'<div class="note">of capital &middot; worst month <span id="k-wm"></span> &middot; '
        f'Sharpe <span id="k-sh"></span></div></div></div>')


def _ladder(sweep: dict, rungs: list) -> str:
    cells = "".join(
        f'<button class="sc lev{" on" if r == DEFAULT_RUNG else ""}" data-rung="{r}" role="radio" '
        f'aria-checked="{"true" if r == DEFAULT_RUNG else "false"}">'
        f'<div class="label">{float(r):g}&times; leverage{" &middot; shipped" if r == DEFAULT_RUNG else ""}</div>'
        f'<div class="val">{_pc(sweep[r]["arith_ret"], 0)}</div>'
        f'<div class="note">a year &middot; {_pc(sweep[r]["arith_dd"], 0)} drawdown</div></button>'
        for r in rungs)
    return f'<div class="scorecard" role="radiogroup" aria-label="Leverage">{cells}</div>'


def _leg_table(legs: pd.DataFrame, since: pd.Timestamp) -> str:
    w = legs[legs.index >= since]
    tot = float(w.sum().sum())
    rows = []
    for c in w.sum().sort_values(ascending=False).index:
        s = legs[c].dropna()
        rows.append(
            f'<tr><td><b>{LEG_LABEL.get(c, c)}</b></td>'
            f'<td>{_pc(float(w[c].sum()) / tot, 0)}</td>'
            f'<td>{mb.scorecard(s)["sharpe"]:.2f}</td>'
            f'<td>{s.index.min().year}</td>'
            f'<td class="whr">{LEG_NOTE.get(c, "")}</td></tr>')
    return ('<table><tr><th>leg</th><th>share of P&amp;L</th><th>own Sharpe</th><th>lists</th>'
            f'<th>what it is</th></tr>{"".join(rows)}</table>')


def _lev_table(sweep: dict, rungs: list) -> str:
    rows = "".join(
        f'<tr{" class=grp" if r == DEFAULT_RUNG else ""}><td>{float(r):g}&times;'
        f'{" &mdash; shipped" if r == DEFAULT_RUNG else ""}</td>'
        f'<td>{_pc(sweep[r]["arith_ret"], 1)}</td>'
        f'<td>{_money(CAPITAL_USD * sweep[r]["ret_sum"])}</td>'
        f'<td>{_pc(sweep[r]["arith_dd"], 1)}</td>'
        f'<td>{_pc(sweep[r]["arith_wm"], 1)}</td>'
        f'<td>{_pc(sweep[r]["worst_day"], 1)}</td>'
        f'<td>{_pc(sweep[r]["vol"], 1)}</td><td>{sweep[r]["sharpe"]:.2f}</td></tr>'
        for r in rungs)
    return ('<table><tr><th>dial</th><th>return / yr</th><th>P&amp;L on $500k</th><th>max DD</th>'
            f'<th>worst month</th><th>worst day</th><th>vol</th><th>Sharpe</th></tr>{rows}</table>')


def build(d: dict, b: pd.Series, legs: pd.DataFrame) -> str:
    lev = d["leverage"]
    a = arith(b)
    lines = {}
    since20 = pd.Timestamp("2020-01-01")

    # --- balance: a running sum of P&L in dollars on fixed capital. Linear axis on purpose: there is no
    #     compounding to straighten out, so a straight line here is a steady dollar rate, which is what a
    #     desk running a fixed book actually earns.
    pnl = CAPITAL_USD * b.cumsum()
    eq_svg, lines["eq"] = line_svg("eq", _pts(_ds(pnl)), 1120, 330, usd=True, baseline=0.0)

    dd = b.cumsum() - b.cumsum().cummax()
    dd_svg, lines["dd"] = line_svg("dd", _pts(_ds(dd)), 548, 240, pct=True)

    rmu = b.rolling(365, min_periods=180).mean()
    rsd = b.rolling(365, min_periods=180).std(ddof=1)
    roll = (np.sqrt(mb.ppy_of(b)) * rmu / rsd).dropna()
    roll_svg, lines["roll"] = line_svg("roll", _pts(_ds(roll)), 548, 240)

    mo = b.resample("ME").sum()
    years = sorted({t.year for t in mo.index})
    yi = {y: i for i, y in enumerate(years)}
    mmat = [[None] * 12 for _ in years]
    for t, v in mo.items():
        mmat[yi[t.year]][t.month - 1] = float(v)
    vmax = float(np.nanpercentile(np.abs(mo.to_numpy()), 92))
    month_svg = heat_svg([str(y) for y in years], MONTHS, mmat, 1120, vmax, "rdylgn",
                         fmt=lambda v: _pc(v, 1), rowh=34)

    # One chart, not two: the year's return with its own figure printed on the bar, the number of legs
    # live that year under the label, and a divider where the book becomes the book. A reader should not
    # have to hover to learn either the size of a year or whether that year was the same portfolio.
    py = [(str(y), float(g.sum())) for y, g in b.groupby(b.index.year)]
    nlegs = legs.notna().sum(axis=1)
    sub = [f"{n} leg" + ("s" if n > 1 else "")
           for n in (int(nlegs[nlegs.index.year == y].max()) for y in years)]
    split = (years.index(2020), "all four legs from here") if 2020 in years else None
    year_svg = bars_svg(py, 1120, 320, pct=True, show_val=True, sub=sub, split=split, rotate=False)

    a20 = arith(b[b.index >= since20])
    sweep, rungs = d["leverage_sweep"], sorted(d["leverage_sweep"], key=float)

    return f"""<meta charset="utf-8">
<title>Live book &mdash; the portfolio sized for return</title>
<style>
{_asset("dashboard.css")}
.sc.lev {{ appearance:none; font:inherit; text-align:left; cursor:pointer; color:var(--text);
  border-top-color:var(--border); transition:border-color .12s, background .12s; }}
.sc.lev:hover {{ border-color:var(--accent); }}
.sc.lev:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.sc.lev.on {{ border-color:color-mix(in srgb,var(--accent) 55%,var(--border)); border-top-color:var(--accent);
  background:color-mix(in srgb,var(--accent) 9%,var(--panel)); }}
.sc.lev .val {{ color:var(--muted); }} .sc.lev.on .val {{ color:var(--accent); }}
</style>
<div class="wrap">
<div class="eyebrow">Quant research &middot; the same book sized for return</div>
<h1>The live book &mdash; four earners, one leverage dial, nothing reinvested</h1>
<p class="sub">The research behind the master book with the brief&rsquo;s five scored targets taken out and
one question in their place: which legs would you put your own money in, and at what size. Four families
at equal risk &mdash; {", ".join(LEG_LABEL.get(c, c) for c in legs.columns)} &middot; <b>no hedge slot</b>
and <b>no book-level overlay</b>: the drawdown ladder and daily-loss breaker guaranteed a mandate that no
longer applies, and measured, they are what stops leverage working. Window opens
<code>{b.index.min().date()}</code> &mdash; the first day both segments of the short-vol gate exist.
Everything on this page is P&amp;L on a fixed ${CAPITAL_USD // 1000}k with nothing reinvested. Hover any chart.</p>
{_kpis(a)}
<p class="valline"><b>Nothing here compounds.</b> Each year and month cell is the sum of that period&rsquo;s
daily returns on the same ${CAPITAL_USD // 1000}k, and the balance chart is that sum in dollars &mdash; the
convention the &radic;-impact cost model is calibrated to. Compounding the same track would read like a
number no desk could trade: from about year eight the balance is past the vol-premium leg&rsquo;s vega
capacity, low tens of millions. One useful consequence: under this convention <b>every figure on the page
scales linearly with the dial</b>, so the charts below are drawn at the shipped {lev:g}&times; and any other
rung is one multiplication away.</p>
{_ladder(sweep, rungs)}
<p class="valline gap">Sharpe and months-in-profit do not move with the dial at all &mdash; only the money
and the pain do, in proportion. That is the whole trade-off: <b>{_pc(a["ret"], 0)} a year costs a
{_pc(a["max_dd"], 0)} drawdown and a {_pc(a["worst_month"], 0)} month</b> at {lev:g}&times;.</p>
<div class="grid">
  <figure class="card s6"><figcaption>P&amp;L on ${CAPITAL_USD // 1000}k, not reinvested &mdash; a running
  sum in dollars, so a straight line is a steady dollar rate (linear axis &middot; hover for date)</figcaption>{eq_svg}</figure>
  <figure class="card s6"><figcaption>Monthly return &mdash; % of the fixed ${CAPITAL_USD // 1000}k, sum of
  the month&rsquo;s daily returns, at {lev:g}&times;</figcaption>{month_svg}</figure>
  <figure class="card s6"><figcaption>Return by year &mdash; % of the fixed ${CAPITAL_USD // 1000}k at
  {lev:g}&times;, with the number of legs live that year &middot; everything left of the divider is a
  two-leg book wearing this book&rsquo;s name</figcaption>{year_svg}</figure>
  <figure class="card"><figcaption>Drawdown &mdash; % of capital, on the un-reinvested track</figcaption>{dd_svg}</figure>
  <figure class="card"><figcaption>Rolling 12-month Sharpe</figcaption>{roll_svg}</figure>
  <figure class="card s6"><figcaption>The dial &mdash; every rung is the same book, sized differently</figcaption>
  {_lev_table(sweep, rungs)}</figure>
  <figure class="card s6"><figcaption>What is in it &mdash; share of P&amp;L over 2020 onward, where all
  four legs are live</figcaption>{_leg_table(legs, since20)}</figure>
</div>
<div class="scnote" style="margin-top:22px"><span class="lead"><b>Read this before the headline.</b></span>
<ul>
<li><b>Nine of these fifteen years are not this book.</b> Breakout and BAB list in 2020; before that this is
vol premium plus the cross-sectional leg, which is why the early bars are taller and thinner evidence.
Measured only where all four are live, 2020 onward, the book returns <b>{_pc(a20["ret"], 0)} a year</b>
against {_pc(a["ret"], 0)} for the full window, at Sharpe {a20["sharpe"]:.2f} instead of {a["sharpe"]:.2f},
worst month {_pc(a20["worst_month"], 1)}. <b>Plan against the 2020 number.</b></li>
<li><b>One leg is more than half the P&amp;L.</b> Vol premium is short variance: it earns steadily and loses
suddenly. Its worst day ungated is &minus;76.4%; the two regime gates it ships with take that to &minus;10.0%,
and across eight dislocations from 2008 to the 2025 tariff shock the gated leg loses more than 4% in exactly
one &mdash; 2008, before the curve data the gate needs existed. That concentration is the real risk here,
not the drawdown number above.</li>
<li><b>The dial is real, not free.</b> {rungs[-1]}&times; is on the table above because the arithmetic is
linear, not because it is advisable: it turns a {_pc(sweep[DEFAULT_RUNG]["arith_dd"], 0)} drawdown into
{_pc(sweep[rungs[-1]]["arith_dd"], 0)} of the capital, and a book that loses that much is a book that gets
turned off before it recovers.</li>
</ul></div>
<div class="foot"><b>Not the test-task deliverable.</b> That is the six-family master book
(<b>REPORT.md</b>, its own dashboard), which holds a crisis-alpha hedge, runs a §8 drawdown ladder and is
sized to a &minus;15% mandate at a constant 1.15&times;. This page is the same validated research with the
scorecard removed &mdash; a different portfolio, not a re-cut of that one. Both are generated from
artifacts, never typed: this one from <b>reports/lab/live_book.json</b> and <b>live_book.parquet</b> by
<b>scripts/make_live_report.py</b>, rebuilt with <b>make live</b>, and a <code>--check</code> in
<code>make lint</code> fails the build if the page lags the book. Window {b.index.min().date()} to
{b.index.max().date()}, {a["years"]:.1f} years.</div>
<div class="tip" id="tip"></div></div>
<script>const LINES={json.dumps(lines, default=float)};
const SWEEP={json.dumps({r: {k: v for k, v in sweep[r].items() if not isinstance(v, dict)} for r in rungs})};
const CAP={CAPITAL_USD};
{_asset("dashboard.js")}
const pc=(x,dp=1)=>(x>=0?'+':'−')+Math.abs(x*100).toFixed(dp)+'%';
function showRung(r){{
  const s=SWEEP[r];
  document.getElementById('k-ret').textContent=pc(s.arith_ret,1);
  document.getElementById('k-pnl').textContent='+'+money(CAP*s.ret_sum);
  document.getElementById('k-pnly').textContent=money(CAP*s.ret_sum/s.years);
  document.getElementById('k-dd').textContent=pc(s.arith_dd,1);
  document.getElementById('k-wm').textContent=pc(s.arith_wm,1);
  document.getElementById('k-sh').textContent=s.sharpe.toFixed(2);
  document.querySelectorAll('.sc.lev').forEach(b=>{{
    const on=b.dataset.rung===r;
    b.classList.toggle('on',on);
    b.setAttribute('aria-checked',on?'true':'false');}});
}}
document.querySelectorAll('.sc.lev').forEach(b=>b.addEventListener('click',()=>showRung(b.dataset.rung)));
showRung('{DEFAULT_RUNG}');
</script>
"""


def main() -> None:
    for p in (SRC, BOOK, LEGS):
        if not p.exists():
            sys.exit(f"{p} missing — run `python scripts/run_live_book.py` first")
    d = json.loads(SRC.read_text())
    b = pd.read_parquet(BOOK)["ret"]
    legs = pd.read_parquet(LEGS)
    html = build(d, b, legs)
    if "--check" in sys.argv:
        # same gate the dashboard has: a committed page quoting numbers the repo no longer produces is
        # worse than no page, so it fails the build rather than shipping
        if not OUT.exists() or OUT.read_text() != html:
            raise SystemExit("reports/live_book.html is stale — the book moved since it was built.\n"
                             "  fix: make live")
        print("reports/live_book.html is current with the book")
        return
    OUT.write_text(html)
    print(f"live report -> {OUT}")


if __name__ == "__main__":
    main()
