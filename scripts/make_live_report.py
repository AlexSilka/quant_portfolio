"""Render the live book's one-page control sheet from its own artifact.

The master book has `make_report.py`; this is the same idea for the book in `run_live_book.py`, and it
exists for the same reason: a page with numbers typed into it drifts the moment anything is re-run, and
this repository has already been bitten by exactly that. Every figure below is read from
`reports/lab/live_book.json`, so the page cannot disagree with the book it describes.

The page is a control sheet, not a report. Its job is one decision — how much leverage — so the leverage
ladder is the first thing on it and picking a rung re-reads the whole page.

    python scripts/make_live_report.py   ->  reports/live_book.html
"""
from __future__ import annotations

import json
import sys

from src.config import CAPITAL_USD, LAB_DIR, REPORTS_DIR

OUT = REPORTS_DIR / "live_book.html"
SRC = LAB_DIR / "live_book.json"
DEFAULT_RUNG = "2.0"
LEG_LABEL = {"volprem": "vol-premium", "xs_momentum": "cross-sectional", "breakout": "breakout", "bab": "BAB"}
LEG_NOTE = {
    "volprem": "Short variance across 18 Cboe underlyings, standing down whenever either the shared VIX "
               "curve or the sleeve's own curve inverts. Carries the book and its concentration risk.",
    "breakout": "Donchian break with a chandelier trail and an ML confidence gate, crypto perps, "
                "long on spot and short on perp so the funding bill is collected rather than paid.",
    "bab": "Betting-against-beta, beta-neutral, the concentrated top-25 crypto book — long low-beta "
           "against short high-beta with Frazzini-Pedersen leg scaling.",
    "xs_momentum": "Cross-sectional momentum on the liquid crypto and equity cross-sections, "
                   "dollar-neutral, ranked each bar on trailing liquidity rather than a fixed list.",
}


def pct(x: float, dp: int = 1) -> str:
    return f"{100 * x:+.{dp}f}%".replace("-", "−")


def money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) < 1e6 else f"${x / 1e6:.2f}M"


def build(d: dict) -> str:
    sweep = d["leverage_sweep"]
    rungs = sorted(sweep, key=float)
    per_year = d["per_year"]
    w0, w1 = d["window"]
    yrs = d["stats"]["years"]

    rung_cards = "".join(
        f'<button class="rung{" is-on" if r == DEFAULT_RUNG else ""}" data-rung="{r}" '
        f'role="radio" aria-checked="{"true" if r == DEFAULT_RUNG else "false"}">'
        f'<span class="rung-x">{float(r):g}&times;</span>'
        f'<span class="rung-ret">{pct(sweep[r]["cagr"], 0)}</span>'
        f'<span class="rung-dd">{pct(sweep[r]["max_dd"], 0)} drawdown</span>'
        f'{"<span class=rung-tag>suggested</span>" if r == DEFAULT_RUNG else ""}</button>'
        for r in rungs)

    share = d["pnl_share_2020"]
    legs = "".join(
        f'<li class="leg"><div class="leg-top"><span class="leg-name">{LEG_LABEL.get(k, k)}</span>'
        f'<span class="leg-share">{100 * share[k]:.0f}%</span></div>'
        f'<div class="bar"><i style="width:{100 * share[k]:.0f}%"></i></div>'
        f'<p class="leg-note">{LEG_NOTE.get(k, "")}</p></li>'
        for k in sorted(share, key=lambda x: -share[x]))

    live = d["legs_live_per_year"]
    ys = sorted(per_year, key=int)
    bars = "".join(
        f'<div class="yr" data-year="{y}"><div class="yr-bar"><i></i></div>'
        f'<span class="yr-v"></span><span class="yr-y">{y[2:]}</span>'
        f'<span class="yr-n">{live[y]}</span></div>' for y in ys)

    return f"""<meta charset="utf-8">
<title>Live book — the portfolio to actually run</title>
<style>
:root {{
  --ground:#eef2f4; --surface:#ffffff; --sunk:#dde5ea; --line:#d3dde3;
  --text:#0d151a; --muted:#5b6f7c; --accent:#9a6b18; --accent-soft:#f0e2c6;
  --critical:#bb3f37; --good:#2c7a58;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0b1014; --surface:#121a20; --sunk:#182430; --line:#22303a;
    --text:#e3ecf1; --muted:#8298a6; --accent:#e0a53f; --accent-soft:#2a2113;
    --critical:#e2685f; --good:#57ad86;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0b1014; --surface:#121a20; --sunk:#182430; --line:#22303a;
  --text:#e3ecf1; --muted:#8298a6; --accent:#e0a53f; --accent-soft:#2a2113;
  --critical:#e2685f; --good:#57ad86;
}}
*{{box-sizing:border-box}}
body {{ margin:0; background:var(--ground); color:var(--text); font-family:var(--sans);
  line-height:1.55; padding:clamp(20px,4vw,56px) clamp(16px,4vw,40px); }}
.wrap{{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:34px}}
h1 {{ font-family:var(--mono); font-size:clamp(21px,2.6vw,30px); font-weight:600; letter-spacing:-.02em;
  margin:0; text-wrap:balance; }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--accent); margin:0 0 10px; }}
.lede {{ margin:10px 0 0; color:var(--muted); max-width:66ch; }}
h2 {{ font-family:var(--mono); font-size:12px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); margin:0 0 14px; font-weight:600; }}
section{{display:block}}
.headline {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
.hcell{{background:var(--surface);padding:16px 18px}}
.hcell .k {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); }}
.hcell .v {{ font-family:var(--mono); font-size:clamp(22px,3vw,30px); font-weight:600; margin-top:8px;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }}
.hcell .n{{font-size:11.5px;color:var(--muted);margin-top:5px}}
.v.up{{color:var(--accent)}} .v.down{{color:var(--critical)}}
.ladder{{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px}}
.rung {{ appearance:none; text-align:left; cursor:pointer; font:inherit; color:inherit; position:relative;
  background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:13px 14px 12px;
  display:flex; flex-direction:column; gap:5px; transition:border-color .12s, background .12s; }}
.rung:hover{{border-color:var(--accent)}}
.rung:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.rung.is-on{{border-color:var(--accent);background:var(--accent-soft)}}
.rung-x {{ font-family:var(--mono); font-size:17px; font-weight:600; font-variant-numeric:tabular-nums; }}
.rung-ret {{ font-family:var(--mono); font-size:14px; color:var(--accent); font-variant-numeric:tabular-nums; }}
.rung-dd {{ font-family:var(--mono); font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.rung-tag {{ position:absolute; top:-8px; right:10px; font-family:var(--mono); font-size:9px;
  letter-spacing:.12em; text-transform:uppercase; background:var(--accent); color:var(--ground);
  padding:2px 7px; border-radius:99px; }}
.legs{{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(238px,1fr));gap:18px}}
.leg{{display:flex;flex-direction:column;gap:8px}}
.leg-top{{display:flex;justify-content:space-between;align-items:baseline;gap:10px}}
.leg-name{{font-family:var(--mono);font-size:13.5px;font-weight:600}}
.leg-share{{font-family:var(--mono);font-size:13.5px;color:var(--accent);font-variant-numeric:tabular-nums}}
.bar{{height:6px;background:var(--sunk);border:1px solid var(--line);border-radius:99px;overflow:hidden}}
.bar i{{display:block;height:100%;background:var(--accent);border-radius:99px}}
.leg-note{{margin:0;font-size:12.5px;color:var(--muted)}}
.cap{{margin:12px 0 0;font-size:12.5px;color:var(--muted);max-width:74ch}}
.years{{display:flex;gap:5px;align-items:flex-end;overflow-x:auto;padding-bottom:2px}}
.yr{{flex:1 0 34px;display:flex;flex-direction:column;align-items:center;gap:5px}}
.yr-bar{{height:96px;width:100%;display:flex;align-items:flex-end;background:var(--sunk);border-radius:4px;overflow:hidden}}
.yr-bar i{{display:block;width:100%;background:var(--accent);border-radius:4px 4px 0 0}}
.yr-bar i.neg{{background:var(--critical)}}
.yr-v{{font-family:var(--mono);font-size:10.5px;color:var(--muted);font-variant-numeric:tabular-nums}}
.yr-y{{font-family:var(--mono);font-size:10px;color:var(--muted)}}
.yr-n {{ font-family:var(--mono); font-size:9px; color:var(--muted); border:1px solid var(--line);
  border-radius:99px; padding:0 5px; line-height:1.5; }}
.yr[data-full] .yr-n{{border-color:var(--accent);color:var(--accent)}}
.note {{ border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:10px;
  background:var(--surface); padding:15px 18px; }}
.note h3 {{ font-family:var(--mono); font-size:12px; letter-spacing:.08em; text-transform:uppercase;
  margin:0 0 8px; color:var(--text); }}
.note p{{margin:0 0 10px;font-size:13.5px;color:var(--muted);max-width:74ch}}
.note p:last-child{{margin-bottom:0}}
.note b{{color:var(--text);font-weight:600}}
code{{font-family:var(--mono);font-size:.92em;color:var(--text)}}
.foot {{ border-top:1px solid var(--line); padding-top:16px; font-size:12px; color:var(--muted);
  font-family:var(--mono); }}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style>
<div class="wrap">
  <header>
    <p class="eyebrow">Live book &middot; not the test-task deliverable</p>
    <h1>Four legs, one dial, and what each setting actually costs</h1>
    <p class="lede">The same research as the master book with the five scored targets taken out, and one
    question in their place: which legs would you put your own money in, and at what size. Window opens
    {w0} &mdash; the first day both segments of the short-vol gate exist. No book-level drawdown ladder,
    no daily-loss breaker: those guaranteed a mandate that no longer applies, and they are what stops
    leverage working.</p>
  </header>

  <section>
    <h2>Pick the dial</h2>
    <div class="ladder" role="radiogroup" aria-label="Leverage">{rung_cards}</div>
  </section>

  <section>
    <h2>What that setting gives you &mdash; {w0} to {w1}</h2>
    <div class="headline">
      <div class="hcell"><div class="k">Return</div><div class="v up" id="m-cagr"></div>
        <div class="n">a year, compounded</div></div>
      <div class="hcell"><div class="k">Max drawdown</div><div class="v down" id="m-dd"></div>
        <div class="n">peak to trough, whole window</div></div>
      <div class="hcell"><div class="k">Worst month</div><div class="v down" id="m-wm"></div>
        <div class="n">worst day <span id="m-wd"></span></div></div>
      <div class="hcell"><div class="k">Volatility</div><div class="v" id="m-vol"></div>
        <div class="n">Sharpe <span id="m-sh"></span> &middot; months up <span id="m-mp"></span></div></div>
      <div class="hcell"><div class="k">P&amp;L on {money(CAPITAL_USD)}</div><div class="v up" id="m-pnl"></div>
        <div class="n">not reinvested &middot; <span id="m-pnly"></span> a year</div></div>
    </div>
  </section>

  <section>
    <h2>What is in it &mdash; share of P&amp;L since all four legs list</h2>
    <ul class="legs">{legs}</ul>
  </section>

  <section>
    <h2>Year by year &mdash; bar height is the year&rsquo;s return, the count under it is legs live</h2>
    <div class="years">{bars}</div>
    <p class="cap">Four legs only from 2020. Before that the two crypto legs do not list yet and this is
    vol-premium plus the cross-sectional book, which is why the early years are not evidence for the
    portfolio as drawn above.</p>
  </section>

  <section class="note">
    <h3>Read this before the headline</h3>
    <p><b>Nine of these fifteen years are not this book.</b> Breakout and BAB list in 2020; before that it
    is vol-premium plus the cross-sectional leg. Measured only where all four are live, 2020 onward, the
    book returns <b><span id="m-s20"></span> a year</b> against <span id="m-full"></span> for the full
    window, at Sharpe <span id="m-s20sh"></span> instead of <span id="m-fullsh"></span>. The first number
    is the honest one, and it is the one to plan against.</p>
    <p><b>One leg is more than half the P&amp;L.</b> Vol-premium is short variance: it earns steadily and
    loses suddenly. Its worst day ungated is &minus;76.4%; the two regime gates it ships with take that to
    &minus;10.0%, and across eight dislocations from 2008 to the 2025 tariff shock the gated leg loses more
    than 4% in exactly one &mdash; 2008, before the curve data the gate needs existed. That is the risk
    this book is really carrying.</p>
    <p><b>The compounded multiple is arithmetic, not a result.</b> At these rates the balance passes the
    vol-premium leg's vega capacity &mdash; low tens of millions &mdash; around year eight, and leaves the
    order size the cost model charges for. The dollar figure above is deliberately the un-reinvested one.</p>
    <p><b>Leverage is a real dial, not a free one.</b> Sharpe does not improve with it; only the return and
    the pain do, in proportion. Every rung above is the same book, sized differently.</p>
  </section>

  <p class="foot">Generated from <code>reports/lab/live_book.json</code> by
  <code>scripts/make_live_report.py</code> &middot; {yrs:.1f} years &middot;
  rebuild with <code>python scripts/run_live_book.py</code></p>
</div>
<script>
const SWEEP = {json.dumps(sweep, separators=(",", ":"))};
const CAP = {CAPITAL_USD};
const NLEGS = {len(d["legs"])};
const pct = (x, dp = 1) => (x >= 0 ? "+" : "\\u2212") + Math.abs(x * 100).toFixed(dp) + "%";
const usd = x => (Math.abs(x) >= 1e6 ? "$" + (x / 1e6).toFixed(2) + "M" : "$" + Math.round(x).toLocaleString());
function show(r) {{
  const s = SWEEP[r];
  document.getElementById("m-cagr").textContent = pct(s.cagr, 0);
  document.getElementById("m-dd").textContent = pct(s.max_dd, 1);
  document.getElementById("m-wm").textContent = pct(s.worst_month, 1);
  document.getElementById("m-wd").textContent = pct(s.worst_day, 1);
  document.getElementById("m-vol").textContent = (s.vol * 100).toFixed(1) + "%";
  document.getElementById("m-sh").textContent = s.sharpe.toFixed(2);
  document.getElementById("m-mp").textContent = Math.round(s.months_in_profit * 100) + "%";
  // the un-reinvested figure is the sum of returns on fixed capital, which scales with the dial
  const pnl = CAP * s.ret_sum;
  document.getElementById("m-pnl").textContent = usd(pnl);
  document.getElementById("m-pnly").textContent = usd(pnl / s.years);
  const py = s.per_year, ymax = Math.max(...Object.values(py).map(Math.abs)) || 1;
  document.querySelectorAll(".yr").forEach(el => {{
    const v = py[el.dataset.year];
    const bar = el.querySelector("i");
    bar.style.height = Math.max(2, 100 * Math.abs(v) / ymax).toFixed(0) + "%";
    bar.classList.toggle("neg", v < 0);
    el.querySelector(".yr-v").textContent = Math.round(v * 100) + "%";
    el.toggleAttribute("data-full", +el.querySelector(".yr-n").textContent === NLEGS);
  }});
  document.getElementById("m-s20").textContent = pct(s.since_2020.cagr, 0);
  document.getElementById("m-s20sh").textContent = s.since_2020.sharpe.toFixed(2);
  document.getElementById("m-full").textContent = pct(s.cagr, 0);
  document.getElementById("m-fullsh").textContent = s.sharpe.toFixed(2);
  document.querySelectorAll(".rung").forEach(b => {{
    const on = b.dataset.rung === r;
    b.classList.toggle("is-on", on);
    b.setAttribute("aria-checked", on ? "true" : "false");
  }});
}}
document.querySelectorAll(".rung").forEach(b => b.addEventListener("click", () => show(b.dataset.rung)));
show("{DEFAULT_RUNG}");
</script>
"""


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC} missing — run `python scripts/run_live_book.py` first")
    html = build(json.loads(SRC.read_text()))
    if "--check" in sys.argv:
        # same gate the dashboard has: a committed page quoting numbers the repo no longer produces is
        # worse than no page, so it fails the build rather than shipping
        if not OUT.exists() or OUT.read_text() != html:
            raise SystemExit("reports/live_book.html is stale — the book moved since it was built.\n"
                             "  fix: python scripts/run_live_book.py && python scripts/make_live_report.py")
        print("reports/live_book.html is current with the book")
        return
    OUT.write_text(html)
    print(f"live report -> {OUT}")


if __name__ == "__main__":
    main()
