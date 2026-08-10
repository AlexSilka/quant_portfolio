"""Every headline number the report quotes, resolved from the committed artifacts — one place.

The report used to carry these as literals, so a book re-run left ~30 sentences quoting a Sharpe the
artifacts no longer held, silently and in the deliverable a reviewer reads first. Now the prose lives in
`scripts/report_assets/report.md` with `{{name}}` placeholders, and `scripts/render_report.py` fills them
from here. Adding a number to the report means adding it here first, which is the point: if it cannot be
resolved from an artifact, it is not a measurement.

Naming is by what the reader sees, not by which file it came from: `book_sharpe`, `oos_worst_month`,
`zoo_trials`. Percent values carry the typographic minus the report uses, so a rendered figure and a
hand-written target read alike on the same line.

    python scripts/report_numbers.py        # print the whole registry (what the template may use)
"""
import json
from pathlib import Path

import pandas as pd

R = Path("reports")


def _pc(v, dp=1):
    return f"{v:+.{dp}%}".replace("-", "−")


def _pcu(v, dp=1):
    """Unsigned percent (for shares and hit rates, where a leading + reads oddly)."""
    return f"{v:.{dp}%}".replace("-", "−")


def _n(v, dp=2):
    return f"{v:+.{dp}f}".replace("-", "−")


def _word(n):
    """Small counts spelled out, so prose reads as prose ("all six legs") and still tracks the artifact."""
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}.get(n, str(n))


def _ordinal(n):
    """"fourth-highest", not "four-highest" — a rank is a different word from a count."""
    return {1: "highest", 2: "second-highest", 3: "third-highest", 4: "fourth-highest",
            5: "fifth-highest", 6: "sixth-highest", 7: "seventh-highest", 8: "eighth-highest"}.get(n, f"{n}th-highest")


def _load(path):
    p = R / path
    return json.loads(p.read_text()) if p.exists() else {}


LEVERAGE_ROWS = (1.00, 1.15, 1.20, 1.25, 1.35, 1.45, 1.65, 2.00)   # the rungs §4b argues over


def _leverage_table(shipped):
    """The §4b leverage grid, emitted from run_risk_budget's own CSV. It used to be transcribed by hand,
    which is eleven numbers a row that go stale together the moment the book moves."""
    import csv
    p = R / "book" / "risk_budget_grid.csv"
    if not p.exists():
        return "_(leverage grid unavailable — run `make risk-budget`)_"
    rows = [r for r in csv.DictReader(p.open()) if r["limits"] == "book_equity"]
    by_lev = {round(float(r["leverage"]), 2): r for r in rows}
    head = ("| leverage | Sharpe | CAGR | max-DD | worst month | months | targets | boot-P5 DD | "
            "boot-P5 month | 2010-event DD | 2010-event month |\n|---|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for lev in LEVERAGE_ROWS:
        r = by_lev.get(lev)
        if not r:
            continue
        f = float(r["leverage"])
        is_shipped = shipped is not None and f"{f:.2f}×" == shipped
        b = (lambda x: f"**{x}**") if is_shipped else (lambda x: x)
        label = f"**{f:.2f}× (shipped)**" if is_shipped else f"{f:.2f}×"
        out.append("| " + " | ".join([
            label, b(f"{float(r['full_sharpe']):.2f}"), b(_pc(float(r["full_cagr"]))),
            b(_pc(float(r["full_max_dd"]))), b(_pc(float(r["full_worst_month"]))),
            b(_pcu(float(r["full_months_in_profit"]))), b(f"{int(r['full_targets'])}/5"),
            b(_pc(float(r["mc_dd_p5"]))), b(_pc(float(r["mc_wmonth_p5"]))),
            b(_pc(float(r["stress_max_dd"]))), b(_pc(float(r["stress_worst_month"])))]) + " |")
    return "\n".join(out)


def _marginal(marg):
    """The §7 marginal-contribution curve, as the report narrates it: the anchor it starts from, the book
    it lands on, and the drawdown the additions buy. Sourced from the same rows the chart plots."""
    if not marg:
        return {}
    rows = sorted(marg, key=lambda r: r["n"])
    first, last = rows[0], rows[-1]
    return {
        "marginal_first_sharpe": f"{first['sharpe']:.2f}",
        "marginal_last_sharpe": f"{last['sharpe']:.2f}",
        "marginal_first_dd": _pc(first["max_dd"]),
        "marginal_last_dd": _pc(last["max_dd"]),
        "marginal_first_months": _pcu(first["months_in_profit"]),
        "marginal_last_months": _pcu(last["months_in_profit"]),
        # the path the report quotes as "the tail the diversifiers buy"
        "marginal_dd_path": " → ".join(_pc(r["max_dd"]) for r in rows[-4:]),
        "marginal_chain": " → ".join(f"{r['sharpe']:.2f}" for r in rows),
    }


def _gate_table():
    """The §5c gate A/B, emitted from run_vol_premium_gates' own CSV: the shipped two-segment rule against
    the ungated leg and the rejected variants, on the same book."""
    import csv
    p = R / "volprem" / "volprem_gates_book.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))
    label = {"none (ungated)": "ungated", "SHIPPED VIX3M/VIX>=1": "long segment only (the previous rule)",
             "fast VIX/VIX9D>=1": "fast segment only", "both segments": "**both segments (shipped)**",
             "both + re-entry 5d": "both + re-entry 5d"}
    out = ["| volprem leg | Sharpe | max-DD | worst month | months | streak | targets |",
           "|---|---|---|---|---|---|---|"]
    vals = {}
    for r in rows:
        name = r["volprem leg"]
        shipped = name == "both segments"
        b = (lambda x: f"**{x}**") if shipped else (lambda x: x)
        out.append("| " + " | ".join([
            label.get(name, name), b(f"{float(r['sharpe']):.2f}"), b(_pc(float(r["max_dd"]))),
            b(_pc(float(r["worst_month"]))), b(_pcu(float(r["months_pos"]))), b(r["streak_mo"]),
            b(r["targets"])]) + " |")
        key = "gate_on" if shipped else ("gate_off" if name == "none (ungated)" else None)
        if key:
            vals[f"{key}_targets"] = r["targets"]
            vals.update({f"{key}_sharpe": f"{float(r['sharpe']):.2f}",
                         f"{key}_oos_sharpe": f"{float(r['oos_sharpe']):.2f}",
                         f"{key}_months": _pcu(float(r["months_pos"])),
                         f"{key}_worst_month": _pc(float(r["worst_month"])),
                         f"{key}_dd": _pc(float(r["max_dd"])),
                         f"{key}_streak": r["streak_mo"]})
    vals["gate_table"] = "\n".join(out)
    vals["gate_table"] = vals["gate_table"].replace(
        "| volprem leg |", "| volprem leg, book scored 2011-01 → 2024-06 |", 1)
    return vals


def _selective_leverage():
    """§4b's "then don't lever the aggressive leg" A/B: every leg at the shipped level against the
    vol-matched alternative that holds vol-prem at 1.00× and gives the risk to the others. From
    run_risk_budget's own json — including the leg counts, so neither the table nor the prose around it
    can go on naming a book composition the experiment no longer ran."""
    sl = _load("book/risk_budget.json").get("selective_leverage") or {}
    a, b = sl.get("all legs"), sl.get("ex-volprem")
    if not (a and b):
        return {}

    def col(d, key, fmt):
        return fmt(d[key]) if key in d else fmt(d["full"][key])
    lev_b = next(iter(b["leverage"].values()))
    rows = [f"| | all {_word(len(a['leverage']))} legs {next(iter(a['leverage'].values())):.2f}× | "
            f"{_word(len(b['leverage']))} legs {lev_b:.2f}×, vol-prem 1.00× |", "|---|---|---|"]
    for lab, key, fmt in (("Sharpe (full / OOS)", None, None), ("CAGR", "cagr", _pc),
                          ("max-DD", "max_dd", _pc), ("worst month", "worst_month", _pc),
                          ("months-in-profit", "months_in_profit", _pcu)):
        if key is None:
            rows.append(f"| {lab} | **{a['full']['sharpe']:.2f} / {a['oos']['sharpe']:.2f}** | "
                        f"{b['full']['sharpe']:.2f} / {b['oos']['sharpe']:.2f} |")
            continue
        rows.append(f"| {lab} | **{fmt(a['full'][key])}** | {fmt(b['full'][key])} |")
    rows.append(f"| targets, full window | **{a['full']['targets']}/5** | {b['full']['targets']}/5 |")
    rows.append(f"| 2010-event DD / month | {_pc(a['stress_max_dd'])} / {_pc(a['stress_worst_month'])} | "
                f"{_pc(b['stress_max_dd'])} / {_pc(b['stress_worst_month'])} |")
    return {"selective_leverage_table": "\n".join(rows),
            "selective_others_word": _word(len(b["leverage"])),
            "selective_stress_month": _pc(b["stress_worst_month"]),
            "selective_d_sharpe": f"{a['full']['sharpe'] - b['full']['sharpe']:.2f}",
            "selective_d_cagr": f"{100 * (a['full']['cagr'] - b['full']['cagr']):.1f}pp",
            "selective_d_months":
                f"{100 * (a['full']['months_in_profit'] - b['full']['months_in_profit']):.1f}pp",
            "selective_targets": f"{b['full']['targets']}/5"}


def _ml_overlay():
    """§5d's whole-book ML arms, from run_ml_portfolio_overlay's own json: the base book and the range the
    six gate engines land in. Quoted as a range because the point of the section is that no engine escapes
    it, not that any one of them is interesting."""
    d = _load("book/ml_portfolio_overlay.json")
    base, arms = d.get("base (no overlay)"), [v for k, v in d.items() if k.startswith("A gate")]
    if not (base and arms):
        return {}

    def rng(key, fmt):
        lo, hi = min(a["full"][key] for a in arms), max(a["full"][key] for a in arms)
        return fmt(lo) if abs(hi - lo) < 1e-9 else f"{fmt(lo)}–{fmt(hi)}"
    return {
        "ml_base_sharpe": f"{base['full']['sharpe']:.2f}",
        "ml_base_months": _pcu(base["full"]["months_in_profit"]),
        "ml_base_cagr": _pcu(base["full"]["cagr"], 0),
        "ml_base_growth": f"{base['full']['growth_x']:.0f}×",
        "ml_gate_sharpe": rng("sharpe", lambda v: f"{v:.2f}"),
        "ml_gate_months": rng("months_in_profit", lambda v: f"{v * 100:.0f}") + "%",
        "ml_gate_cagr": rng("cagr", lambda v: f"{v * 100:.0f}") + "%",
        "ml_gate_growth": rng("growth_x", lambda v: f"{v:.0f}") + "×",
        "ml_gate_targets": rng("targets", lambda v: f"{int(v)}/5"),
        **({"ml_random_gate": f"{d['random_gate_full_sharpe']['min']:.2f}–"
                              f"{d['random_gate_full_sharpe']['max']:.2f}"}
           if isinstance(d.get("random_gate_full_sharpe"), dict) else {}),
    }


def _grid_verdict():
    """How the OOS block scores across the leverage grid — the claim §4b makes about it never setting the
    choice. Phrased from the grid so "at every leverage" cannot outlive being true."""
    import csv
    p = R / "book" / "risk_budget_grid.csv"
    if not p.exists():
        return {}
    rows = [(float(r["leverage"]), int(r["oos_targets"])) for r in csv.DictReader(p.open())
            if r["limits"] == "book_equity"]
    if not rows:
        return {}
    best = max(t for _, t in rows)
    holds = [lev for lev, t in rows if t == best]
    if len(holds) == len(rows):
        return {"oos_grid_verdict": f"scores {best}/5 at every leverage on the grid"}
    misses = sorted(lev for lev, t in rows if t != best)
    return {"oos_grid_verdict": f"scores {best}/5 at {len(holds)} of the grid's {len(rows)} rungs, dropping "
                                f"to {min(t for _, t in rows)}/5 only at "
                                + ", ".join(f"{lev:.2f}×" for lev in misses)}


LEG_SWAP = (("baseline (shipped)", "baseline", None, True),
            ("breakout raw (no ML)", "breakout", "book_raw", False),
            ("breakout + ML *(shipped)*", "breakout", "book_ml", False),
            ("trend raw", "trend", "book_raw", False),
            ("trend + LightGBM gate", "trend", "book_lgbm_gate", False),
            ("trend + RF gate", "trend", "book_rf_gate", False),
            ("carry + timing overlay", "carry", "book_gated", False))


def _leg_swap_table():
    """§5d's leg-swap table, emitted from run_ml_book_contribution's own json — seven rows of seven
    figures that used to be transcribed, and that all move together whenever the book does."""
    d = _load("book/ml_book_contribution.json")
    if not d:
        return {}
    out = ["| book, leg swapped | Sharpe full / OOS | **CAGR full / OOS** | max-DD | worst month | months | streak |",
           "|---|---|---|---|---|---|---|"]
    for label, fam, key, bold in LEG_SWAP:
        rec = d.get(fam) if key is None else (d.get(fam) or {}).get(key)
        if not rec:
            continue
        b = (lambda x: f"**{x}**") if bold else (lambda x: x)
        out.append("| " + " | ".join([
            b(label), b(f"{_n(rec['sharpe_full'])} / {_n(rec['sharpe_oos'])}"),
            f"**{_pcu(rec['cagr_full'])} / {_pcu(rec['cagr_oos'])}**" if bold
            else f"{_pcu(rec['cagr_full'])} / {_pcu(rec['cagr_oos'])}",
            b(_pc(rec["dd_full"])), b(_pc(rec["worst_full"])),
            b(_pcu(rec["months_full"], 0)), b(str(int(rec["streak_full"])))]) + " |")
    return {"leg_swap_table": "\n".join(out)} if len(out) > 2 else {}


def _grid_ranges():
    """How each metric moves across the leverage grid — the claims §4b makes about what scales and what
    does not, read off the grid instead of transcribed row by row."""
    import csv
    p = R / "book" / "risk_budget_grid.csv"
    if not p.exists():
        return {}
    rows = sorted(((float(r["leverage"]), r) for r in csv.DictReader(p.open())
                   if r["limits"] == "book_equity"), key=lambda t: t[0])
    if not rows:
        return {}
    lo, hi = rows[0][1], rows[-1][1]
    lo_lev, hi_lev = rows[0][0], rows[-1][0]
    oos = [float(r["oos_sharpe"]) for _, r in rows]
    return {
        "grid_span": f"{lo_lev:.2f}× to {hi_lev:.2f}×",
        "grid_oos_sharpe_range": f"[{min(oos):.2f}, {max(oos):.2f}]",
        "grid_oos_streak": (str(int(lo["oos_streak"])) if len({r["oos_streak"] for _, r in rows}) == 1
                            else f"{min(int(r['oos_streak']) for _, r in rows)}–"
                                 f"{max(int(r['oos_streak']) for _, r in rows)}"),
        "grid_dd_path": f"{_pc(float(lo['oos_max_dd']))} → {_pc(float(hi['oos_max_dd']))}",
        "grid_worst_month_path": f"{_pc(float(lo['oos_worst_month']))} → {_pc(float(hi['oos_worst_month']))}",
        "grid_cagr_path": f"{_pcu(float(lo['oos_cagr']))} → {_pcu(float(hi['oos_cagr']))}",
        "grid_full_streak": (str(int(lo["full_streak"])) if len({r["full_streak"] for _, r in rows}) == 1
                             else f"{min(int(r['full_streak']) for _, r in rows)}–"
                                  f"{max(int(r['full_streak']) for _, r in rows)}"),
    }


def _grid_span():
    """How much the Sharpe actually moves across the whole leverage grid — the claim §4b makes about it
    being flat, taken from the grid instead of asserted."""
    import csv
    p = R / "book" / "risk_budget_grid.csv"
    if not p.exists():
        return {}
    sh = [float(r["full_sharpe"]) for r in csv.DictReader(p.open()) if r["limits"] == "book_equity"]
    return {"grid_sharpe_range": f"{max(sh):.2f} → {min(sh):.2f}"} if sh else {}


def _targets_hit(sc: dict) -> int:
    """How many of the brief's five scorecard targets a window clears. Counted here rather than
    written into the prose, because the prose is what goes stale when the book moves."""
    return int(sum([sc["sharpe"] >= 1.5, sc["max_dd"] >= -0.15, sc["months_in_profit"] >= 0.80,
                    sc["worst_month"] >= -0.06, sc["longest_losing_streak_mo"] <= 2]))


def _family_costs():
    """§9/§12 per family: cost as a share of gross P&L and the break-even multiple, from
    measure_family_costs (four re-run with their cost model off) plus the four their deep-dives already
    publish. Ordered worst-first, because the question the section answers is which leg is fragile."""
    d = _load("book/family_cost_shares.json")
    if not d:
        return {}
    rows, frag = [], []
    for src in (d.get("re_run_here") or {}, d.get("from_deep_dives") or {}):
        for name, v in src.items():
            if "cost_share_of_gross_pnl" not in v:
                continue
            rows.append((name, v["cost_share_of_gross_pnl"], v.get("breakeven_cost_mult"),
                         bool(v.get("cost_fragile")), v.get("reproduces_published_series", True)))
            if v.get("cost_fragile"):
                frag.append(name)
    if not rows:
        return {}
    rows.sort(key=lambda r: -r[1])
    out = ["| family | cost / gross P&L | break-even | cost-fragile |", "|---|---|---|---|"]
    for name, share, be, fragile, exact in rows:
        mark = "" if exact else " *"
        out.append(f"| {name}{mark} | {_pcu(share)} | {be:.1f}× | "
                   f"{'**yes**' if fragile else 'no'} |")
    return {"family_cost_table": "\n".join(out),
            "n_family_cost_fragile": str(len(frag)),
            "family_cost_fragile_names": ", ".join(frag) if frag else "none",
            "family_cost_worst": rows[0][0], "family_cost_worst_share": _pcu(rows[0][1])}


# §11's five targets, as tests rather than prose — so "meets all five" is a computed verdict and cannot
# survive the metric that made it true moving underneath it.
TARGETS = (("Sharpe", lambda d: 2.5 <= d["sharpe"] <= 4.0, "Sharpe outside the 2.5–4.0 band"),
           ("months in profit", lambda d: d["months_in_profit"] >= 0.80, "months-in-profit under 80%"),
           ("max drawdown", lambda d: d["max_dd"] >= -0.15, "a drawdown past −15%"),
           ("losing streak", lambda d: d["longest_losing_streak_mo"] <= 2,
            "a {longest_losing_streak_mo:.0f}-month losing streak against ≤2"),
           ("worst month", lambda d: d["worst_month"] >= -0.06, "a worst month past −6%"))


#: What each family is paid for, and where its deep-dive lives. Prose per family, numbers from the run —
#: so the README's source table lists exactly the legs the book assembled, in P&L order, and a family
#: entering or leaving the book cannot leave a stale row behind.
FAMILY_BLURB = {
    "volprem": ("[short-vol / VRP](docs/strategies/VOLPREM.md)",
                "selling insurance against volatility across 18 Cboe underlyings"),
    "gmacro": ("[global-macro](scripts/run_gmacro.py)",
               "trend on EM FX + commodities — asset classes no other family trades"),
    "trend": ("[trend](docs/strategies/TREND.md)", "price trend, the only family spanning both asset classes"),
    "bab": ("[BAB / low-vol](docs/strategies/BAB.md)",
            "the leverage-constraint premium: long low-beta, short high-beta"),
    "breakout": ("[breakout](docs/strategies/BREAKOUT.md)",
                 "channel breakouts held on a trailing stop, ML-gated on fast bars"),
    "xs_momentum": ("[x-sect momentum](docs/strategies/XSECT.md)", "relative strength, market-neutral"),
    "carry": ("[carry](docs/strategies/CARRY.md)",
              "perpetual funding: being paid to hold the unpopular side"),
    "crisis": ("[crisis-alpha](scripts/run_crisis.py)",
               "long-gamma managed futures — it pays when the others bleed"),
}


def _family_sources(s):
    """The README's one-page source table, emitted from the run's own family list."""
    share, sharpe = s.get("pnl_share", {}), s.get("standalone_sharpe", {})
    pretty = {"xs_momentum": "x-sect", "bab": "BAB", "gmacro": "gmacro", "trend_momentum": "trend"}
    ranked = sorted(share, key=lambda f: -share[f])
    line = ", ".join(f"**{pretty.get(f, f)} {_pcu(share[f], 0)}**" if i == 0 else
                     f"{pretty.get(f, f)} {_pcu(share[f], 0)}" for i, f in enumerate(ranked))
    fams = [f for f in sorted(s["families"], key=lambda f: -share.get(f, 0)) if f in FAMILY_BLURB]
    rows = ["| family | what it earns on | Sharpe | share of P&L |", "|---|---|---|---|"]
    for i, f in enumerate(fams):
        name, what = FAMILY_BLURB[f]
        pc = _pcu(share.get(f, float("nan")), 0)
        rows.append(f"| {name} | {what} | {_n(sharpe.get(f, float('nan')))} | "
                    f"{'**' + pc + '**' if i == 0 else pc} |")
    # §4's family table: the traded legs with their standalone Sharpe and correlation to the book they
    # are part of. Same source, so the two tables cannot disagree about which families the book holds.
    desc = {"volprem": "short-vol / VRP across 18 Cboe underlyings (incl. gold-miners), 2005+ ([docs/strategies/VOLPREM.md](docs/strategies/VOLPREM.md))",
            "breakout": "crypto trend+ML / PIT top-30 x-sect ([docs/strategies/BREAKOUT.md](docs/strategies/BREAKOUT.md))",
            "bab": "beta-neutral top-25 crypto, betting-against-beta ([docs/strategies/BAB.md](docs/strategies/BAB.md))",
            "gmacro": "EM-FX + commodities TSMOM (`scripts/run_gmacro.py`)",
            "xs_momentum": "crypto residual (idio) + equity, top-100 liquid ([docs/strategies/XSECT.md](docs/strategies/XSECT.md))",
            "crisis": "multi-asset managed-futures trend (`scripts/run_crisis.py`)"}
    label = {"volprem": "vol-premium", "xs_momentum": "x-sect momentum", "bab": "BAB / low-vol",
             "gmacro": "global-macro", "crisis": "crisis-alpha", "breakout": "breakout"}
    ctb = s.get("corr_to_book", {})
    brows = ["| family | honest series | standalone Sharpe | corr to book |", "|---|---|---|---|"]
    for f in sorted(s["families"], key=lambda f: -sharpe.get(f, 0)):
        brows.append(f"| **{label.get(f, f)}** | {desc.get(f, '')} | {sharpe.get(f, float('nan')):.2f} | "
                     f"{_n(ctb.get(f, float('nan')))} |")
    return {"family_source_table": "\n".join(rows), "pnl_share_line": line,
            "book_family_table": "\n".join(brows)}


def _risk_budget_extras():
    """The §4b numbers that live in run_risk_budget's json rather than in its grid CSV: the unlevered
    stack's own volatility, the shipped book's, the flash-crash day, and the rung above the shipped one.

    The last of these is what the section now argues from — "the worst month breaks immediately" is only
    honest if the number quoted is the next rung the grid actually holds, not a remembered one."""
    import csv
    d = _load("book/risk_budget.json")
    if not d:
        return {}
    out = {}
    if "stack_vol" in d:
        out["stack_vol"] = _pcu(d["stack_vol"]["full"])
    if "book_vol" in d:
        out["book_vol"] = _pcu(d["book_vol"])
    ev = d.get("event") or {}
    if "leg_day_loss_at_book_weight" in ev:
        out["event_leg_at_weight"] = _pc(ev["leg_day_loss_at_book_weight"])
        out["event_book_day"] = _pc(ev["book_day_loss_unlevered"])
        out["event_quarters"] = _word(len(ev.get("replayed_into_quarters", [])))
    shipped = d.get("leverage")
    p = R / "book" / "risk_budget_grid.csv"
    if shipped and p.exists():
        rows = sorted(({round(float(r["leverage"]), 2): r for r in csv.DictReader(p.open())
                        if r["limits"] == "book_equity"}).items())
        nxt = next((r for lev, r in rows if lev > round(float(shipped), 2) + 1e-9), None)
        if nxt:
            out["worst_month_next_rung"] = _pc(float(nxt["full_worst_month"]))
            out["next_rung"] = f"{float(nxt['leverage']):.2f}×"
    return out


def _cap_binding():
    """The per-leg vol target is not a scalar, and §4b now says so with the measurement rather than with a
    remembered leg name: which legs sit on `_scale`'s 3× cap, how often, and what raising the target by the
    shipped leverage actually produces — a different book on the days the cap binds, not a bigger one.

    Which legs exist in 2005 is measured here too, for the same reason: the sentence naming them used to be
    typed, and it went on naming a leg the book had dropped."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_master_book as mb
    pretty = {"trend_momentum": "trend", "xs_momentum": "x-sect", "bab": "BAB", "gmacro": "global-macro",
              "volprem": "vol-premium", "crisis": "crisis", "breakout": "breakout", "carry": "carry"}
    legs = {lab: v for lab, f, c in mb.FAMILIES if (v := mb.load(lab, f, c)) is not None}
    if not legs:
        return {}
    share = {k: float((mb._scale(v.dropna(), mb.VOL_TARGET_ANNUAL) >= 3.0 - 1e-12).mean())
             for k, v in legs.items()}
    top = sorted(share, key=lambda k: -share[k])[:2]

    def stack(target):
        df = pd.DataFrame({k: mb.rescale(v, target) for k, v in legs.items()}).sort_index()
        df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
        return df[df.notna().sum(axis=1) >= 2].mean(axis=1, skipna=True).dropna()

    lev = mb.BOOK_LEVERAGE
    base, alt = stack(mb.VOL_TARGET_ANNUAL), stack(mb.VOL_TARGET_ANNUAL * lev)
    differ = int((alt - lev * base).abs().gt(1e-4).sum())
    card = mb.scorecard(mb.risk_overlay(alt, leverage=1.0)[0])
    early = [k for k, v in legs.items() if (v.dropna().index < pd.Timestamp("2006-01-01")).any()]
    # the two accounting conventions at the shipped rung and the one above — the sentence built on these
    # used to be typed, and it had the two conventions the wrong way round after the book moved
    import run_risk_budget as RB
    ew = RB.assemble()[0].mean(axis=1, skipna=True).dropna()
    conv = {}
    for tag, L in (("shipped", lev), ("next", round(lev + 0.05, 2)), ("next2", round(lev + 0.10, 2))):
        b = mb.risk_overlay(ew, leverage=L)[0]
        conv[tag] = (mb.scorecard(b)["worst_month"], mb.fixed_size_scorecard(b)["worst_month"])

    # which month actually sets the floor, and the one behind it — the sentence about "a single month
    # sitting close to the floor" named Oct-2018 long after the book's worst month had moved to Apr-2020
    _b = mb.risk_overlay(ew, leverage=lev)[0]
    _mo = ((1 + _b).resample("ME").prod() - 1).nsmallest(2)
    worst_when = _mo.index[0].strftime("%b-%Y")
    worst_next = _pc(_mo.iloc[1])
    _q = ((1 + _b).resample("QE").prod() - 1).nsmallest(1)
    worst_q = f"Q{_q.index[0].quarter}-{_q.index[0].year}"

    def verdict(t):
        c, f = conv[t]
        both = (c >= -0.06, f >= -0.06)
        if all(both):
            return "clears both"
        if not any(both):
            return "fails both"
        which = "compounded" if both[0] else "fixed-size"
        other = "fixed-size" if both[0] else "compounded"
        gap = abs((c if both[0] else f) + 0.06), abs((f if both[0] else c) + 0.06)
        return (f"clears the {which} one by {gap[0] * 1e4:.0f}bp and fails the {other} one by "
                f"{gap[1] * 1e4:.0f}bp")
    return {"cap_leg_names": " and ".join(pretty.get(k, k) for k in top),
            "cap_leg_share": f"at most {_pcu(max(share.values()), 1)}",
            "cap_days_differ": f"{differ} of {len(base):,} days",
            "cap_alt_dd": _pc(card["max_dd"]), "cap_alt_worst": _pc(card["worst_month"]),
            "early_legs_word": _word(len(early)),
            "early_legs": ", ".join(pretty.get(k, k) for k in early),
            "worst_month_when": worst_when, "worst_month_next": worst_next,
            "worst_quarter_when": worst_q,
            "conv_shipped": verdict("shipped"), "conv_next": verdict("next"),
            "conv_next2": verdict("next2"),
            "next2_rung": f"{lev + 0.10:.2f}×",
            "fixed_worst_month_shipped": _pc(conv["shipped"][1], 2)}


def _composition():
    """§6d-ter's composition search, from run_composition_search's own json.

    This is the one section where the denominator matters more than the winner, so the counts are read
    from the artifact rather than typed: how many configurations were tried, how many cleared both
    windows, and what the shipped one gave up. A hand-written "37" would go on reading right the day a
    ninth family made it 46."""
    d = _load("book/composition_search.json")
    if not d:
        return {}
    cfg, base = d["configurations"], d["configurations"]["all eight"]
    ship = cfg[d["shipped"]]
    pretty = {"trend_momentum": "trend", "xs_momentum": "x-sect", "bab": "BAB", "gmacro": "global-macro",
              "crisis": "crisis", "breakout": "breakout", "carry": "carry", "volprem": "vol-premium"}

    def name(label):
        return " + ".join(pretty.get(x, x) for x in cfg[label]["dropped"]) or "all eight"

    def cell(r, win):
        # the Sharpe is printed on every row, so a row that fails the *corridor* says which side it fell
        # off rather than repeating the number in a miss list
        sh = r[win]["sharpe"]
        m = [x for x in r[f"misses_{win}"] if not x.startswith("Sharpe")]
        edge = " (under 2.5)" if sh < 2.5 else (" (over 4.0)" if sh > 4.0 else "")
        got = f"**{r['targets_' + win]}/5**" if r["targets_" + win] == 5 else f"{r['targets_' + win]}/5"
        return (f"{got} — Sharpe {sh:.2f}{edge}" + (f", {', '.join(m)}" if m else "")).replace("-", "−")

    # the rows the section argues over: the baseline, every single removal that changes a verdict, and
    # both survivors — the other pairs are counted, not listed, and the table says so
    shown = ["all eight"] + [k for k in cfg if len(cfg[k]["dropped"]) == 1] + list(d["passing"])
    rows = ["| configuration | full window | frozen block |", "|---|---|---|"]
    for k in dict.fromkeys(shown):
        lab = ("**drop " + name(k) + "** *(shipped)*" if k == d["shipped"] else
               "all eight" if not cfg[k]["dropped"] else "drop " + name(k))
        rows.append(f"| {lab} | {cell(cfg[k], 'full')} | {cell(cfg[k], 'oos')} |")
    rest = d["n_configurations"] - len(dict.fromkeys(shown))
    rows.append(f"| *({rest} further pairs)* | — | fail at least one |")

    vm, cost = d.get("vol_matched_eight") or {}, d["cost_of_passing"]
    ctab = ["| | eight families | eight, vol-matched | {N} families *(shipped)* |", "|---|---|---|---|",
            f"| targets, full / block | {base['targets_full']}/5 · {base['targets_oos']}/5 | "
            f"{vm.get('targets_full', '—')}/5 · {vm.get('targets_oos', '—')}/5 | "
            f"**{ship['targets_full']}/5 · {ship['targets_oos']}/5** |",
            f"| Sharpe, full / block | {base['full']['sharpe']:.2f} / {base['oos']['sharpe']:.2f} | "
            f"{vm['full']['sharpe']:.2f} / {vm['oos']['sharpe']:.2f} | "
            f"{ship['full']['sharpe']:.2f} / **{ship['oos']['sharpe']:.2f}** |",
            f"| **CAGR, full / block** | {_pc(base['full']['cagr'])} / {_pc(base['oos']['cagr'])} | "
            f"{_pc(vm['full']['cagr'])} / {_pc(vm['oos']['cagr'])} | "
            f"**{_pc(ship['full']['cagr'])} / {_pc(ship['oos']['cagr'])}** |",
            f"| **P&L /yr on $500k** | ${base['full']['pnl_usd_per_year'] / 1e3:.0f}k | "
            f"${vm['full']['pnl_usd_per_year'] / 1e3:.0f}k | "
            f"**${ship['full']['pnl_usd_per_year'] / 1e3:.0f}k** |",
            f"| book volatility | {_pcu(base['full']['vol'])} | {_pcu(vm['full']['vol'])} | "
            f"**{_pcu(ship['full']['vol'])}** |",
            f"| max-DD / worst month | {_pc(base['full']['max_dd'])} / {_pc(base['full']['worst_month'])} | "
            f"{_pc(vm['full']['max_dd'])} / {_pc(vm['full']['worst_month'])} | "
            f"**{_pc(ship['full']['max_dd'])} / {_pc(ship['full']['worst_month'])}** |",
            f"| vol-premium share of P&L | {_pcu(cost['volprem_pnl_share'][0], 0)} | — | "
            f"**{_pcu(cost['volprem_pnl_share'][1], 0)}** |"]
    solo = d["standalone_sharpe"]
    ranked = sorted(solo, key=lambda k: -solo[k])
    return {
        "comp_table": "\n".join(rows),
        "comp_cost_table": "\n".join(ctab).replace("{N}", _word(ship["n_families"])),
        "comp_vm_leverage": f"{vm.get('leverage', float('nan')):.2f}×",
        # deltas between two rates are POINTS, not percent — "+4.1% more CAGR" reads as a relative gain
        # and is a different (wrong) number. The vol-matched pair is quoted as the wider book's advantage,
        # so its sign matches the sentence that carries it.
        "comp_d_cagr_full": f"{100 * cost['cagr_full']:+.1f}pp".replace("-", "−"),
        "comp_d_cagr_oos": f"{100 * cost['cagr_oos']:+.1f}pp".replace("-", "−"),
        "comp_d_cagr_full_vm": f"{-100 * cost['cagr_full_vol_matched']:+.1f}pp".replace("-", "−"),
        "comp_d_cagr_oos_vm": f"{-100 * cost['cagr_oos_vol_matched']:+.1f}pp".replace("-", "−"),
        "comp_d_pnl_yr": f"${abs(ship['full']['pnl_usd_per_year'] - base['full']['pnl_usd_per_year']) / 1e3:.0f}k",
        "comp_share_before_vol": f"from {_pcu(base['full']['vol'])} to {_pcu(ship['full']['vol'])}",
        "comp_n_configs": str(d["n_configurations"]),
        "comp_n_configs_word": _word(d["n_configurations"]),
        "comp_n_passing_word": _word(len(d["passing"])),
        "comp_n_passing_word_cap": _word(len(d["passing"])).capitalize(),
        "comp_passing": " and ".join(sorted((name(k) for k in d["passing"]), key=len)),
        "comp_base_targets_full": f"{base['targets_full']}/5",
        "comp_base_targets_oos": f"{base['targets_oos']}/5",
        "comp_base_miss_full": ", ".join(base["misses_full"]),
        "comp_base_miss_oos": ", ".join(base["misses_oos"]),
        "comp_base_sharpe_full": f"{base['full']['sharpe']:.2f}",
        "comp_base_sharpe_oos": f"{base['oos']['sharpe']:.2f}",
        "comp_ship_sharpe_full": f"{ship['full']['sharpe']:.2f}",
        "comp_ship_sharpe_oos": f"{ship['oos']['sharpe']:.2f}",
        "comp_cost_sharpe_oos": _n(d["cost_of_passing"]["sharpe_oos"]),
        "comp_share_before": _pcu(d["cost_of_passing"]["volprem_pnl_share"][0], 0),
        "comp_share_after": _pcu(d["cost_of_passing"]["volprem_pnl_share"][1], 0),
        "comp_trend_solo": f"{solo['trend_momentum']:.2f}",
        "comp_carry_solo": f"{solo['carry']:.2f}",
        "comp_carry_rank": _ordinal(ranked.index("carry") + 1),
        "comp_trend_neighbours": " and ".join(
            f"{pretty.get(k, k)} ({solo[k]:.2f})" for k in
            (ranked[ranked.index("trend_momentum") - 1], ranked[ranked.index("trend_momentum") + 1])),
        "comp_drop_trend_months_oos": _pcu(cfg["drop trend_momentum"]["oos"]["months_in_profit"]),
        "comp_ship_months_oos": _pcu(ship["oos"]["months_in_profit"]),
    }


def _verdict(sc, prefix):
    passed = [name for name, test, _ in TARGETS if test(sc)]
    missed = [why.format(**sc) for name, test, why in TARGETS if not test(sc)]
    word = {5: "all five", 4: "four of the five", 3: "three of the five",
            2: "two of the five", 1: "one of the five", 0: "none of the five"}[len(passed)]
    return {f"{prefix}_targets_met": f"{len(passed)} of 5",
            f"{prefix}_targets_word": word,
            f"{prefix}_miss": (" and ".join(missed) if missed else "nothing"),
            f"{prefix}_miss_short": (missed[0] if missed else "")}


def _longgamma_table(rungs=("0.15", "0.25", "0.40"), tag="E curve-timed long vol") -> str:
    """The §6c size-sweep table, from run_longgamma_search's artifact. It used to be typed, on a book
    two data fixes and a composition change ago, which is how it came to claim 1.9pp of worst-month
    headroom for a leg that buys 0.3pp."""
    d = _load("lab/longgamma_search.json").get("size_sweep") or {}
    if not d or tag not in d:
        return "_(long-gamma sweep unavailable — run `python scripts/run_longgamma_search.py`)_"
    head = ("| E, share of one slot | selection window: Sharpe / CAGR / worst month / months "
            "| frozen block: Sharpe / CAGR |\n|---|---|---|")
    rows = [head]

    def row(label, c, bold=False):
        b = (lambda x: f"**{x}**") if bold else (lambda x: x)
        o = c.get("oos", {})
        oos_sharpe = b(f"{o.get('sharpe', float('nan')):+.2f}")
        return (f"| {label} | {c['sharpe']:+.2f} / {_pc(c['cagr'])} / {b(_pc(c['worst_month']))} / "
                f"{_pcu(c['months_in_profit'], 0)} | {oos_sharpe} / {_pc(o.get('cagr', float('nan')), 0)} |")

    base = (d.get("baseline (no extra leg)") or {}).get("0.00")
    if base:
        rows.append(row("0 (shipped)", base, bold=True))
    for w in rungs:
        if w in d[tag]:
            rows.append(row(w, d[tag][w]))
    return "\n".join(rows)


def build():
    """The registry: {placeholder name -> rendered string}. Missing artifacts drop their keys rather
    than resolving to a guess, so render_report fails loudly instead of publishing a blank."""
    s = _load("master_book_summary.json")
    z = _load("book/zoo_summary.json")
    c = _load("book/cscv_pbo.json")
    out = {}

    if s:
        m, oos = s["master"], s["scorecard_oos"]
        # the scorecard block is rounded to 4dp on the way into the json, which is enough to move a
        # percentage by 0.1pp when it is re-rounded for display — take the unrounded master fields where
        # they exist so two sentences quoting the same metric cannot disagree
        full = {**s["scorecard_full"], **{k: m[f"full_{k}"] for k in
                                          ("sharpe", "max_dd", "months_in_profit", "worst_month")
                                          if f"full_{k}" in m}}
        full["months_in_profit"] = m.get("months_in_profit", full["months_in_profit"])
        full["max_dd"] = m.get("max_dd", full["max_dd"])
        full["sharpe"] = m.get("sharpe", full["sharpe"])
        s_window, OOS_DATE = s["window"], s["oos_start"]
        fx_full, fx_oos = s.get("fixed_size_full", {}), s.get("fixed_size_oos", {})
        bb = (m.get("mc_variants") or {}).get("block_bootstrap") or {}
        held = s.get("scorecard_weights_held", {})
        out.update({
            "book_sharpe": f"{full['sharpe']:.2f}",
            "book_months": _pcu(full["months_in_profit"]),
            "book_months_round": _pcu(full["months_in_profit"], 0),
            "book_dd": _pc(full["max_dd"]),
            "book_dd_2dp": _pc(full["max_dd"], 2),
            "book_worst_month": _pc(full["worst_month"]),
            "book_worst_month_2dp": _pc(full["worst_month"], 2),
            "book_streak": str(full["longest_losing_streak_mo"]),
            "book_targets": f"{_targets_hit(full)}/5",
            "oos_targets": f"{_targets_hit(oos)}/5",
            "oos_sharpe": f"{oos['sharpe']:.2f}",
            "oos_months": _pcu(oos["months_in_profit"]),
            "oos_months_round": _pcu(oos["months_in_profit"], 0),
            "oos_dd": _pc(oos["max_dd"]),
            "oos_worst_month": _pc(oos["worst_month"]),
            "oos_streak": str(oos["longest_losing_streak_mo"]),
            # §9 fixed-$500k reading of the same track — the stricter convention on the two limits
            "fixed_dd": _pc(fx_full.get("max_dd", float("nan")), 2),
            "fixed_worst_month": _pc(fx_full.get("worst_month", float("nan")), 2),
            "pnl_usd": f"${fx_full.get('pnl_usd', 0) / 1e6:.2f}M",
            "pnl_usd_per_year": f"${fx_full.get('pnl_usd_per_year', 0) / 1e3:.0f}k",
            "worst_month_usd": f"−${abs(fx_full.get('worst_month_usd', 0)):,.0f}",
            "dd_usd": f"−${abs(fx_full.get('max_dd_usd', 0)):,.0f}",
            "oos_pnl_usd": f"${fx_oos.get('pnl_usd', 0) / 1e3:.0f}k",
            "mean_corr": _n(s["mean_correlation"]),
            "top_removed_family": s["top_removed"]["family"],
            "top_removed_sharpe": _n(s["top_removed"]["sharpe"]),
            "n_families": str(len(s["families"])),
            "window": f"{s['window'][0][:4]}–{s['window'][1][:4]}",
            "capital": f"${s.get('sizing_capital_usd', 0) // 1000:,.0f}k",
            "mc_sharpe_p5": _n(m["mc_p5"]), "mc_sharpe_p50": _n(m["mc_p50"]), "mc_sharpe_p95": _n(m["mc_p95"]),
            "mc_dd_p5": _pc(bb.get("maxdd_p5", m.get("mc_maxdd_p5", float("nan")))),
            "mc_dd_p50": _pc(bb.get("maxdd_p50", m.get("mc_maxdd_p50", float("nan")))),
            "mc_hit_p5": _pcu(bb.get("hit_p5", m.get("mc_hit_p5", float("nan"))), 0),
            "mc_wmonth_p5": _pc(bb.get("wmonth_p5", float("nan"))),
            "turnover": f"{s.get('annual_turnover', float('nan')):.0f}×",
            "volprem_pnl_share": _pcu(s.get("pnl_share", {}).get("volprem", float("nan")), 0),
            "n_years": str(len(s.get("per_year", {}))),
            # the per-year line was a typed list of nine years and every one of them had drifted; it is
            # the whole series now, so it cannot be partly stale
            "per_year_line": " · ".join(f"{y} **{v:+.1f}**" for y, v in
                                        sorted(s.get("per_year", {}).items())),
            # spelled out for prose ("a six-family book"); unsigned for "≈ 0.06" style
            "n_families_word": _word(len(s["families"])),
            "n_families_word_cap": _word(len(s["families"])).capitalize(),
            "n_families_less_one_word": _word(len(s["families"]) - 1),
            # the standalone spread of everything except the anchor — "the other five families run 0.4-1.4"
            "solo_range_ex_anchor": (lambda v: f"{min(v):.1f}–{max(v):.1f}")(
                [x for k, x in s["standalone_sharpe"].items()
                 if k != max(s["standalone_sharpe"], key=s["standalone_sharpe"].get)]),
            "mean_corr_abs": f"{abs(s['mean_correlation']):.2f}",
            # §7.2's stability line — typed, and every one of its four numbers had drifted
            "corr_stability": (lambda c: f"first-half {c['first_half_mean']:.2f} / second-half "
                               f"{c['second_half_mean']:.2f} / OOS-block {c['oos_mean']:.2f}, max pairwise "
                               f"shift {c['max_pairwise_shift']:.2f}")(s["correlation_stability"]),
            "n_years_positive": str(sum(1 for v in s.get("per_year", {}).values() if v > 0)),
            "weakest_year_sharpe": _n(min(s.get("per_year", {"x": 0.0}).values(), key=float), 1),
            "weakest_year": min(s.get("per_year", {"x": 0.0}), key=lambda k: s["per_year"][k]),
            "turnover_held": f"{s.get('annual_turnover_weights_held', float('nan')):.1f}×",
        })
        if held:
            out.update({"held_sharpe": f"{held['sharpe']:.2f}", "held_dd": _pc(held["max_dd"]),
                        "held_worst_month": _pc(held["worst_month"])})
        # the two return rates the report quotes side by side: what the fixed $500k earns with P&L taken
        # out, and the rate the same track compounds at (which is what the risk metrics are measured on)
        yrs = (pd.Timestamp(s["window"][1]) - pd.Timestamp(s["window"][0])).days / 365.25
        cap = s.get("sizing_capital_usd") or 1
        out["return_not_reinvested"] = _pc(fx_full.get("pnl_usd_per_year", 0) / cap)
        if yrs > 0 and full.get("total_return") is not None:
            out["return_compounded"] = _pc((1.0 + full["total_return"]) ** (1 / yrs) - 1.0)

        # the leverage the book ships at, and the constraint that currently binds it
        rb = _load("book/risk_budget.json")
        if rb.get("leverage"):
            out["leverage"] = f"{rb['leverage']:.2f}×"
        allowed = rb.get("leverage_allowed_by") or {}
        live = {k: v for k, v in allowed.items() if isinstance(v, (int, float))}
        if live:
            binding = min(live, key=live.get)
            out["binding_constraint"] = binding
            out["binding_leverage"] = f"{live[binding]:.2f}×"
            if "realised worst month" in live:
                out["worst_month_allows"] = f"{live['realised worst month']:.2f}×"
        if allowed:
            rows = ["| constraint | largest leverage that still holds |", "|---|---|"]
            for k, v in allowed.items():
                cell = f"**{v:.2f}×**" if v == live.get(binding) else (f"{v:.2f}×" if v is not None
                                                                       else "fails already at 1.00×")
                rows.append(f"| {k} | {cell} |")
            out["constraint_table"] = "\n".join(rows)

        out["leverage_table"] = _leverage_table(out.get("leverage"))
        out.update(_verdict(s["scorecard_full"], "book"))
        out.update(_verdict(oos, "oos"))
        # the same full window read on the brief's fixed-$500k convention — Sharpe is scale-free, so it
        # carries over; the point of the sentence that uses this is whether the verdict survives the switch
        if fx_full:
            out.update(_verdict({**fx_full, "sharpe": s["scorecard_full"]["sharpe"]}, "fixed"))
        out.update(_grid_verdict())
        # the same a-priori book scored on three reporting windows — §1's "nothing hinges on the window"
        wr = _load("master_book_wf_summary.json").get("window_robustness") or {}
        lab = {"full_21y_2005": "full_history", "15y_2011": "window_15y", "10y_2016": "window_10y"}
        for k, name in lab.items():
            if k in wr:
                out[f"{name}_sharpe"] = f"{wr[k]['sharpe']:.2f}"
                out[f"{name}_months"] = _pcu(wr[k]["months_in_profit"])
                out[f"{name}_worst_month"] = _pc(wr[k]["worst_month"])
                out[f"{name}_streak"] = str(wr[k]["longest_losing_streak_mo"])
                out[f"{name}_dd"] = _pc(wr[k]["max_dd"])

        # the book-level walk-forward (run_wf_book) — the wider out-of-sample evidence §5c/§6e leans on
        w = _load("master_book_wf_summary.json").get("headline_wf_oos") or {}
        if w:
            out.update({"wf_sharpe": f"{w['sharpe']:.2f}", "wf_dd": _pc(w["max_dd"]),
                        "wf_window": f"{w['start'][:4]}→{w['end'][:4]}",
                        "wf_months": _pcu(w["months_in_profit"])})
        wfs = _load("master_book_wf_summary.json")
        rng = wfs.get("window_cadence_invariance_range")
        if rng:
            out["wf_invariance"] = f"[+{rng[0]:.2f}, +{rng[1]:.2f}]"
            out["wf_invariance_spread"] = f"{rng[1] - rng[0]:.2f}"
        for lab, key in (("gfc", "2008 GFC"), ("volmageddon", "2018 Volmageddon"), ("covid", "COVID crash")):
            st = (wfs.get("stress") or {}).get(key)
            if st:
                out[f"wf_stress_{lab}"] = _pc(st["max_dd"])
                out[f"wf_stress_{lab}_legs"] = _word(st["n_legs"])
        out.update(_grid_span())
        out.update(_grid_ranges())
        out.update(_leg_swap_table())
        # SE of an annualised Sharpe (Lo 2002): sqrt((1 + S²/2)/T) on T independent observations, scaled
        # by the annualisation — the report leans on it to say what a 2-year block can and cannot show
        n_obs, sh = oos.get("n_obs"), oos.get("sharpe")
        if n_obs and sh is not None:
            ppy = n_obs / max((pd.Timestamp(s_window[1]) - pd.Timestamp(OOS_DATE)).days / 365.25, 1e-9)
            se = ((1.0 + 0.5 * (sh / ppy ** 0.5) ** 2) / n_obs) ** 0.5 * ppy ** 0.5
            out["oos_sharpe_se"] = f"±{se:.2f}"
            out["oos_months_n"] = str(round(n_obs / (ppy / 12)))
        # the 2010 flash-crash replay at unlevered risk — the tail the sizing argument turns on
        rb0 = _load("book/risk_budget.json").get("selective_leverage", {}).get("all legs", {})
        if rb0.get("stress_worst_month") is not None:
            out["stress_worst_month_1x"] = _pc(rb0["stress_worst_month"])
        out.update(_marginal(s.get("marginal") or []))
        out.update(_gate_table())
        out.update(_selective_leverage())
        out.update(_family_sources(s))
        out.update(_composition())
        out.update(_risk_budget_extras())
        out.update(_cap_binding())
        out.update(_ml_overlay())
        out.update(_family_costs())

        cl = _load("master_book_cost_levels.json")
        for lv in cl.get("levels", []):
            out[f"cost_{lv['label'].split('x')[0]}x_sharpe"] = _n(lv["sharpe"])
            out[f"cost_{lv['label'].split('x')[0]}x_dd"] = _pc(lv["max_dd"])
        if cl.get("breakeven_mult"):
            out["cost_breakeven"] = f"{cl['breakeven_mult']:.0f}×"

    if z:
        fn = dict((lab, n) for lab, n in z.get("funnel", []))
        out.update({
            "zoo_trials": f"{z['n_trials']:,}",
            "zoo_survivors": f"{z['n_survivors']:,}",
            "zoo_sharpe_is": _n(z["portfolio"]["sharpe_ann"]),
            "zoo_wf_oos": _n(z["wf_oos_sharpe"]),
            "zoo_dsr": f"{z['best_sleeve_dsr']:.2f}",
            "zoo_fdr": _pcu(z["placebo_fdr"]),
        })
        for lab, n in fn.items():
            key = ("funnel_generated" if "generated" in lab else
                   "funnel_in_sample" if "in-sample" in lab else
                   "funnel_walk_forward" if "walk-forward" in lab else
                   "funnel_monte_carlo" if "Monte Carlo" in lab else "funnel_entered")
            out[key] = f"{n:,}"
        cps = z.get("cost_per_sleeve") or {}
        if cps:
            out.update({"sleeve_cost_median": _pcu(cps["median_cost_share_of_gross_pnl"]),
                        "sleeve_cost_worst": _pcu(cps["worst_cost_share"]),
                        "sleeve_cost_worst_name": cps["worst_sleeve"],
                        "n_cost_fragile": str(cps["n_cost_fragile"])})

    if c:
        out.update({"cscv_pbo": _pcu(c["pbo"], 0), "cscv_n": f"{c['n_strategies']:,}",
                    "cscv_is_bar": _n(c["is_sharpe_mean"], 3), "cscv_oos_bar": _n(c["oos_sharpe_mean"], 3),
                    "cscv_p_loss": _pcu(c["prob_oos_loss"], 0)})
    out["longgamma_table"] = _longgamma_table()
    return out


if __name__ == "__main__":
    reg = build()
    print(f"{len(reg)} resolved numbers:\n")
    for k, v in sorted(reg.items()):
        print(f"  {{{{{k}}}}}".ljust(34), v)
