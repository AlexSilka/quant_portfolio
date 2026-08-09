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
    """§4b's "then don't lever the aggressive leg" A/B: all eight legs at the shipped level against the
    vol-matched alternative that holds vol-prem at 1.00× and gives the risk to the other seven. From
    run_risk_budget's own json, so the table cannot drift from the experiment."""
    sl = _load("book/risk_budget.json").get("selective_leverage") or {}
    a, b = sl.get("all legs"), sl.get("ex-volprem")
    if not (a and b):
        return {}

    def col(d, key, fmt):
        return fmt(d[key]) if key in d else fmt(d["full"][key])
    lev_b = next(iter(b["leverage"].values()))
    rows = [f"| | all eight legs {next(iter(a['leverage'].values())):.2f}× | "
            f"seven legs {lev_b:.2f}×, vol-prem 1.00× |", "|---|---|---|"]
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
    return {"selective_leverage_table": "\n".join(rows)}


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


def _grid_span():
    """How much the Sharpe actually moves across the whole leverage grid — the claim §4b makes about it
    being flat, taken from the grid instead of asserted."""
    import csv
    p = R / "book" / "risk_budget_grid.csv"
    if not p.exists():
        return {}
    sh = [float(r["full_sharpe"]) for r in csv.DictReader(p.open()) if r["limits"] == "book_equity"]
    return {"grid_sharpe_range": f"{max(sh):.2f} → {min(sh):.2f}"} if sh else {}


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
            "mc_hit_p5": _pcu(bb.get("hit_p5", m.get("mc_hit_p5", float("nan"))), 0),
            "mc_wmonth_p5": _pc(bb.get("wmonth_p5", float("nan"))),
            "turnover": f"{s.get('annual_turnover', float('nan')):.0f}×",
            "volprem_pnl_share": _pcu(s.get("pnl_share", {}).get("volprem", float("nan")), 0),
            "n_years": str(len(s.get("per_year", {}))),
            # spelled out for prose ("an eight-family book"); unsigned for "≈ 0.06" style
            "n_families_word": {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(
                len(s["families"]), str(len(s["families"]))),
            "mean_corr_abs": f"{abs(s['mean_correlation']):.2f}",
            "n_years_positive": str(sum(1 for v in s.get("per_year", {}).values() if v > 0)),
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
        out.update(_grid_span())
        # the 2010 flash-crash replay at unlevered risk — the tail the sizing argument turns on
        rb0 = _load("book/risk_budget.json").get("selective_leverage", {}).get("all legs", {})
        if rb0.get("stress_worst_month") is not None:
            out["stress_worst_month_1x"] = _pc(rb0["stress_worst_month"])
        out.update(_marginal(s.get("marginal") or []))
        out.update(_gate_table())
        out.update(_selective_leverage())
        out.update(_ml_overlay())

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
    return out


if __name__ == "__main__":
    reg = build()
    print(f"{len(reg)} resolved numbers:\n")
    for k, v in sorted(reg.items()):
        print(f"  {{{{{k}}}}}".ljust(34), v)
