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
            "turnover": f"{s.get('annual_turnover', float('nan')):.0f}×",
            "turnover_held": f"{s.get('annual_turnover_weights_held', float('nan')):.1f}×",
        })
        if held:
            out.update({"held_sharpe": f"{held['sharpe']:.2f}", "held_dd": _pc(held["max_dd"]),
                        "held_worst_month": _pc(held["worst_month"])})
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
