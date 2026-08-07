"""Phase 6 — portfolio-level risk management (task §8): does volatility-managing the book and a
drawdown-responsive de-risking ladder RAISE the Sharpe (not just the return) and pin the tail
drawdown to the 15% budget?

For each candidate book it compares: raw · book-level vol-target · drawdown ladder · both, on
Sharpe (IS/OOS), point drawdown, Monte-Carlo P5 tail drawdown, and CAGR. Then it sizes the book to
the 15% *tail* (MC-P5) drawdown budget and reports what that costs/earns.

    python scripts/trend/run_trend_risk.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import sh  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.overlay import apply_overlay, drawdown_ladder, vol_managed  # noqa: E402
from src.validation.monte_carlo import mc_metrics  # noqa: E402


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (365 / len(r)) - 1) if len(r) else 0.0


def stat(r: pd.Series, mc: bool = True) -> dict:
    r = r.dropna()
    s = summarise(r, 365)
    oos = r[r.index >= T.OOS_START]
    out = {"sharpe": round(s["sharpe_ann"], 2), "sharpe_oos": sh(oos), "cagr": round(cagr(r), 3),
           "vol": round(float(r.std() * np.sqrt(365)), 3), "dd_point": round(s["max_dd"], 3)}
    if mc:
        out["dd_tail_p5"] = mc_metrics(r, 365, 600, T.SEED).get("maxdd_p5")
    return out


def load_books() -> dict[str, pd.Series]:
    books = {}
    p = T.REPORTS / "trend_book_asym.parquet"
    if p.exists():
        books["ema_asym"] = pd.read_parquet(p)["ret"]
    p = T.REPORTS / "trend_book_blend_long_only.parquet"
    if p.exists():
        books["blend_LO"] = pd.read_parquet(p)["ret"]
    e = T.CACHE / "sleeves_ema_reversal_long_only_lag2_1d4h1h.parquet"
    b = T.CACHE / "sleeves_blend_reversal_long_only_lag2_1d4h1h.parquet"
    if e.exists() and b.exists():
        books["ema+blend_LO"] = pd.concat([pd.read_parquet(e), pd.read_parquet(b)], axis=1).mean(axis=1)
    return books


def budget_target_vol(ret: pd.Series, dd_budget: float = 0.15, cap: float = 3.0) -> dict:
    """Scan book-level vol targets; return the one whose MC-P5 tail drawdown ≈ the budget."""
    best = None
    for tv in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]:
        vm, expo = vol_managed(ret, target_vol=tv, cap=cap)
        fin, _ = drawdown_ladder(vm)
        s = stat(fin)
        s["target_vol"] = tv
        s["avg_gross"] = round(float((expo).mean()), 2)
        if best is None or abs((s["dd_tail_p5"] or -9) + dd_budget) < abs((best["dd_tail_p5"] or -9) + dd_budget):
            best = s
        if s["dd_tail_p5"] is not None and s["dd_tail_p5"] <= -dd_budget:
            return s          # first target whose tail DD reaches the budget
    return best


def main():
    books = load_books()
    print(f"=== Portfolio risk overlays on the trend book(s): {list(books)} ===\n")
    results = {}
    for name, ret in books.items():
        raw = stat(ret)
        vm, vm_expo = vol_managed(ret, target_vol=0.12, cap=3.0)
        vms = stat(vm)
        dl, dl_expo = drawdown_ladder(ret)
        dls = stat(dl)
        both, expos = apply_overlay(ret, target_vol=0.12, cap=3.0)
        boths = stat(both)
        budget = budget_target_vol(ret)
        results[name] = {"raw": raw, "vol_managed_12": vms, "ladder": dls, "both_12": boths, "budget15": budget}

        print(f"--- {name} ---")
        print(f"  {'variant':16s} {'Sharpe':>7s} {'OOS':>6s} {'CAGR':>7s} {'vol':>6s} {'DDpoint':>8s} {'DDtail_P5':>10s}")
        for lab, s in [("raw 1x", raw), ("vol-managed 12%", vms), ("drawdown ladder", dls), ("vol+ladder 12%", boths)]:
            print(f"  {lab:16s} {s['sharpe']:>+7.2f} {s['sharpe_oos']:>+6.2f} {s['cagr']:>+7.1%} "
                  f"{s['vol']:>6.1%} {s['dd_point']:>+8.1%} {(s['dd_tail_p5'] or float('nan')):>+10.1%}")
        print(f"  -> sized to 15% TAIL budget: target_vol {budget['target_vol']:.0%}, avg gross {budget['avg_gross']}x"
              f"  =>  Sharpe {budget['sharpe']:+.2f} (OOS {budget['sharpe_oos']:+.2f})  CAGR {budget['cagr']:+.1%}  "
              f"DD point {budget['dd_point']:+.1%} / tail {budget['dd_tail_p5']:+.1%}\n")

    (T.REPORTS / "trend_risk.json").write_text(json.dumps(results, indent=2, default=float))
    print("wrote reports/trend/trend_risk.json")
    # headline read: did vol-management raise Sharpe?
    for name, r in results.items():
        d = r["vol_managed_12"]["sharpe"] - r["raw"]["sharpe"]
        print(f"  {name}: vol-managing Sharpe delta {d:+.2f} ({r['raw']['sharpe']:+.2f} -> {r['vol_managed_12']['sharpe']:+.2f})")


if __name__ == "__main__":
    main()
