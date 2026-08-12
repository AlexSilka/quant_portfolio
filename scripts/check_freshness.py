"""Fail the build when a figure in the report was measured against a book that no longer exists.

Every other gate here compares one artifact to its own inputs, which is why this whole class of defect
walked straight through them: a leverage grid, a composition search, an ML contribution and a crisis
sizing study were all internally consistent and all describing a previous portfolio, and the report
quoted them beside numbers from the current one.

Each derived artifact records the `book_id` it was computed against (src/book_id.stamp). This compares
those to the book on disk and names every one that has fallen behind, with the command that refreshes it.

    python scripts/check_freshness.py            report, exit 1 if anything is behind
    python scripts/check_freshness.py --list     show every registry entry and its state
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.book_id import current

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

# artifact -> the command that rebuilds it. Anything measured AGAINST the assembled book belongs here;
# a family's own deep-dive series does not, because it is an input to the book rather than a reading of it.
REGISTRY = {
    "master_book_wf_summary.json": "python scripts/run_wf_book.py",
    "book/cscv_pbo.json": "python scripts/run_cscv.py",
    "book/family_cost_shares.json": "python scripts/measure_family_costs.py",
    "book/risk_budget.json": "python scripts/run_risk_budget.py",
    "book/composition_search.json": "python scripts/run_composition_search.py",
    "book/ml_book_contribution.json": "python scripts/run_ml_book_contribution.py",
    "book/ml_portfolio_overlay.json": "python scripts/run_ml_portfolio_overlay.py",
    "lab/crisis_lab.json": "python scripts/run_crisis_lab.py",
    "lab/longgamma_search.json": "python scripts/run_longgamma_search.py",
    "lab/live_book.json": "python scripts/run_live_book.py",
}


def main() -> None:
    book = current()
    if book is None:
        raise SystemExit("no reports/master_book_summary.json — run `python scripts/run_master_book.py` first")

    stale, unstamped, missing = [], [], []
    for rel, cmd in sorted(REGISTRY.items()):
        p = REP / rel
        if not p.exists():
            missing.append((rel, cmd))
            continue
        got = json.loads(p.read_text()).get("book_id")
        if got is None:
            unstamped.append((rel, cmd))
        elif got != book:
            stale.append((rel, got, cmd))

    if "--list" in sys.argv:
        print(f"book_id {book}\n")
        for rel, cmd in sorted(REGISTRY.items()):
            p = REP / rel
            got = json.loads(p.read_text()).get("book_id") if p.exists() else None
            state = "missing" if not p.exists() else ("unstamped" if got is None else
                                                      ("current" if got == book else f"stale ({got})"))
            print(f"  {state:22s} {rel}")
        return

    for rel, cmd in missing:
        print(f"  missing   {rel} — {cmd}")
    for rel, cmd in unstamped:
        print(f"  unstamped {rel} — records no book_id; {cmd}")
    if stale:
        lines = "\n".join(f"    {rel}  (measured against {got}, book is {book})\n      fix: {cmd}"
                          for rel, got, cmd in stale)
        raise SystemExit(f"{len(stale)} artifact(s) describe a book that is no longer assembled:\n{lines}")
    print(f"all {len(REGISTRY) - len(missing)} book-derived artifacts are current with book {book}")


if __name__ == "__main__":
    main()
