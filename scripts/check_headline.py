"""Fail loudly when README/REPORT claim a scorecard the artifacts do not support.

This exists because the failure keeps happening and is invisible to reading. Every family re-run moves
the book, the prose around the numbers stays internally consistent, and a target that has started
failing keeps its tick. It has now happened three times in one day — most recently with the README's
one-page table showing "months in profit >= 80% | 76.9% ✓".

So the check is mechanical: recompute the five §11 targets from reports/master_book_summary.json, then
assert that every scorecard figure and every pass/fail mark in the documents agrees. Run it before
calling anything done, and after any concurrent commit that touches a family series.

    python scripts/check_headline.py        # exit 1 on any disagreement
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"sharpe": (2.5, 4.0), "months_in_profit": (0.80, None), "max_dd": (-0.15, None),
           "longest_losing_streak_mo": (None, 2), "worst_month": (-0.06, None)}


def passes(card: dict) -> dict:
    return {k: (lo is None or card[k] >= lo) and (hi is None or card[k] <= hi)
            for k, (lo, hi) in TARGETS.items()}


def main() -> int:
    d = json.loads((ROOT / "reports" / "master_book_summary.json").read_text())
    full, oos = d["scorecard_full"], d["scorecard_oos"]
    n_full, n_oos = sum(passes(full).values()), sum(passes(oos).values())
    print(f"artifacts: full {n_full}/5  Sharpe {full['sharpe']:.2f}  months "
          f"{100 * full['months_in_profit']:.1f}%  streak {full['longest_losing_streak_mo']}")
    print(f"           OOS  {n_oos}/5  Sharpe {oos['sharpe']:.2f}  months "
          f"{100 * oos['months_in_profit']:.1f}%  streak {oos['longest_losing_streak_mo']}")

    bad = []
    readme = (ROOT / "README.md").read_text()
    # the one-page table's totals row: "| | **N / 5** | **M / 5** |" — OOS first, full second
    m = re.search(r"\|\s*\|\s*\*\*(\d) / 5\*\*\s*\|\s*\*\*(\d) / 5\*\*\s*\|", readme)
    if not m:
        bad.append("README: one-page scorecard totals row not found")
    elif (int(m.group(1)), int(m.group(2))) != (n_oos, n_full):
        bad.append(f"README totals say {m.group(1)}/5 OOS, {m.group(2)}/5 full; artifacts say "
                   f"{n_oos}/5, {n_full}/5")
    # a target row must not carry a tick while its own figure fails
    for row, key, fmt in [("months in profit", "months_in_profit", lambda c: 100 * c[key]),
                          ("longest losing streak", "longest_losing_streak_mo", lambda c: c[key])]:
        line = next((ln for ln in readme.splitlines() if ln.startswith(f"| {row}")), None)
        if line is None:
            bad.append(f"README: row '{row}' missing")
            continue
        marks = re.findall(r"[✓✗]", line)
        want = [passes(oos)[key], passes(full)[key]]
        if len(marks) != 2 or [x == "✓" for x in marks] != want:
            bad.append(f"README row '{row}' marks {marks} disagree with artifacts {want}")

    # "5/5 on both windows" must not survive a window that no longer passes
    for doc in ("README.md", "REPORT.md"):
        txt = (ROOT / doc).read_text()
        if (n_full < 5 or n_oos < 5) and re.search(r"all five targets on both windows", txt, re.I):
            bad.append(f"{doc}: claims 'all five targets on both windows' at {n_oos}/5 and {n_full}/5")

    for b in bad:
        print(f"  MISMATCH: {b}")
    print("HEADLINE OK" if not bad else f"HEADLINE STALE — {len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
