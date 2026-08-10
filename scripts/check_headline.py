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
    # The one-page table scores the FROZEN BLOCK and shows the 15-year column beside it as supporting
    # evidence — the brief scores the block, and presenting both as scorecards is what let a self-invented
    # both-windows tally distort real decisions. So the totals row carries the block's count and a dash,
    # and each target row carries ONE mark, the block's. This check used to demand two of each and had
    # been failing on the current README's shape rather than on anything the README got wrong.
    m = re.search(r"\|\s*\|\s*\*\*(\d) / 5\*\*\s*\|\s*(?:—|\*\*(\d) / 5\*\*)\s*\|", readme)
    if not m:
        bad.append("README: one-page scorecard totals row not found")
    else:
        if int(m.group(1)) != n_oos:
            bad.append(f"README totals say {m.group(1)}/5 on the block; artifacts say {n_oos}/5")
        if m.group(2) is not None and int(m.group(2)) != n_full:
            bad.append(f"README totals say {m.group(2)}/5 full; artifacts say {n_full}/5")
    # a target row must not carry a tick while the block's own figure fails
    for row, key in [("months in profit", "months_in_profit"),
                     ("longest losing streak", "longest_losing_streak_mo"),
                     ("Sharpe", "sharpe"), ("max drawdown", "max_dd"), ("worst month", "worst_month")]:
        line = next((ln for ln in readme.splitlines() if ln.lower().startswith(f"| {row.lower()}")), None)
        if line is None:
            continue                      # the table names its rows freely; only score the ones present
        marks = [x == "\u2713" for x in re.findall(r"[\u2713\u2717]", line)]
        if marks and marks[0] != passes(oos)[key]:
            bad.append(f"README row '{row}' marks the block {marks[0]} against artifacts {passes(oos)[key]}")

    # A LIVE claim that the long window also clears everything, made while it does not. The phrase is
    # allowed in the past tense — §6d-ter and the README both recount that trend+carry once cleared both
    # windows, and immediately say it is no longer the reason to hold the composition — so only an
    # unqualified present-tense assertion counts.
    for doc in ("README.md", "REPORT.md"):
        txt = (ROOT / doc).read_text()
        live = re.search(r"(?:clears|meets|passes|hits)\s+all\s+five\s+targets\s+on\s+both\s+windows", txt, re.I)
        if n_full < 5 and live:
            bad.append(f"{doc}: claims the book 'clears all five targets on both windows' at {n_full}/5 full")

    for b in bad:
        print(f"  MISMATCH: {b}")
    print("HEADLINE OK" if not bad else f"HEADLINE STALE — {len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
