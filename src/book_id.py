"""Identity of the assembled book, so a figure computed against an older one cannot pass for current.

Every gate in this repo compares one thing to its own inputs: the report checks it matches the artifacts,
each producer's output matches what it read. Nothing checked that the artifacts agree with EACH OTHER —
so a leverage grid, a composition search or an ML contribution measured against last week's book stayed
in the report, byte-consistent with itself, quoting a portfolio that no longer exists. Seven of them were.

A fingerprint fixes that in the one place it can be fixed: the book stamps its own identity, everything
derived from it records the identity it used, and `scripts/check_freshness.py` compares. Modification
times cannot do this job — a checkout, a copy or a re-run that changes nothing all move them, and none
of those mean what a stale figure means.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SUMMARY = Path(__file__).resolve().parents[1] / "reports" / "master_book_summary.json"


def fingerprint(s: pd.Series) -> str:
    """Twelve hex chars over the book's own returns — enough to distinguish books, short enough to read.

    Rounded to 1e-9 first: a rebuild that changes nothing must produce the same id, or the gate cries
    wolf on every re-run and gets ignored, which is worse than not having it.
    """
    s = s.dropna().round(9)
    payload = f"{s.index[0]}|{s.index[-1]}|{len(s)}|" + ",".join(f"{v:.9f}" for v in s.to_numpy())
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def current() -> str | None:
    """The id of the book on disk right now, or None if it has never been built."""
    if not SUMMARY.exists():
        return None
    return json.loads(SUMMARY.read_text()).get("book_id")


def stamp(doc: dict) -> dict:
    """Record which book a derived artifact was computed against. Call it on the dict being written."""
    doc["book_id"] = current()
    return doc
