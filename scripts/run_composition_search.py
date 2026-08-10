"""Which family composition clears all five §11 targets — the whole search, not just its winner.

The book trades six families. That number is the one choice in this deliverable made *against* the
scorecard rather than before it, so the search behind it is published in full rather than summarised by
its result. Every other decision — equal weight, the frozen universes, the gate thresholds, the leverage
— is fixed before its outcome is seen; this one is not, and a report that quotes only the surviving
configuration is hiding the denominator.

So: take the eight validated families, run every single- and double-removal, and score all five targets
on both windows. The output records each configuration, how many targets it meets, and what the passing
ones cost — because "two of thirty-seven pass" is the honest headline, and the ratio only exists if the
thirty-seven are counted.

Removal only. Adding a family the deep-dives rejected would be a second search on top of this one, and
the point of publishing the denominator is not to enlarge it.

    python scripts/run_composition_search.py   ->  reports/book/composition_search.json
"""
from __future__ import annotations

import itertools
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
from src.config import OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402

OOS = pd.Timestamp(OOS_START).tz_localize(None)

#: The eight families that pass validation, with the file each publishes. The book's own FAMILIES list
#: carries six of them; trend and carry are named here because dropping them is the thing being measured
#: — the search cannot count a configuration it cannot build.
ALL_FAMILIES = [
    ("volprem", "volprem/volprem_book.parquet", "ret_gated"),
    ("xs_momentum", "xs/xs_book.parquet", "ret"),
    ("breakout", "breakout/bo_combined_portfolio.parquet", "ret"),
    ("crisis", "book/crisis_sleeve.parquet", "ret"),
    ("gmacro", "book/gmacro_sleeve.parquet", "ret"),
    ("bab", "bab/bab_book_c25.parquet", "ret"),
    ("trend_momentum", "trend/trend_block_returns.parquet", "ret"),
    ("carry", "carry/carry_breadth_headline.parquet", "ret"),
]
TARGETS = {"sharpe": (2.5, 4.0), "months_in_profit": (0.80, None), "max_dd": (-0.15, None),
           "longest_losing_streak_mo": (None, 2), "worst_month": (-0.06, None)}


def n_targets(c: dict) -> int:
    return sum((lo is None or c[k] >= lo) and (hi is None or c[k] <= hi)
               for k, (lo, hi) in TARGETS.items())


def misses(c: dict) -> list[str]:
    """What a configuration fails on, in the report's own words — a count alone hides which target broke."""
    out = []
    if not (2.5 <= c["sharpe"] <= 4.0):
        out.append(f"Sharpe {c['sharpe']:.2f}")
    if c["months_in_profit"] < 0.80:
        out.append(f"months {100 * c['months_in_profit']:.1f}%")
    if c["max_dd"] < -0.15:
        out.append(f"max-DD {100 * c['max_dd']:.1f}%")
    if c["longest_losing_streak_mo"] > 2:
        out.append(f"streak {c['longest_losing_streak_mo']}")
    if c["worst_month"] < -0.06:
        out.append(f"worst month {100 * c['worst_month']:.1f}%")
    return out


def frame(legs: dict[str, pd.Series]) -> pd.DataFrame:
    """The rescaled leg matrix on the book's own window — run_master_book.assemble, for a leg subset."""
    df = pd.DataFrame({k: mb.rescale(v) for k, v in legs.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
    return df[df.notna().sum(axis=1) >= 2]


def book(legs: dict[str, pd.Series]) -> pd.Series:
    """The canonical assembly, restricted to a subset of families — same rescale, same overlay, same
    leverage. Nothing here may differ from run_master_book except which columns enter the mean."""
    return mb.risk_overlay(frame(legs).mean(axis=1, skipna=True).dropna(), leverage=mb.BOOK_LEVERAGE)[0]


def score(legs: dict[str, pd.Series]) -> dict:
    b = book(legs)
    full, oos = mb.scorecard(b), mb.scorecard(b.loc[OOS:])
    # concentration is what a removal actually buys or costs — same definition as run_master_book's
    # pnl_share (each leg's contribution to the equal-weight sum, over the book's own window)
    df = frame(legs)
    contrib = (df / len(df.columns)).sum()
    share = round(float((contrib / contrib.sum()).get("volprem", float("nan"))), 4)
    return {"full": full, "oos": oos, "targets_full": n_targets(full), "targets_oos": n_targets(oos),
            "misses_full": misses(full), "misses_oos": misses(oos), "volprem_pnl_share": share}


def main():
    raw = {lab: mb.load(lab, f, c) for lab, f, c in ALL_FAMILIES}
    missing = [k for k, v in raw.items() if v is None]
    if missing:                                        # a family whose series is absent is not a removal
        raise SystemExit(f"missing family series, cannot count the denominator: {', '.join(missing)}")
    names = list(raw)

    configs = [("all eight", ())]
    configs += [(f"drop {d}", (d,)) for d in names]
    configs += [(f"drop {a} + {b}", (a, b)) for a, b in itertools.combinations(names, 2)]

    rows, passing = {}, []
    for label, drop in configs:
        legs = {k: v for k, v in raw.items() if k not in drop}
        r = score(legs)
        r["dropped"] = list(drop)
        r["n_families"] = len(legs)
        rows[label] = r
        both = r["targets_full"] == 5 and r["targets_oos"] == 5
        if both:
            passing.append(label)
        print(f"  {label:28s} {r['n_families']}f  full {r['targets_full']}/5  OOS {r['targets_oos']}/5"
              f"  Sh {r['full']['sharpe']:.2f}/{r['oos']['sharpe']:.2f}"
              f"  months {100 * r['full']['months_in_profit']:.1f}%/{100 * r['oos']['months_in_profit']:.1f}%"
              + ("   <-- clears both" if both else ""))

    # the shipped label is derived, not typed: whatever run_master_book actually assembles is the row
    # this search has to call "shipped", or the report's cost-of-passing would describe a different book
    shipped_drop = [n for n in names if n not in {lab for lab, _, _ in mb.FAMILIES}]
    shipped = next(lab for lab, drop in configs if sorted(drop) == sorted(shipped_drop))
    wide = frame(raw)
    solo_legs = {c: wide[c].dropna() for c in wide.columns}
    base, ship = rows["all eight"], rows[shipped]
    out = {"n_configurations": len(configs), "passing": passing, "shipped": shipped,
           # same convention as run_master_book's own standalone_sharpe — the vol-targeted leg over the
           # book's window, not the raw series, so §6d-ter's "trend sits between x-sect and gmacro" is
           # comparing against the numbers the README's source table prints
           "standalone_sharpe": {k: round(summarise(v, mb.ppy_of(v))["sharpe_ann"], 2)
                                 for k, v in solo_legs.items()},
           "cost_of_passing": {
               "sharpe_full": round(ship["full"]["sharpe"] - base["full"]["sharpe"], 2),
               "sharpe_oos": round(ship["oos"]["sharpe"] - base["oos"]["sharpe"], 2),
               "volprem_pnl_share": [base["volprem_pnl_share"], ship["volprem_pnl_share"]]},
           "configurations": rows}
    p = mb.R / "book" / "composition_search.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{len(passing)} of {len(configs)} configurations clear all five targets on both windows: "
          f"{', '.join(passing)}\nwrote {p}")


if __name__ == "__main__":
    main()
