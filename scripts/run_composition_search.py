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
from src.config import CAPITAL_USD, OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.book_id import stamp  # noqa: E402

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
    return mb.risk_overlay(mb.book_stack(frame(legs)).dropna(), leverage=mb.BOOK_LEVERAGE)[0]


def ret(s: pd.Series) -> dict:
    """Return and risk, not only the ratio.

    A composition change reweights every leg, so it moves the book's volatility as well as its return, and
    Sharpe cannot tell those apart. Reporting only the ratio made a change that RAISED return by ~10% read
    as a cost, because the ratio fell — the same blindness this report criticises in its ML section."""
    yrs = (s.index.max() - s.index.min()).days / 365.25
    return {"cagr": round(float((1 + s).prod() ** (1 / yrs) - 1), 4),
            "pnl_usd_per_year": round(float(s.sum()) * CAPITAL_USD / yrs, 0),
            "vol": round(float(s.std(ddof=1) * (365 ** 0.5)), 4)}


def score(legs: dict[str, pd.Series]) -> dict:
    b = book(legs)
    full, oos = {**mb.scorecard(b), **ret(b)}, {**mb.scorecard(b.loc[OOS:]), **ret(b.loc[OOS:])}
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
    legs_shipped = {k: v for k, v in raw.items() if k not in shipped_drop}
    wide = frame(raw)
    solo_legs = {c: wide[c].dropna() for c in wide.columns}
    # the honest like-for-like: any composition change moves book vol, so the eight-family book is also
    # scored at the leverage that puts it on the SHIPPED book's realised volatility. Same risk, same
    # question — does the wider book actually earn more?
    b8, b6 = mb.book_stack(frame(raw)).dropna(), mb.book_stack(frame(legs_shipped)).dropna()
    lev = mb.BOOK_LEVERAGE * float(b6.std(ddof=1) / b8.std(ddof=1))
    m8 = mb.risk_overlay(b8, leverage=lev)[0]
    vol_matched = {"leverage": round(lev, 3), "full": {**mb.scorecard(m8), **ret(m8)},
                   "oos": {**mb.scorecard(m8.loc[OOS:]), **ret(m8.loc[OOS:])}}
    vol_matched["targets_full"] = n_targets(vol_matched["full"])
    vol_matched["targets_oos"] = n_targets(vol_matched["oos"])
    base, ship = rows["all eight"], rows[shipped]
    out = {"n_configurations": len(configs), "passing": passing, "shipped": shipped,
           # same convention as run_master_book's own standalone_sharpe — the vol-targeted leg over the
           # book's window, not the raw series, so §6d-ter's "trend sits between x-sect and gmacro" is
           # comparing against the numbers the README's source table prints
           "standalone_sharpe": {k: round(summarise(v, mb.ppy_of(v))["sharpe_ann"], 2)
                                 for k, v in solo_legs.items()},
           "vol_matched_eight": vol_matched,
           "cost_of_passing": {
               "sharpe_full": round(ship["full"]["sharpe"] - base["full"]["sharpe"], 2),
               "sharpe_oos": round(ship["oos"]["sharpe"] - base["oos"]["sharpe"], 2),
               "cagr_full": round(ship["full"]["cagr"] - base["full"]["cagr"], 4),
               "cagr_oos": round(ship["oos"]["cagr"] - base["oos"]["cagr"], 4),
               "cagr_full_vol_matched": round(ship["full"]["cagr"] - vol_matched["full"]["cagr"], 4),
               "cagr_oos_vol_matched": round(ship["oos"]["cagr"] - vol_matched["oos"]["cagr"], 4),
               "volprem_pnl_share": [base["volprem_pnl_share"], ship["volprem_pnl_share"]]},
           "configurations": rows}
    p = mb.R / "book" / "composition_search.json"
    p.write_text(json.dumps(stamp(out), indent=2, default=str))
    print(f"\n{len(passing)} of {len(configs)} configurations clear all five targets on both windows: "
          f"{', '.join(passing)}\nwrote {p}")


if __name__ == "__main__":
    main()
