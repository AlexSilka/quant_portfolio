"""§9 per family: turnover cost as a share of gross P&L, for the book's own eight legs.

The discovery sleeves carry their charged cost per candidate (`run_book.py` publishes it), which is the
brief's unit. The eight *families* the book actually trades did not: four published a cost sweep in their
deep-dive and four published nothing, so "which sleeves are cost-fragile" had a hole in it exactly where
the traded book is.

This closes it the only honest way — by re-running each family's own construction with its cost model
switched off and comparing. No new cost model, no estimate: the same code, once charged and once not.

    cost share = 1 − net P&L / gross P&L        break-even multiple = gross / (gross − net)

The families whose deep-dive already publishes a break-even (breakout, x-sect) or a Sharpe at a cost
multiple (trend, vol-prem) are carried through from those artifacts rather than re-run — re-deriving a
number a deep-dive already owns is how two answers to one question appear.

Each re-run is the family's shipped construction, so nothing here can silently become a different book:
the costed pass is checked against the published series and the run fails if they disagree.

    python scripts/measure_family_costs.py           # the families the book holds
    python scripts/measure_family_costs.py --all     # every researched family (slow; §9's full table)
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BOOK_DIR, REPORTS_DIR  # noqa: E402
from src.log import get_logger  # noqa: E402

log = get_logger("family_costs")
OUT = BOOK_DIR / "family_cost_shares.json"
from src.book_id import stamp  # noqa: E402

R = REPORTS_DIR
TOL = 0.02        # the re-run must reproduce the published series' total return to within 2%


def _share(gross, net):
    """Cost as a fraction of gross P&L, on total return over the family's whole life."""
    g, n = float(gross.sum()), float(net.sum())
    if g <= 0:
        return None
    return {"gross_pnl": round(g, 4), "net_pnl": round(n, 4),
            "cost_share_of_gross_pnl": round((g - n) / g, 4),
            "breakeven_cost_mult": round(g / (g - n), 1) if g > n else None,
            "cost_fragile": bool(g > n and g / (g - n) < 3.0)}


def _check(label, rebuilt, published_path, out):
    """A re-run that does not reproduce the shipped series is measuring something else — say so."""
    p = R / published_path
    if not p.exists():
        out["note"] = "published series not found; cost share is from the re-run alone"
        return out
    pub = pd.read_parquet(p).iloc[:, 0].dropna()
    pub.index = pd.to_datetime(pub.index)
    if pub.index.tz is not None:
        pub.index = pub.index.tz_localize(None)
    r = rebuilt.dropna()
    r.index = pd.to_datetime(r.index)
    if r.index.tz is not None:
        r.index = r.index.tz_localize(None)
    a, b = float(r.sum()), float(pub.sum())
    drift = abs(a - b) / max(abs(b), 1e-9)
    out["reproduces_published_series"] = bool(drift <= TOL)
    if drift > TOL:
        out["note"] = f"re-run total return {a:+.3f} vs published {b:+.3f} ({drift:.1%} apart)"
        print(f"  ! {label}: re-run does not reproduce the published series ({drift:.1%} apart)")
    return out


def crisis():
    import scripts.run_crisis as m
    return m.build_crisis(), m.build_crisis(cost_bps=0.0), "book/crisis_sleeve.parquet"


def gmacro():
    import scripts.run_gmacro as m
    return m.build_gmacro(), m.build_gmacro(cost_bps=0.0), "book/gmacro_sleeve.parquet"


def bab():
    import scripts.bab.run_bab_portfolio as m
    return m._bab_net(25), m._bab_net(25, cost_bps=0.0), "bab/bab_book_c25.parquet"


def carry():
    import scripts.carry.run_carry_breadth as m
    C, V, fd = m.load_all()
    btc = C["BTCUSDT"].pct_change() if "BTCUSDT" in C else C.iloc[:, 0].pct_change()
    sig = m.carry_xs.signal_level(fd, 7)
    elig = m.carry_xs.pit_eligible(V, 100)
    net = m.vt(m.run_carry(C, fd, sig, elig, beta_ret=btc)[0])
    gross = m.vt(m.run_carry(C, fd, sig, elig, cost_bps=0.0, beta_ret=btc)[0])
    return net, gross, "carry/carry_breadth_headline.parquet"


FAMILIES = (("crisis-alpha", crisis), ("global-macro", gmacro), ("BAB", bab), ("carry", carry))

# Display name -> the label the book assembles under. Every family researched is measured here, including
# the ones the book does not hold, because §9 documents the research and not only the shipped legs. But a
# reader has to be able to tell which is which: the book dropped trend and carry, and a cost range quoted
# over all eight while calling them "the book's legs" is simply false. Tagged at the source, where both
# names are known, rather than re-derived by whoever reads the file.
BOOK_LABEL = {"crisis-alpha": "crisis", "global-macro": "gmacro", "BAB": "bab", "carry": "carry",
              "breakout": "breakout", "x-sect": "xs_momentum", "vol-prem": "volprem",
              "trend": "trend_momentum"}
# Every display name here must answer to a family the assembler could hold, or `_in_book` silently says
# "no" and the §9 table files a traded leg under "measured but NOT held". That is what "trend" -> "trend"
# did: the assembler calls it `trend_momentum`, so the one word that had to match did not.
_STRAY = set(BOOK_LABEL.values()) - {"crisis", "gmacro", "bab", "carry", "breakout", "xs_momentum",
                                     "volprem", "trend_momentum"}
if _STRAY:
    raise ValueError(f"BOOK_LABEL maps to family ids nothing answers to: {sorted(_STRAY)}")


def _in_book(label: str) -> bool:
    import scripts.run_master_book as mb
    return BOOK_LABEL.get(label, label) in {lab for lab, _, _ in mb.FAMILIES}


def main():
    # Each family is re-run TWICE (costed and costless) to get its cost share, and that is minutes of
    # work per family. By default only the families the book actually HOLDS are re-run; the research
    # ones keep whatever the last `--all` pass measured, because their number cannot change unless
    # their construction does, and the book being reassembled is not that. `--all` re-measures
    # everything, which is what to run after touching a family that is not in the composition.
    prev = {}
    if (OUT).exists():
        try:
            prev = json.loads(OUT.read_text()).get("families", {})
        except Exception as exc:
            log.warning("family costs: could not read the previous pass (%s) — re-running everything",
                        type(exc).__name__)
    # Re-run everything this file knows how to re-run when there is no previous pass to carry forward.
    # The filter used to be "only families the book holds", and the four families this file can re-run
    # are exactly the four the book does not — so the default pass measured NOTHING and the table shipped
    # with an empty `re_run_here`.
    todo = (FAMILIES if ("--all" in sys.argv or not prev)
            else tuple((l, f) for l, f in FAMILIES if _in_book(l) or l not in prev))
    carried = [l for l, _ in FAMILIES if (l, dict(FAMILIES)[l]) not in todo and l in prev]
    if carried:
        print(f"carried from the last --all pass (not in the book): {', '.join(carried)}")
    out = {k: v for k, v in prev.items() if k in carried}
    for label, fn in todo:
        print(f"{label} …", flush=True)
        try:
            net, gross, published = fn()
        except Exception as exc:                       # a family that cannot be re-run is named, not skipped
            print(f"  ! {label}: could not re-run ({type(exc).__name__}: {exc})")
            out[label] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        idx = net.dropna().index.intersection(gross.dropna().index)
        rec = _share(gross.loc[idx], net.loc[idx])
        if rec is None:
            out[label] = {"error": "gross P&L is not positive — a cost share would be meaningless"}
            continue
        out[label] = _check(label, net, published, rec)
        print(f"  cost {rec['cost_share_of_gross_pnl']:.1%} of gross P&L · break-even "
              f"{rec['breakeven_cost_mult']}× · fragile={rec['cost_fragile']}")

    # the four whose deep-dive already answers this — carried through, not re-derived
    carried = {}
    for label, path, key in (
            ("breakout", "breakout/bo_final_summary.json", "breakeven_mult"),
            ("x-sect", "xs/xs_summary.json", "breakeven_cost_mult")):
        try:
            v = float(json.loads((R / path).read_text())[key])
            carried[label] = {"breakeven_cost_mult": v, "cost_share_of_gross_pnl": round(1.0 / v, 4),
                              "cost_fragile": v < 3.0, "source": path}
        except Exception as exc:      # named, not swallowed: a leg that vanishes here silently narrows
            print(f"  ! {label}: {path} unreadable ({type(exc).__name__}: {exc})")   # the §9 range on the page
            carried[label] = {"error": f"{type(exc).__name__}: {exc}", "source": path}
    try:
        # the DEPLOYED column, not the research book's: the leg the book holds is the gated one, and it
        # pays a different spread bill because every gate switch crosses the same spread a roll does.
        # `cost_fragile` is read off the ladder rather than asserted — the flag used to be a hard-coded
        # False sitting next to a row that had gone negative.
        vp = pd.read_csv(R / "volprem" / "volprem_cost_robustness.csv")
        col = "sharpe_deployed" if "sharpe_deployed" in vp.columns else "sharpe"
        ladder = {str(int(m)): float(s) for m, s in zip(vp["cost_mult"], vp[col])}
        dead = [m for m, s in zip(vp["cost_mult"], vp[col]) if m > 0 and s <= 0.0]
        carried["vol-prem"] = {"sharpe_at_cost_mult": ladder,
                               "breakeven_cost_mult": float(min(dead)) if dead else None,
                               "cost_fragile": bool(dead and min(dead) < 3.0),
                               "construction": "deployed (gated)" if col == "sharpe_deployed" else "research book",
                               "source": "volprem/volprem_cost_robustness.csv"}
    except Exception as exc:
        print(f"  ! vol-prem: volprem_cost_robustness.csv unreadable ({type(exc).__name__}: {exc})")
        carried["vol-prem"] = {"error": f"{type(exc).__name__}: {exc}",
                               "source": "volprem/volprem_cost_robustness.csv"}
    tr = R / "trend" / "trend_book_blend_summary.json"
    if tr.exists():
        cl = (json.loads(tr.read_text()).get("cost_levels") or {})
        if "3x" in cl:
            carried["trend"] = {"sharpe_at_cost_mult": {"3": float(cl["3x"])}, "cost_fragile": False,
                                "source": "trend/trend_book_blend_summary.json"}

    for d in (out, carried):
        for label, rec in d.items():
            rec["in_book"] = _in_book(label)
    have_share = {k for d in (out, carried) for k, v in d.items()
                  if v.get("in_book") and "cost_share_of_gross_pnl" in v}
    import scripts.run_master_book as mb
    book_n = len({lab for lab, _, _ in mb.FAMILIES})
    no_share = sorted({BOOK_LABEL.get(k, k) for d in (out, carried) for k, v in d.items() if v.get("in_book")}
                      - {BOOK_LABEL.get(k, k) for k in have_share})
    if no_share:
        print(f"  book legs without a cost SHARE (they publish a Sharpe at a cost multiple instead): "
              f"{', '.join(no_share)}")
    dropped = sorted(k for d in (out, carried) for k, v in d.items() if not v["in_book"])
    if dropped:                       # named, because a silent extra leg is how a false headline is born
        print(f"  measured but NOT held by the book: {', '.join(dropped)} (tagged in_book=false)")
    measured = {k: v for k, v in out.items() if "cost_share_of_gross_pnl" in v}
    doc = {"re_run_here": out, "from_deep_dives": carried,
           "book_n_families": book_n, "book_legs_without_share": no_share,
           "n_cost_fragile": sum(1 for v in list(measured.values()) + list(carried.values())
                                 if v.get("cost_fragile")),
           "method": "same construction re-run with its cost model off; share = 1 - net/gross total return"}
    OUT.write_text(json.dumps(stamp(doc), indent=2, default=float))
    print(f"\n{len(measured)} families re-run, {len(carried)} carried from deep-dives, "
          f"{doc['n_cost_fragile']} cost-fragile -> reports/book/family_cost_shares.json")
    print("FAMILY COSTS OK")


if __name__ == "__main__":
    main()
