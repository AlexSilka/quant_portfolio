"""How much leverage the book can carry — and what sets the limit.

The assembled book runs at ~8.4% annualised vol against a -15% drawdown mandate, so it looks
under-risked and the obvious move is to lever it. This script answers *how far* from the stated risk
budget rather than from the scorecard, through the canonical assembler (run_master_book), on three
independent readings of the tail:

  1. GRID          — constant leverage 1.00-2.00x in 0.05 steps, all five targets, full window and the
                     frozen OOS block, under both §8 limit conventions (below). Diagnostic only: the
                     level is never chosen by reading the best row off this table (the scorecard is flat
                     at 4/5 from 1.00x to 1.45x, so it cannot pick a level even if asked).
  2. BOOTSTRAP     — the same book resampled (stationary block bootstrap): the 5th-percentile max-DD
                     and worst month, i.e. what the SAME return distribution does on an unlucky path.
                     Sizing to a single realised path is sizing to one draw.
  3. EVENT STRESS  — the vol-premium leg's documented systemic tail replayed inside the eight-leg book.
                     That tail is real and dated: 2010-05-06, the flash crash, -76.4% on the leg in one
                     day (-50.9% at its book weight) — and it sits OUTSIDE the 2011+ reporting window,
                     so no realised metric in the report contains it. Every leg that existed in 2010 is
                     replayed at its actual path, so the diversifiers get credit for what they really
                     did that day (crisis +2.0%, gmacro +4.1%).

It also answers the obvious "then don't lever the aggressive leg" — measured vol-matched, not argued.

§8 limit convention — the ladder (-6/-9/-12% -> 0.66/0.33/flat), its -4% restore and the -4%/day
breaker are quoted as percentages of *something*, and leverage forces the choice:
  book_equity  — percent of the levered book's equity: a trigger always means the same loss to the
                 investor, so the triggers stay in the same units as the -15% mandate they enforce;
  risk_budget  — percent of the unlevered stack, i.e. triggers scale with leverage, so the exposure
                 path is leverage-invariant and the levered book is a pure constant scaling.
Both are measured over the whole grid; the verdict below is what ships.

    make risk-budget      # ~2 min; prints the table, writes reports/book/risk_budget{,_grid.csv}
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scripts.run_master_book import (  # noqa: E402
    OOS, R, assemble, book_stack, ppy_of, risk_overlay, scorecard)
from src.config import BOOK_LEVERAGE, SEED  # noqa: E402
from src.validation.monte_carlo import mc_metrics  # noqa: E402

GRID = [round(x, 2) for x in np.arange(1.00, 2.001, 0.05)]
LIMIT_POLICIES = ("book_equity", "risk_budget")
MC_REPS = 1000                     # block-bootstrap replicates per grid point (42 points)

# the five task targets, as limits (Sharpe is a corridor, not "more is better")
SHARPE_LO, SHARPE_HI = 2.5, 4.0
MONTHS_MIN, DD_LIMIT, WORST_MONTH_LIMIT, STREAK_MAX = 0.80, -0.15, -0.06, 2

# ── event stress: the leg tail the reporting window excludes ───────────────────────────────────────
EVENT = ("2010-04-01", "2010-06-30")   # contains 2010-05-06; starts on a Thursday
REPLAY_FROM, REPLAY_TO = "2021-01-01", "2024-03-01"   # every quarter the event is replayed into: all
                                       # eight legs live (crypto legs start 2020) and the whole span
                                       # ends before the frozen OOS block. The reported stress is the
                                       # WORST placement, so the answer cannot ride on a lucky quarter.


def cagr(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((1 + s).prod() ** (1 / yrs) - 1) if yrs > 0 else 0.0


def targets_hit(sc):
    return sum([SHARPE_LO <= sc["sharpe"] <= SHARPE_HI, sc["months_in_profit"] >= MONTHS_MIN,
                sc["max_dd"] >= DD_LIMIT, sc["worst_month"] >= WORST_MONTH_LIMIT,
                sc["longest_losing_streak_mo"] <= STREAK_MAX])


def card(s):
    sc = scorecard(s)
    return {"sharpe": sc["sharpe"], "cagr": round(cagr(s), 4), "max_dd": sc["max_dd"],
            "worst_month": sc["worst_month"], "months_in_profit": sc["months_in_profit"],
            "streak": sc["longest_losing_streak_mo"], "targets": targets_hit(sc)}


def row_str(c):
    return (f"Sh {c['sharpe']:.2f} CAGR {c['cagr']:+6.1%} DD {c['max_dd']:+7.2%} worst {c['worst_month']:+7.2%} "
            f"mo {c['months_in_profit']:.1%} strk {c['streak']} [{c['targets']}/5]")


def tail(s):
    """Max-DD and worst month of the SAME return distribution on a resampled path (P5 = unlucky path)."""
    mc = mc_metrics(s.dropna(), ppy_of(s), MC_REPS, SEED)
    return {"mc_dd_p5": mc.get("maxdd_p5"), "mc_wmonth_p5": mc.get("wmonth_p5"),
            "mc_wmonth_p50": mc.get("wmonth_p50")}


def replay_starts(index):
    """One splice date per quarter of the replay span, weekday-aligned to the event window so the
    event's Thursday stays a Thursday (the equity legs' calendar has to survive the move)."""
    src0 = pd.Timestamp(EVENT[0])
    span = pd.Timestamp(EVENT[1]) - src0
    out = []
    for q in pd.date_range(REPLAY_FROM, REPLAY_TO, freq="QS"):
        d = q + pd.Timedelta(days=(src0.dayofweek - q.dayofweek) % 7)   # same weekday => whole weeks apart
        if d in index and d + span in index:
            out.append(d)
    return out


def book_of(legs, lev_map):
    """Equal-weight mean over the legs live each day, each leg scaled by its OWN leverage — the same
    assembly as run_master_book, generalised so leverage need not be uniform across families."""
    return book_stack(legs.mul(pd.Series({c: lev_map.get(c, 1.0) for c in legs.columns}), axis=1)).rename("ret")


def event_replay(df, wide, start):
    """Splice the 2010 systemic event into an all-eight-legs-live window of the current book.

    Legs that existed in 2010 (volprem, crisis, gmacro, x-sect) are replayed at their ACTUAL 2010 daily
    path, so the diversifiers get credit for what they really did; the four crypto legs, which had no
    2010, keep their actual returns in the target window — the neutral assumption, since how a crypto
    premium behaves in an equity flash crash is unknowable, not zero. The replayed legs are blanked
    first, so the 2010 equity calendar (no weekends) carries over and the book averages over exactly the
    legs live that day, as it does everywhere else.
    """
    src = wide.loc[EVENT[0]:EVENT[1]]
    delta = pd.Timestamp(start) - src.index[0]
    cols = [c for c in src.columns if src[c].notna().any()]
    out = df.copy()
    span = (out.index >= src.index[0] + delta) & (out.index <= src.index[-1] + delta)
    out.loc[span, cols] = np.nan
    for dt, r in src.iterrows():
        if (t := dt + delta) in out.index:
            out.loc[t, cols] = r[cols].to_numpy()
    return out                                        # the leg matrix, so callers can re-weight it


def worst_stress(stressed, lev, policy, lev_map=None):
    """The event replayed into every candidate quarter — reported at its WORST placement, so the
    verdict cannot ride on a splice that happened to land in a strong month."""
    cards = [card(risk_overlay(book_of(s, lev_map or {}), leverage=lev, limits=policy)[0])
             for s in stressed.values()]
    return {"stress_max_dd": min(c["max_dd"] for c in cards),
            "stress_worst_month": min(c["worst_month"] for c in cards),
            "stress_streak": max(c["streak"] for c in cards),
            "stress_months_in_profit": min(c["months_in_profit"] for c in cards)}


def largest_passing(grid, col, ok):
    """The largest grid leverage at which a constraint still holds (the grid is the exact answer —
    the overlay makes every metric a step function of leverage, not a straight line)."""
    good = [r.leverage for r in grid.itertuples() if ok(getattr(r, col))]
    return max(good) if good else None


def main():
    df, _ = assemble()
    wide, _ = assemble(start="2005-01-01")                 # same legs, extended back over the 2010 event
    ew = book_stack(df).rename("ret")
    ppy = ppy_of(ew)
    stack_vol = float(ew.std(ddof=1) * np.sqrt(ppy))
    pre = ew[ew.index < OOS]
    pre_vol = float(pre.std(ddof=1) * np.sqrt(ppy_of(pre)))
    oos_vol = float(ew[ew.index >= OOS].std(ddof=1) * np.sqrt(ppy_of(ew[ew.index >= OOS])))
    print(f"stack {ew.index.min().date()}..{ew.index.max().date()}  n={len(ew)}  ppy={ppy:.0f}\n"
          f"realised vol of the unlevered stack: full {stack_vol:.2%}  pre-OOS {pre_vol:.2%}  OOS {oos_vol:.2%}\n"
          f"mandate: max-DD {DD_LIMIT:.0%}, worst month {WORST_MONTH_LIMIT:.0%}, months >={MONTHS_MIN:.0%}, "
          f"streak <={STREAK_MAX}, Sharpe corridor {SHARPE_LO}-{SHARPE_HI}\n")

    starts = replay_starts(df.index)
    stressed = {str(s.date()): event_replay(df, wide, s) for s in starts}
    leg_day = float(wide["volprem"].loc[EVENT[0]:EVENT[1]].min())      # the leg's crash day at book weight
    book_day = min(float(book_of(s, {}).min()) for s in stressed.values())   # worst book day, any placement
    print(f"event stress: {EVENT[0]}..{EVENT[1]} replayed into {len(starts)} quarters "
          f"({starts[0].date()}..{starts[-1].date()}), reported at its worst placement; the leg loses "
          f"{leg_day:.1%} at book weight in one day, costing the unlevered book {book_day:.2%}\n")

    # ── the grid ───────────────────────────────────────────────────────────────────────────────────
    rows = []
    for policy in LIMIT_POLICIES:
        print(f"=== §8 limits measured in {policy.upper()} — constant leverage grid ===")
        print(f"{'lev':>5} | {'FULL WINDOW 2011+':<61} | {'OOS 2024-07+':<61} | 2010-event "
              f"{'DD':>7} {'month':>7} | bootstrap-P5 {'DD':>7} {'month':>7}")
        for lev in GRID:
            managed, gross, n_breaker = risk_overlay(ew, leverage=lev, limits=policy)
            full, oos = card(managed), card(managed[managed.index >= OOS])
            st, tl = worst_stress(stressed, lev, policy), tail(managed)
            rows.append({"limits": policy, "leverage": lev,
                         **{f"full_{k}": v for k, v in full.items()},
                         **{f"oos_{k}": v for k, v in oos.items()}, **st, **tl,
                         "breaker_days": n_breaker, "mean_gross": round(float(gross.mean()), 3),
                         "days_derisked": round(float((gross < lev - 1e-9).mean()), 4)})
            print(f"{lev:5.2f} | {row_str(full):<61} | {row_str(oos):<61} | "
                  f"          {st['stress_max_dd']:+7.2%} {st['stress_worst_month']:+7.2%} | "
                  f"             {tl['mc_dd_p5']:+7.2%} {tl['mc_wmonth_p5']:+7.2%}")
        print()

    grid = pd.DataFrame(rows)
    grid.to_csv(R / "book" / "risk_budget_grid.csv", index=False)

    # ── sanity: months-in-profit must be leverage-invariant under a PURE constant scaling ──────────
    # Any move in it is the §8 overlay changing the exposure path, not scaling. Under "risk_budget"
    # limits the overlay path is leverage-invariant by construction, so that column must be flat too.
    print("=== sanity: months-in-profit vs leverage (pure scaling must not move it) ===")
    base = risk_overlay(ew, leverage=1.0)[0]
    for lev in (1.0, 1.25, 1.5, 1.75, 2.0):
        pure = scorecard(base * lev)["months_in_profit"]
        be = grid[(grid.limits == "book_equity") & (grid.leverage == lev)].iloc[0]
        rb = grid[(grid.limits == "risk_budget") & (grid.leverage == lev)].iloc[0]
        print(f"  {lev:.2f}x  pure scaling {pure:.2%} | risk_budget limits {rb.full_months_in_profit:.2%} "
              f"| book_equity limits {be.full_months_in_profit:.2%} "
              f"(de-risked {be.days_derisked:.1%} of days, breaker {int(be.breaker_days)}d)")

    # ── what each reading of the tail allows, under the shipped (book_equity) convention ──────────
    ship_grid = grid[grid.limits == "book_equity"]
    b1 = ship_grid[ship_grid.leverage == 1.0].iloc[0]
    caps = {"realised max-DD": largest_passing(ship_grid, "full_max_dd", lambda v: v >= DD_LIMIT),
            "realised worst month": largest_passing(ship_grid, "full_worst_month", lambda v: v >= WORST_MONTH_LIMIT),
            "realised months-in-profit": largest_passing(ship_grid, "full_months_in_profit", lambda v: v >= MONTHS_MIN),
            "bootstrap-P5 max-DD": largest_passing(ship_grid, "mc_dd_p5", lambda v: v >= DD_LIMIT),
            "bootstrap-P5 worst month": largest_passing(ship_grid, "mc_wmonth_p5", lambda v: v >= WORST_MONTH_LIMIT),
            "2010-event max-DD": largest_passing(ship_grid, "stress_max_dd", lambda v: v >= DD_LIMIT),
            "2010-event worst month": largest_passing(ship_grid, "stress_worst_month", lambda v: v >= WORST_MONTH_LIMIT)}
    print("\n=== largest leverage each constraint still allows (exact, off the grid) ===")
    for k, v in caps.items():
        print(f"  {k:26s} {f'{v:.2f}x' if v else 'fails already at 1.00x':>21}")
    print(f"  the realised worst month is a typical draw, not a cushion: bootstrap median 15-year path "
          f"{b1.mc_wmonth_p50:.2%} vs realised {b1.full_worst_month:.2%} — the unlucky-path P5 is "
          f"{b1.mc_wmonth_p5:.2%}, already past the {WORST_MONTH_LIMIT:.0%} floor at 1.00x")

    # ── is leverage better spent on the seven non-tail legs, holding the aggressive one at 1.0x? ───
    # Judged VOL-MATCHED (same book risk), because any sizing change that raises risk raises return.
    print("\n=== selective leverage: lever everything vs lever everything EXCEPT the tail leg (volprem) ===")
    others = [c for c in df.columns if c != "volprem"]
    lo, hi = 1.0, 4.0
    for _ in range(40):                              # bisect the seven legs' leverage to the same book vol
        mid = (lo + hi) / 2
        b = book_of(df, {c: mid for c in others})
        v = float(b[b.index < OOS].std(ddof=1) * np.sqrt(ppy_of(b[b.index < OOS])))
        lo, hi = (mid, hi) if v < BOOK_LEVERAGE * pre_vol else (lo, mid)
    selective = {}
    for tag, lm in (("all legs", {c: BOOK_LEVERAGE for c in df.columns}),
                    ("ex-volprem", {c: (lo + hi) / 2 for c in others}),
                    ("volprem cut", {**{c: 2.0 for c in others}, "volprem": 0.5})):
        b = book_of(df, lm)
        m = risk_overlay(b)[0]                       # leverage already inside the legs; absolute limits
        full, oos = card(m), card(m[m.index >= OOS])
        st, tl = worst_stress(stressed, 1.0, "book_equity", lm), tail(m)
        share = (df.mul(pd.Series({c: lm.get(c, 1.0) for c in df.columns}), axis=1)).sum()
        selective[tag] = {"leverage": {k: round(v, 3) for k, v in lm.items()}, "full": full, "oos": oos,
                          **st, **tl, "volprem_pnl_share": round(float((share / share.sum())["volprem"]), 3)}
        print(f"  {tag:11s} ({max(lm.values()):.2f}x)  FULL {row_str(full)}\n  {'':11s} {'':7s} OOS  {row_str(oos)}\n"
              f"  {'':11s} {'':7s} tail: boot-P5 DD {tl['mc_dd_p5']:+.2%} month {tl['mc_wmonth_p5']:+.2%} · "
              f"2010-event DD {st['stress_max_dd']:+.2%} month {st['stress_worst_month']:+.2%} · "
              f"volprem {selective[tag]['volprem_pnl_share']:.0%} of P&L")

    # ── verdict: the shipped level, both limit conventions side by side ────────────────────────────
    print(f"\n=== VERDICT: constant leverage {BOOK_LEVERAGE:.2f}x = book vol {BOOK_LEVERAGE * pre_vol:.2%} "
          f"(the -15% mandate allows up to 1.35x; 1.40x is a cliff, so the level sits below it) ===")
    shipped = {}
    for policy in LIMIT_POLICIES:
        managed = risk_overlay(ew, leverage=BOOK_LEVERAGE, limits=policy)[0]
        full, oos = card(managed), card(managed[managed.index >= OOS])
        st = worst_stress(stressed, BOOK_LEVERAGE, policy)
        shipped[policy] = {"full": full, "oos": oos, **st}
        print(f"  {policy:12s} FULL   {row_str(full)}\n  {'':12s} OOS    {row_str(oos)}\n"
              f"  {'':12s} STRESS DD {st['stress_max_dd']:+.2%} worst month {st['stress_worst_month']:+.2%} "
              f"mo {st['stress_months_in_profit']:.1%} strk {st['stress_streak']}")

    (R / "book" / "risk_budget.json").write_text(json.dumps({
        "stack_vol": {"full": round(stack_vol, 4), "pre_oos": round(pre_vol, 4), "oos": round(oos_vol, 4)},
        "leverage": BOOK_LEVERAGE, "book_vol": round(BOOK_LEVERAGE * pre_vol, 4), "limits": "book_equity",
        "selective_leverage": selective,
        "mandate": {"max_dd": DD_LIMIT, "worst_month": WORST_MONTH_LIMIT, "months_in_profit": MONTHS_MIN,
                    "streak": STREAK_MAX, "sharpe_corridor": [SHARPE_LO, SHARPE_HI]},
        "leverage_allowed_by": caps, "shipped": shipped,
        "event": {"window": list(EVENT), "replayed_into_quarters": [str(s.date()) for s in starts],
                  "date": str(wide["volprem"].loc[EVENT[0]:EVENT[1]].idxmin().date()),
                  "leg_day_loss_at_book_weight": round(leg_day, 4),
                  "book_day_loss_unlevered": round(book_day, 4)},
        "mc_reps": MC_REPS, "grid": rows}, indent=2, default=float))
    print(f"\nartifacts -> {R / 'book' / 'risk_budget.json'}, {R / 'book' / 'risk_budget_grid.csv'}")
    print("RISK BUDGET OK")


if __name__ == "__main__":
    main()
