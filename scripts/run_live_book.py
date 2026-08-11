"""The book as it would actually be run — return-first, from the date its risk control exists.

This is NOT the test-task deliverable. That one is `run_master_book.py`: eight validated families, six
traded at equal risk, a §8 drawdown ladder on top, and a constant 1.15x sized to a −15% drawdown mandate.
Everything there is shaped by a scorecard. This is the same research with the scorecard removed and one
question in its place: which of these legs would you put your own money in, and at what size.

Three differences from the deliverable, each with its reason.

  WINDOW — starts 2011-01-03, not 2011-01-01 by coincidence: that is the first day BOTH segments of the
  short-vol gate exist. The gate wants a normal slope on VIX3M/VIX and on VIX/VIX9D, and VIX9D's history
  begins 2011-01-04 (published as VXST from Oct-2013, backfilled to there). Before that the leg runs on
  one segment or none, and "none" is what makes 2008 the one dislocation the gate does not cover. So the
  honest live window opens where the control does. VIX3M itself reaches back to 2007-12 at Cboe, but the
  public CSV feed truncates it at 2009-09-18 — recovering those two years needs a source this project
  does not use, and it would not help anyway while VIX9D does not exist.

  COMPOSITION — volprem, breakout, BAB, x-sect. Crisis-alpha and global-macro are dropped: both earn
  ~0.5-0.9 standalone and existed to buy risk targets that no longer bind, and dropping them takes the
  book from 49.5% a year to 88.9% at the deliverable's own leverage. Carry stays out for the same reason
  it was dropped there — it does not add at this size — and trend stays out on its own merits.

  NO §8 OVERLAY — the drawdown ladder (−6/−9/−12% → two-thirds/one-third/flat) and the daily-loss breaker
  guarantee a mandate that no longer exists, and they are not free: they cut after a loss and restore
  after a recovery, which on a mean-reverting equity curve sells low and buys back higher. Measured, they
  are what stops leverage working — at 3x they turn 422% a year into 181%.

    python scripts/run_live_book.py [leverage]     ->  reports/lab/live_book.parquet
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
from src.config import CAPITAL_USD, LAB_DIR  # noqa: E402

START = "2011-01-03"        # first day both gate segments exist (VIX9D lists 2011-01-04)
LEGS = ["volprem", "breakout", "bab", "xs_momentum"]
DEFAULT_LEVERAGE = 2.0


def legs() -> pd.DataFrame:
    """The chosen families, each rescaled to the common per-leg vol target — the assembler's own step."""
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES if lab in LEGS}
    raw = {k: v for k, v in raw.items() if v is not None}
    missing = [k for k in LEGS if k not in raw]
    if missing:                                   # a leg silently absent would change the book, not fail it
        print(f"WARNING: legs not found and therefore not held: {missing}")
    df = pd.DataFrame({k: mb.rescale(raw[k]) for k in raw}).sort_index()
    return df[df.index >= pd.Timestamp(START)].dropna(how="all")


def book(df: pd.DataFrame, leverage: float) -> pd.Series:
    """Equal risk over the legs live each day, at one constant leverage. No book-level overlay."""
    return (mb.book_stack(df) * leverage).dropna()


def stats(s: pd.Series) -> dict:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    eq = (1 + s).cumprod()
    dd = eq / eq.cummax() - 1
    mo = (1 + s).resample("ME").prod() - 1
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": mb.scorecard(s)["sharpe"],
            "max_dd": float(dd.min()), "worst_month": float(mo.min()), "worst_day": float(s.min()),
            "vol": float(s.std() * np.sqrt(mb.ppy_of(s))), "months_in_profit": float((mo > 0).mean()),
            "years": yrs, "growth_x": float(eq.iloc[-1])}


def main() -> None:
    lev = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LEVERAGE
    df = legs()
    b = book(df, lev)
    st = stats(b)
    live = df.notna().sum(axis=1)
    print(f"=== LIVE BOOK — {', '.join(df.columns)} @ {lev:.2f}x, no §8 overlay ===")
    print(f"window {b.index.min().date()}..{b.index.max().date()}  ({st['years']:.1f} years; "
          f"{int(live.min())}-{int(live.max())} legs live/day)\n")
    # The compounded multiple is arithmetic, not a deliverable: at this rate the balance passes the
    # vol-premium leg's vega capacity (low tens of $M) around year eight and leaves the size the cost
    # model charges for, so it is printed as a caveat rather than as a headline.
    print(f"  return        {st['cagr']:+.1%} a year")
    print(f"  volatility    {st['vol']:.1%}      Sharpe {st['sharpe']:+.2f}")
    print(f"  max drawdown  {st['max_dd']:+.1%}      worst month {st['worst_month']:+.1%}   "
          f"worst day {st['worst_day']:+.1%}")
    print(f"  months in profit {st['months_in_profit']:.0%}")
    print(f"  on ${CAPITAL_USD // 1000}k, P&L NOT reinvested: ${CAPITAL_USD * b.sum():,.0f} "
          f"(${CAPITAL_USD * b.sum() / st['years']:,.0f}/yr)  <- the honest dollar figure")
    print(f"  (compounded it reads {st['growth_x']:,.0f}x, which is past this book's capacity from ~year 8 "
          f"— arithmetic, not a result)")

    print(f"\n{'year':>6s} {'return':>9s} {'max DD':>8s} {'worst mo':>9s} {'legs':>5s}")
    per_year = {}
    for y, g in b.groupby(b.index.year):
        eq = (1 + g).cumprod()
        m = (1 + g).resample("ME").prod() - 1
        per_year[int(y)] = round(float(eq.iloc[-1] - 1), 4)
        print(f"{y:6d} {100 * (eq.iloc[-1] - 1):8.1f}% {100 * (eq / eq.cummax() - 1).min():7.1f}% "
              f"{100 * m.min():8.1f}% {int(df.loc[g.index].notna().sum(axis=1).max()):5d}")

    # The window opens in 2011 because that is when the gate does, but the FOURTH leg lists in 2020 — for
    # nine of these fifteen years "the portfolio" is vol-prem plus x-sect. Both windows are printed so the
    # headline cannot be read as fifteen years of the thing actually being run.
    full = df[df.index >= pd.Timestamp("2020-01-01")]
    sf = stats(book(full, lev))
    print(f"\n  since all four legs list (2020-01+): {sf['cagr']:+.1%} a year, Sharpe {sf['sharpe']:+.2f}, "
          f"max DD {sf['max_dd']:+.1%} — the same book measured only where it is the book")

    print(f"\n{'leverage':>9s} {'return':>9s} {'max DD':>8s} {'worst mo':>9s} {'worst day':>10s} {'vol':>7s}")
    sweep = {}
    for x in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        sx = stats(book(df, x))
        sweep[f"{x:.1f}"] = {k: round(v, 4) for k, v in sx.items()}
        print(f"{x:9.1f} {100 * sx['cagr']:8.1f}% {100 * sx['max_dd']:7.1f}% {100 * sx['worst_month']:8.1f}% "
              f"{100 * sx['worst_day']:9.1f}% {100 * sx['vol']:6.1f}%")

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    b.rename("ret").to_frame().to_parquet(LAB_DIR / "live_book.parquet")
    (LAB_DIR / "live_book.json").write_text(json.dumps(
        {"window": [str(b.index.min().date()), str(b.index.max().date())], "legs": list(df.columns),
         "leverage": lev, "overlay": None, "stats": {k: round(v, 4) for k, v in st.items()},
         "per_year": per_year, "leverage_sweep": sweep,
         "since_all_legs_2020": {k: round(v, 4) for k, v in stats(book(full, lev)).items()}}, indent=2))
    print(f"\nwrote {LAB_DIR / 'live_book.parquet'} and live_book.json")


if __name__ == "__main__":
    main()
