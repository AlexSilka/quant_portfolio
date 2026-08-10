"""VIX term-structure regime gate for the short-vol (VRP) strategy — a leg-level timing signal.

The short-vol crash that clusters the leg's losing months is preceded by the VIX curve inverting: when
a near-dated VIX sits above a longer-dated one (*backwardation*), a vol shock is under way and short-vol
bleeds. This gate flattens the leg's exposure in that regime and holds it in contango — every sleeve of
it, metals and oil and duration included, not because the VIX forecasts their volatility (it does not)
but because an inverted VIX curve is a read on SYSTEMIC stress, and in that state the sleeves fall
together whatever they sell. It is part of the
volprem strategy's own signal — run_vol_premium_book.py publishes the gated series as `ret_gated`, and the
master book simply consumes that. It is NOT a portfolio overlay like the book's drawdown ladder / daily-loss
breaker (those react to whole-book state); this reacts to one leg's own regime.

The curve has three free points — VIX9D (9 days), VIX (30 days), VIX3M (3 months) — so it has two
segments, and the gate requires a normal slope on BOTH: `VIX3M/VIX >= 1` AND `VIX/VIX9D >= 1`. The short
segment inverts earlier (it prices the days a shock is actually happening in), the long segment is the
steadier read; demanding both is what times the leg out of the episodes a single segment misses, and it
catches 9 of the leg's 10 worst days against 4 for the long segment alone (`run_vol_premium_gates.py`).

Design choices (kept honest, not fitted to the scorecard):
  - both thresholds = 1.0 — the contango/backwardation boundary itself, an a-priori economic line, not a
    number picked from results. The 5x5 surface around them is a plateau, not a spike;
  - causal — the decision uses the *prior* close (shift 1); no same-bar look-ahead;
  - point-in-time — the Cboe curve indices are published intraday and never revised;
  - parameter-light — one binary regime switch, no per-asset tuning;
  - a segment with no data does not gate. VIX9D lists 2011-01-04 and VIX3M 2009-09-18, so the leg simply
    runs ungated before then rather than sitting flat on a signal that does not exist.

Validated in `run_vol_premium_gates.py` (`make volprem`) against the nulls a lucky rule has to beat: it
beats 100% of 200 block-random gates at the same average exposure on Sharpe, drawdown, worst month and
months-in-profit, and the constant-exposure control is far worse — so the value is the *timing*, not the
de-risking. Switching the leg off and back on is charged the vega spread at the sleeve, so the timing is
not free. What it cannot reach: a one-session dislocation out of a calm curve — into the 2010 flash crash
VIX3M/VIX stood at 1.059 and the curve only inverted on the crash day itself.
"""
from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR

SEGMENTS = (("VIX", "VIX3M"), ("VIX9D", "VIX"))    # (near, far) — far/near >= 1 is a normal slope


def _index(sym: str, index) -> pd.Series:
    """A Cboe curve index aligned to `index`, forward-filled — information at t."""
    s = pd.read_parquet(RAW_DIR / "cboe" / f"{sym}.parquet")["close"]
    s.index = _dates(s.index)
    return s.reindex(_dates(index)).ffill()


def _dates(ix) -> pd.DatetimeIndex:
    ix = pd.to_datetime(ix)
    return (ix.tz_localize(None) if ix.tz is not None else ix).normalize()


def term_structure(index, near: str = "VIX", far: str = "VIX3M") -> pd.Series:
    """far/near for one curve segment aligned to `index` (>1 contango, <1 backwardation)."""
    return _index(far, index) / _index(near, index)


def short_vol_gate(index, threshold: float = 1.0) -> pd.Series:
    """Exposure multiplier on `index` for the short-vol leg: 1.0 while EVERY curve segment is in contango
    (far/near >= threshold), 0.0 as soon as one inverts, decided on the prior close (shift 1).

    A segment whose index has no history yet is not a signal, so it does not gate — the ratio is NaN and
    treated as contango, leaving the leg fully live rather than flat on a rule that cannot be evaluated."""
    live = pd.Series(True, index=_dates(index))
    for near, far in SEGMENTS:
        ratio = term_structure(index, near, far)
        live &= (ratio >= threshold) | ratio.isna()            # no data on a segment => that segment abstains
    gate = live.astype(float).shift(1).fillna(1.0)
    gate.index = pd.Index(index)          # restore the caller's original index for element-wise use
    return gate


OWN_CURVE_LOOKBACK = 63           # trading days in three months — the calendar a 3M vol index spans


def own_curve_gate(implied: pd.Series, index, lookback: int = OWN_CURVE_LOOKBACK,
                   threshold: float = 1.0) -> pd.Series:
    """The same contango test as `short_vol_gate`, for a sleeve that has no far-dated index of its own.

    `short_vol_gate` reads the VIX curve, which is the volatility of the S&P 500. That makes it a good
    read on a SYSTEMIC shock, which is why it gates all eighteen sleeves — but it is blind to a vol
    event that one market has on its own. Thirteen of the sleeves sell variance on gold, silver,
    gold-miners, oil, duration, EM ETFs and single names, and an idiosyncratic repricing in any of them
    leaves the VIX flat. That is a coverage hole rather than a calibration one: it stays open however
    the VIX thresholds are set. In 2026 it opened all the way — the VIX averaged 19.0 with 93% of days
    in contango while silver's realised volatility went 32% -> 73%, so the shared gate never stood the
    metals sleeves down and they sold variance the whole way. This gate is what closes that, per sleeve;
    the two are ANDed and cover different failures.

    Cboe publishes no GVZ3M/OVX3M, so the far leg is proxied by the sleeve's OWN trailing mean implied
    vol: `mean(iv, lookback) / iv >= threshold` is "spot vol sits below its own three-month level", the
    same economic line the VIX segments draw. Neither knob is fitted — `lookback` is the calendar length
    of a 3M index and `threshold` is the contango boundary `short_vol_gate` already uses. The 4x3 surface
    around them is a plateau: every cell beats the ungated book on Sharpe, drawdown AND 2026 at once
    (`run_gate_coverage.py`).

    Decided on the prior close (shift 1); `short_vol_book` shifts the side by `exec_lag` again, so three
    days separate signal from exposure. Validated against the null a leg that merely sits out would
    produce — a random gate at each sleeve's own duty cycle clears Sharpe +3.03 at p95 against this
    rule's +7.36, so the value is the timing and not the absence.
    """
    idx = _dates(index)
    iv = implied.copy()
    iv.index = _dates(iv.index)
    iv = iv[~iv.index.duplicated(keep="last")].reindex(idx).ffill(limit=5)
    live = (iv.rolling(lookback).mean() / iv) >= threshold
    # Before `lookback` bars the trailing mean does not exist. Unlike a missing VIX segment — where the
    # test cannot be run but the leg is otherwise fine, so it abstains — here the gap is the sleeve's own
    # first three months, and standing down through them costs a quarter once per sleeve, ever.
    gate = live.astype(float).shift(1).fillna(0.0)
    gate.index = pd.Index(index)
    return gate


def gate_short_vol_leg(leg: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Return the short-vol return series `leg` with the backwardation gate applied (crash regime → flat).

    NOTE: this multiplies a finished P&L series, so it does NOT charge the vega spread the switch really
    crosses. It is kept for A/B diagnostics only — the shipped `ret_gated` is built by
    `run_vol_premium_book.gated_book`, which passes the gate into the sleeve so switching is paid for."""
    return (leg * short_vol_gate(leg.index, threshold)).rename(getattr(leg, "name", "ret"))
