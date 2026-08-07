"""VIX term-structure regime gate for the short-vol (VRP) strategy — a leg-level timing signal.

The short-vol crash that clusters the leg's losing months is preceded by the VIX curve inverting: when
spot VIX rises above 3-month VIX (VIX3M/VIX < 1, *backwardation*), a vol shock is under way and short-vol
bleeds. This gate flattens the leg's exposure in that regime and holds it in contango. It is part of the
volprem strategy's own signal — run_vol_premium_book.py publishes the gated series as `ret_gated`, and the
master book simply consumes that. It is NOT a portfolio overlay like the book's drawdown ladder / daily-loss
breaker (those react to whole-book state); this reacts to one leg's own regime.

Design choices (kept honest, not fitted to the scorecard):
  - threshold = 1.0 — the contango/backwardation boundary itself (VIX3M = VIX), an a-priori economic
    line, not a number picked from results;
  - causal — the decision uses the *prior* close (shift 1); no same-bar look-ahead;
  - point-in-time — VIX and VIX3M are published intraday and never revised;
  - parameter-light — one binary regime switch, no per-asset tuning.

Validated in `scripts/run_ml_book_contribution.py` (`make ml-contribution`): it is the *timing*, not
de-risking — a constant cut to the same average exposure does nothing and a random gate hurts — and no
ML engine (logistic / RF / HistGB / LightGBM / MLP) beats this rule: the value is the VIX signal, not
the model. It is what closes the master book's full-window scorecard from 3/5 to 5/5.
"""
from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR


def _dates(ix) -> pd.DatetimeIndex:
    ix = pd.to_datetime(ix)
    return (ix.tz_localize(None) if ix.tz is not None else ix).normalize()


def vix_term_structure(index) -> pd.Series:
    """VIX3M / spot-VIX aligned to `index` (>1 contango, <1 backwardation), forward-filled."""
    vix = pd.read_parquet(RAW_DIR / "rates" / "VIXCLS.parquet")["val"]
    v3 = pd.read_parquet(RAW_DIR / "vol_etp" / "VIX3M_yf.parquet")["close"]
    vix.index, v3.index = _dates(vix.index), _dates(v3.index)
    nidx = _dates(index)
    return v3.reindex(nidx).ffill() / vix.reindex(nidx).ffill()


def short_vol_gate(index, threshold: float = 1.0) -> pd.Series:
    """Exposure multiplier on `index` for the short-vol leg: 1.0 in contango (VIX3M/VIX >= threshold),
    0.0 in backwardation, decided on the prior close (shift 1). Missing history defaults to full."""
    gate = (vix_term_structure(index) >= threshold).astype(float).shift(1).fillna(1.0)
    gate.index = pd.Index(index)          # restore the caller's original index for element-wise use
    return gate


def gate_short_vol_leg(leg: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Return the short-vol return series `leg` with the backwardation gate applied (crash regime → flat)."""
    return (leg * short_vol_gate(leg.index, threshold)).rename(getattr(leg, "name", "ret"))
