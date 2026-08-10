"""Market-stress reading — how much protection the book should be carrying, from market state alone.

This is a SIZING input, not a return forecast and not a signal any strategy trades. It exists because
the book's one long-gamma leg was held at the same flat risk budget as its five earners, which is the
wrong shape for a hedge: through a calm decade the leg is the book's weakest earner (standalone Sharpe
~0.6 against a book above 3) and dilutes it every month, and through a crash it is the only leg paying.
A hedge should be small when protection is cheap to skip and large when it is not, and "when" has to be
read off the market at t-1 rather than off anyone's P&L.

Two components, both free and both causal:

  * VIX term structure (VIX / VIX3M) — the front month above the three-month is the classic risk-off
    tell, and the same reading the vol-premium leg already gates its own exposure on. Calm (deep
    contango) sits near 0.85-0.90; inversion is 1.00+.
  * S&P drawdown from its trailing-year high — the slower, price-based confirmation, so a one-day vol
    spike that the market shrugs off does not buy a year of protection on its own.

Stress is the MAX of the two: either firing is enough. A crypto-drawdown component was built and
rejected — the book's crypto legs are dollar-neutral, so a BTC drawdown is not stress for this book
(correlation +0.09, same worst month with and without), and including it raised the hedge's average
weight by a third while moving neither the worst month nor the drawdown. The measurement lives in
scripts/run_crisis_lab.py; it is recorded here because the opposite error — a stress reading blind to
a class the book actually holds — has been made in this project before.
"""
from __future__ import annotations

import pandas as pd

from src.data.cboe import load_cboe_vol
from src.data.equity import load_equity_daily

# The ramp ends, stated rather than fitted: below CURVE_CALM / above CURVE_STRESS the vol component is
# pinned at 0 and 1, and likewise for the drawdown. Chosen from what the levels mean (deep contango vs
# an inverted curve; a normal pullback vs a correction), not from a sweep of book outcomes — the
# neighbourhood of every one of them is published in reports/lab/crisis_lab.json.
CURVE_CALM, CURVE_STRESS = 0.90, 1.05
DRAWDOWN_FULL = -0.12               # equity drawdown at which the stress reading is fully on
_CACHE: pd.Series | None = None


def market_stress(index: pd.Index | None = None) -> pd.Series:
    """A 0..1 stress reading, lagged one bar so bar t is sized on information through t-1.

    Returned on its own daily index when `index` is None, otherwise forward-filled onto `index` — a
    crypto leg trades on days the Cboe and NYSE are shut, and on those days the last published reading
    is the only one that exists. The lag is taken on the CONSUMER's calendar, after the fill: lagging on
    the exchange calendar first would leave a 7-day leg reading Thursday's close all weekend, when
    Friday's has been published and is the honest most-recent observation."""
    global _CACHE
    if _CACHE is None:
        vix, vix3m = load_cboe_vol("VIX"), load_cboe_vol("VIX3M")
        spy = load_equity_daily("SPY", start="2004-01-01")["close"]
        for s in (vix, vix3m, spy):
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
        curve = (vix / vix3m.reindex(vix.index).ffill()).dropna()
        dd = spy / spy.rolling(252, min_periods=60).max() - 1.0
        vol_part = ((curve - CURVE_CALM) / (CURVE_STRESS - CURVE_CALM)).clip(0.0, 1.0)
        dd_part = (dd / DRAWDOWN_FULL).clip(0.0, 1.0)
        # union, not the vol index: VIX3M only starts in 2009, and the drawdown component is readable
        # from 2005. Intersecting would hand the whole pre-2009 window a default weight and silently
        # drop the one component that covers the GFC — the deepest stress in the sample.
        both = pd.concat([vol_part, dd_part], axis=1).sort_index().ffill()
        _CACHE = both.max(axis=1, skipna=True).dropna().rename("stress")   # unlagged; the lag is below
    if index is None:
        return _CACHE.shift(1).dropna()
    ix = pd.DatetimeIndex(index)
    return _CACHE.reindex(_CACHE.index.union(ix)).ffill().reindex(ix).shift(1).ffill().rename("stress")


def hedge_weight(index: pd.Index, calm: float = 0.25, stressed: float = 1.5) -> pd.Series:
    """The share of one equal-risk slot a long-gamma leg carries, ramped linearly on `market_stress`.

    `calm` and `stressed` are the two ends: a quarter of a slot when nothing is moving, a slot and a
    half when the curve is inverted or the market is 12% off its high. The average over 2005-2026 is
    ~0.7, so this is not a way of holding more hedge — it is the same average protection bought at the
    times it pays. Rotating the stress path (same re-sizing, wrong days) gives back the whole gain,
    which is what says the timing rather than the smaller average is doing the work."""
    z = market_stress(index)
    return (calm + (stressed - calm) * z).fillna(1.0).rename("hedge_weight")


def stress_summary(index: pd.Index | None = None) -> dict:
    """Duty cycle of the reading — how often the book is paying for full protection."""
    z = market_stress(index).dropna()
    return {"days": int(len(z)), "mean": round(float(z.mean()), 3),
            "share_calm_below_0.2": round(float((z < 0.2).mean()), 3),
            "share_stressed_above_0.8": round(float((z > 0.8).mean()), 3),
            "window": [str(z.index.min().date()), str(z.index.max().date())]}


__all__ = ["market_stress", "hedge_weight", "stress_summary", "CURVE_CALM", "CURVE_STRESS", "DRAWDOWN_FULL"]
