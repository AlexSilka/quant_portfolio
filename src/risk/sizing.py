"""Vol-target position sizing — the one implementation.

This formula was written out thirty-seven times across the repo before it lived here, and the copies are
not interchangeable trivia: the `.shift(1)` is what keeps the size computable at the bar rather than from
the day's own volatility, and a single copy missing it is a look-ahead nobody would notice in a Sharpe.
One implementation means that discipline is written once.

The ceiling and the lookback stay ARGUMENTS, not shared constants. A sleeve's cap is part of a
construction that was validated with it in place, and quietly re-pointing every sleeve at one number
would re-open every family's published series. The book-assembly layer passes `VOL_SCALE_CAP` because
that one was measured; everyone else passes what they were already using, visibly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_scale(net: pd.Series, target: float, ppy: float,
                     lookback: int = 60, cap: float = 3.0) -> pd.Series:
    """Multiplier that puts `net` on `target` annualised vol, from trailing info only.

    net      : a return series, already net of its own costs.
    target   : annualised vol to size to (e.g. 0.15).
    ppy      : that series' observations per year — 365 for crypto, ~252 for an exchange calendar.
    lookback : bars of trailing vol to size off.
    cap      : ceiling on the multiplier, so a quiet stretch cannot hand the leg unbounded leverage
               just before a shock.

    Lagged one bar and zero over the warm-up: before there is enough history to size a position there is
    no position.
    """
    return (target / (net.rolling(lookback).std() * np.sqrt(ppy))).clip(upper=cap).shift(1).fillna(0.0)
