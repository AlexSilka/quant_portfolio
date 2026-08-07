"""Breakout / range sleeve: Donchian channel breakout (the non-ML baseline)."""
from __future__ import annotations

import pandas as pd


def primary_side(close: pd.Series, high: pd.Series, low: pd.Series,
                 lookback: int = 55) -> pd.Series:
    """+1 on a close above the trailing N-bar high, -1 below the trailing N-bar low."""
    hh = high.rolling(lookback).max().shift(1)
    ll = low.rolling(lookback).min().shift(1)
    side = pd.Series(0.0, index=close.index)
    side[close > hh] = 1.0
    side[close < ll] = -1.0
    return side
