"""Momentum / trend-following sleeve: EMA fast-vs-slow crossover (the non-ML baseline)."""
from __future__ import annotations

import pandas as pd


def primary_side(close: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """+1 when the fast EMA is above the slow EMA (uptrend), -1 when below."""
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    side = pd.Series(0.0, index=close.index)
    side[f > s] = 1.0
    side[f < s] = -1.0
    return side
