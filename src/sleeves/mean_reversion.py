"""Mean-reversion sleeve: Bollinger z-score primary rule (the non-ML baseline).

Primary side: +1 (long) when price is stretched below its band (expect reversion up),
-1 (short) when stretched above. This rule is the baseline; a meta-model then gates it,
and the sleeve's ML value is measured as gated minus baseline.
"""
from __future__ import annotations

import pandas as pd

EPS = 1e-12


def primary_side(close: pd.Series, lookback: int = 20, entry_z: float = 1.5) -> pd.Series:
    """Return a per-bar side series in {-1, 0, +1} from a z-score reversion rule."""
    m = close.rolling(lookback).mean()
    s = close.rolling(lookback).std()
    z = (close - m) / (s + EPS)
    side = pd.Series(0.0, index=close.index)
    side[z <= -entry_z] = 1.0    # oversold -> expect bounce up
    side[z >= entry_z] = -1.0    # overbought -> expect fade down
    return side
