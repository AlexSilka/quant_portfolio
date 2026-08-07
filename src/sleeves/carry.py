"""Carry sleeve: perpetual funding carry (crypto-native, structurally distinct family).

When funding is richly positive (longs pay shorts) the perp is expensive to be long, so the
carry trade is to be short and collect funding; when funding is negative, be long. Sided off
a z-score of the funding rate so it adapts across regimes.
"""
from __future__ import annotations

import pandas as pd

EPS = 1e-12


def primary_side(funding: pd.Series, close: pd.Series, z_lookback: int = 270,
                 entry_z: float = 1.0) -> pd.Series:
    """-1 when funding is richly positive (collect by shorting), +1 when richly negative."""
    fr = funding.reindex(close.index).ffill()
    z = (fr - fr.rolling(z_lookback).mean()) / (fr.rolling(z_lookback).std() + EPS)
    side = pd.Series(0.0, index=close.index)
    side[z >= entry_z] = -1.0
    side[z <= -entry_z] = 1.0
    return side
