"""Shared sleeve-modelling helpers: the LightGBM meta-label model factory and signal-event
detection. Used by the discovery layer (`scripts/run_book.py`) and the per-sleeve ML drivers
(`run_meta_overlay`, `pairs`, `meanrev`).
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.config import SEED


def model_factory():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=-1, verbose=-1)


def signal_events(side: pd.Series) -> pd.DatetimeIndex:
    """Event bars = onset of a new directional signal (dedup consecutive same-side bars)."""
    prev = side.shift(1).fillna(0.0)
    return side.index[(side != 0.0) & (side != prev)]
