"""Purged, embargoed cross-validation for event-based labels (in-house, auditable).

Folds are contiguous in time. For each test fold, training events whose label window
[t0, t1] overlaps the test span are purged, and an embargo removes training events for a
short window after the test span (serial-correlation leakage). OOS predictions are stitched
across folds into a single series covering the whole period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def purged_kfold(t0: pd.DatetimeIndex, t1: pd.DatetimeIndex, n_splits: int = 6,
                 embargo: pd.Timedelta = pd.Timedelta(hours=24)):
    """Yield (train_positions, test_positions) integer arrays over the event array."""
    n = len(t0)
    order = np.argsort(t0.values)
    for fold in np.array_split(order, n_splits):
        test_idx = np.sort(fold)
        test_start = t0[test_idx].min()
        test_end = t1[test_idx].max()
        # purge training events whose label window overlaps [test_start, test_end + embargo]
        overlap = (t1.values >= np.datetime64(test_start)) & \
                  (t0.values <= np.datetime64(test_end + embargo))
        train_mask = ~overlap
        train_mask[test_idx] = False
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx


def cv_oos_predictions(X: pd.DataFrame, y: pd.Series, t1: pd.Series, model_factory,
                       n_splits: int = 6, embargo: pd.Timedelta = pd.Timedelta(hours=24)):
    """Return (oos P(win) series, per-fold info). Each fold trained on its purged complement."""
    t0 = pd.DatetimeIndex(X.index)
    t1i = pd.DatetimeIndex(t1.reindex(X.index).values)
    oos = pd.Series(np.nan, index=X.index)
    stats = []
    for k, (tr, te) in enumerate(purged_kfold(t0, t1i, n_splits, embargo)):
        model = model_factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        oos.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
        stats.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
                      "test_start": t0[te].min(), "test_end": t0[te].max()})
    return oos, pd.DataFrame(stats)
