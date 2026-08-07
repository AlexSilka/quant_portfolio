"""Triple-barrier labelling and meta-labelling (Lopez de Prado, AFML ch. 3).

For each event at time t0: an upper barrier at +pt*sigma, a lower at -sl*sigma (in
log-return units off the entry price), and a vertical barrier at t0 + horizon bars. The
label is the sign of the first barrier touched (vertical -> 0). Meta-labelling turns a
primary side into a binary "did the primary's bet win" target, trained only on bars where
the primary fired.

Causality: sigma is trailing; each event looks forward only to its own barrier touch. The
touch time t1 and its return are realised information at t1 (used as a training target),
never as a feature stamped at t0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """EWM std of log returns — the per-bar volatility unit sigma_t (causal)."""
    return np.log(close).diff().ewm(span=span).std()


def triple_barrier_labels(close: pd.Series, events_idx: pd.DatetimeIndex,
                          sigma: pd.Series, pt: float = 1.0, sl: float = 1.0,
                          horizon: int = 24) -> pd.DataFrame:
    """Label each event by first-barrier-touch. Returns columns t1, ret, bin (indexed by t0)."""
    close = close.sort_index()
    idx = close.index
    c = close.to_numpy()
    pos = {t: i for i, t in enumerate(idx)}
    n = len(idx)
    sig = sigma.reindex(idx)
    rows = []
    for t0 in events_idx:
        i = pos.get(t0)
        if i is None or i >= n - 1:
            continue
        s = sig.iat[i]
        if not np.isfinite(s) or s <= 0:
            continue
        j_end = min(i + horizon, n - 1)
        p0 = c[i]
        up = p0 * np.exp(pt * s)
        dn = p0 * np.exp(-sl * s)
        window = c[i + 1: j_end + 1]
        up_hits = np.flatnonzero(window >= up)
        dn_hits = np.flatnonzero(window <= dn)
        first_up = up_hits[0] if up_hits.size else np.inf
        first_dn = dn_hits[0] if dn_hits.size else np.inf
        if first_up < first_dn:
            k, bin_ = int(first_up), 1
        elif first_dn < first_up:
            k, bin_ = int(first_dn), -1
        else:
            k, bin_ = len(window) - 1, 0          # vertical barrier
        t1 = idx[i + 1 + k]
        rows.append((t0, t1, c[i + 1 + k] / p0 - 1.0, bin_))
    return pd.DataFrame(rows, columns=["t0", "t1", "ret", "bin"]).set_index("t0")


def meta_labels(labels: pd.DataFrame, side: pd.Series) -> pd.Series:
    """Binary target: 1 if the primary `side` (+1/-1) profited over the event, else 0.

    Defined only on events where the primary fired (side != 0).
    """
    s = side.reindex(labels.index).fillna(0.0)
    pnl = s * labels["ret"]
    y = (pnl > 0).astype(int)
    return y[s != 0]
