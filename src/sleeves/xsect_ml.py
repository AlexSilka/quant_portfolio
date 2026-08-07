"""ML layers for the cross-sectional sleeve — two honest, leakage-controlled variants.

1. Learning-to-rank (cross-sectional): per (name, bar) build a feature vector (multi-horizon
   momentum, reversal, vol, volume, distance-from-high, beta), train a model to predict the
   *cross-sectionally demeaned* forward return, then long the top / short the bottom of the
   prediction. Demeaning the target removes the market factor so the model learns *relative*
   ranking, not direction — and cannot cheat by learning "the market went up".

2. Meta-label gate (time-series): take the rule-based book's own return stream and, per
   rebalance, predict P(the book wins the next period) from regime features (dispersion,
   breadth, panel vol, own-recent-state). Trade only when P > threshold — the company's
   "confidence factor". Measures ML's incremental value as risk reduction, not a Sharpe boost.

Both are fit strictly inside expanding walk-forward folds (no future rows in any training set)
and every feature is stamped at bar t from data <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── feature panels (each a wide bars×names frame, computable-at-bar) ───────────────────────
def rank_features(px: pd.DataFrame, advol: pd.DataFrame | None, bpd: int) -> dict[str, pd.DataFrame]:
    """A dictionary of name-level features for learning-to-rank, all lagged/rolling."""
    r = px.pct_change()
    feats: dict[str, pd.DataFrame] = {}
    for d in (5, 10, 20, 30, 60, 120, 180):
        lb = d * bpd
        feats[f"mom_{d}"] = px / px.shift(lb) - 1.0
        feats[f"radj_{d}"] = (px / px.shift(lb) - 1.0) / r.rolling(lb).std().replace(0, np.nan)
    feats["rev_5"] = -(px / px.shift(5 * bpd) - 1.0)                 # short-term reversal
    for d in (20, 60):
        feats[f"vol_{d}"] = r.rolling(d * bpd).std()
    feats["dist_high"] = px / px.rolling(120 * bpd).max() - 1.0      # distance from 120d high
    mkt = r.mean(axis=1)                                             # equal-weight panel = "market"
    feats["beta_60"] = r.rolling(60 * bpd).cov(mkt).div(mkt.rolling(60 * bpd).var(), axis=0)
    if advol is not None:
        av = advol.reindex_like(px)
        feats["advtrend"] = av / av.rolling(60 * bpd).mean() - 1.0   # relative volume expansion
    return feats


def stack_xy(feats: dict[str, pd.DataFrame], px: pd.DataFrame, fwd_bars: int
             ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Long-format (row = name×bar) design matrix X, demeaned forward-return target y, and the
    bar timestamp per row (for purged folds). Forward return is cross-sectionally demeaned so
    the label is a *relative* rank target, not market direction."""
    fwd = px.shift(-fwd_bars) / px - 1.0
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)                          # cross-sectional demean
    cols = list(feats)
    long = {c: feats[c].stack(dropna=False) for c in cols}
    X = pd.DataFrame(long)
    y = fwd.stack(dropna=False).reindex(X.index)
    ts = X.index.get_level_values(0).to_series(index=X.index)
    good = X.notna().all(axis=1) & y.notna()
    return X[good], y[good], ts[good]


# ── regime features for the meta-gate (one row per bar, from the panel) ────────────────────
def regime_features(px: pd.DataFrame, signal: pd.DataFrame, book_ret: pd.Series,
                    bpd: int) -> pd.DataFrame:
    """Bar-level regime features driving the meta-gate — all data <= t."""
    r = px.pct_change()
    ranks = signal.rank(axis=1, pct=True)
    df = pd.DataFrame(index=px.index)
    df["dispersion"] = signal.std(axis=1)                           # signal spread across names
    df["breadth"] = (signal > 0).mean(axis=1)                       # fraction trending up
    df["panel_vol"] = r.std(axis=1).rolling(20 * bpd).mean()        # avg cross-sectional vol
    df["mkt_mom"] = px.mean(axis=1) / px.mean(axis=1).shift(20 * bpd) - 1.0
    df["rank_concentration"] = ranks.sub(0.5).abs().mean(axis=1)    # how separated the tails are
    df["book_r5"] = book_ret.rolling(5 * bpd).mean()               # own recent performance
    df["book_dd"] = book_ret.cumsum() - book_ret.cumsum().cummax()  # own drawdown state
    return df


def expanding_predict(X: pd.DataFrame, y: pd.Series, ts: pd.Series, model_factory,
                      n_folds: int = 6, embargo_bars: int = 10) -> pd.Series:
    """Expanding walk-forward OOS predictions: train on all rows strictly before a fold's start
    (minus an embargo), predict the fold. No future row ever enters a training set."""
    order = np.argsort(ts.values, kind="stable")
    Xs, ys, tss = X.iloc[order], y.iloc[order], ts.iloc[order]
    uniq = np.array(sorted(tss.unique()))
    bounds = [uniq[min(int(i * len(uniq) / (n_folds + 1)), len(uniq) - 1)]
              for i in range(n_folds + 2)]
    pred = pd.Series(np.nan, index=Xs.index)
    embargo = pd.Timedelta(0)
    for k in range(1, n_folds + 1):
        te0, te1 = bounds[k], bounds[k + 1]
        if hasattr(pd.Timestamp(te0), "to_pydatetime"):
            embargo = (uniq[1] - uniq[0]) * embargo_bars if len(uniq) > 1 else pd.Timedelta(0)
        tr = tss < (pd.Timestamp(te0) - embargo)
        te = (tss >= te0) & (tss < te1)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = model_factory()
        m.fit(Xs[tr].to_numpy(), ys[tr].to_numpy())
        pred.iloc[np.flatnonzero(te.to_numpy())] = m.predict(Xs[te].to_numpy())
    return pred


def predictions_to_panel(pred: pd.Series, px: pd.DataFrame) -> pd.DataFrame:
    """Reshape long-format (bar,name) predictions back to a wide signal panel for xs_backtest."""
    wide = pred.unstack()
    return wide.reindex(index=px.index, columns=px.columns)
