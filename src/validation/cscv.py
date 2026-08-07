"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu, 2015).

The brief (§6) makes multiple-testing control mandatory and names "probability of backtest
overfitting (CSCV) or equivalent". Combinatorially-Symmetric Cross-Validation answers: when we pick
the in-sample-best configuration out of N trials, how often does it land *below the median* out of
sample? That fraction is the PBO — a direct estimate of how much the selection is fitting noise.

Algorithm (no external dependency; the `purgedcv` package is NOT used):
  1. Trim T to a multiple of S, split into S disjoint contiguous blocks.
  2. For every way to assign S/2 blocks to in-sample (the rest to out-of-sample) — C(S, S/2) splits —
     rank the N strategies by IS Sharpe, take the IS-best n*, find its OOS rank.
  3. ω = OOS_rank / (N+1) ∈ (0,1); logit λ = ln(ω/(1-ω)). Overfit if λ < 0 (n* below OOS median).
  4. PBO = P(λ < 0) across all splits.

Also reported: IS-vs-OOS Sharpe of the selected strategy (performance degradation), and P(OOS<0 |
selected) — the chance the chosen strategy actually loses live.

Vectorised over strategies via per-block (sum, sumsq, count) so C(16,8)=12,870 splits run in seconds.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16, min_coverage: float = 0.95,
             window: tuple[str, str] | None = None) -> dict:
    """Compute PBO by CSCV on a T×N matrix of per-strategy return series.

    returns      : DataFrame indexed by time, one column per strategy/config (the trial set).
    n_blocks     : S, must be even; C(S, S/2) splits are evaluated.
    min_coverage : keep only strategies observed on at least this fraction of the analysis window
                   (sparse tails are 0-filled after the filter, so blocks are equal length).
    window       : optional (start, end) to restrict to a dense common window.
    """
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    df = returns.copy()
    df.index = pd.to_datetime(df.index)
    if window is not None:
        df = df.loc[window[0]:window[1]]
    cov = df.notna().mean()
    keep = cov[cov >= min_coverage].index
    df = df[keep].fillna(0.0)
    T, N = df.shape
    if N < 4 or T < n_blocks * 4:
        return {"error": "insufficient data", "n_strategies": int(N), "n_obs": int(T)}

    T = (T // n_blocks) * n_blocks
    X = df.to_numpy()[:T]
    blocks = X.reshape(n_blocks, T // n_blocks, N)          # (S, block_len, N)
    b_sum = blocks.sum(axis=1)                              # (S, N)
    b_sq = (blocks ** 2).sum(axis=1)
    b_cnt = np.full((n_blocks, N), T // n_blocks, dtype=float)

    def sharpe(idx):
        s, sq, c = b_sum[idx].sum(0), b_sq[idx].sum(0), b_cnt[idx].sum(0)
        mean = s / c
        var = np.maximum(sq / c - mean ** 2, 1e-18)
        return mean / np.sqrt(var)

    all_blocks = set(range(n_blocks))
    lambdas, is_sr, oos_sr, oos_rel = [], [], [], []
    for is_idx in combinations(range(n_blocks), n_blocks // 2):
        oos_idx = tuple(all_blocks - set(is_idx))
        s_is, s_oos = sharpe(list(is_idx)), sharpe(list(oos_idx))
        n_star = int(np.argmax(s_is))
        # OOS rank of n* (1 = worst … N = best); relative rank ω, logit λ
        rank = int((s_oos <= s_oos[n_star]).sum())          # 1..N
        omega = rank / (N + 1.0)
        lambdas.append(np.log(omega / (1.0 - omega)))
        oos_rel.append(omega)
        is_sr.append(float(s_is[n_star]))
        oos_sr.append(float(s_oos[n_star]))
    lam = np.asarray(lambdas)
    oos_sr = np.asarray(oos_sr)
    return {
        "pbo": float((lam < 0).mean()),                     # probability of backtest overfitting
        "n_strategies": int(N),
        "n_obs": int(T),
        "n_blocks": int(n_blocks),
        "n_splits": int(len(lam)),
        "prob_oos_loss": float((oos_sr < 0).mean()),        # P(selected strategy loses OOS)
        "is_sharpe_mean": float(np.mean(is_sr)),            # per-bar Sharpe of the IS-best
        "oos_sharpe_mean": float(np.mean(oos_sr)),          # …and its OOS Sharpe (degradation)
        "window": [str(df.index.min().date()), str(df.index.max().date())],
        "_lambdas": lam,                                    # for the histogram (not serialised)
        "_oos_rel_rank": np.asarray(oos_rel),
    }
