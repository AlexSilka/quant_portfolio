"""Monte Carlo robustness via the stationary block bootstrap (Politis-Romano, arch).

The stationary bootstrap preserves serial dependence (unlike an iid bootstrap, which gives
falsely tight bands), with a data-driven block length. Reports P5/P50/P95 of Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length


def bootstrap_sharpe(net_ret: pd.Series, periods_per_year: float,
                     n_reps: int = 1000, seed: int = 7) -> dict:
    r = net_ret.dropna().to_numpy()
    if len(r) < 50:
        return {}
    block = float(np.asarray(optimal_block_length(r)["stationary"])[0])
    bs = StationaryBootstrap(block, r, seed=seed)
    out = np.empty(n_reps)
    for i, (pos_args, _) in enumerate(bs.bootstrap(n_reps)):
        x = pos_args[0]
        sd = x.std(ddof=1)
        out[i] = np.sqrt(periods_per_year) * x.mean() / sd if sd > 0 else 0.0
    return {
        "sharpe_p5": float(np.percentile(out, 5)),
        "sharpe_p50": float(np.percentile(out, 50)),
        "sharpe_p95": float(np.percentile(out, 95)),
        "block_len": block,
        "n_reps": n_reps,
    }


def _maxdd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _pct3(a: np.ndarray):
    a = np.asarray(a)[np.isfinite(a)]
    return [round(float(np.percentile(a, p)), 4) for p in (5, 50, 95)] if len(a) else [None] * 3


def _pack(sh, dd, hit, wm, **extra) -> dict:
    s5, s50, s95 = _pct3(sh)
    d5, d50, d95 = _pct3(dd)
    h5, h50, h95 = _pct3(hit)
    w5, w50, w95 = _pct3(wm)
    return {"sharpe_p5": s5, "sharpe_p50": s50, "sharpe_p95": s95,
            "maxdd_p5": d5, "maxdd_p50": d50, "maxdd_p95": d95,
            "hit_p5": h5, "hit_p50": h50, "hit_p95": h95,
            "wmonth_p5": w5, "wmonth_p50": w50, "wmonth_p95": w95, **extra}


def _months_chunk(x: np.ndarray, mlen: int) -> np.ndarray:
    n_full = (len(x) // mlen) * mlen
    if not n_full:
        return np.empty(0)
    return (1.0 + x[:n_full].reshape(-1, mlen)).prod(axis=1) - 1.0


def _monthly_hit_chunk(x: np.ndarray, mlen: int) -> float:
    m = _months_chunk(x, mlen)
    return float((m > 0).mean()) if len(m) else np.nan


def _monthly_worst_chunk(x: np.ndarray, mlen: int) -> float:
    """Worst month of a resample — the brief's -6% floor gets the same tail band as Sharpe/max-DD."""
    m = _months_chunk(x, mlen)
    return float(m.min()) if len(m) else np.nan


def mc_metrics(net_ret: pd.Series, periods_per_year: float, n_reps: int = 1000,
               seed: int = 7, monthly_ppy: int = 12) -> dict:
    """Block-bootstrap P5/P50/P95 for Sharpe, max-drawdown, monthly hit rate and worst month together.

    One resampling scheme (stationary block bootstrap, dependence-preserving) drives all four
    metrics so they are mutually consistent. The monthly figures are computed by chunking each
    resample into month-length blocks (periods_per_year / 12) — a bootstrap proxy that needs no
    calendar index. Returns {metric_p5/p50/p95}.
    """
    r = net_ret.dropna().to_numpy()
    if len(r) < 50:
        return {}
    block = float(np.asarray(optimal_block_length(r)["stationary"])[0])
    bs = StationaryBootstrap(block, r, seed=seed)
    mlen = max(int(round(periods_per_year / monthly_ppy)), 1)
    sh, dd, hit, wm = (np.empty(n_reps) for _ in range(4))
    for i, (pos_args, _) in enumerate(bs.bootstrap(n_reps)):
        x = pos_args[0]
        sd = x.std(ddof=1)
        sh[i] = np.sqrt(periods_per_year) * x.mean() / sd if sd > 0 else 0.0
        dd[i] = _maxdd(x)
        hit[i] = _monthly_hit_chunk(x, mlen)
        wm[i] = _monthly_worst_chunk(x, mlen)
    return _pack(sh, dd, hit, wm, block_len=round(block, 2), n_reps=n_reps)


def _real_monthly_hit(r: pd.Series) -> float:
    m = (1.0 + r).resample("ME").prod() - 1.0
    return float((m > 0).mean()) if len(m) else np.nan


def _real_monthly_worst(r: pd.Series) -> float:
    m = (1.0 + r).resample("ME").prod() - 1.0
    return float(m.min()) if len(m) else np.nan


def trade_order_mc(net_ret: pd.Series, periods_per_year: float, n_reps: int = 1000,
                   seed: int = 7, monthly_ppy: int = 12) -> dict:
    """Trade-order resampling: iid permutation of the realised returns.

    Destroys serial dependence entirely, so the P&L path (and thus max-drawdown / streaks) is re-drawn
    from the same return *distribution* in random order — the brief's "trade-order resampling". Sharpe
    is permutation-invariant by construction (mean/std unchanged), so its band is ~degenerate; the point
    of this arm is the drawdown / hit-rate spread that path-order alone produces. Monthly hit uses the
    month-length chunk proxy (permuted returns have no calendar).
    """
    r = net_ret.dropna().to_numpy()
    if len(r) < 50:
        return {}
    rng = np.random.default_rng(seed)
    mlen = max(int(round(periods_per_year / monthly_ppy)), 1)
    sh, dd, hit, wm = (np.empty(n_reps) for _ in range(4))
    for i in range(n_reps):
        x = rng.permutation(r)
        sd = x.std(ddof=1)
        sh[i] = np.sqrt(periods_per_year) * x.mean() / sd if sd > 0 else 0.0
        dd[i] = _maxdd(x)
        hit[i] = _monthly_hit_chunk(x, mlen)
        wm[i] = _monthly_worst_chunk(x, mlen)
    return _pack(sh, dd, hit, wm, n_reps=n_reps)


def entry_jitter_mc(net_ret: pd.Series, periods_per_year: float, n_reps: int = 1000,
                    seed: int = 7, max_lag: int = 3) -> dict:
    """Entry jitter ±1..max_lag bars: delay/advance execution by a random small lag each replicate.

    Perturbs *when* the book is on: r_jittered = net_ret.shift(k), k ∈ {±1,±2,±3}. NOTE — applied to a
    *realised* P&L series a uniform shift only drops |k| endpoints and re-bins month boundaries, so Sharpe
    and max-DD are near-invariant by construction (no signal↔return misalignment is left to perturb); the
    informative dimension here is monthly-hit-rate boundary sensitivity. Real calendar index is preserved,
    so that hit rate is the true resample("ME") value, not a proxy. (Signal-level entry jitter — perturbing
    the position series before it earns returns — is done per-sleeve inside the deep-dives.)
    """
    r = net_ret.dropna()
    if len(r) < 50:
        return {}
    rng = np.random.default_rng(seed)
    lags = [k for k in range(-max_lag, max_lag + 1) if k != 0]
    sh, dd, hit, wm = (np.empty(n_reps) for _ in range(4))
    for i in range(n_reps):
        k = int(rng.choice(lags))
        x = r.shift(k).dropna()
        sd = x.std(ddof=1)
        sh[i] = np.sqrt(periods_per_year) * x.mean() / sd if sd > 0 else 0.0
        dd[i] = _maxdd(x.to_numpy())
        hit[i] = _real_monthly_hit(x)
        wm[i] = _real_monthly_worst(x)
    return _pack(sh, dd, hit, wm, n_reps=n_reps, max_lag=max_lag)


def random_start_mc(net_ret: pd.Series, periods_per_year: float, n_reps: int = 1000,
                    seed: int = 7, frac: float = 0.8) -> dict:
    """Randomised start dates: metrics on a random contiguous sub-window of length frac·T each replicate.

    Tests whether the result depends on the sample's start/end — a robust book scores similarly on most
    contiguous 80% windows; one carried by a single regime does not. Real index preserved (true monthly
    hit).
    """
    r = net_ret.dropna()
    T = len(r)
    win = int(T * frac)
    if T < 50 or win < 30:
        return {}
    rng = np.random.default_rng(seed)
    sh, dd, hit, wm = (np.empty(n_reps) for _ in range(4))
    for i in range(n_reps):
        s = int(rng.integers(0, T - win + 1))
        x = r.iloc[s:s + win]
        sd = x.std(ddof=1)
        sh[i] = np.sqrt(periods_per_year) * x.mean() / sd if sd > 0 else 0.0
        dd[i] = _maxdd(x.to_numpy())
        hit[i] = _real_monthly_hit(x)
        wm[i] = _real_monthly_worst(x)
    return _pack(sh, dd, hit, wm, n_reps=n_reps, window_frac=frac)


def mc_all_variants(net_ret: pd.Series, periods_per_year: float, n_reps: int = 1000,
                    seed: int = 7) -> dict:
    """All four Monte-Carlo schemes the brief requires, each with P5/P50/P95 of Sharpe, max-DD,
    monthly hit rate and worst month: stationary block bootstrap, trade-order resampling, entry jitter
    (±1-3 bars) and randomised start dates. Returns {variant: {metric_pXX}}."""
    return {
        "block_bootstrap": mc_metrics(net_ret, periods_per_year, n_reps, seed),
        "trade_order": trade_order_mc(net_ret, periods_per_year, n_reps, seed),
        "entry_jitter": entry_jitter_mc(net_ret, periods_per_year, n_reps, seed),
        "random_start": random_start_mc(net_ret, periods_per_year, n_reps, seed),
    }
