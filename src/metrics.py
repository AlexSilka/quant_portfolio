"""Performance metrics + Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado).

The DSR/PSR inputs are per-bar (non-annualised) Sharpe with the return series' own skew and
kurtosis and T = number of observations. `sharpe()` annualises for readability only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1))


def sortino(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    downside = r[r < 0].std(ddof=1)
    if not np.isfinite(downside) or downside == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / downside)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def monthly_returns(net_ret: pd.Series) -> pd.Series:
    return (1.0 + net_ret).resample("ME").prod() - 1.0


def months_in_profit(net_ret: pd.Series) -> float:
    m = monthly_returns(net_ret)
    return float((m > 0).mean()) if len(m) else 0.0


def probabilistic_sharpe(sr_hat: float, T: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float:
    """PSR = P(true SR > benchmark), non-normal SR variance (Mertens/Lo). Per-bar SR inputs."""
    denom = np.sqrt(max(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2, 1e-12))
    return float(norm.cdf((sr_hat - sr_benchmark) * np.sqrt(max(T - 1, 1)) / denom))


def expected_max_sharpe(n_trials: int, var_across_trials: float) -> float:
    """E[max SR] under the null across N independent trials (false-strategy theorem)."""
    gamma = 0.5772156649015329
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_across_trials) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe(sr_hat: float, T: int, skew: float, kurt: float,
                    n_trials: int, var_across_trials: float) -> float:
    """DSR = PSR evaluated at benchmark = E[max SR under the null]. Per-bar SR inputs."""
    sr0 = expected_max_sharpe(n_trials, var_across_trials)
    return probabilistic_sharpe(sr_hat, T, skew, kurt, sr_benchmark=sr0)


def summarise(net_ret: pd.Series, periods_per_year: float) -> dict:
    r = net_ret.dropna()
    eq = (1.0 + r).cumprod()
    sr_bar = 0.0 if r.std(ddof=1) == 0 else r.mean() / r.std(ddof=1)
    return {
        "sharpe_ann": sharpe(r, periods_per_year),
        "sortino_ann": sortino(r, periods_per_year),
        "max_dd": max_drawdown(eq),
        "months_in_profit": months_in_profit(r),
        "psr_gt0": probabilistic_sharpe(sr_bar, len(r), r.skew(), r.kurt() + 3.0),
        "total_return": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
        "n_obs": int(len(r)),
    }
