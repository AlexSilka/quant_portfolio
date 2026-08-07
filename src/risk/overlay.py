"""Book-level risk overlays: volatility targeting and a drawdown-responsive de-risking ladder.

These sit ABOVE the per-sleeve vol-targeting — they size the *whole book* by its own realised risk,
implementing the portfolio-level risk logic the task requires (§8): a vol target that holds the book
at a constant risk budget, and a drawdown ladder that cuts gross exposure in stated steps as losses
deepen and restores it as they recover.

Both are causal — exposure applied to bar t uses information through t-1 only (realised vol is lagged;
the ladder sizes bar t from the drawdown of the managed equity through t-1). Applying an exposure to a
book *return* series is post-hoc leverage: legitimate as portfolio risk-budgeting, with `cap` standing
in for the maximum-leverage limit and costs/funding assumed to scale ~linearly with it (a small extra
drag not re-modelled here — noted where the numbers are reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realised_vol(ret: pd.Series, lookback: int = 30, ppy: int = 365) -> pd.Series:
    """Annualised trailing realised volatility (causal)."""
    return ret.rolling(lookback).std() * np.sqrt(ppy)


def vol_managed(ret: pd.Series, target_vol: float = 0.12, lookback: int = 30,
                cap: float = 3.0, floor: float = 0.0, ppy: int = 365) -> tuple[pd.Series, pd.Series]:
    """Scale a return series to a constant target annualised vol using LAGGED realised vol.

    Moreira-Muir / AQR 'volatility-managed portfolios': because volatility clusters and high-vol
    episodes precede weak momentum returns, cutting exposure when recent vol is high can *raise*
    Sharpe, not merely slide along the risk axis. Returns (scaled_ret, exposure).
    """
    expo = (target_vol / realised_vol(ret, lookback, ppy)).clip(lower=floor, upper=cap)
    expo = expo.shift(1).fillna(0.0)                       # size bar t from vol through t-1
    return (expo * ret).rename("ret"), expo.rename("exposure")


def drawdown_ladder(ret: pd.Series,
                    triggers: tuple[tuple[float, float], ...] = ((-0.06, 0.66), (-0.09, 0.33), (-0.12, 0.0)),
                    restore: float = -0.04) -> tuple[pd.Series, pd.Series]:
    """Drawdown-responsive de-risking ladder (task §8: stated triggers, step sizes, stop & restore).

    The drawdown SIGNAL is the STRATEGY's own (full-exposure) running drawdown — NOT the realised managed
    drawdown. This matters at the deepest trigger: cutting gross to 0 ('stop trading') freezes the *managed*
    equity, so a restore keyed off it could never fire and would strand the book flat forever. Keying off the
    always-moving strategy equity means the ladder de-risks in step as the strategy falls (cut to that step)
    and restores to full once the strategy itself recovers above `restore` (hysteresis, no threshold
    flip-flop). `triggers` must be ordered shallow→deep. Causal: bar t is sized from the drawdown through t-1.
    Returns (scaled_ret, exposure).
    """
    r = ret.fillna(0.0).to_numpy()
    n = len(r)
    managed = np.empty(n)
    expo = np.empty(n)
    ref_eq = ref_peak = 1.0                                # STRATEGY (full-exposure) equity — the dd signal
    cur = 1.0                                              # current exposure state (hysteresis)
    for i in range(n):
        dd = ref_eq / ref_peak - 1.0                       # strategy drawdown through t-1 (always moves)
        target = 1.0
        for thr, step in triggers:                         # deepest matching trigger wins
            if dd <= thr:
                target = step
        if target < cur:
            cur = target                                   # de-risk immediately
        elif dd >= restore:
            cur = 1.0                                       # re-risk only after the strategy recovers (hysteresis)
        expo[i] = cur
        managed[i] = cur * r[i]
        ref_eq *= 1.0 + r[i]                               # advance on the RAW return, so the signal never freezes
        ref_peak = max(ref_peak, ref_eq)
    idx = ret.index
    return pd.Series(managed, index=idx, name="ret"), pd.Series(expo, index=idx, name="exposure")


def apply_overlay(ret: pd.Series, target_vol: float = 0.12, lookback: int = 30, cap: float = 3.0,
                  triggers=((-0.06, 0.66), (-0.09, 0.33), (-0.12, 0.0)), restore: float = -0.04,
                  ppy: int = 365) -> tuple[pd.Series, pd.DataFrame]:
    """Full overlay: vol-target the book, then apply the drawdown ladder on the vol-managed series.
    Returns (final_ret, exposures[vol, ladder, gross])."""
    vm_ret, vm_expo = vol_managed(ret, target_vol, lookback, cap, ppy=ppy)
    fin_ret, dd_expo = drawdown_ladder(vm_ret, triggers, restore)
    expos = pd.DataFrame({"vol": vm_expo, "ladder": dd_expo, "gross": (vm_expo * dd_expo)})
    return fin_ret.rename("ret"), expos
