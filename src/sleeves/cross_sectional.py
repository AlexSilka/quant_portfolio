"""Cross-sectional momentum sleeve — long recent winners, short recent losers, dollar-neutral.

Structurally distinct from time-series trend: it bets on *relative* ranking across a panel, not
on each asset's own trend, so it is largely market-neutral. A documented, robust equity anomaly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(panel_close: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    """Trailing total return per name — the cross-sectional ranking signal."""
    return panel_close.pct_change(lookback)


def breakout_signal(panel_close: pd.DataFrame, kind: str = "nearness",
                    lookback: int = 252) -> pd.DataFrame:
    """Cross-sectional breakout ranking signal (higher = more broken-out, ranked long).

    kind='nearness'  : close / trailing max (George & Hwang 52-week-high nearness; the evidenced
                       cross-sectional breakout proxy — long names nearest their high, short farthest).
    kind='donchian'  : position in the trailing N-bar close range in [0,1] (0=at low, 1=at high).
    All windows are backward-only; xs_returns lags the resulting weights t+2, so nothing looks ahead.
    """
    if kind == "nearness":
        return panel_close / panel_close.rolling(lookback).max()
    if kind == "donchian":
        hh = panel_close.rolling(lookback).max()
        ll = panel_close.rolling(lookback).min()
        return (panel_close - ll) / (hh - ll + 1e-12)
    raise ValueError(kind)


def xs_returns(panel_close: pd.DataFrame, signal: pd.DataFrame,
               top_frac: float = 0.3) -> tuple[pd.Series, pd.Series]:
    """Dollar-neutral long-top / short-bottom returns and turnover from a ranking signal.

    Weights are lagged two bars against the pct_change return so a signal stamped at bar t is
    filled at close(t+1), never at the signal bar's own close. Returns (gross, turnover) per bar.
    """
    rets = panel_close.pct_change()
    ranks = signal.rank(axis=1, pct=True)
    longs = (ranks >= 1.0 - top_frac).astype(float)
    shorts = (ranks <= top_frac).astype(float)
    wl = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0)
    ws = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).shift(2).fillna(0.0)
    return (w * rets).sum(axis=1), w.diff().abs().sum(axis=1)
