"""Vectorised sleeve backtest with bar-close->execution delay, costs and funding.

Return convention: gross_ret[t] = held_position[t] * close.pct_change()[t], where the held
position is `target_pos.shift(exec_lag)`. pct_change[t] is realised over (t-1, t], so a shift
of 1 would fill at the signal bar's OWN close — the spec forbids executing at the price of the
bar that generated the signal. exec_lag therefore defaults to 2: a signal stamped at bar t is
filled at close(t+1) and first earns the (t+1, t+2] return (a genuine one-bar delay).

Costs are charged on turnover (|change in position|); funding is charged on the held position,
summed over every settlement that falls inside each bar (so no settlement is dropped when the
bar is coarser than the 8h funding grid, e.g. three settlements per 1d bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import funding_pnl, trade_cost_bps


def backtest(prices: pd.Series, target_pos: pd.Series, *, capital: float,
             commission_bps: float, half_spread_bps: float, impact_k: float = 0.1,
             adv: pd.Series | None = None, funding: pd.Series | None = None,
             exec_lag: int = 2) -> pd.DataFrame:
    """Return a per-bar frame: position, ret, gross_ret, cost, funding, net_ret, equity."""
    close = prices.sort_index()
    r = close.pct_change().fillna(0.0)

    # signal stamped at bar t is filled exec_lag bars later (default 2 = one bar AFTER the
    # signal bar's close), held until changed — never executes at the signal bar's own price
    pos = target_pos.reindex(close.index).ffill().fillna(0.0).shift(exec_lag).fillna(0.0)
    gross_ret = pos * r

    dpos = pos.diff().abs().fillna(pos.abs())           # turnover
    sigma_bar = r.rolling(20).std().bfill().fillna(r.std())
    notional = dpos * capital
    adv_ser = (adv.reindex(close.index).ffill() if adv is not None
               else pd.Series(np.inf, index=close.index))
    cost_bps = trade_cost_bps(notional.to_numpy(), adv_ser.to_numpy(),
                              sigma_bar.to_numpy(), commission_bps, half_spread_bps, impact_k)
    cost = dpos * pd.Series(cost_bps, index=close.index) / 1e4

    fund = (funding_pnl(pos, funding) if funding is not None
            else pd.Series(0.0, index=close.index))

    net_ret = gross_ret - cost + fund
    equity = (1.0 + net_ret).cumprod()
    return pd.DataFrame({"position": pos, "ret": r, "gross_ret": gross_ret,
                         "cost": cost, "funding": fund, "net_ret": net_ret,
                         "equity": equity})


def vol_target(position: pd.Series, close: pd.Series, target_ann: float,
               ppy_bar: float, lookback: int = 100, cap: float = 3.0) -> pd.Series:
    """Scale a raw position to a constant annualised volatility (risk parity across sleeves,
    assets and timeframes). The vol estimate is lagged one bar so it uses no future data."""
    rv = close.pct_change().rolling(lookback).std() * np.sqrt(ppy_bar)
    scale = (target_ann / rv).clip(upper=cap).shift(1).fillna(0.0)
    return position * scale


def positions_from_events(index: pd.DatetimeIndex, side: pd.Series,
                          t1: pd.Series, keep: pd.Index) -> pd.Series:
    """Build a held-position series: for each kept event t0, hold `side` over [t0, t1]."""
    pos = pd.Series(0.0, index=index)
    for t0 in keep:
        s = side.get(t0, 0.0)
        end = t1.get(t0, t0)
        if s != 0.0:
            pos.loc[t0:end] = s
    return pos
