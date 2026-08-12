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
             symbol: str | None = None, exec_lag: int = 2) -> pd.DataFrame:
    """Return a per-bar frame: position, ret, gross_ret, cost, funding, net_ret, equity.

    `symbol` is how a single-asset caller gets carry for free. The panel backtests read the venue off
    the panel's own names; a bare price Series carries none, so funding here was an explicit argument
    that a caller could simply not pass — and holding a perp for nothing is the defect
    `src/backtest/carry` exists to make impossible. Naming the symbol resolves its funding from the
    archive; passing `funding=` still wins, and a symbol the venue never settled funding on (an
    equity, a spot pair) resolves to none, so it is always safe to name it.
    """
    close = prices.sort_index()
    if funding is None and symbol is not None:
        from src.backtest.carry import perp_symbols, settlements
        if symbol in perp_symbols():
            funding = settlements(symbol)
    r = close.pct_change().fillna(0.0)

    # signal stamped at bar t is filled exec_lag bars later (default 2 = one bar AFTER the
    # signal bar's close), held until changed — never executes at the signal bar's own price
    pos = target_pos.reindex(close.index).ffill().fillna(0.0).shift(exec_lag).fillna(0.0)
    gross_ret = pos * r

    # Turnover is the target change PLUS the drift back onto it. `gross_ret = pos * r` prices a
    # position held at a constant fraction of NAV, and a fraction does not stay constant on its own:
    # hold p through a return r and it becomes p(1+r)/(1+p·r). At p = 1 that is exactly 1 and there is
    # nothing to charge, which is why this never showed on an unlevered sleeve — but the vol-target
    # runs these legs at whatever size 15% needs, and on the trend legs `pos.diff()` alone charges
    # 0.4-0.9x/yr against 1.2-1.6x actually traded. Same defect as `xsect.held_turnover` fixes for the
    # panel books; a single asset just has one column.
    drifted = pos * (1.0 + r) / (1.0 + pos * r).replace(0.0, np.nan)
    dpos = (pos - drifted.shift(1)).abs().fillna(pos.abs())
    # bar vol for the √-impact cost term — causal: expands over the first 20 bars then rolls,
    # lagged one bar, zero on the warm-up. (No .bfill()/full-sample .std() — those would seed the
    # early bars from future/whole-series vol; harmless to the return path but a real look-ahead.)
    sigma_bar = r.rolling(20, min_periods=1).std().shift(1).fillna(0.0)
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
    # `turnover` is published rather than left to be re-derived from `position`: the two differ now
    # that the drift back onto a held size is charged, and a consumer re-deriving it would report a
    # turnover the book never paid — the shape of defect this whole pass exists to remove.
    return pd.DataFrame({"position": pos, "ret": r, "gross_ret": gross_ret, "turnover": dpos,
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
