"""Sector-ETF pairs stat-arb sleeve — trade the mean-reverting spread between cointegrated sector
ETFs. The one mean-reversion family that survives realistic costs on liquid instruments: sector
spreads revert around a slow economic relationship, and each pair is market- and largely
sector-neutral, so the return source is distinct from trend and cross-sectional momentum.

Pairs are selected by an Engle-Granger cointegration test on a FORMATION window only (no selection
look-ahead), then traded out-of-sample. Each spread uses a rolling, past-only hedge ratio and
z-score; per-pair parameters are walk-forward selected; positions execute at t+2 and pay both legs'
costs. The sleeve return is the equal-weight basket of the selected pairs, vol-targeted to ~15%.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from src.risk.sizing import vol_target_scale

# The ten SPDR select-sector ETFs — liquid, tight spreads, long history.
SECTOR_ETFS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"]
TVOL = 0.15


def _sharpe(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    return float(np.sqrt(ppy) * r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else 0.0


def _cointegration(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Engle-Granger on logs: OLS hedge ratio, ADF p-value on the residual, OU half-life (bars).

    Degenerate pairs (near-constant residual -> no tradeable spread) return p=1 rather than raising.
    """
    df = pd.concat([np.log(y), np.log(x)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 100:
        return 1.0, np.nan
    ly, lx = df.iloc[:, 0], df.iloc[:, 1]
    res = ly - np.polyfit(lx, ly, 1)[0] * lx
    if res.std() < 1e-8:
        return 1.0, np.nan
    try:
        p = float(adfuller(res.to_numpy(), maxlag=1, autolag=None)[1])
    except Exception:
        return 1.0, np.nan
    dres, lag = res.diff().dropna(), res.shift(1).dropna()
    lam = np.polyfit(lag.loc[dres.index], dres, 1)[0]
    hl = -np.log(2) / lam if lam < 0 else np.nan
    return p, float(hl) if np.isfinite(hl) else np.nan


def _positions_from_z(z: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    """Enter short spread at z > +entry, long at z < -entry; flatten as z crosses back through exit_."""
    pos = np.zeros(len(z))
    state = 0.0
    for i, zi in enumerate(z):
        if not np.isnan(zi):
            if state == 0.0:
                state = -1.0 if zi > entry else (1.0 if zi < -entry else 0.0)
            elif state > 0 and zi >= -exit_:
                state = 0.0
            elif state < 0 and zi <= exit_:
                state = 0.0
        pos[i] = state
    return pos


def _spread_return(y: pd.Series, x: pd.Series, lookback: int, entry: float,
                   ppy: float, cost_bps: float, exit_: float = 0.5) -> pd.Series:
    """One pair's vol-targeted daily return: past-only rolling hedge ratio + z-score, t+2, two legs."""
    y, x = y.align(x, join="inner")
    ry, rx = y.pct_change(), x.pct_change()
    beta = (ry.rolling(lookback).cov(rx) / (rx.rolling(lookback).var() + 1e-12)).shift(1)
    spread = np.log(y) - beta * np.log(x)
    z = (spread - spread.rolling(lookback).mean()) / (spread.rolling(lookback).std() + 1e-12)
    pos = pd.Series(_positions_from_z(z.to_numpy(), entry, exit_), index=y.index)
    pair_ret = ry - beta * rx
    scale = vol_target_scale(pair_ret, TVOL, ppy)
    held = (pos * scale).shift(2).fillna(0.0)                                   # t+2 execution
    return (held * pair_ret - held.diff().abs().fillna(0.0) * 2 * cost_bps / 1e4).dropna()


def pairs_basket(panel: pd.DataFrame, ppy: float = 252, cost_bps: float = 2.0,
                 form_years: float = 2.0, step_months: int = 6,
                 lookbacks: tuple = (60, 90), entries: tuple = (1.5, 2.0),
                 adf_p: float = 0.05, hl_bars: tuple = (3, 250)) -> tuple[pd.Series | None, float, int]:
    """Walk-forward stat-arb: every `step_months`, re-select cointegrated pairs on a TRAILING
    `form_years` window and pick each pair's params on that window, then trade them out-of-sample for
    the next step; stitch the periods. Both the pair set AND the parameters refresh over time, so the
    sleeve adapts to regime change and never selects on data it trades (no look-ahead).

    Returns (stitched OOS basket return, mean pairs traded per period, number of re-selection periods);
    the basket is the vol-targeted daily P&L stream ready to drop into the portfolio (None if no pair
    is ever cointegrated-and-tradeable).
    """
    names = list(panel.columns)
    aligned = panel.dropna(how="all").ffill().dropna()
    anchors = list(pd.date_range(aligned.index[0] + pd.DateOffset(years=form_years),
                                 aligned.index[-1], freq=f"{step_months}MS"))
    periods, n_sel = [], []
    for i, t0 in enumerate(anchors):
        t1 = anchors[i + 1] if i + 1 < len(anchors) else aligned.index[-1] + pd.Timedelta(days=1)
        form = aligned.loc[t0 - pd.DateOffset(years=form_years):t0]
        legs = []
        for a, b in itertools.combinations(names, 2):
            if len(form) < 100:
                continue
            p, hl = _cointegration(form[a], form[b])
            if not (p < adf_p and hl_bars[0] < hl < hl_bars[1]):
                continue
            lb, ez = max(((lb, ez) for lb in lookbacks for ez in entries),
                         key=lambda k: _sharpe(_spread_return(form[a], form[b], k[0], k[1], ppy, cost_bps), ppy))
            traded = _spread_return(aligned[a].loc[:t1], aligned[b].loc[:t1], lb, ez, ppy, cost_bps)
            legs.append(traded.loc[t0:t1])
        if legs:
            n_sel.append(len(legs))
            periods.append(pd.concat(legs, axis=1).mean(axis=1))
    if not periods:
        return None, 0.0, len(anchors)
    basket = pd.concat(periods)
    basket = basket[~basket.index.duplicated(keep="first")].dropna()
    return basket, float(np.mean(n_sel)), len(anchors)
