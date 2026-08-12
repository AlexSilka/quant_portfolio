"""Trend-following research lab: entries, regime filters, direction modes and exits as
composable, causal building blocks — the deep-dive counterpart to `breakout_lab`.

The base `momentum.primary_side` answers one question (EMA-fast-vs-slow, held to reversal).
This module widens every axis the trend premium can be expressed on, so each can be measured
as a controlled variable against the same harness:

  entries   EMA cross · SMA cross · time-series-momentum sign (Moskowitz-Ooi-Pedersen) · MACD ·
            Donchian channel (breakout-as-trend) · multi-lookback blend (AQR continuous forecast)
  direction long-short · long-only · short-only · asymmetric 70/30 (crypto's positive drift)
  regime    ADX gate · long-term-EMA gate · realised-vol gate (trade trend only when it pays)
  exits     held-to-reversal · chandelier ATR-trail · Donchian channel · time stop (from breakout_lab)

Every construction is computable-at-bar: a decision stamped at bar t uses only close/high/low/
volume up to and including t (backward-only rolling windows, one-bar-lagged levels). The engine
applies the t+2 execution delay downstream, so nothing here fills at the signal bar's own price.

Two position conventions:
  * discrete  — a signed side (+1/-1/0), consumed by an exit to make a held position.
  * continuous — a trend *strength* in [-1, 1] (blend / z-scored forecast) used directly as the
    position, so conviction sizes the bet (vol-targeting rescales it downstream).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .breakout_lab import (  # exits + ATR are already causal and unit-shared
    atr,
    donchian_side,
    entry_events,
    hold_atr_trailing,
    hold_channel_exit,
    hold_time_stop,
    hold_to_reversal,
)

EPS = 1e-12

__all__ = [
    "ema_cross_side", "sma_cross_side", "tsmom_side", "macd_side", "donchian_side",
    "multi_lookback_blend", "trend_strength",
    "adx", "adx_gate", "efficiency_ratio", "efficiency_gate", "long_term_gate", "vol_gate", "apply_gate",
    "apply_direction",
    "hold_to_reversal", "hold_atr_trailing", "hold_channel_exit", "hold_time_stop",
    "entry_events",
]


# --- entries: discrete signed side (+1/-1/0) --------------------------------------

def ema_cross_side(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """+1 while the fast EMA is above the slow EMA, -1 below. The canonical trend rule."""
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    side = pd.Series(0.0, index=close.index)
    side[f > s] = 1.0
    side[f < s] = -1.0
    return side


def sma_cross_side(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """SMA analogue of the EMA cross (slower to react, fewer whipsaws)."""
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    side = pd.Series(0.0, index=close.index)
    side[f > s] = 1.0
    side[f < s] = -1.0
    return side


def tsmom_side(close: pd.Series, lookback: int = 90) -> pd.Series:
    """Time-series-momentum: +1 if the trailing `lookback`-bar return is positive, else -1.
    The Moskowitz-Ooi-Pedersen construction (their 12-month sign, generalised to any bar TF)."""
    r = close.pct_change(lookback)
    side = pd.Series(0.0, index=close.index)
    side[r > 0] = 1.0
    side[r < 0] = -1.0
    return side


def macd_side(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """+1 while the MACD line (EMA_fast - EMA_slow) is above its signal EMA, -1 below."""
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    side = pd.Series(0.0, index=close.index)
    side[macd > sig] = 1.0
    side[macd < sig] = -1.0
    return side


# --- entries: continuous trend strength in [-1, 1] --------------------------------

def multi_lookback_blend(close: pd.Series,
                         pairs=((8, 24), (16, 48), (32, 96), (64, 192)),
                         cap: float = 3.0) -> pd.Series:
    """AQR-style continuous forecast: average the vol-normalised EMA-cross signals across a set of
    fast/slow speeds, squashed to [-1, 1]. A single smooth conviction that blends short and long
    trend horizons instead of one binary cross — diversifies the speed choice away.

    Each pair contributes (ema_fast - ema_slow) normalised by the price's rolling stdev, tanh-
    squashed; the mean across pairs is the forecast. All windows are backward-only.
    """
    vol = close.pct_change().rolling(64).std().replace(0.0, np.nan)
    sig = pd.Series(0.0, index=close.index)
    for fast, slow in pairs:
        f = close.ewm(span=fast, adjust=False).mean()
        s = close.ewm(span=slow, adjust=False).mean()
        z = (f - s) / (close * vol + EPS)          # cross in units of price-vol
        sig = sig + np.tanh(np.clip(z / cap, -3.0, 3.0))
    return (sig / len(pairs)).fillna(0.0)


def trend_strength(close: pd.Series, lookback: int = 90, vol_lb: int = 60) -> pd.Series:
    """Continuous risk-adjusted momentum: trailing return scaled by its realised vol, tanh-squashed
    to [-1, 1]. The single-horizon continuous forecast (conviction-sized TSMOM)."""
    r = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_lb).std() * np.sqrt(lookback)
    return np.tanh((r / (vol + EPS)).clip(-4.0, 4.0)).fillna(0.0)


# --- regime gates (boolean masks; trade the trend only when the regime favours it) ---

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index (Wilder) in [0, 100]: trend *strength* irrespective of sign.
    Rising, high ADX = a directional regime; low ADX = chop. Fully causal (Wilder EMAs)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0.0), 0.0)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0)
    atr_ = atr(high, low, close, n)                 # Wilder-smoothed mean of true range
    alpha = 1.0 / n                                 # same smoothing on ±DM: the /n cancels in DI
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr_ + EPS)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr_ + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def adx_gate(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14,
             threshold: float = 25.0) -> pd.Series:
    """Boolean: ADX >= threshold (a directional regime worth trend-trading)."""
    return (adx(high, low, close, n) >= threshold).fillna(False)


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman efficiency ratio in [0, 1]: net move over n bars / total path length. Near 1 = a clean
    directional trend; near 0 = choppy/rangebound. A regime measure of trend *quality* (causal)."""
    direction = close.diff(n).abs()
    volatility = close.diff().abs().rolling(n).sum()
    return (direction / (volatility + EPS)).fillna(0.0)


def efficiency_gate(close: pd.Series, n: int = 20, threshold: float = 0.3) -> pd.Series:
    """Boolean: efficiency ratio >= threshold — trade trend only in a clean-trend regime, stand aside
    in chop (the regime where trend-following whipsaws). Causal."""
    return (efficiency_ratio(close, n) >= threshold).fillna(False)


def long_term_gate(close: pd.Series, span: int = 200) -> pd.Series:
    """Signed long-horizon trend (+1 above a long EMA, -1 below) — used with `apply_gate` to keep
    only entries aligned with the dominant trend (the classic 'trade with the tide' filter)."""
    ema = close.ewm(span=span, adjust=False).mean()
    return pd.Series(np.sign((close - ema).fillna(0.0)), index=close.index)


def vol_gate(close: pd.Series, lookback: int = 100, lo_q: float = 0.0, hi_q: float = 0.9,
             ref: int = 500) -> pd.Series:
    """Boolean: realised vol in the [lo_q, hi_q] quantile band of its own recent history — drop the
    extreme-vol bars where trend signals are noise/blow-off. Causal (backward quantiles)."""
    rv = close.pct_change().rolling(lookback).std()
    lo = rv.rolling(ref).quantile(lo_q)
    hi = rv.rolling(ref).quantile(hi_q)
    return ((rv >= lo) & (rv <= hi)).fillna(False)


def apply_gate(side: pd.Series, *masks: pd.Series, align: pd.Series | None = None) -> pd.Series:
    """Zero out entries failing any boolean mask; if `align` (signed) is given, keep only entries
    whose direction matches its sign. Per-bar, causal — the trend analogue of breakout's filters."""
    out = side.copy()
    for m in masks:
        out = out.where(m.reindex(out.index).fillna(False), 0.0)
    if align is not None:
        a = align.reindex(out.index).fillna(0.0)
        out = out.where(np.sign(out) == np.sign(a), 0.0)
    return out


# --- direction modes --------------------------------------------------------------

def apply_direction(pos: pd.Series, mode: str = "ls", short_weight: float = 0.3) -> pd.Series:
    """Re-weight a signed/continuous position by trading direction.

      ls          long-short, symmetric (the default).
      long_only   drop shorts (structural-drift markets: the long leg is the engine).
      short_only  drop longs (isolates the short leg's standalone value — usually a bear hedge).
      asym        full long, short scaled to `short_weight` (crypto's positive-drift 70/30 scheme).
    """
    if mode == "ls":
        return pos
    if mode == "long_only":
        return pos.clip(lower=0.0)
    if mode == "short_only":
        return pos.clip(upper=0.0)
    if mode == "asym":
        return pos.where(pos >= 0.0, pos * short_weight)
    raise ValueError(mode)


# ── panel time-series momentum: ONE implementation for the two macro sleeves ─────────────────
def tsmom_panel(close: pd.DataFrame, lookbacks, ppy: float, cost_bps: float = 2.0,
                target: float = 0.15, vol_lb: int = 40, cap: float = 3.0,
                carry_pa: pd.DataFrame | None = None) -> pd.Series:
    """Equal-weight sign-blend TSMOM over a price panel, per-asset vol-scaled, net of turnover cost.

    This was written twice — once in `scripts/run_crisis.py` and once in `scripts/run_gmacro.py` —
    and both copies carried the same two defects, which is what a copy does:

      * `sig.shift(1)` earning `close.pct_change()` fills at the CLOSE OF THE BAR THAT GENERATED THE
        SIGNAL. `backtest.engine` forbids exactly that and defaults to `exec_lag=2` for it; every
        other leg in the book obeys it. Correcting it alone took the global-macro commodity tranche
        from Sharpe +0.82 to +0.37 and the crisis commodity tranche from +0.56 to +0.42.
      * the per-asset vol scaler was NOT lagged, so a name's weight on day t was divided by a 40-day
        standard deviation containing day t's own return — the position shrinks exactly on the day it
        moves. Lagging it costs the crisis equity tranche +0.21 -> +0.05.

    Both are fixed here, once. `sig.shift(2)` is the same convention as the engine: a signal stamped
    on close(t) is filled at close(t+1) and first earns the (t+1, t+2] return.

    `carry_pa` is what a PRICE series does not pay: an annualised % rate per name, added to the
    position's return. A spot FX cross is a funded position and the interest differential is the whole
    reason its forward differs from its spot; an ETF price series drops every distribution, which on
    this universe averages 2.3% a year and reaches 6.2% on HYG. Both were simply absent, and both
    turn out to have been understating the leg — the bond tranche is a loser without them and a
    (marginal) winner with them.
    """
    import numpy as _np
    r = close.pct_change()
    vol = r.rolling(vol_lb).std().shift(1)                  # lagged: computable at the bar
    sig = sum(_np.sign(close / close.shift(h) - 1.0) for h in lookbacks) / len(lookbacks)
    pos = sig.shift(2) * (target / _np.sqrt(ppy) / vol).clip(upper=cap)
    n = close.shape[1]
    gross = (pos * r).sum(axis=1) / n
    if carry_pa is not None:
        gross = gross + (pos * carry_pa.reindex_like(pos).fillna(0.0) / 100.0 / ppy).sum(axis=1) / n
    cost = (pos.diff().abs().sum(axis=1) / n) * cost_bps / 1e4
    return gross - cost
