"""Breakout research lab: entries, filters and exits as composable, causal building blocks.

Every construction here is computable-at-bar: a decision stamped at bar t uses only close/high/
low/volume up to and including t (backward-only rolling windows, one-bar-lagged channel levels).
Execution delay (t+2) is applied downstream by the backtest engine, so nothing here fills at the
signal bar's own price.

The module exists to answer one question the base `breakout.primary_side` + triple-barrier cannot:
does breakout edge live in the *fat tail* of the big move (like trend-following), so that a short,
bounded exit throws it away? It therefore offers the full exit menu — bounded (time / triple-
barrier) versus trend-riding (opposite channel / ATR-trailing / held-to-reversal) — over the same
entries, so the exit is the controlled variable.

Entries return a *persistent* side series (+1 while the up-breakout condition holds, -1 while the
down condition holds, 0 otherwise). Exits consume that side and return a *held position* series
(+1/-1/0 per bar) ready for vol-targeting and the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# --- shared -----------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average true range (Wilder EMA of true range), in price units. Causal."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def entry_events(side: pd.Series) -> pd.DatetimeIndex:
    """Onset bars: the first bar of each new non-zero directional run (dedup consecutive)."""
    prev = side.shift(1).fillna(0.0)
    return side.index[(side != 0.0) & (side != prev)]


def fresh_side(side: pd.Series) -> pd.Series:
    """Collapse a persistent side to an impulse that fires only on the onset bar.

    The entries here are *persistent* (+1 for every bar price sits above the channel), so a
    trend-riding exit that stops out mid-move re-enters on the very next bar while the condition
    still holds — it pays the round trip and buys the same position back. Feeding the impulse
    instead makes a stop-out final until price makes a genuinely new breakout.
    """
    out = pd.Series(0.0, index=side.index)
    onset = entry_events(side)
    out.loc[onset] = side.reindex(onset)
    return out


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ADX — trend strength, direction-blind, 0-100. Causal.

    Used as a regime gate: the standard reading is that below ~20-25 the market is ranging and
    breakouts fail more often than they follow through.
    """
    up, dn = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = atr(high, low, close, n)
    ew = dict(alpha=1.0 / n, adjust=False, min_periods=n)
    plus_di = 100.0 * plus_dm.ewm(**ew).mean() / (tr + EPS)
    minus_di = 100.0 * minus_dm.ewm(**ew).mean() / (tr + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(**ew).mean()


# --- entries (persistent side) ----------------------------------------------------

def donchian_side(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 55,
                  buffer_atr: float = 0.0) -> pd.Series:
    """+1 while close breaks the trailing N-bar high (optionally by buffer_atr*ATR), -1 the low.

    buffer_atr > 0 is the classic false-breakout filter: the close must clear the channel by a
    fraction of ATR, not merely tag it. The channel is shifted one bar so bar t's own extreme
    never defines its own breakout level.
    """
    hh = high.rolling(lookback).max().shift(1)
    ll = low.rolling(lookback).min().shift(1)
    side = pd.Series(0.0, index=close.index)
    if buffer_atr:
        a = atr(high, low, close, 14)
        side[close > hh + buffer_atr * a] = 1.0
        side[close < ll - buffer_atr * a] = -1.0
    else:
        side[close > hh] = 1.0
        side[close < ll] = -1.0
    return side


def bollinger_side(close: pd.Series, lookback: int = 20, k: float = 2.0) -> pd.Series:
    """+1 while close is above the upper Bollinger band (MA + k*std), -1 below the lower."""
    m = close.rolling(lookback).mean()
    s = close.rolling(lookback).std()
    side = pd.Series(0.0, index=close.index)
    side[close > m + k * s] = 1.0
    side[close < m - k * s] = -1.0
    return side


def keltner_side(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 20,
                 k: float = 2.0) -> pd.Series:
    """+1 while close breaks EMA + k*ATR (Keltner upper), -1 below EMA - k*ATR."""
    ema = close.ewm(span=lookback, adjust=False).mean()
    a = atr(high, low, close, lookback)
    side = pd.Series(0.0, index=close.index)
    side[close > ema + k * a] = 1.0
    side[close < ema - k * a] = -1.0
    return side


# --- entry filters (gate the side to reduce whipsaw) ------------------------------

def volume_filter(volume: pd.Series, lookback: int = 20, z_min: float = 0.5) -> pd.Series:
    """Boolean: bar's dollar/coin volume z-score over the window >= z_min (expansion confirm)."""
    m = volume.rolling(lookback).mean()
    s = volume.rolling(lookback).std()
    z = (volume - m) / (s + EPS)
    return (z >= z_min).fillna(False)


def trend_filter(close: pd.Series, span: int = 100) -> pd.Series:
    """Signed long-term trend: +1 if above a long EMA, -1 if below. Align breakouts with it."""
    ema = close.ewm(span=span, adjust=False).mean()
    return np.sign(close - ema)


def squeeze_filter(close: pd.Series, lookback: int = 20, pct: float = 0.5,
                   ref: int = 100) -> pd.Series:
    """Boolean: band width (rolling std / price) in the lower `pct` quantile of its recent history
    — i.e. the breakout follows a volatility contraction (NR/squeeze), the classic setup."""
    width = close.rolling(lookback).std() / (close + EPS)
    thresh = width.rolling(ref).quantile(pct)
    return (width <= thresh).fillna(False)


def apply_filters(side: pd.Series, *masks: pd.Series, align: pd.Series | None = None) -> pd.Series:
    """Zero out entries that fail any boolean mask; if `align` (signed) is given, keep only
    entries whose direction matches its sign. Applied per bar, causally."""
    out = side.copy()
    for m in masks:
        out = out.where(m.reindex(out.index).fillna(False), 0.0)
    if align is not None:
        a = align.reindex(out.index).fillna(0.0)
        out = out.where(np.sign(out) == np.sign(a), 0.0)
    return out


# --- exits (held position from a persistent side) ---------------------------------
# Each returns a +1/-1/0 position per bar. Trend-riding exits use compact numpy loops
# (running state cannot be vectorised); all state at bar t is realised at t.

def hold_to_reversal(side: pd.Series) -> pd.Series:
    """Stop-and-reverse: hold the last non-zero breakout direction until the opposite fires.
    Always in the market (the pure Donchian trend system). Fully vectorised."""
    return side.replace(0.0, np.nan).ffill().fillna(0.0)


def hold_time_stop(side: pd.Series, horizon: int) -> pd.Series:
    """Enter on each onset event, hold exactly `horizon` bars (fixed vertical barrier), flat
    between. Overlapping same-direction re-entries extend the hold."""
    events = entry_events(side)
    idx = side.index
    pos = np.zeros(len(idx))
    loc = {t: i for i, t in enumerate(idx)}
    for t0 in events:
        i = loc[t0]
        s = side.iloc[i]
        pos[i:min(i + horizon, len(idx))] = s
    return pd.Series(pos, index=idx)


def hold_channel_exit(close: pd.Series, high: pd.Series, low: pd.Series, side: pd.Series,
                      exit_lookback: int = 20) -> pd.Series:
    """Turtle exit: enter on the breakout, ride until price closes through the opposite
    `exit_lookback`-bar channel (a shorter channel than entry), then flat until the next entry.

    This is the trend-riding exit — it holds through the fat tail instead of a fixed barrier.
    """
    idx = side.index
    c = close.to_numpy()
    exit_hi = high.rolling(exit_lookback).max().shift(1).to_numpy()  # for shorts
    exit_lo = low.rolling(exit_lookback).min().shift(1).to_numpy()   # for longs
    s = side.to_numpy()
    pos = np.zeros(len(idx))
    cur = 0.0
    for i in range(len(idx)):
        if cur == 0.0:
            if s[i] != 0.0:
                cur = s[i]
        elif cur > 0.0:
            if np.isfinite(exit_lo[i]) and c[i] < exit_lo[i]:
                cur = s[i] if s[i] < 0 else 0.0     # exit; flip if an opposite breakout coincides
        else:
            if np.isfinite(exit_hi[i]) and c[i] > exit_hi[i]:
                cur = s[i] if s[i] > 0 else 0.0
        pos[i] = cur
    return pd.Series(pos, index=idx)


def chandelier_trades(close: pd.Series, high: pd.Series, low: pd.Series, side: pd.Series,
                      k: float = 3.0, atr_n: int = 14) -> pd.DataFrame:
    """Segment the chandelier-exit held position into discrete trades: one row per entry with its
    entry time t0, exit time t1 and side. Used to meta-label and gate at the trade level (only enter
    breakouts the ML confidence model likes) while keeping the fat-tail-preserving chandelier exit."""
    pos = hold_atr_trailing(close, high, low, side, k, atr_n)
    idx = pos.index
    p = pos.to_numpy()
    rows, t0, s = [], None, 0.0
    for i in range(len(idx)):
        if p[i] != 0.0 and t0 is None:
            t0, s = idx[i], p[i]
        elif t0 is not None and (p[i] == 0.0 or p[i] != s):
            rows.append((t0, idx[i], s))          # trade closed (flat or flipped) at bar i
            t0, s = (idx[i], p[i]) if p[i] != 0.0 else (None, 0.0)
    if t0 is not None:
        rows.append((t0, idx[-1], s))
    return pd.DataFrame(rows, columns=["t0", "t1", "side"]).set_index("t0")


def hold_atr_trailing(close: pd.Series, high: pd.Series, low: pd.Series, side: pd.Series,
                      k: float = 3.0, atr_n: int = 14) -> pd.Series:
    """Chandelier exit: ride until close falls k*ATR below the highest close since entry (long)
    or rises k*ATR above the lowest close since entry (short). Trend-riding, fat-tail-friendly."""
    idx = side.index
    c = close.to_numpy()
    a = atr(high, low, close, atr_n).to_numpy()
    s = side.to_numpy()
    pos = np.zeros(len(idx))
    cur = 0.0
    anchor = np.nan  # highest close (long) / lowest close (short) since entry
    for i in range(len(idx)):
        if cur == 0.0:
            if s[i] != 0.0 and np.isfinite(a[i]):
                cur, anchor = s[i], c[i]
        elif cur > 0.0:
            anchor = max(anchor, c[i])
            if c[i] < anchor - k * a[i]:
                cur = 0.0
                if s[i] < 0 and np.isfinite(a[i]):
                    cur, anchor = -1.0, c[i]
        else:
            anchor = min(anchor, c[i])
            if c[i] > anchor + k * a[i]:
                cur = 0.0
                if s[i] > 0 and np.isfinite(a[i]):
                    cur, anchor = 1.0, c[i]
        pos[i] = cur
    return pd.Series(pos, index=idx)
