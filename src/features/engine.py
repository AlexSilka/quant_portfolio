"""Feature library — 80-120 candidate features, all computable-at-bar.

Every feature at bar t uses only information available up to and including bar t
(backward-only rolling windows, no centering, no future shift). This is enforceable:
`scripts/smoke_features.py` recomputes on a truncated series and asserts past values
are unchanged — the leakage audit as code.

Families (Task A §4): trend/MA, momentum/ROC, mean-reversion, volatility (incl.
Parkinson & Garman-Klass), range/breakout, volume/flow, oscillators, statistical
structure (Hurst, autocorr, variance ratio, entropy), higher moments, cross-asset,
calendar/session. Normalisation is point-in-time (rolling z-score), never full-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _logret(c: pd.Series) -> pd.Series:
    return np.log(c).diff()


# --- families -------------------------------------------------------------------

def trend(df, lookbacks=(10, 20, 50, 100, 200)):
    c = df["close"]
    out = {}
    for n in lookbacks:
        sma = c.rolling(n).mean()
        ema = c.ewm(span=n, adjust=False).mean()
        out[f"px_sma_{n}"] = c / sma - 1.0
        out[f"px_ema_{n}"] = c / ema - 1.0
        out[f"sma_slope_{n}"] = sma.diff(n) / (sma.shift(n).abs() + EPS)
    out["sma_50_200"] = c.rolling(50).mean() / (c.rolling(200).mean() + EPS) - 1.0
    return pd.DataFrame(out, index=df.index)


def momentum(df, horizons=(1, 5, 10, 20, 60, 120)):
    c = df["close"]
    out = {}
    for h in horizons:
        out[f"ret_{h}"] = c.pct_change(h)
        out[f"logret_{h}"] = np.log(c).diff(h)
    return pd.DataFrame(out, index=df.index)


def mean_reversion(df, lookbacks=(20, 50, 100)):
    c = df["close"]
    out = {}
    for n in lookbacks:
        m = c.rolling(n).mean()
        s = c.rolling(n).std()
        out[f"zscore_{n}"] = (c - m) / (s + EPS)
        # Bollinger %B
        out[f"bb_pctb_{n}"] = (c - (m - 2 * s)) / (4 * s + EPS)
    # distance from rolling VWAP anchor
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    for n in (20, 50):
        vwap = (tp * df["volume"]).rolling(n).sum() / (df["volume"].rolling(n).sum() + EPS)
        out[f"vwap_dist_{n}"] = c / (vwap + EPS) - 1.0
    return pd.DataFrame(out, index=df.index)


def volatility(df, lookbacks=(20, 60)):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    r = _logret(c)
    hl = np.log(h / l)
    co = np.log(c / o)
    out = {}
    for n in lookbacks:
        out[f"realvol_{n}"] = r.rolling(n).std()
        out[f"parkinson_{n}"] = np.sqrt((hl ** 2).rolling(n).mean() / (4 * np.log(2)))
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        out[f"garmanklass_{n}"] = np.sqrt(gk.rolling(n).mean().clip(lower=0))
    # ATR (Wilder-style, backward rolling mean of true range)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / (c + EPS)
    # vol of vol & regime
    rv = r.rolling(20).std()
    out["vol_of_vol_20"] = rv.rolling(20).std()
    out["vol_regime"] = rv / (r.rolling(100).std() + EPS)
    return pd.DataFrame(out, index=df.index)


def range_breakout(df, lookbacks=(20, 55)):
    h, l, c = df["high"], df["low"], df["close"]
    out = {}
    for n in lookbacks:
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        out[f"donchian_pos_{n}"] = (c - ll) / (hh - ll + EPS)
        out[f"breakout_up_{n}"] = c / (hh.shift(1) + EPS) - 1.0
        out[f"breakout_dn_{n}"] = c / (ll.shift(1) + EPS) - 1.0
        out[f"range_width_{n}"] = (hh - ll) / (c + EPS)
    return pd.DataFrame(out, index=df.index)


def volume_flow(df, lookbacks=(20, 50)):
    c, v = df["close"], df["volume"]
    out = {}
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv_z_50"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + EPS)
    dv = c * v
    for n in lookbacks:
        out[f"dollar_vol_z_{n}"] = (dv - dv.rolling(n).mean()) / (dv.rolling(n).std() + EPS)
        out[f"vol_z_{n}"] = (v - v.rolling(n).mean()) / (v.rolling(n).std() + EPS)
    # taker imbalance proxy (Binance klines expose taker-buy volume)
    if "taker_buy_volume" in df:
        imb = 2.0 * df["taker_buy_volume"] / (v + EPS) - 1.0
        out["taker_imbalance"] = imb
        out["taker_imbalance_ma20"] = imb.rolling(20).mean()
    return pd.DataFrame(out, index=df.index)


def oscillators(df):
    h, l, c = df["high"], df["low"], df["close"]
    out = {}
    for n in (14, 28):
        delta = c.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        out[f"rsi_{n}"] = 100 - 100 / (1 + up / (dn + EPS))
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    out["stoch_k_14"] = 100 * (c - ll) / (hh - ll + EPS)
    out["williams_r_14"] = -100 * (hh - c) / (hh - ll + EPS)
    tp = (h + l + c) / 3.0
    mad = (tp - tp.rolling(20).mean()).abs().rolling(20).mean()
    out["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + EPS)
    return pd.DataFrame(out, index=df.index)


def _hurst(x: np.ndarray) -> float:
    # rescaled-range Hurst estimate over a few sub-window sizes
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return np.nan
    lags = [2, 4, 8, 16]
    lags = [k for k in lags if k < len(x)]
    tau = [np.std(x[k:] - x[:-k]) for k in lags]
    tau = np.asarray(tau)
    if np.any(tau <= 0):
        return np.nan
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def _entropy(x: np.ndarray, bins: int = 8) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan
    hist, _ = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def statistical(df, window=100, fast=False):
    c = df["close"]
    r = _logret(c)
    out = {}
    out["autocorr_1_100"] = r.rolling(window).corr(r.shift(1))  # vectorised lag-1 autocorr
    # variance ratio VR(5): Var(5-step) / (5 * Var(1-step))
    v1 = r.rolling(window).var()
    v5 = r.rolling(5).sum().rolling(window).var()
    out["variance_ratio_5"] = v5 / (5 * v1 + EPS)
    if not fast:  # apply-based; skipped on high-bar-count intraday for speed
        out["hurst_100"] = np.log(c).rolling(window).apply(_hurst, raw=True)
        out["entropy_ret_100"] = r.rolling(window).apply(_entropy, raw=True)
    return pd.DataFrame(out, index=df.index)


def higher_moments(df, window=60):
    r = _logret(df["close"])
    out = {
        f"skew_{window}": r.rolling(window).skew(),
        f"kurt_{window}": r.rolling(window).kurt(),
    }
    up = r.rolling(window).quantile(0.95)
    dn = r.rolling(window).quantile(0.05)
    out["tail_ratio_60"] = up / (dn.abs() + EPS)
    return pd.DataFrame(out, index=df.index)


def calendar(df):
    idx = df.index
    out = {
        "hour": idx.hour,
        "dayofweek": idx.dayofweek,
    }
    frame = pd.DataFrame(out, index=idx).astype(float)
    # minutes since first bar of the UTC day (session proxy for intraday)
    minutes = idx.hour * 60 + idx.minute
    day = idx.normalize()
    frame["min_since_day_open"] = (minutes - pd.Series(minutes, index=idx).groupby(day).transform("min")).values
    return frame


def cross_asset(df, benchmark: pd.Series, lookbacks=(20, 60)):
    """Relative strength, rolling beta and correlation stability vs a benchmark close."""
    c = df["close"]
    bench = benchmark.reindex(c.index).ffill()
    r = _logret(c)
    rb = _logret(bench)
    out = {}
    for n in lookbacks:
        out[f"rs_bench_{n}"] = c.pct_change(n) - bench.pct_change(n)
        cov = r.rolling(n).cov(rb)
        var = rb.rolling(n).var()
        out[f"beta_{n}"] = cov / (var + EPS)
        out[f"corr_{n}"] = r.rolling(n).corr(rb)
    out["corr_stability"] = out["corr_20"] - out["corr_60"]
    return pd.DataFrame(out, index=df.index)


# --- assembly -------------------------------------------------------------------

def compute_features(df: pd.DataFrame, benchmark: pd.Series | None = None,
                     fast: bool = False) -> pd.DataFrame:
    """Return the wide candidate-feature matrix for one instrument (computable-at-bar).

    fast=True skips the two apply-based statistical features (Hurst, entropy) so the engine
    stays vectorised for high-bar-count intraday (5m/15m over years = 100k+ bars).
    """
    parts = [
        trend(df), momentum(df), mean_reversion(df), volatility(df),
        range_breakout(df), volume_flow(df), oscillators(df),
        statistical(df, fast=fast), higher_moments(df), calendar(df),
    ]
    if benchmark is not None:
        parts.append(cross_asset(df, benchmark))
    feats = pd.concat(parts, axis=1)
    return feats.replace([np.inf, -np.inf], np.nan)


def pit_normalize(feats: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """Point-in-time z-score: each value scaled by its own trailing window only."""
    mean = feats.rolling(window, min_periods=window // 2).mean()
    std = feats.rolling(window, min_periods=window // 2).std()
    return (feats - mean) / (std + EPS)
