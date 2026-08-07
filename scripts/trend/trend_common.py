"""Shared harness for the trend-following deep-dive.

Reuses the breakout deep-dive's tested harness verbatim (`bo_common`): the same offline data
loaders, cost/timeframe/execution conventions, vol-targeting and daily-resampled `evaluate` — so
every trend number here is directly comparable to the book and to the breakout numbers. This
module adds only the trend-specific piece: turning an (entry, regime-gate, direction, exit) spec
into a held position via `src.sleeves.trend_lab`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402  (data loaders, evaluate, cost maps, universe lists)
from src.config import OOS_START  # noqa: E402,F401  — re-exported as T.OOS_START for the trend/ scripts
from src.sleeves import trend_lab as tl  # noqa: E402

REPORTS = bo.ROOT / "reports" / "trend"
REPORTS.mkdir(parents=True, exist_ok=True)
CACHE = bo.ROOT / "data" / "cache" / "trend"
CACHE.mkdir(parents=True, exist_ok=True)

# re-export the harness constants so trend scripts read one namespace
SEED, CAP, TVOL = bo.SEED, bo.CAP, bo.TVOL
CC, EC, FXC = bo.CC, bo.EC, bo.FXC
CRYPTO_TF, EQUITY_TF, FX_TF = bo.CRYPTO_TF, bo.EQUITY_TF, bo.FX_TF
CRYPTO, STOCKS, FX = bo.CRYPTO, bo.STOCKS, bo.FX

# the frozen equity universe (large caps + liquid ETFs), not
# the 734-name equity_td glob (too broad/noisy for a single-name trend edge map).
EQ_CORE = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM"]
_EQ_STORE = bo.ROOT / "data" / "raw" / "equity"        # deep-history OHLCV (SPY 1993+, IWM 2000+)
_SPLICE = pd.Timestamp("2020-01-01", tz="UTC")         # spot before, perp from here (funding era)


def load_equity(sym: str, start: str = "2012-01-01") -> pd.DataFrame | None:
    """Split-adjusted daily OHLCV from the deep-history store (all 10 core names incl. IWM)."""
    p = _EQ_STORE / f"{sym}_1d.parquet"
    if not p.exists():
        return None
    px = pd.read_parquet(p)
    px = px[px.index >= pd.Timestamp(start, tz="UTC")]
    return px if len(px) >= 250 else None


def load_crypto_long(sym: str, tf: str) -> pd.DataFrame | None:
    """Crypto price history spliced spot(pre-2020) + perp(2020+) for 1d/4h — extends the trend
    backtest to 2017 (the pre-perp bull/bear, incl. Q4-2018). Intraday (<=1h) is perp-only.

    Funding applies only to the perp era; on the spot leg there is no perp, so trend there is a
    spot long/short with zero funding (the honest pre-2020 approximation). Columns match the perp
    contract so the engine and cost model are unchanged.
    """
    perp = bo.load_crypto(sym, tf)
    if tf in ("5m", "15m", "1h") or perp is None:
        return perp
    # read spot months directly (bo._read_months clips to the 2020 floor — we want 2017+)
    sdir = bo.ROOT / "data" / "raw" / "spot" / "klines" / sym / tf
    files = sorted(sdir.glob("[0-9]*.parquet")) if sdir.exists() else []
    if not files:
        return perp
    spot = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    spot = spot[~spot.index.duplicated(keep="first")]
    pre = spot[spot.index < _SPLICE]
    if len(pre) == 0:
        return perp
    cols = [c for c in perp.columns if c in pre.columns]
    out = pd.concat([pre[cols], perp[perp.index >= _SPLICE][cols]]).sort_index()
    return out[~out.index.duplicated(keep="first")]


# --- entries ----------------------------------------------------------------------

def entry_signal(px: pd.DataFrame, kind: str, **kw):
    """Return an entry as either a signed side (+1/-1/0) or a continuous strength in [-1,1]."""
    c, h, l = px["close"], px["high"], px["low"]
    if kind == "ema":
        return tl.ema_cross_side(c, kw.get("fast", 50), kw.get("slow", 200))
    if kind == "sma":
        return tl.sma_cross_side(c, kw.get("fast", 50), kw.get("slow", 200))
    if kind == "tsmom":
        return tl.tsmom_side(c, kw.get("lookback", 90))
    if kind == "macd":
        return tl.macd_side(c, kw.get("fast", 12), kw.get("slow", 26), kw.get("signal", 9))
    if kind == "donchian":
        return tl.donchian_side(c, h, l, kw.get("lookback", 55))
    if kind == "blend":                                   # continuous
        return tl.multi_lookback_blend(c, kw.get("pairs", ((8, 24), (16, 48), (32, 96), (64, 192))))
    if kind == "strength":                                # continuous
        return tl.trend_strength(c, kw.get("lookback", 90))
    raise ValueError(kind)


CONTINUOUS = {"blend", "strength"}


# --- position from a full spec ----------------------------------------------------

def trend_position(px: pd.DataFrame, spec: dict, tf: str) -> pd.Series:
    """Build a held position (+1/-1/0 or continuous) from an entry/gate/direction/exit spec.

    spec keys: entry (str), params (dict), gate (None|'adx'|'ltf'|'vol'|'adx+ltf'),
               exit ('reversal'|'atr_trailing'|'channel'|'time'), direction ('ls'|'long_only'|
               'short_only'|'asym'). Continuous entries ('blend'/'strength') ignore the exit
               (the forecast itself is the held position; conviction re-sizes each bar).
    """
    c, h, l = px["close"], px["high"], px["low"]
    entry = spec["entry"]
    params = spec.get("params", {})
    sig = entry_signal(px, entry, **params)

    # regime gate (applied to the signed side; for continuous, gate zeroes the forecast)
    gate = spec.get("gate")
    if gate:
        masks, align = [], None
        for g in gate.split("+"):
            if g == "adx":
                masks.append(tl.adx_gate(h, l, c, threshold=params.get("adx_thr", 25.0)))
            elif g == "ltf":
                align = tl.long_term_gate(c, params.get("ltf_span", 200))
            elif g == "vol":
                masks.append(tl.vol_gate(c, hi_q=params.get("vol_hi_q", 0.9)))
            elif g == "eff":
                masks.append(tl.efficiency_gate(c, params.get("eff_n", 20), params.get("eff_thr", 0.3)))
        if entry in CONTINUOUS:
            m = pd.Series(True, index=c.index)
            for msk in masks:
                m &= msk.reindex(c.index).fillna(False)
            sig = sig.where(m, 0.0)
            if align is not None:
                sig = sig.where(np.sign(sig) == align.reindex(c.index).fillna(0.0), 0.0)
        else:
            sig = tl.apply_gate(sig, *masks, align=align)

    # exit → held position (discrete only; continuous forecast is already a held position)
    if entry in CONTINUOUS:
        pos = sig
    else:
        ex = spec.get("exit", "reversal")
        if ex == "reversal":
            pos = tl.hold_to_reversal(sig)
        elif ex == "atr_trailing":
            pos = tl.hold_atr_trailing(c, h, l, sig, params.get("atr_k", 3.0), params.get("atr_n", 14))
        elif ex == "channel":
            pos = tl.hold_channel_exit(c, h, l, sig, params.get("exit_lb", 20))
        elif ex == "time":
            pos = tl.hold_time_stop(sig, params.get("horizon", bo.HORIZON[tf]))
        else:
            raise ValueError(ex)

    return tl.apply_direction(pos, spec.get("direction", "ls"),
                              short_weight=spec.get("short_weight", 0.3))


# --- evaluation (thin wrapper over bo.evaluate) -----------------------------------

def eval_spec(px: pd.DataFrame, spec: dict, tf: str, ppy_bar: float, costs: dict,
              fund=None, adv=None, with_mc: bool = False, ppy_daily: int = 365):
    """Vol-target → backtest → daily-resample → summarise. Returns (stats_dict, daily_ret).

    ppy_daily annualises the daily-resampled Sharpe: 365 for 24/7 crypto, 252 for equities (whose
    calendar-'D' resample drops weekends), so an equity sleeve's Sharpe is not inflated by √(365/252).
    """
    pos = trend_position(px, spec, tf)
    return bo.evaluate(px["close"], pos, ppy_bar, costs, fund=fund, adv=adv,
                       with_mc=with_mc, freq="D", ppy_daily=ppy_daily)


def crypto_adv(px: pd.DataFrame) -> pd.Series:
    """20-bar median dollar-volume, lagged one bar (the liquidity input to √-impact costs)."""
    return px["quote_volume"].rolling(20).median().shift(1)


def spec_label(spec: dict) -> str:
    p = spec.get("params", {})
    pstr = ",".join(f"{k}{v}" for k, v in p.items() if k in ("fast", "slow", "lookback", "atr_k"))
    bits = [spec["entry"]]
    if pstr:
        bits.append(pstr)
    if spec.get("gate"):
        bits.append(f"g:{spec['gate']}")
    if spec.get("exit") and spec["entry"] not in CONTINUOUS:
        bits.append(spec["exit"])
    bits.append(spec.get("direction", "ls"))
    return "|".join(bits)
