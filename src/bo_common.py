"""Shared harness for the breakout deep-dive: data loading, exit construction, evaluation.

Cost/timeframe/execution conventions are copied verbatim from scripts/run_book.py so every
breakout number here is directly comparable to the book (t+2 execution, liquidity-aware costs,
funding at every settlement, vol-target to 15% annualised, daily-resampled returns).
"""
from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pandas as pd

# Any accidental network call (a cache gap) must fail fast, never hang an unattended sweep.
socket.setdefaulttimeout(12)

from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.config import (BINANCE_SPOT_TAKER_BPS, BREAKOUT_DIR, CACHE_DIR,  # noqa: E402
                        CAPITAL_USD, CRYPTO_HALF_SPREAD_BPS, IMPACT_K, RAW_DIR, REPORTS_DIR,
                        ROOT_DIR, SEED, VOL_TARGET_ANNUAL)
from src.labels.triple_barrier import triple_barrier_labels, trailing_vol  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

ROOT = ROOT_DIR                          # repo root, re-exported for callers (bo.ROOT anchors data paths)
REPORTS = REPORTS_DIR                    # reports ROOT — non-breakout callers build their own subpath
BREAKOUT = BREAKOUT_DIR                  # the breakout family's sub-book: every bo_* artifact lands here
BREAKOUT.mkdir(parents=True, exist_ok=True)
# `reports/` is one folder per family and `run_master_book.py` reads breakout/bo_combined_portfolio
# .parquet. The bo_* writers used to point at the reports ROOT while the artifacts sat in the
# sub-folder, so re-running the breakout chain wrote files nobody read and the master book silently
# kept scoring a stale series. Breakout writers use BREAKOUT; everyone else keeps REPORTS.

SEED, CAP, TVOL = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL
CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)   # crypto perp taker
SC = dict(commission_bps=BINANCE_SPOT_TAKER_BPS, half_spread_bps=CRYPTO_HALF_SPREAD_BPS,
          impact_k=IMPACT_K, exec_lag=2)                                       # crypto spot taker (2x perp)
EC = dict(commission_bps=1.0, half_spread_bps=2.0, impact_k=0.1, exec_lag=2)   # US equity
FXC = dict(commission_bps=0.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)  # FX
CRYPTO_TF = {"5m": 288 * 365, "15m": 96 * 365, "1h": 24 * 365, "4h": 6 * 365, "1d": 365}
EQUITY_TF = {"5m": 78 * 252, "15m": 26 * 252, "1h": 7 * 252, "4h": 2 * 252, "1d": 252}
FX_TF = {"5m": 288 * 252, "15m": 96 * 252, "1h": 24 * 252, "4h": 6 * 252, "1d": 252}
HORIZON = {"5m": 48, "15m": 32, "1h": 24, "4h": 30, "1d": 10}   # triple-barrier vertical, per TF
START, END = "2020-01", "2026-07"

# 50 liquid USD-M perps (CoinGecko mcap-ranked, >=3y history) — the run_book crypto set
CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "TRXUSDT", "DOGEUSDT", "ZECUSDT",
          "ADAUSDT", "XMRUSDT", "LINKUSDT", "XLMUSDT", "BCHUSDT", "LTCUSDT", "HBARUSDT", "1000SHIBUSDT",
          "AVAXUSDT", "SUIUSDT", "UNIUSDT", "NEARUSDT", "DOTUSDT", "AAVEUSDT", "1000PEPEUSDT", "ICPUSDT",
          "ETCUSDT", "ALGOUSDT", "ATOMUSDT", "FILUSDT", "ARBUSDT", "APTUSDT", "INJUSDT",
          "DASHUSDT", "VETUSDT", "FETUSDT", "CRVUSDT", "1000LUNCUSDT", "STXUSDT", "LDOUSDT", "IMXUSDT",
          "XTZUSDT", "JASMYUSDT", "CFXUSDT", "OPUSDT", "1000FLOKIUSDT", "ENSUSDT", "COMPUSDT", "GRTUSDT",
          "IOTAUSDT", "RUNEUSDT", "MKRUSDT"]

_EQTD = RAW_DIR / "equity_td"
_eqfx = sorted(p.name[:-11] for p in _EQTD.glob("*_1d.parquet")) if _EQTD.exists() else []
STOCKS = [t for t in _eqfx if "=X" not in t]
FX = [t for t in _eqfx if "=X" in t]
TD_ITV = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
rng = np.random.default_rng(SEED)


# --- data -------------------------------------------------------------------------

_TD_CACHE = RAW_DIR / "twelvedata"
_UM = RAW_DIR / "futures/um"
_SPOT = RAW_DIR / "spot/klines"
_LO = pd.Timestamp(START + "-01", tz="UTC")
_HI = pd.Timestamp(END + "-01", tz="UTC") + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)


def _read_months(d: Path) -> pd.DataFrame:
    """Concat every cached monthly parquet under a dir, sorted & de-duplicated. Pure offline —
    reads only what is on disk (the bulk loader's network fetch of unpublished recent months
    stalls an unattended sweep, so we never call it here)."""
    files = sorted(d.glob("[0-9]*.parquet"))
    if not files:
        return pd.DataFrame()
    out = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out[(out.index >= _LO) & (out.index < _HI)]


def load_crypto(sym, tf):
    px = _read_months(_UM / "klines" / sym / tf)
    return px if len(px) >= 500 else None


def pit_universe(n: int = 10) -> pd.DataFrame:
    """Boolean (day × symbol): is this perp among the `n` most liquid by TRAILING dollar volume?

    The breakout book traded `CORE10` — BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC, typed once and
    used from 2020-01. That is the 2026 mega-cap list applied to 2020, and it was worth more to the
    leg's Sharpe than the future-fitted ML gate beside it (1.12 -> 0.69 on the universe alone).

    This is the same rule the cross-sectional legs already use — `xsect.top_n_liquid`: rank on a
    trailing 30-day median of dollar volume, lagged one bar, over EVERY perp on disk including the
    delisted ones. 137 distinct names pass through the top ten over the window, and on 2020-02-01 it
    holds BCH EOS ETC TRX XLM where the frozen list holds SOL BNB DOGE ADA AVAX.
    """
    A = pd.read_parquet(CACHE_DIR / "xs" / "crypto_1d_adv.parquet")
    trail = A.replace(0.0, np.nan).rolling(30, min_periods=10).median().shift(1)
    return trail.rank(axis=1, ascending=False) <= n


def pit_mask(sym: str, index, universe: pd.DataFrame) -> pd.Series:
    """`universe` membership for one symbol, carried onto that sleeve's own (possibly intraday) bars."""
    if sym not in universe.columns:
        return pd.Series(False, index=index)
    return universe[sym].reindex(index, method="ffill").fillna(False)


def safe_funding(sym):
    """USD-M funding series (offline), or an empty series if not cached."""
    df = _read_months(_UM / "fundingRate" / sym)
    return df["last_funding_rate"] if "last_funding_rate" in df else pd.Series(dtype=float)


def load_spot(sym, tf):
    """Binance SPOT klines for the same symbol — identical schema to the perp cache, so this is a
    drop-in price series. Spot reaches back to 2017-08 (perps list 2020-01) and pays no funding."""
    px = _read_months(_SPOT / sym / tf)
    return px if len(px) >= 500 else None


def backtest_split(sym, tf, perp, posv, fund=None):
    """Fill the LONG leg on spot and the SHORT leg on perps, and return the combined per-bar frame.

    The two venues price the same position very differently and the difference is not symmetric.
    Measured on the core-10 since 2020, a perp long pays 23.4%/yr in funding *conditional on this
    book being long* — a trend book is long exactly when funding is extreme — against a 10.3%
    unconditional average, while a perp short collects only 2.3% because funding has normalised by
    the time the book flips. Spot has no funding and charges 2x the taker fee, which at ~8 round
    turns a year is 40bps against ~1pp of avoided funding. So longs belong on spot and shorts on
    perps, and neither venue alone can express that.

    Only the FILL changes: the caller's signal, universe and sizing are untouched, so any difference
    against a single-venue run is attributable to execution. Falls back to the perp fill when the
    symbol has no spot history, which keeps the book's composition unchanged.
    """
    sp = load_spot(sym, tf)
    if sp is None:
        return backtest(perp["close"], posv, capital=CAP, funding=fund,
                        adv=perp["quote_volume"].rolling(20).median().shift(1), **CC)
    idx = perp.index.intersection(sp.index)
    sp, pp, pv = sp.loc[idx], perp.loc[idx], posv.reindex(idx).fillna(0.0)
    long_ = backtest(sp["close"], pv.clip(lower=0.0), capital=CAP, funding=None,
                     adv=sp["quote_volume"].rolling(20).median().shift(1), **SC)
    short = backtest(pp["close"], pv.clip(upper=0.0), capital=CAP, funding=fund,
                     adv=pp["quote_volume"].rolling(20).median().shift(1), **CC)
    return pd.DataFrame({"position": long_["position"] + short["position"],
                         "cost": long_["cost"] + short["cost"],
                         "funding": short["funding"],
                         "turnover": long_["turnover"] + short["turnover"],
                         "net_ret": long_["net_ret"] + short["net_ret"]})


def _td_symbol(sym):
    from src.data.equity import _to_td_symbol
    return _to_td_symbol(sym).replace("/", "-")


def load_eqfx(sym, tf):
    """Strictly offline: read the widest cached parquet for this symbol/interval; never hit the
    network (Twelve Data rate limits stall an unattended sweep). Returns None if not cached."""
    if tf == "1d":
        p = _EQTD / f"{sym}_1d.parquet"
        if not p.exists():
            return None
        px = pd.read_parquet(p)
        px = px[px.index >= pd.Timestamp("2012-01-01", tz="UTC")]
    else:
        itag = TD_ITV[tf]
        matches = sorted(_TD_CACHE.glob(f"{_td_symbol(sym)}_{itag}_*.parquet"),
                         key=lambda q: q.stat().st_size, reverse=True)
        if not matches:
            return None
        px = pd.read_parquet(matches[0])
    return px if len(px) >= 500 else None


# --- entries ----------------------------------------------------------------------

def entry_side(px, kind="donchian", lookback=55, **kw):
    c, h, l = px["close"], px["high"], px["low"]
    if kind == "donchian":
        return bl.donchian_side(c, h, l, lookback, kw.get("buffer_atr", 0.0))
    if kind == "bollinger":
        return bl.bollinger_side(c, lookback, kw.get("k", 2.0))
    if kind == "keltner":
        return bl.keltner_side(c, h, l, lookback, kw.get("k", 2.0))
    raise ValueError(kind)


# --- exits (held position from a persistent side) ---------------------------------

def held_position(exit_style, px, side, tf, **kw):
    """Dispatch a persistent side to a +1/-1/0 held position under the chosen exit."""
    c, h, l = px["close"], px["high"], px["low"]
    if exit_style == "triple_barrier":
        events = bl.entry_events(side)
        lab = triple_barrier_labels(c, events, trailing_vol(c, 100),
                                    kw.get("pt", 1.0), kw.get("sl", 1.0), HORIZON[tf])
        return positions_from_events(c.index, side, lab["t1"], events)
    if exit_style == "reversal":
        return bl.hold_to_reversal(side)
    if exit_style == "channel":
        return bl.hold_channel_exit(c, h, l, side, kw.get("exit_lookback", 20))
    if exit_style == "atr_trailing":
        return bl.hold_atr_trailing(c, h, l, side, kw.get("k", 3.0), kw.get("atr_n", 14))
    if exit_style == "time":
        return bl.hold_time_stop(side, kw.get("horizon", HORIZON[tf]))
    raise ValueError(exit_style)


# --- evaluation -------------------------------------------------------------------

def evaluate(close, pos, ppy_bar, costs, fund=None, adv=None, ppy_daily=365,
             with_mc=True, freq="D"):
    """Vol-target -> backtest -> daily-resample -> summarise (+ MC once Sharpe clears 0.5)."""
    posv = vol_target(pos, close, TVOL, ppy_bar)
    bt = backtest(close, posv, capital=CAP, funding=fund, adv=adv, **costs)
    daily = pd.DataFrame({
        "ret": (1 + bt["net_ret"]).resample(freq).prod() - 1,
        "cost": bt["cost"].resample(freq).sum(),
        "gross": bt["position"].abs().resample(freq).mean(),
        "turnover": bt["turnover"].resample(freq).sum(),
    }).dropna(subset=["ret"])
    s = summarise(daily["ret"], ppy_daily)
    if with_mc and s["sharpe_ann"] > 0.5:
        mc = bootstrap_sharpe(daily["ret"], ppy_daily, n_reps=500, seed=SEED)
        s["mc_p5"], s["mc_p50"] = mc.get("sharpe_p5", np.nan), mc.get("sharpe_p50", np.nan)
    else:
        s["mc_p5"] = s["mc_p50"] = np.nan
    s["turnover"] = float(bt["turnover"].sum())
    s["ann_turnover"] = float(daily["turnover"].mean() * ppy_daily) if len(daily) else np.nan
    return s, daily["ret"].rename("ret")


def cfg_for(kind, sym):
    """(ppy_bar map, costs, adv-getter kind) for an asset class."""
    if kind == "crypto":
        return CRYPTO_TF, CC
    if kind == "fx":
        return FX_TF, FXC
    return EQUITY_TF, EC
