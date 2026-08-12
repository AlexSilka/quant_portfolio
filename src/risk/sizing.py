"""Sizing a book: how big each leg is, and what changing that size costs.

Four things live here, and they are here because each of them was written out again and again across the
repo and drifted every time:

  * `vol_target_scale` — the multiplier. The `.shift(1)` is what keeps the size computable at the bar
    rather than from the day's own volatility, and a single copy missing it is a look-ahead nobody would
    notice in a Sharpe.
  * `equal_risk_weights` / `equal_risk_combine` — equal RISK over legs, which is not equal notional over
    legs. A blend that weights two legs of very different volatility equally by notional is one leg, and
    saying "risk parity" beside it does not make it two.
  * `held_weight_turnover` — what a layer of weights actually trades, target change PLUS the drift back
    onto it.
  * `resize_cost` — what re-sizing costs. Multiplying a finished net series by a lagged leverage is exact
    for the P&L and is NOT free, and nothing charged it until this function existed.

The last two exist because the same defect kept appearing at every layer: the legs charge their own
trading, the book above them charges its own, and whatever sat in between traded for nothing.

The ceiling and the lookback stay ARGUMENTS, not shared constants. A sleeve's cap is part of a
construction that was validated with it in place, and quietly re-pointing every sleeve at one number
would re-open every family's published series. The book-assembly layer passes `VOL_SCALE_CAP` because
that one was measured; everyone else passes what they were already using, visibly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_scale(net: pd.Series, target: float, ppy: float,
                     lookback: int = 60, cap: float = 3.0) -> pd.Series:
    """Multiplier that puts `net` on `target` annualised vol, from trailing info only.

    net      : a return series, already net of its own costs.
    target   : annualised vol to size to (e.g. 0.15).
    ppy      : that series' observations per year — 365 for crypto, ~252 for an exchange calendar.
    lookback : bars of trailing vol to size off.
    cap      : ceiling on the multiplier, so a quiet stretch cannot hand the leg unbounded leverage
               just before a shock.

    Lagged one bar and zero over the warm-up: before there is enough history to size a position there is
    no position.
    """
    return (target / (net.rolling(lookback).std() * np.sqrt(ppy))).clip(upper=cap).shift(1).fillna(0.0)


def equal_risk_weights(df: pd.DataFrame, lookback: int = 252, min_periods: int = 60,
                       cap_q: float = 0.99) -> pd.DataFrame:
    """Trailing inverse-vol weights over the columns that have STARTED — genuine equal risk.

    Equal *notional* over legs of different volatility is not equal risk, and calling it that is how a
    blend ends up 95% one leg while its docstring says 50/50. The weight is a TRAILING, lagged vol, so
    it is computable at the bar; a full-sample one would let the end of the series decide the start.

    A started leg keeps its weight on the days its own market is shut, instead of the book renormalising
    onto whoever is open — the same rule `run_master_book.hold_started` applies one layer up.

    The clip guards the one failure a trailing inverse-vol weight has: a leg whose trailing vol is near
    zero would otherwise take the whole book. It is an EXPANDING quantile, never a full-sample one.
    """
    started = df.notna().cummax()
    iv = (1.0 / df.rolling(lookback, min_periods=min_periods).std()).shift(1)
    cap = iv.max(axis=1).expanding(min_periods=min_periods).quantile(cap_q).ffill()
    iv = iv.clip(upper=cap, axis=0).where(started)
    return iv.div(iv.sum(axis=1).replace(0.0, np.nan), axis=0)


def equal_risk_combine(df: pd.DataFrame, lookback: int = 252, min_periods: int = 60,
                       cap_q: float = 0.99) -> tuple[pd.Series, pd.DataFrame]:
    """`equal_risk_weights` applied — returns (combined series, the weights it used).

    The weights come back with the series because they are what the blend TRADES: moving them costs
    money, and a caller that cannot see them cannot charge for them.
    """
    w = equal_risk_weights(df, lookback, min_periods, cap_q)
    started = df.notna().cummax()
    ret = df.where(df.notna(), 0.0).where(started)      # a shut market earns nothing; it does not hand
    combined = (ret * w).sum(axis=1, min_count=1).where(df.notna().any(axis=1))   # its weight to the open one
    return combined, w


def held_weight_turnover(w: pd.DataFrame, legs: pd.DataFrame) -> pd.Series:
    """Notional a layer of weights actually trades each bar — the target change PLUS the drift back.

    A weight held flat still has to be traded back onto: the leg that earned more than the book is a
    larger share of it by the close. `Σ|Δw|` sees only the bars the target moves, so any layer charged
    that way pays less than it trades. Same arithmetic as `xsect.held_turnover` and `backtest.engine`
    one level down — one implementation, so a blend inside a family and the book above it cannot end up
    charged by two different rules.
    """
    W = w.fillna(0.0)
    r = legs.reindex_like(W).fillna(0.0)
    b = (W * r).sum(axis=1)
    drifted = W.mul(1.0 + r).div((1.0 + b).replace(0.0, np.nan), axis=0)
    return (W - drifted.shift(1).fillna(0.0)).abs().sum(axis=1)


def resize_cost(scale: pd.Series, cost_bps: float, gross=1.0) -> pd.Series:
    """What the vol-target's own trading costs, per bar.

    Multiplying a finished net series by a lagged leverage L is exact for the P&L — L scales gross and
    the sleeve's own charged costs together — but it is NOT free, and nothing was charging it. Moving a
    position from L(t-1) to L(t) is a trade of |ΔL| times whatever notional is on, every bar, and the
    sleeve's cost model never sees it because the sleeve was already finished when the scaling happened.

    That is the same hole `run_master_book`'s BOOK_REBALANCE_BPS closed one layer up: the book pays for
    re-sizing its legs, while the legs re-sized themselves for nothing.

    scale    : the multiplier from `vol_target_scale`.
    cost_bps : what one unit of notional costs to move — the sleeve's own trading cost where the sleeve
               names instruments, the book rebalance rate where the layer moves whole sleeves.
    gross    : notional per unit of scale. A panel book passes its own Σ|w| (a dollar-neutral book runs
               ~2.0, so re-sizing it trades twice what a directional one does); a layer that moves a
               finished book passes 1.0, the default.
    """
    moved = scale.diff().abs().fillna(0.0)
    return moved * gross * (cost_bps / 1e4)
