"""Cross-sectional (relative-value) momentum engine — the research superset.

Panel-level long-short: each bar, rank the names by a momentum signal, go long the top
quantile and short the bottom, dollar-neutral. Structurally market-neutral — it bets on
*relative* ranking, not each name's own trend — so it is a genuine diversifier against the
time-series trend book, not a re-labelled copy of it.

Rich enough to sweep the whole construction grid the literature cares about
(Jegadeesh-Titman skip, risk-adjusted / raw signal, decile↔tercile breadth, equal / rank /
inverse-vol leg weights, rebalance cadence) and to feed an ML ranking or meta-label layer.

Every signal is stamped at bar t using only data <= t (a shift audit lives in the driver).
Execution is delayed exec_lag bars so a signal never fills at its own bar's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import panel_impact_cost


# ── signals: each returns a wide (bars × names) frame, value at t from data <= t ──────────
def mom(px: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Trailing total return over (t-skip-lookback, t-skip] — the raw ranking signal.

    skip > 0 drops the most-recent `skip` bars (Jegadeesh-Titman gap) so short-term reversal
    does not contaminate the momentum measurement.
    """
    return px.shift(skip) / px.shift(skip + lookback) - 1.0


def risk_adj_mom(px: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Momentum divided by trailing return-volatility over the same window (Sharpe-like).

    Rewards smooth trends, penalises names whose gains came with high volatility — usually a
    higher-Sharpe signal than raw return.
    """
    r = px.pct_change()
    ret = px.shift(skip) / px.shift(skip + lookback) - 1.0
    vol = r.shift(skip).rolling(lookback).std()
    return ret / vol.replace(0.0, np.nan)


def resid_mom(px: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Residual (idiosyncratic) momentum — risk-adjusted momentum of the market-beta residual.

    Regress each name's returns on the equal-weight panel ("market") over the window, strip the
    beta·market component, and rank on the residual's mean÷vol. Removes the shared factor so the
    signal is a cleaner *relative* bet (Blitz-Huij-Martens) — usually higher-Sharpe, lower-beta
    than raw momentum on equities. All rolling, lagged by `skip`, so it is computable-at-bar.
    """
    r = px.pct_change()
    mkt = r.mean(axis=1)
    var_m = mkt.rolling(lookback).var()
    beta = r.rolling(lookback).cov(mkt).div(var_m, axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    m = resid.shift(skip).rolling(lookback).mean()
    v = resid.shift(skip).rolling(lookback).std()
    return m / v.replace(0.0, np.nan)


def idio_mom(px: pd.DataFrame, form_lb: int, beta_lb: int | None = None, skip: int = 0,
             market: pd.Series | None = None) -> pd.DataFrame:
    """Idiosyncratic (residual) momentum with a *separate* beta-estimation and formation window.

    The canonical Blitz-Huij-Martens construction estimates each name's market beta over a **long**
    trailing window (`beta_lb`, e.g. ~3y for equities), strips the beta·market component to get the
    residual, then ranks on the residual's mean÷std over a **shorter** formation window (`form_lb`,
    e.g. the classic t-12..t-2), skipping the most-recent `skip` bars. Standardising by residual vol
    (the ÷std) is the step that turns a raw residual return into an information-ratio signal.

    `resid_mom` is the special case beta_lb == form_lb (one window for both). Decoupling them is the
    literature's recipe — a stable long-window beta, momentum measured on the recent formation window —
    and typically lifts Sharpe / lowers factor beta further than the single-window form. All rolling,
    lagged by `skip`, so it is computable-at-bar. market defaults to the equal-weight panel mean
    return (single market factor); pass a series (e.g. BTC / SPY returns) for an external market.
    """
    beta_lb = form_lb if beta_lb is None else beta_lb
    r = px.pct_change()
    mkt = r.mean(axis=1) if market is None else market.reindex(r.index)
    var_m = mkt.rolling(beta_lb).var()
    beta = r.rolling(beta_lb).cov(mkt).div(var_m.replace(0.0, np.nan), axis=0)
    resid = r.sub(beta.mul(mkt, axis=0))
    m = resid.shift(skip).rolling(form_lb).mean()
    v = resid.shift(skip).rolling(form_lb).std()
    return m / v.replace(0.0, np.nan)


def blend_rank(signals: list[pd.DataFrame]) -> pd.DataFrame:
    """Average percentile-rank across several signals (e.g. multi-horizon momentum).

    Ranking first, then averaging, makes horizons commensurable regardless of scale.
    """
    ranks = [s.rank(axis=1, pct=True) for s in signals]
    return sum(ranks) / len(ranks)


# ── portfolio: signal → dollar-neutral weights → net return ───────────────────────────────
def xs_weights(signal: pd.DataFrame, top_frac: float = 0.3, weighting: str = "equal",
               vol: pd.DataFrame | None = None, min_names: int = 6) -> pd.DataFrame:
    """Dollar-neutral long-top / short-bottom weights (long side sums to +1, short to −1).

    weighting: 'equal' (each selected name equal), 'rank' (score-proportional — weight by
    distance from the cross-sectional median, using the whole tail), or 'volinv' (inverse
    trailing vol within each leg — risk-parity legs). Bars with < min_names valid names are
    flat (nothing to rank across).
    """
    ranks = signal.rank(axis=1, pct=True)
    n_valid = signal.notna().sum(axis=1)
    longs = ranks >= (1.0 - top_frac)
    shorts = ranks <= top_frac
    if weighting == "equal":
        wl, ws = longs.astype(float), shorts.astype(float)
    elif weighting == "rank":
        wl = (ranks - 0.5).clip(lower=0.0) * longs
        ws = (0.5 - ranks).clip(lower=0.0) * shorts
    elif weighting == "volinv":
        iv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        wl, ws = longs.astype(float) * iv, shorts.astype(float) * iv
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    wl = wl.div(wl.sum(axis=1).replace(0.0, np.nan), axis=0)
    ws = ws.div(ws.sum(axis=1).replace(0.0, np.nan), axis=0)
    w = (wl - ws).where(n_valid >= min_names, 0.0).fillna(0.0)
    return w


def _no_trade_buffer(w: pd.DataFrame, buffer: float) -> pd.DataFrame:
    """Hold each name's weight until its target moves more than `buffer` (fraction of book
    notional) away from the held level — a per-name no-trade band.

    On a thin, noisy cross-section (crypto pre-2020: 4-18 names) rebalancing to every wiggle
    both racks up turnover and lets a single volatile name dominate a held book between rebalances;
    the band only trades when the target has moved materially, which both cuts cost and keeps the
    book from chasing noise. Sequential by construction (each bar depends on the held state).
    """
    tgt = np.nan_to_num(w.to_numpy())
    held = np.zeros(tgt.shape[1])
    out = np.empty_like(tgt)
    for i in range(tgt.shape[0]):
        moved = np.abs(tgt[i] - held) > buffer
        held = np.where(moved, tgt[i], held)
        out[i] = held
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def xs_backtest(px: pd.DataFrame, signal: pd.DataFrame, *, top_frac: float = 0.3,
                weighting: str = "equal", rebal: int = 1, exec_lag: int = 2, buffer: float = 0.0,
                cost_bps: float = 6.0, commission_bps: float | None = None,
                half_spread_bps: float | None = None, vol_lb: int = 20, min_names: int = 6,
                adv: pd.DataFrame | None = None, capital: float = 500_000.0,
                impact_k: float = 0.0, borrow_bps_annual: float = 0.0, ppy: int = 252) -> dict:
    """Backtest a signal on a price panel; return net/gross return, turnover and the cost breakdown.

    Weights are formed each bar, optionally held for `rebal` bars (cadence), passed through a
    no-trade `buffer` band, then delayed `exec_lag` bars before earning returns — so a bar-t signal
    never fills at close(t). The per-trade cost is split into three visible pieces, never one opaque
    number (`src/config.py` holds the constants, verified against the Binance schedule):
      - commission : the exchange taker fee (spot 10bps vs USD-M futures 5bps — venue matters),
      - half-spread: the bid/ask floor,      both charged on turnover (bps of traded notional);
      - √-impact   : Almgren  k·σ·√(order/ADV), added per name when an ADV panel is supplied, so the
                     illiquid mid-cap tail pays its true wider cost instead of a flat spread.
    Pass `commission_bps`+`half_spread_bps` for the split (the honest form); a lone `cost_bps` is the
    legacy single-number path still used by the other sleeves' drivers.
    """
    if commission_bps is not None or half_spread_bps is not None:
        commission_bps = commission_bps or 0.0
        half_spread_bps = half_spread_bps or 0.0
    else:                                             # legacy single-number path
        commission_bps, half_spread_bps = cost_bps, 0.0

    rets = px.pct_change()
    vol = rets.rolling(vol_lb).std() if weighting == "volinv" else None
    w = xs_weights(signal, top_frac, weighting, vol, min_names)
    if rebal > 1:
        keep = np.zeros(len(w), dtype=bool)
        keep[::rebal] = True
        w = w.where(pd.Series(keep, index=w.index), axis=0).ffill().fillna(0.0)
    if buffer > 0.0:
        w = _no_trade_buffer(w, buffer)
    w = w.shift(exec_lag).fillna(0.0)

    gross_ret = (w * rets).sum(axis=1)
    dw = w.diff().abs()
    turn = dw.sum(axis=1)
    commission = turn * commission_bps / 1e4
    spread = turn * half_spread_bps / 1e4
    if adv is not None and impact_k > 0.0:
        impact = panel_impact_cost(dw, rets.rolling(vol_lb).std(), adv, capital, impact_k)
    else:
        impact = pd.Series(0.0, index=w.index)
    # borrow on the SHORT leg (equities): per-bar cost on short gross notional. Crypto perps pay funding,
    # not borrow, so those callers leave borrow_bps_annual=0 and charge funding separately.
    if borrow_bps_annual:
        borrow = w.clip(upper=0.0).abs().sum(axis=1) * (borrow_bps_annual / 1e4) / ppy
    else:
        borrow = pd.Series(0.0, index=w.index)
    cost = commission + spread + impact + borrow
    net = gross_ret - cost
    return {"net": net, "gross": gross_ret, "turnover": turn, "commission": commission,
            "spread": spread, "impact": impact, "borrow": borrow, "cost": cost, "weights": w}


def liquidity_mask(signal: pd.DataFrame, adv: pd.DataFrame | None, daily_floor: float,
                   bpd_: int = 1, lookback_days: int = 30) -> pd.DataFrame:
    """Mask a signal to names above a trailing-median *daily* dollar-volume floor that bar.

    On a broad universe (hundreds of names, many illiquid micro-caps) the raw decile tails fill
    with unfillable names; ranking only the tradable subset each bar keeps the book realistic.
    `adv` is per-bar notional, so it is scaled to a daily figure (×bars_per_day) before the
    comparison — the floor is quoted in $/day and is therefore consistent across timeframes.
    min_periods guards the gappy-panel trap where a full-window median is NaN everywhere.
    """
    if adv is None or daily_floor <= 0:
        return signal
    win = max(5, lookback_days * bpd_)
    daily = adv.reindex_like(signal).replace(0.0, np.nan) * bpd_
    trail = daily.rolling(win, min_periods=max(5, win // 3)).median().shift(1)
    return signal.where(trail >= daily_floor)


def top_n_liquid(signal: pd.DataFrame, adv: pd.DataFrame | None, n: int,
                 bpd_: int = 1, lookback_days: int = 30) -> pd.DataFrame:
    """Restrict ranking to the N most-liquid names *at each bar* (by trailing daily $-volume).

    This is the honest, tradable "top-N universe": survivorship-free (the membership rotates as
    liquidity changes, using only past data) and focused (small-cap microstructure noise is simply
    excluded, not survived-into). Contrast a "top-N by today's market cap" list, which is chosen
    with hindsight and inflates a momentum long-short.
    """
    if adv is None or not n:
        return signal
    win = max(5, lookback_days * bpd_)
    daily = adv.reindex_like(signal).replace(0.0, np.nan) * bpd_
    trail = daily.rolling(win, min_periods=max(5, win // 3)).median().shift(1)
    rank = trail.rank(axis=1, ascending=False)      # 1 = most liquid that bar
    return signal.where(rank <= n)


def vol_target(net: pd.Series, ppy: float, target: float = 0.15,
               lb: int = 60, cap: float = 3.0) -> pd.Series:
    """Scale a net-return series to constant annualised vol (leverage lagged one bar).

    Scaling the *net* series is exact here: a lagged leverage L_t multiplies both gross return
    and turnover-cost, so net_scaled = L_t · net_raw. Risk-parity across sleeves/timeframes.
    """
    scale = (target / (net.rolling(lb).std() * np.sqrt(ppy))).clip(upper=cap).shift(1).fillna(0.0)
    return net * scale
