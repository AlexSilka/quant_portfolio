"""Cross-sectional funding carry — the construction under which perp carry actually works.

A single-asset directional funding-timer (src/sleeves/carry.py) takes full price risk in a
~60%-vol asset to collect ~10%/yr funding, so funding is swamped by price noise and the sleeve
is ~0 Sharpe everywhere. The carry premium is harvested cross-sectionally and dollar-neutral:

  rank the panel by trailing funding, LONG the names the market pays you to hold (low/negative
  funding) and SHORT the names that are expensive to hold (high funding), dollar-neutral.

Then market beta nets out and the book collects the cross-sectional funding *spread* (top-quantile
minus bottom-quantile funding) as a steady drip, while the price legs cancel to first order. The
open empirical question — answered by the runner, not asserted here — is whether the residual price
leg helps (high funding = crowded longs that mean-revert down) or hurts (funding just tracks
momentum). So the runner also builds the price-only reversal/momentum books and a funding signal
residualised on trailing return, to isolate carry's *incremental* information beyond past price.

Funding P&L convention matches src/backtest/costs.funding_pnl: a holder of position w in a name
with funding f pays w*f each settlement, so book funding P&L = -sum_i(w_i * f_i). Longing the
lowest-funding names and shorting the highest makes every term positive = the funding spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

# Non-crypto-native perps on Binance USD-M that pollute a crypto funding-carry cross-section: their
# funding/price dynamics belong to another asset class, so they are excluded from the carry universe.
#   - stablecoins: ~1.0 price, floor funding -> land in the long leg as empty slots (a de-pegged one
#     like FRAX shows extreme funding and would be longed into a crash)
#   - tokenized gold (PAXG/XAUT): a commodity, tracks gold not crypto (PAXG is liquid enough to be
#     selected, so this one actually matters)
#   - synthetic index perps (BTC dominance, DeFi basket): not a single token at all
NON_CRYPTO_NATIVE = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "FRAXUSDT", "USTCUSDT", "USTUSDT", "BUSDUSDT", "USDPUSDT",
    "AEURUSDT", "EURUSDT", "EURIUSDT",                    # stablecoins / fiat
    "PAXGUSDT", "XAUTUSDT",                                # tokenized gold
    "BTCDOMUSDT", "DEFIUSDT", "BLUEBIRDUSDT",             # synthetic index perps
}


def funding_daily(funding_panel: pd.DataFrame) -> pd.DataFrame:
    """Sum the 8h (or finer) settlements into a per-name daily funding cost (what a 1x holder
    pays over the day). Aligns the funding grid to the daily price grid used for rebalancing."""
    return funding_panel.sort_index().resample("1D").sum()


# ---- cross-sectional ranking signals (higher value = more expensive to be long = short it) ----

def signal_level(fd: pd.DataFrame, lookback: int = 7) -> pd.DataFrame:
    """Trailing mean daily funding per name — the carry estimate. Lookback in days (EWM)."""
    return fd.ewm(span=lookback, min_periods=max(1, lookback // 2)).mean()


def signal_vol_adj(fd: pd.DataFrame, ret: pd.DataFrame, lookback: int = 7,
                   vol_lb: int = 30) -> pd.DataFrame:
    """Carry per unit of price risk: trailing funding / realised vol. Prefers high, *cheap-to-hold*
    carry — down-weights a rich funding on a name whose price is too volatile to be worth it."""
    vol = ret.rolling(vol_lb).std()
    return fd.ewm(span=lookback, min_periods=max(1, lookback // 2)).mean() / (vol + EPS)


def signal_macd(fd: pd.DataFrame, short: int = 3, long: int = 14) -> pd.DataFrame:
    """MACD-style funding: fast decay minus slow decay of funding (Presto Research construction).
    Ranks by whether funding is *accelerating* rich, not just its level."""
    return fd.ewm(span=short).mean() - fd.ewm(span=long).mean()


def signal_resid(fd: pd.DataFrame, ret: pd.DataFrame, lookback: int = 7,
                 mom_lb: int = 14) -> pd.DataFrame:
    """Funding residualised on trailing return, cross-sectionally, per date. Strips the component
    of funding that is explained by recent momentum, isolating carry's incremental information."""
    car = fd.ewm(span=lookback, min_periods=max(1, lookback // 2)).mean()
    mom = ret.rolling(mom_lb).sum()
    resid = pd.DataFrame(np.nan, index=car.index, columns=car.columns)
    for t in car.index:
        y, x = car.loc[t], mom.loc[t]
        m = y.notna() & x.notna()
        if m.sum() < 5:
            continue
        yv, xv = y[m].to_numpy(), x[m].to_numpy()
        b = np.polyfit(xv, yv, 1)
        resid.loc[t, m[m].index] = yv - (b[0] * xv + b[1])
    return resid


def pit_eligible(dollar_vol: pd.DataFrame, n: int, lookback: int = 30) -> pd.DataFrame:
    """Point-in-time universe mask: True where a name is in the top-`n` by trailing median dollar
    volume on that date (lagged, look-ahead-free). Names drop out when they stop trading (a delisted
    coin simply leaves the eligible set), so ranking on this mask is survivorship-honest."""
    liq = dollar_vol.rolling(lookback, min_periods=lookback // 2).median().shift(1)
    rank = liq.rank(axis=1, ascending=False)      # 1 = most liquid
    return rank.le(n)


def pit_eligible_band(dollar_vol: pd.DataFrame, lo: int, hi: int, lookback: int = 30) -> pd.DataFrame:
    """PIT universe mask for a LIQUIDITY BAND: names ranked (lo, hi] by trailing dollar volume. Lets a
    sleeve target a specific tier — e.g. carry lives in the mid-cap band, excluding the megacaps whose
    funding is compressed (no dispersion to harvest) and the illiquid tail (cost/noise)."""
    liq = dollar_vol.rolling(lookback, min_periods=lookback // 2).median().shift(1)
    rank = liq.rank(axis=1, ascending=False)
    return (rank > lo) & (rank <= hi)


# ---- dollar-neutral cross-sectional book ----

def xs_book(close: pd.DataFrame, fd: pd.DataFrame, signal: pd.DataFrame, *,
            direction: float = -1.0, top_frac: float = 0.3, exec_lag: int = 2,
            cost_bps: float = 6.0, rebalance: int = 1, weight: str = "equal",
            vol_lb: int = 30, buffer: float = 0.0) -> pd.DataFrame:
    """Dollar-neutral long/short book from a cross-sectional ranking signal.

    direction=-1 (carry): LONG the lowest-signal names, SHORT the highest (collect funding).
    direction=+1: LONG highest / SHORT lowest (for the funding-momentum / price-momentum controls).
    rebalance: hold weights for `rebalance` days between refreshes (turnover control).
    weight: within-leg weighting — "equal", "inv_vol" (risk-parity, down-weight volatile names),
            or "signal" (weight by |signal| distance from the cross-sectional median).
    buffer: no-trade band — a name's weight is only moved once its target drifts by more than
            `buffer` from what is held (cuts turnover on small rank shuffles; Robot Wealth method).

    Returns a per-day frame with net return and its price / funding / cost attribution.
    """
    rets = close.pct_change()
    ranks = signal.rank(axis=1, pct=True)
    hi = (ranks >= 1.0 - top_frac).astype(float)
    lo = (ranks <= top_frac).astype(float)
    if weight == "inv_vol":
        iv = 1.0 / (rets.rolling(vol_lb).std() + EPS)
        hi, lo = hi * iv, lo * iv                          # risk-parity within each leg
    elif weight == "signal":
        strength = (signal.sub(signal.median(axis=1), axis=0)).abs()
        hi, lo = hi * strength, lo * strength              # more capital to stronger carry
    wl = (lo if direction < 0 else hi)      # long leg
    ws = (hi if direction < 0 else lo)      # short leg
    wl = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0)
    ws = ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).fillna(0.0)
    if rebalance > 1:                        # refresh weights every `rebalance` days, else hold
        keep = pd.Series(np.arange(len(w)) % rebalance == 0, index=w.index)
        w = w.where(keep, np.nan).ffill().fillna(0.0)
    if buffer > 0.0:                         # no-trade band: hold a name until its target drifts > buffer
        arr = w.to_numpy(); held = np.zeros(arr.shape[1]); out = np.empty_like(arr)
        for t in range(arr.shape[0]):
            move = np.abs(arr[t] - held) > buffer
            held = np.where(move, arr[t], held)
            out[t] = held
        w = pd.DataFrame(out, index=w.index, columns=w.columns)
    wl_h = w.shift(exec_lag).fillna(0.0)     # bar-close -> execution delay

    price = (wl_h * rets).sum(axis=1)
    funding = -(wl_h * fd.reindex_like(wl_h).fillna(0.0)).sum(axis=1)   # collect the spread
    turn = wl_h.diff().abs().sum(axis=1)
    cost = turn * cost_bps / 1e4
    net = price + funding - cost
    return pd.DataFrame({"ret": net, "price": price, "funding": funding,
                         "cost": cost, "turnover": turn, "gross": wl_h.abs().sum(axis=1)}).dropna(subset=["ret"])


def beta_hedge(book_ret: pd.Series, btc_ret: pd.Series, lookback: int = 60) -> pd.Series:
    """Neutralise residual market beta: even a dollar-neutral book can carry net BTC-beta if its
    long and short baskets have different betas. Estimate a rolling (lagged) beta of the book to BTC
    and subtract that much BTC return. The research flagged this as a gap — practitioners usually
    rely on long/short symmetry alone; here it is measured, not assumed."""
    btc = btc_ret.reindex(book_ret.index).fillna(0.0)
    cov = book_ret.rolling(lookback).cov(btc)
    var = btc.rolling(lookback).var()
    beta = (cov / (var + EPS)).shift(1).fillna(0.0)
    return (book_ret - beta * btc).rename("ret")


def basis_carry(spot_close: pd.DataFrame, perp_close: pd.DataFrame, fd: pd.DataFrame, *,
                fund_gate: float = 0.0, exec_lag: int = 2, cost_bps: float = 6.0,
                spot_cost_bps: float = 10.0) -> pd.DataFrame:
    """Delta-neutral cash-and-carry: for each name, short perp + long spot when funding is richly
    positive (collect), flip to long perp + short spot when richly negative. The price legs cancel
    up to the spot-perp basis change; the harvest is funding minus two-leg costs.

    Equal-weight across the names that clear |funding| > fund_gate on the day. This is the textbook
    carry — high Sharpe, low vol, but capacity/borrow-limited (reported honestly, not hidden).
    """
    common = perp_close.columns.intersection(spot_close.columns)
    perp, spot, f = perp_close[common], spot_close[common], fd.reindex(columns=common)
    perp_r = perp.pct_change()
    spot_r = spot.reindex_like(perp).ffill().pct_change()
    side = pd.DataFrame(0.0, index=perp.index, columns=common)      # +1 short-perp/long-spot
    side = side.where(f.reindex_like(side).abs() <= fund_gate,
                      np.sign(f).reindex_like(side)).fillna(0.0)     # short perp when funding>0
    n = side.abs().sum(axis=1).replace(0, np.nan)
    w = side.div(n, axis=0).shift(exec_lag).fillna(0.0)             # equal weight, exec delay

    # short-perp leg earns -w*perp_r; long-spot leg earns +w*spot_r; funding collected = +w*f (short perp>0)
    leg_pnl = (-w * perp_r + w * spot_r.reindex_like(w)).sum(axis=1)
    funding = (w * f.reindex_like(w).fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1)
    cost = turn * (cost_bps + spot_cost_bps) / 1e4                  # both legs pay
    net = leg_pnl + funding - cost
    return pd.DataFrame({"ret": net, "basis": leg_pnl, "funding": funding,
                         "cost": cost, "turnover": turn, "gross": w.abs().sum(axis=1)}).dropna(subset=["ret"])


def basis_carry_hold(spot_close: pd.DataFrame, perp_close: pd.DataFrame, fd: pd.DataFrame, *,
                     enter: float = 2e-4, smooth: int = 7, rebalance: int = 7, exec_lag: int = 2,
                     cost_bps: float = 6.0, spot_cost_bps: float = 10.0) -> pd.DataFrame:
    """Basis carry that HOLDS through the contango regime instead of flipping on every funding
    sign-change (which the naive version does — 119x/yr turnover destroys the harvest). Uses
    smoothed funding + a dead-band with hysteresis (keep the position while funding hovers, only
    reverse when smoothed funding decisively crosses to the other side) + a weekly rebalance. The
    persistent state means turnover is a fraction of the naive version, so the funding harvest survives.
    """
    common = perp_close.columns.intersection(spot_close.columns)
    perp, spot, f = perp_close[common], spot_close[common], fd.reindex(columns=common)
    perp_r = perp.pct_change()
    spot_r = spot.reindex_like(perp).ffill().pct_change()
    sf = f.ewm(span=smooth).mean()                                 # smoothed daily funding per name
    state = pd.DataFrame(np.nan, index=sf.index, columns=common)   # +1 short-perp when funding rich +
    state = state.mask(sf > enter, 1.0).mask(sf < -enter, -1.0)    # decisive -> set; dead-band -> NaN
    state = state.ffill().fillna(0.0)                              # hysteresis: hold through dead-band
    if rebalance > 1:                                             # refresh weights weekly, else hold
        keep = pd.Series(np.arange(len(state)) % rebalance == 0, index=state.index)
        state = state.where(keep, np.nan).ffill().fillna(0.0)
    n = state.abs().sum(axis=1).replace(0, np.nan)
    w = state.div(n, axis=0).shift(exec_lag).fillna(0.0)
    leg_pnl = (-w * perp_r + w * spot_r.reindex_like(w)).sum(axis=1)
    funding = (w * f.reindex_like(w).fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1)
    cost = turn * (cost_bps + spot_cost_bps) / 1e4
    net = leg_pnl + funding - cost
    return pd.DataFrame({"ret": net, "basis": leg_pnl, "funding": funding,
                         "cost": cost, "turnover": turn, "gross": w.abs().sum(axis=1)}).dropna(subset=["ret"])
