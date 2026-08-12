"""Betting-against-beta / low-volatility sleeve — the leverage-constraint premium.

Frazzini-Pedersen (2014): leverage-constrained and lottery-seeking investors overbid high-beta
assets, so low-beta earns positive risk-adjusted alpha. The trade is long the low-beta names,
short the high-beta names. Two constructions, and the gap between them is the whole point of BAB:

    dollar-neutral : long-leg $ = short-leg $ = 1. Simple, but because low-beta<high-beta the book
                     keeps a *residual negative market beta* (long low-β, short high-β) — in a rising
                     market that short-beta tilt is a headwind that is NOT the leverage premium.
    beta-neutral   : de-lever the short (high-β) leg so the two legs' betas cancel (net-zero beta).
                     This is the honest Frazzini-Pedersen construction — the leverage premium with
                     the market tilt removed. The dollar↔beta gap measures how much of the raw
                     dollar-neutral number is just the market-short tilt vs the actual BAB alpha.

The ranking signal is a trailing panel beta βᵢ (rolling regression of a name's return on the
equal-weight panel "market"), computable at bar t — same vectorised machinery as
`xsect.resid_mom`. A trailing-vol proxy (−vol) and a trailing-skew control (−skew, the H2 lottery
signal) are provided so the driver can orthogonalise and show the premium is *beta*, not lottery.

The dollar-neutral book is a literal signal-swap into `xsect.xs_backtest` (signal = −beta / −vol);
this module adds only what that engine cannot express — the beta-neutral leg-scaling — plus a
prebuilt-weights backtest that shares `xs_backtest`'s exact cost model (commission + half-spread on
turnover + √-impact, never flat) so the two constructions are directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.carry import resolve as resolve_carry
from src.backtest.costs import panel_impact_cost
from src.sleeves.xsect import held_turnover


# ── data integrity: winsorise the panel before anything reads it ──────────────────────────
def winsorize_panel(close: pd.DataFrame, winsor: float = 0.5) -> pd.DataFrame:
    """Return a close panel whose bar-over-bar returns have ±inf dropped and |return| > winsor
    treated as flat (0) — so split/delisting artifacts cannot be earned or dominate a signal.

    Essential, not cosmetic (the same trap the overnight sleeve documents): the survivorship-free
    broad panels carry split/delisting prints — a prior close that rounds to zero gives an ∞
    return, and a mis-adjusted day gives a spurious ±hundreds-of-percent. A single such name-day,
    if it lands in a leg, dominates the vol-target and manufactures a fake cross-sectional edge; a
    handful of them make otherwise-orthogonal books look ~identical (the beta and skew books' return
    correlation collapses from a spurious 0.97 to ~0 once these are removed). The absolute price
    level is arbitrary here — the book is dollar-neutral and ADV is a separate panel — so only the
    cleaned returns matter; pre-listing / delisted NaNs are preserved. The driver reports the
    raw-vs-clean delta so the artifact is visible, not hidden.
    """
    mask = close.notna()
    r = close.pct_change().replace([np.inf, -np.inf], np.nan)
    r = r.where(r.abs() <= winsor, 0.0).fillna(0.0)      # extremes (and gaps) → flat: discard the print
    return (1.0 + r).cumprod().where(mask)


# ── signals: each a wide (bars × names) frame, value at t from data <= t ──────────────────
def panel_beta(px: pd.DataFrame, lookback: int, market: pd.Series | None = None) -> pd.DataFrame:
    """Trailing beta of each name's return on the market over `lookback` bars — the BAB signal.

    market defaults to the equal-weight panel mean return (the repo's market proxy, as in
    `xsect.resid_mom`); pass a series (e.g. BTC returns) for a single-asset market. β = rolling
    cov(rᵢ, mkt) / var(mkt), fully vectorised and backward-only, so it is computable at bar t.
    Rank low→high and long the low tail / short the high tail (signal = −beta) to bet against beta.
    """
    r = px.pct_change()
    mkt = r.mean(axis=1) if market is None else market.reindex(r.index)
    var_m = mkt.rolling(lookback).var()
    return r.rolling(lookback).cov(mkt).div(var_m.replace(0.0, np.nan), axis=0)


def trailing_vol(px: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing return volatility — the simpler low-risk proxy (signal = −vol → long low-vol).

    The low-volatility anomaly is the same leverage-constraint premium seen through total (not
    market) risk; correlating the −vol book with the −beta book shows whether they are one effect.
    """
    return px.pct_change().rolling(lookback).std()


def trailing_skew(px: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing return skew — the lottery (H2 / MAX) control, NOT the BAB signal itself.

    High-skew names are the "lottery" tickets retail overpays for; skew and vol/beta are correlated,
    so the driver regresses the BAB book on a −skew book to prove the premium survives controlling
    for lottery demand — i.e. it is *beta*, not re-labelled skewness. Provided here so that control
    is built from the same panel with the same convention.
    """
    return px.pct_change().rolling(lookback).skew()


# ── beta-neutral weights (what xs_backtest cannot express) ────────────────────────────────
def bab_weights(beta: pd.DataFrame, *, top_frac: float = 0.2, neutral: str = "beta",
                rebal: int = 1, min_names: int = 6) -> pd.DataFrame:
    """Long low-β / short high-β weights, dollar-neutral or beta-neutral (Frazzini-Pedersen).

    Ranks names by trailing beta each bar; the bottom `top_frac` (low beta) is the long leg, the
    top `top_frac` (high beta) the short leg, each equal-weighted within the leg.

      neutral="dollar" : legs sum to +1 / −1 (a residual short-beta book).
      neutral="beta"   : the short (high-β) leg is scaled by β̄_low / β̄_high so the two legs' betas
                         cancel (net book beta ≈ 0). β̄ are the lagged leg-mean betas, so the weights
                         stay computable-at-bar; the scale is clipped to [0, 5] for the rare bar
                         whose high-β leg mean is small. Vol-targeting downstream sets the final size.

    Held `rebal` bars between reforms (monthly cadence keeps a slow signal's turnover — and cost —
    low). Bars with < `min_names` valid betas are flat. Returns wide weights to feed `bab_backtest`.
    """
    ranks = beta.rank(axis=1, pct=True)
    n_valid = beta.notna().sum(axis=1)
    longs = (ranks <= top_frac).astype(float)          # low beta → long
    shorts = (ranks >= 1.0 - top_frac).astype(float)   # high beta → short
    wl = longs.div(longs.sum(axis=1).replace(0.0, np.nan), axis=0)
    ws = shorts.div(shorts.sum(axis=1).replace(0.0, np.nan), axis=0)
    if neutral == "dollar":
        w = wl - ws
    elif neutral == "beta":
        b_lo = (beta * wl).sum(axis=1)                 # mean beta of the long (low-β) leg
        b_hi = (beta * ws).sum(axis=1)                 # mean beta of the short (high-β) leg
        scale = (b_lo / b_hi.replace(0.0, np.nan)).clip(lower=0.0, upper=5.0)
        w = wl - ws.mul(scale, axis=0)                 # net beta = b_lo·1 − b_hi·(b_lo/b_hi) = 0
    else:
        raise ValueError(f"unknown neutral {neutral!r}")
    w = w.where(n_valid >= min_names, 0.0).fillna(0.0)
    if rebal > 1:
        keep = np.zeros(len(w), dtype=bool)
        keep[::rebal] = True
        w = w.where(pd.Series(keep, index=w.index), axis=0).ffill().fillna(0.0)
    return w


def net_book_beta(weights: pd.DataFrame, beta: pd.DataFrame) -> pd.Series:
    """Realised net market beta of a weight book — Σ wᵢ·βᵢ per bar (the tilt beta-neutral removes)."""
    return (weights * beta.reindex_like(weights)).sum(axis=1)


# ── prebuilt-weights backtest (shares xs_backtest's cost model, verbatim) ──────────────────
def bab_backtest(px: pd.DataFrame, weights: pd.DataFrame, *, exec_lag: int = 2,
                 cost_bps: float = 6.0, vol_lb: int = 20, adv: pd.DataFrame | None = None,
                 impact_k: float = 0.0, capital: float = 500_000.0, ppy: float = 365,
                 borrow_bps_annual: float | None = None, carry=None) -> dict:
    """Net/gross/turnover/cost of a *prebuilt* weight book — for the beta-neutral construction.

    Identical execution and cost treatment to `xsect.xs_backtest` (delay `exec_lag` bars so a
    bar-t weight never fills at close(t); commission+half-spread on traded notional plus an optional
    √-impact term scaled to bar $-volume) — the only difference is that weights are supplied rather
    than built from a signal, which is what the leg-scaled beta-neutral book requires. Same return
    shape as xs_backtest so the caller vol-targets the net series identically.

    That "identical treatment" used to stop at the trading costs: this book runs on the same USD-M
    perp panel as the crypto x-sect legs and charged no funding at all, which is worth −1.4% a year
    on the shipped top-25 construction. Carry is now resolved from the panel like everywhere else
    (`src/backtest/carry`) rather than being something each caller has to remember.
    """
    rets = px.pct_change()
    w = weights.shift(exec_lag).fillna(0.0)
    gross_ret = (w * rets).sum(axis=1)
    # the monthly-cadence book still starts every bar on its target weights, so the drift back onto them
    # is a trade — see `xsect.held_turnover`. Charged here for the same reason funding is: the omission
    # only ever flatters, and it grows with the rebalance period the caller picks.
    dw = held_turnover(w, rets)
    turn = dw.sum(axis=1)
    lin_cost = turn * cost_bps / 1e4
    if adv is not None and impact_k > 0.0:
        imp_cost = panel_impact_cost(dw, rets.rolling(vol_lb).std(), adv, capital, impact_k)
    else:
        imp_cost = pd.Series(0.0, index=w.index)
    model = resolve_carry(carry, px, borrow_bps_annual=borrow_bps_annual, where="bab_backtest")
    carry_pnl = model.pnl(w, ppy)
    cost = lin_cost + imp_cost - carry_pnl                 # carry_pnl is signed: a short can collect
    return {"net": gross_ret - cost, "gross": gross_ret, "turnover": turn,
            "carry": -carry_pnl, "cost": cost, "weights": w}
