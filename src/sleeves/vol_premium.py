"""Short-volatility / variance-risk-premium (VRP) sleeve — structurally orthogonal to trend.

Trend is long gamma: it pays in big moves and bleeds in calm. This sleeve is SHORT gamma: it
harvests the variance risk premium (implied variance exceeds realised variance on average, because
option buyers overpay for protection) and loses in vol spikes. That opposite payoff is the whole
point — it diversifies the *source* of return, which adding another trend-adjacent family cannot.

Construction (honest, from free data). A discrete **capped variance swap**, re-struck weekly:
short the implied variance (Deribit DVOL, the only free historical implied series), pay the perp
bars' own realised variance, with the realised leg capped at `var_cap x strike` — a real, traded
instrument whose left tail is bounded (you implicitly own a cheap wing). Per-option historical
marks are NOT free, so this is a variance-swap REPLICATION, not an executed option chain; the cost
of the option spread is modelled in vega terms and stress-tested in the runner.

Leakage. The strike `K` and side are decided from information at the decision bar and shifted
`exec_lag` bars forward, so everything multiplying a given day's squared return was fixed strictly
before that day. The realised leg the short pays is the *outcome*, not a peek.

Non-ML baseline (§5): always-short vol. Family rule: short only when implied is rich vs realised.
Risk control (§8, family-specific): the variance cap bounds the defining left-tail hazard here,
rather than leaning on the uniform portfolio-level stop that suits a trend sleeve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PPY = 365  # crypto trades continuously


def realized_vol(close: pd.Series, lookback: int = 30, ppy: int = PPY) -> pd.Series:
    """Trailing annualised realised vol from daily log returns (uses only info <= t)."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(lookback).std() * np.sqrt(ppy)


def realized_var_ohlc(bars: pd.DataFrame, ppy: int = PPY) -> pd.Series:
    """Per-bar annualised realised variance from OHLC — the overnight gap (close-to-open) plus the
    Rogers-Satchell intraday-range term. A delta-hedged short-vol book pays the intraday *path*, not
    just the net close-to-close move; on choppy/round-trip days the range captures variance that
    close-to-close nets away. Rogers-Satchell is drift-free and non-negative."""
    o, h, l, c = (np.log(bars[k]) for k in ("open", "high", "low", "close"))
    overnight = (o - c.shift(1)) ** 2                              # the gap you can't hedge through
    rs = (h - c) * (h - o) + (l - c) * (l - o)                     # Rogers-Satchell intraday, >= 0
    return (overnight + rs).clip(lower=0.0) * ppy


def short_vol_book(close: pd.Series, dvol: pd.Series, *, bars: pd.DataFrame | None = None,
                   rv_lookback: int = 30, ppy: int = PPY,
                   k_rich: float = 1.0, timed: bool = True, var_cap: float = 2.5,
                   restrike_days: int = 7, vega_cost_volpts: float = 0.75, wing_markup: float = 0.0,
                   spike_degross: float = 0.0, exec_lag: int = 2,
                   gate: pd.Series | None = None) -> pd.DataFrame:
    """Daily capped-variance-swap P&L for a short-vol book on one asset (variance units).

    timed=False is the always-short non-ML baseline; timed=True shorts only when implied is rich
    (DVOL > k_rich * trailing realised vol). Returns a frame: side, strike K, gross, cost, net,
    turnover. The net series is a P&L proxy (variance units); the runner vol-targets it to 15%.

    Tail hedge (the deployable form). `var_cap` truncates the realised charge at var_cap*strike — the
    payoff of a bought wing that bounds the crash — and the wing is PRICED, else the truncation is a
    free lunch that vol-targeting inflates into a fake Sharpe. Its cost self-scales with how tight the
    cap is: `wing_markup` times the trailing mean of the tail it protects (max(realised - cap*strike, 0)),
    charged each bar. A tight cap protects (and so costs) more; a loose cap little. wing_markup=0 with a
    huge var_cap is the naked research book; var_cap~2.5, wing_markup~2 is the tail-bounded deploy book.
    """
    close = close.sort_index()
    if bars is not None:                                      # realistic paid leg: intraday path + gap
        r2 = realized_var_ohlc(bars.reindex(close.index), ppy)
    else:                                                     # close-to-close (net daily move only)
        r2 = (np.log(close / close.shift(1)) ** 2) * ppy
    dv = dvol.reindex(close.index).ffill() / 100.0            # implied vol (decimal), info at t
    rv = realized_vol(close, rv_lookback, ppy)                # trailing realised vol, info at t

    # --- side (decision at t): always-short baseline, or short-only-when-rich ---
    if timed:
        side = -(dv > k_rich * rv).astype(float)              # -1 when implied rich, else flat
    else:
        side = pd.Series(-1.0, index=close.index)             # always short (baseline)
    side = side.where(dv.notna() & rv.notna(), 0.0)

    # --- ex-ante spike de-gross (§8 risk tool, no options needed): cut the short while implied vol is
    # spiking (dv above spike_degross x its 20-bar mean) -> reduce exposure BEFORE the crash deepens,
    # unlike a P&L drawdown stop that fires after the loss. Uses only info at t (lagged with side below).
    if spike_degross:
        spike = dv / dv.rolling(20, min_periods=5).mean()
        side = side * np.minimum(1.0, spike_degross / spike).fillna(1.0)

    # --- regime gate (exposure multiplier in [0,1], decided at t): applied to the SIDE, not to the
    # finished P&L, so switching the leg off and back on pays the vega spread through the same cost
    # model as any other roll. Gating the return series instead would make the timing look free.
    if gate is not None:
        g = pd.Series(gate).reindex(close.index).ffill().fillna(1.0).clip(0.0, 1.0)
        side = side * g

    # --- strike: re-struck every `restrike_days`, held between rolls (turnover control) ---
    K = pd.Series(np.nan, index=close.index)
    K.iloc[::restrike_days] = dv.iloc[::restrike_days]
    K = K.ffill()                                             # implied strike in force, info <= t

    # --- shift decision variables forward: no leg multiplies its own or an earlier day's return ---
    Kx = K.shift(exec_lag)
    sidex = side.shift(exec_lag).fillna(0.0)
    Kvar = Kx ** 2                                            # strike variance (annualised)

    charge = np.minimum(r2, var_cap * Kvar)                  # capped realised-variance charge
    # long-variance payoff is (realised - strike); side = -1 flips it to the short's (strike - realised),
    # which profits in calm (realised < strike) and loses in a spike -> the true short-vol sign.
    gross = sidex * (charge - Kvar)

    # --- costs: vega spread when the strike rolls or the side flips (dVar ~ 2*K*dVol) ---
    roll = (Kx != Kx.shift(1)) | (sidex != sidex.shift(1))
    # charge the LARGER of the two sides of the roll, so unwinding to flat costs the spread as much as
    # putting the position on. (With the always-short baseline |side| is constant, so this is identical
    # to charging |sidex|; it only bites once a gate takes exposure to zero and back.)
    turnover = roll.astype(float) * np.maximum(sidex.abs(), sidex.abs().shift(1).fillna(0.0))
    cost = turnover * 2.0 * Kx.clip(lower=1e-6) * (vega_cost_volpts / 100.0)

    # --- tail-wing premium: pay wing_markup x the trailing tail the cap protects (self-scales with cap) ---
    protected = np.maximum(r2 - var_cap * Kvar, 0.0)          # per-bar tail truncated by the cap (wing payoff)
    wing = wing_markup * protected.rolling(252, min_periods=20).mean().shift(1).fillna(0.0) * sidex.abs()

    net = (gross - cost - wing).where(sidex != 0.0, 0.0)
    return pd.DataFrame({"side": sidex, "K": Kx, "rich": (dv - rv).shift(exec_lag),
                         "gross": gross, "cost": cost, "net": net, "turnover": turnover}).dropna()
