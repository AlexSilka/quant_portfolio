"""Liquidity-aware transaction costs and perpetual funding.

Cost per trade = commission + half-spread + sqrt market impact (Almgren-style), never a
flat constant. Impact scales with order size relative to bar volume, so illiquid fills are
penalised while a $500k order in a multi-billion-ADV name is (correctly) cheap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_cost_bps(notional, adv, sigma_bar, commission_bps: float,
                   half_spread_bps: float, impact_k: float = 0.1):
    """Per-trade cost in bps of notional.

    notional, adv : same currency; adv = bar dollar volume (or ADV proxy).
    sigma_bar     : fractional bar volatility (e.g. rolling std of returns).
    impact        = impact_k * sigma_bar * sqrt(notional / adv), converted to bps.
    """
    adv = np.where(np.asarray(adv) > 0, adv, np.inf)
    q = np.clip(np.asarray(notional) / adv, 0.0, None)
    impact_bps = impact_k * np.asarray(sigma_bar) * np.sqrt(q) * 1e4
    return commission_bps + half_spread_bps + impact_bps


def panel_impact_cost(dw: pd.DataFrame, sig_bar: pd.DataFrame, adv: pd.DataFrame,
                      capital: float, impact_k: float) -> pd.Series:
    """Almgren √-impact cost per bar for a cross-sectional book, summed over names.

    q = order notional / bar dollar-volume; per-name impact = impact_k·sig_bar·√q (fractional),
    charged on the per-name turnover `dw` and summed across the panel. Shared by the
    cross-sectional sleeves (xsect / bab / overnight / seasonal) so the √-impact model lives in
    ONE place instead of an inlined copy per sleeve. Callers pass their own `sig_bar` (rolling
    vol of the sleeve's return panel) because the return source differs slightly per sleeve.
    """
    q = (dw * capital).div(adv.reindex_like(dw).ffill().replace(0.0, np.nan))
    impact_bps = (impact_k * sig_bar * np.sqrt(q.clip(lower=0.0))).fillna(0.0)
    return (impact_bps * dw).sum(axis=1)


def funding_pnl(position: pd.Series, funding_rate: pd.Series) -> pd.Series:
    """Per-bar funding P&L for a perp: -position * (funding accrued over the bar).

    Funding settles on an 8h grid (00/08/16 UTC). When the bar is coarser than that (a 1d bar
    spans three settlements) a plain reindex would keep only the settlement whose timestamp
    equals the bar's and silently drop the rest. Instead we bin every settlement into the bar
    interval it falls in and sum, so all settlements are charged (spec: charge at every
    settlement). At <=4h every bar holds exactly one settlement, so this reduces to the rate.
    """
    idx = position.index
    fr = funding_rate.sort_index()
    if len(idx) > 1:
        bar = idx.to_series().diff().dropna().median()
        fr = fr.resample(bar, origin=idx[0]).sum()
    accrued = fr.reindex(idx).fillna(0.0)
    return -(position * accrued)
