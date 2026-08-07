"""FX carry — the canonical carry trade, and the FX analogue of crypto funding carry.

A currency's short interest rate is its "carry": hold a high-rate currency (funded in USD) and you
earn the rate differential, exactly as shorting a rich-funding perp earns funding. The cross-sectional
book ranks currencies by rate, goes LONG the high-rate names and SHORT the low-rate ones,
dollar-neutral, so the USD funding leg cancels and the book collects the rate spread plus whatever the
exchange rates do. The classic result (Koijen-Moskowitz-Pedersen-Vrugt): carry earns the differential
but crashes in risk-off (negative skew) — the mirror image of crypto carry, harvested here on the same
cross-sectional machinery so the two are directly comparable.

Sign note vs crypto: funding is a COST to a perp long (so you SHORT rich funding); an interest rate is
a BENEFIT to a currency holder (so you LONG high rate). Same principle — go where you are paid — opposite
direction, because funding and interest have opposite sign conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# FX pair -> (currency, is_inverse). is_inverse=True means the pair quotes CCY-per-USD (USDxxx),
# so the USD value of one unit of CCY is 1/price; False means USD-per-CCY (xxxUSD), USD value = price.
PAIR_MAP = {
    "EURUSD": ("EUR", False), "GBPUSD": ("GBP", False), "AUDUSD": ("AUD", False),
    "NZDUSD": ("NZD", False), "USDJPY": ("JPY", True), "USDCAD": ("CAD", True),
    "USDCHF": ("CHF", True), "USDMXN": ("MXN", True), "USDNOK": ("NOK", True),
    "USDSEK": ("SEK", True), "USDZAR": ("ZAR", True),
}


def usd_value_panel(fx_close: dict[str, pd.Series]) -> pd.DataFrame:
    """Turn the raw USD-pair closes into the USD value of one unit of each currency (+ USD ≡ 1)."""
    cols = {}
    for pair, (ccy, inv) in PAIR_MAP.items():
        if pair in fx_close:
            s = fx_close[pair]
            cols[ccy] = (1.0 / s) if inv else s
    panel = pd.DataFrame(cols).sort_index()
    panel["USD"] = 1.0                                  # numeraire: no FX move, earns the USD rate
    return panel


def fx_carry_book(usd_value: pd.DataFrame, rates: pd.DataFrame, *, top_frac: float = 0.33,
                  exec_lag: int = 2, half_spread_bps: float = 1.0, rebalance: int = 5,
                  rate_lag_months: int = 1, ppy_bar: float = 252) -> pd.DataFrame:
    """Dollar-neutral currency carry: LONG high-rate, SHORT low-rate. Returns a per-day frame with
    net return and its FX / carry-accrual / cost attribution (carry accrual = rate/ppy per bar)."""
    idx = usd_value.index
    fx_ret = usd_value.pct_change()
    # rates: monthly %, lagged rate_lag_months to stay point-in-time, ffilled to the daily grid
    r = rates.reindex(columns=usd_value.columns)
    r = r.shift(rate_lag_months).reindex(idx, method="ffill") / 100.0
    daily_carry = r / ppy_bar                            # interest earned per bar holding the currency

    ranks = r.rank(axis=1, pct=True)
    hi = (ranks >= 1.0 - top_frac).astype(float)         # high-rate -> LONG (collect carry)
    lo = (ranks <= top_frac).astype(float)               # low-rate  -> SHORT
    wl = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    ws = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).fillna(0.0)
    if rebalance > 1:
        keep = pd.Series(np.arange(len(w)) % rebalance == 0, index=w.index)
        w = w.where(keep, np.nan).ffill().fillna(0.0)
    w_h = w.shift(exec_lag).fillna(0.0)

    fx = (w_h * fx_ret).sum(axis=1)
    carry = (w_h * daily_carry.reindex_like(w_h).fillna(0.0)).sum(axis=1)   # LONG high-rate earns +
    turn = w_h.diff().abs().sum(axis=1)
    cost = turn * half_spread_bps / 1e4
    net = fx + carry - cost
    return pd.DataFrame({"ret": net, "fx": fx, "carry": carry, "cost": cost,
                         "turnover": turn, "gross": w_h.abs().sum(axis=1)}).dropna(subset=["ret"])
