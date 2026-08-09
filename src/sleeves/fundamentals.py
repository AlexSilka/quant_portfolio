"""Chain-fundamentals signals — valuing a token on what its chain earns, not on how busy it looks.

The on-chain family ([onchain.py](onchain.py)) had to proxy value with address counts, because the
free network data covers only legacy coins. Chain fees and TVL are free for the modern chains, and
they are *cash flows*, so the ratios here are the ones an equity analyst would recognise:

  VALUE     fee_yield   annualised fees ÷ market cap        — the crypto earnings yield (inverse P/F)
            rev_yield   annualised revenue ÷ market cap     — the slice that reaches the token holder
            tvl_yield   TVL ÷ market cap                    — price-to-book: capital hosted per dollar of cap
  GROWTH    fee_growth  fees this quarter vs the last       — fundamental momentum, no price input
            tvl_growth  capital arriving on the chain
  QUALITY   fee_margin  fees ÷ TVL                          — how productive the parked capital is

Every transform is trailing and quarter-length: chain revenue is violently seasonal (one NFT mint or
memecoin week can multiply a day's fees), so a 90-day window is the shortest that measures a business
rather than an event. All are stamped at the bar they are computable on; the driver adds `exec_lag`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 90  # one quarter — short enough to be current, long enough to survive a memecoin week


def _trailing_sum(x: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    """Trailing sum with a half-window minimum, so a chain enters the cross-section once it has a
    meaningful history rather than on its first noisy day."""
    return x.rolling(lb, min_periods=lb // 2).sum()


def _trailing_mean(x: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    return x.rolling(lb, min_periods=lb // 2).mean()


def fee_yield(fees: pd.DataFrame, mktcap: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    """Annualised trailing fees ÷ market cap. High = the market is paying little for each dollar the
    chain collects = cheap → long. The direct analogue of an earnings yield, and the ratio the crypto
    fundamentals literature (Token Terminal's P/F) is built on."""
    ann = _trailing_sum(fees, lb) * (365.0 / lb)
    return ann / mktcap.replace(0.0, np.nan)


def rev_yield(revenue: pd.DataFrame, mktcap: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    """Annualised trailing revenue ÷ market cap. Revenue is the part of fees that accrues to the
    token — burned supply, validator take — so this is the yield a holder actually owns, where fee
    yield is the yield the *network* generates. Long the high-yield names."""
    ann = _trailing_sum(revenue, lb) * (365.0 / lb)
    return ann / mktcap.replace(0.0, np.nan)


def tvl_yield(tvl: pd.DataFrame, mktcap: pd.DataFrame, smooth: int = 30) -> pd.DataFrame:
    """TVL ÷ market cap — capital hosted per dollar of token value, crypto's price-to-book inverted
    so that high = cheap → long. Smoothed, because TVL is a mark-to-market stock and jumps with the
    price of whatever is deposited."""
    t = _trailing_mean(tvl, smooth)
    return t / mktcap.replace(0.0, np.nan)


def fee_growth(fees: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    """Log growth of trailing-quarter fees against the quarter before it. A chain whose economics are
    compounding, measured without touching price — the one signal here that cannot be a re-labelled
    price momentum, since neither leg contains a price."""
    cur = _trailing_sum(fees, lb)
    prev = cur.shift(lb)
    return np.log(cur.clip(lower=1.0) / prev.clip(lower=1.0))


def tvl_growth(tvl: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    """Log growth in TVL over a quarter — capital voting with its feet. Partly mechanical (TVL rises
    when deposited assets appreciate), which is exactly why it is reported next to fee growth rather
    than blended into it."""
    t = _trailing_mean(tvl, 30)
    return np.log(t.clip(lower=1.0) / t.shift(lb).clip(lower=1.0))


def fee_margin(fees: pd.DataFrame, tvl: pd.DataFrame, lb: int = WINDOW) -> pd.DataFrame:
    """Annualised fees ÷ TVL — what the chain extracts per dollar parked on it. A quality measure
    rather than a valuation one: it says nothing about price, only about whether the capital sitting
    on this chain is doing anything."""
    ann = _trailing_sum(fees, lb) * (365.0 / lb)
    return ann / _trailing_mean(tvl, 30).replace(0.0, np.nan)


def value_blend(fees: pd.DataFrame, revenue: pd.DataFrame, tvl: pd.DataFrame,
                mktcap: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight rank blend of the three valuation ratios. Averaging ranks rather than levels
    keeps a single chain with a freak fee week from dominating, and the three ratios disagree often
    enough (a chain can be cheap on fees and rich on TVL) that the blend is not a fourth copy."""
    parts = [fee_yield(fees, mktcap), rev_yield(revenue, mktcap), tvl_yield(tvl, mktcap)]
    ranks = [p.rank(axis=1, pct=True) for p in parts]
    return sum(ranks) / len(ranks)
