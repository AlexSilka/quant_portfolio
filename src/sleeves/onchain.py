"""On-chain / network-data signals (H3) — the one information source not derived from price.

Two shapes, both computable-at-bar (value at t uses only data ≤ t; the driver adds the shared
`exec_lag` delay so nothing fills at its own bar):

  • Cross-sectional (panel bars × names) → swapped straight into `xsect.xs_backtest`, exactly like
    every other family. Two economic angles:
      – ADOPTION MOMENTUM: active-address / transaction-count growth. Network usage accelerating →
        long. The on-chain analogue of momentum; the honest question is whether it adds anything
        *over* price momentum (tested by the network-vs-price divergence signal + orthogonalisation).
      – ON-CHAIN VALUE: MVRV (price vs aggregate cost basis) and NVM / Metcalfe (market cap per
        active user). Cheap-per-network → long, rich → short. The on-chain analogue of value.

  • Time-series (one asset) → BTC/ETH market-timing overlays: MVRV z-score, NVT, Puell (miner),
    and the stablecoin-supply (SSR) risk-on macro tilt. Tested long/flat and long/short vs buy-hold.

Daily counts carry strong day-of-week seasonality and blockchain noise, so activity series are
7-day-smoothed before any growth/ratio is taken — standard on-chain practice (Coin Metrics /
Glassnode). Everything is a rolling transform → no look-ahead beyond the bar it is stamped at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _smooth(x: pd.DataFrame, w: int = 7) -> pd.DataFrame:
    """7-day trailing mean — kills weekday seasonality + block-level noise in raw daily counts.
    w<=1 is a no-op (already-aggregated series, e.g. a weekly-resampled panel)."""
    if w <= 1:
        return x
    return x.rolling(w, min_periods=max(2, w // 2)).mean()


def _z(x: pd.DataFrame | pd.Series, lb: int, min_frac: float = 0.5):
    """Trailing z-score over `lb` bars (computable-at-bar; the current bar is included)."""
    m = x.rolling(lb, min_periods=int(lb * min_frac)).mean()
    s = x.rolling(lb, min_periods=int(lb * min_frac)).std()
    return (x - m) / s.replace(0.0, np.nan)


# ── cross-sectional signals (wide bars × names) ───────────────────────────────────────────────
def adr_momentum(adr: pd.DataFrame, lb: int = 30, smooth: int = 7) -> pd.DataFrame:
    """Active-address growth over `lb` days (log), on the 7d-smoothed series. Adoption momentum:
    a network gaining users faster than its peers → long. Ranked cross-sectionally."""
    a = _smooth(adr, smooth)
    return np.log(a / a.shift(lb))


def tx_momentum(tx: pd.DataFrame, lb: int = 30, smooth: int = 7) -> pd.DataFrame:
    """Transaction-count growth over `lb` days (log, smoothed). Usage momentum — the transaction
    twin of address momentum; correlated but not identical (whales vs breadth)."""
    t = _smooth(tx, smooth)
    return np.log(t / t.shift(lb))


def nvm_ratio(mktcap: pd.DataFrame, adr: pd.DataFrame, smooth: int = 7,
              metcalfe: bool = False) -> pd.DataFrame:
    """Network-value-to-users (NVM). Market cap ÷ active addresses = price per user; the Metcalfe
    variant divides by addresses² (fair value ∝ users², Metcalfe's law). Rich-per-user is expensive
    → the *value* signal is −NVM (long cheap networks). Returned as raw ratio; the driver flips sign."""
    a = _smooth(adr, smooth)
    denom = a ** 2 if metcalfe else a
    return mktcap / denom.replace(0.0, np.nan)


def mvrv_value(mvrv: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional MVRV level. Low = trading near/below realised cost basis (undervalued) →
    long; high = extended above cost basis → short. The driver ranks on −MVRV (long low)."""
    return mvrv.replace(0.0, np.nan)


def net_vs_price_divergence(adr: pd.DataFrame, px: pd.DataFrame, lb: int = 30,
                            smooth: int = 7) -> pd.DataFrame:
    """Address-growth minus price-return over the same window (both cross-sectionally z-scored).
    Positive = on-chain activity is outrunning price (accumulation not yet in the tape) → long.
    This is the *orthogonal-by-construction* signal: it strips the shared move, so it is the honest
    "information not present in price" bet rather than re-labelled price momentum."""
    a = _smooth(adr, smooth)
    adr_g = np.log(a / a.shift(lb))
    px_g = np.log(px / px.shift(lb))
    zc = lambda d: d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1).replace(0.0, np.nan), axis=0)
    return zc(adr_g) - zc(px_g)


# ── time-series overlays (single asset) ───────────────────────────────────────────────────────
def mvrv_zscore(mvrv: pd.Series, lb: int = 365) -> pd.Series:
    """MVRV z-score (Coin Metrics / Woobull). Extended when high, capitulation when low; timing
    signal = −z (fade the extreme). Long window (1y) so the mean is a full-cycle cost basis."""
    return _z(mvrv.replace(0.0, np.nan), lb)


def nvt_signal(mktcap: pd.Series, tx_val_usd: pd.Series, smooth: int = 90) -> pd.Series:
    """NVT signal (Kalichkin): market cap ÷ 90d-avg on-chain transaction value (USD). High = price
    rich vs economic throughput → bearish; timing signal = −z(NVT). BTC only (needs transfer value,
    which is free on blockchain.com but pay-walled on CM)."""
    tv = tx_val_usd.rolling(smooth, min_periods=smooth // 2).mean()
    return mktcap / tv.replace(0.0, np.nan)


def puell_multiple(miner_rev_usd: pd.Series, lb: int = 365) -> pd.Series:
    """Puell multiple: daily miner revenue ÷ its 365d MA. Low = miner capitulation (historable
    bottoms) → bullish; high = distribution → bearish. Timing signal = −z(Puell). BTC only."""
    return miner_rev_usd / miner_rev_usd.rolling(lb, min_periods=lb // 2).mean().replace(0.0, np.nan)


def stablecoin_ssr_growth(stable_supply: pd.Series, lb: int = 30) -> pd.Series:
    """Aggregate stablecoin-supply growth over `lb` days — dry powder entering the system. Expansion
    → risk-on → long BTC/majors (a macro tilt, not cross-sectional). Returned as log-growth."""
    s = stable_supply.rolling(7, min_periods=3).mean()
    return np.log(s / s.shift(lb))


# ── ML feature panel (mirror of xsect_ml.rank_features, but the on-chain axis) ─────────────────
def ml_feature_panel(oc_p: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Name-level on-chain features for a learning-to-rank model — every one trailing/rolling so it
    is computable-at-bar. Deliberately the *full* on-chain axis (adoption momentum at several
    horizons, valuation levels + self-relative z, usage intensity, dilution, network-vs-price
    divergence) so the model can find any non-linear combination the linear books cannot — and so
    feature importance can reveal whether it re-derives price momentum or uses genuine on-chain info.
    `oc_p` holds the aligned wide panels: AdrActCnt, TxCnt, CapMrktCurUSD, CapMVRVCur, PriceUSD, SplyCur."""
    adr, tx = oc_p["AdrActCnt"], oc_p["TxCnt"]
    cap, mvrv, pxu = oc_p["CapMrktCurUSD"], oc_p["CapMVRVCur"], oc_p["PriceUSD"]
    sply = oc_p.get("SplyCur")
    a7 = _smooth(adr, 7)
    t7 = _smooth(tx, 7)
    f: dict[str, pd.DataFrame] = {}
    for d in (7, 14, 30, 90):
        f[f"adr_mom_{d}"] = np.log(a7 / a7.shift(d))
        f[f"tx_mom_{d}"] = np.log(t7 / t7.shift(d))
    f["adr_accel"] = np.log(a7 / a7.shift(14)) - np.log(a7.shift(14) / a7.shift(28))  # 2nd difference
    f["nvm_log"] = np.log(nvm_ratio(cap, adr, 7))
    f["nvm_z"] = _z(np.log(nvm_ratio(cap, adr, 7)), 365)
    f["metcalfe_log"] = np.log(nvm_ratio(cap, adr, 7, metcalfe=True))
    f["mvrv"] = mvrv.replace(0.0, np.nan)
    f["mvrv_z"] = _z(mvrv.replace(0.0, np.nan), 365)
    f["tx_per_adr"] = np.log((t7 / a7).replace(0.0, np.nan))                          # usage intensity
    f["divergence"] = net_vs_price_divergence(adr, pxu, 30, 7)
    if sply is not None:
        f["supply_growth"] = np.log(sply / sply.shift(90))                            # own-token dilution
    return f
