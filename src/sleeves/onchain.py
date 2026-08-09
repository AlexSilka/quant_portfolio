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


def holder_momentum(holders: pd.DataFrame, lb: int = 30, smooth: int = 7) -> pd.DataFrame:
    """Growth in addresses holding a non-zero balance. Distinct from active-address momentum:
    holders is a *stock* (who owns), active addresses a *flow* (who moved this week), so this is
    accumulation breadth rather than usage. Rising holder base → long."""
    h = _smooth(holders, smooth)
    return np.log(h / h.shift(lb))


def mcap_per_holder(mktcap: pd.DataFrame, holders: pd.DataFrame, smooth: int = 7) -> pd.DataFrame:
    """Market cap ÷ holder count = price per owner. The value twin of NVM built on the ownership
    stock instead of the activity flow — a network held by few at a high cap is expensive.
    Returned as the raw ratio; the driver flips sign (long cheap)."""
    h = _smooth(holders, smooth)
    return mktcap / h.replace(0.0, np.nan)


def supply_inflation(issuance: pd.DataFrame, supply: pd.DataFrame, lb: int = 90) -> pd.DataFrame:
    """Annualised token issuance over trailing `lb` days as a fraction of circulating supply — the
    dilution a holder is paid to absorb. High inflation = structural sell pressure → short; the
    driver ranks on −inflation. The on-chain analogue of an equity buyback/issuance factor."""
    iss = issuance.rolling(lb, min_periods=lb // 2).sum()
    return (iss / supply.replace(0.0, np.nan)) * (365.0 / lb)


def fee_yield(fees_ntv: pd.DataFrame, px_usd: pd.DataFrame, mktcap: pd.DataFrame,
              lb: int = 90) -> pd.DataFrame:
    """Trailing fee revenue (native units × price) over `lb` days, annualised, ÷ market cap — the
    crypto earnings yield (inverse P/F). A chain earning more per dollar of cap is cheap → long.
    Only the 14 names whose fees are free on the community tier carry a column."""
    rev = (_smooth(fees_ntv, 7) * px_usd).rolling(lb, min_periods=lb // 2).sum() * (365.0 / lb)
    return rev / mktcap.replace(0.0, np.nan)


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


def exchange_netflow_z(flow_in: pd.Series, flow_out: pd.Series, supply_ex: pd.Series,
                       smooth: int = 7, lb: int = 365) -> pd.Series:
    """Net coin flow onto exchanges, as a share of the exchange-held balance, z-scored. The classic
    practitioner read: coins moving *onto* exchanges are being positioned to sell (bearish), coins
    leaving are moving to cold storage (accumulation). Scaling by exchange supply makes the series
    comparable across a decade in which exchange balances changed by an order of magnitude.

    High z = heavy inflow = bearish, matching the driver's fade convention (position = −signal)."""
    net = _smooth(flow_in - flow_out, smooth)
    return _z(net / supply_ex.replace(0.0, np.nan), lb)


def exchange_supply_trend(supply_ex: pd.Series, supply_cur: pd.Series, lb: int = 30) -> pd.Series:
    """Change over `lb` days in the fraction of circulating supply sitting on exchanges — the
    *stock* counterpart to net-flow, and far less noisy than a daily difference. A rising exchange
    share is distribution (bearish); a falling share is the supply squeeze the thesis is about.

    High = share rising = bearish, matching the driver's fade convention."""
    share = (supply_ex / supply_cur.replace(0.0, np.nan)).rolling(7, min_periods=3).mean()
    return share - share.shift(lb)


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
    sply, hold, iss, fee = oc_p.get("SplyCur"), oc_p.get("AdrBalCnt"), oc_p.get("IssTotNtv"), oc_p.get("FeeTotNtv")
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
    if hold is not None:
        h7 = _smooth(hold, 7)
        f["holder_mom_30"] = np.log(h7 / h7.shift(30))                                # ownership breadth
        f["holder_mom_90"] = np.log(h7 / h7.shift(90))
        f["cap_per_holder_log"] = np.log(mcap_per_holder(cap, hold, 7))
        f["holders_per_active"] = np.log((h7 / a7).replace(0.0, np.nan))              # owners vs users
    if iss is not None and sply is not None:
        f["inflation_90"] = supply_inflation(iss, sply, 90)                           # measured dilution
    if fee is not None:
        f["fee_yield_90"] = fee_yield(fee, pxu, cap, 90)                              # 14-name coverage
    return f
