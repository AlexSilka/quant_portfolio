"""Can crypto SPOT's deeper history (2017+) extend the funding-carry strategy? Funding is perp-only
(2020+), so it cannot — directly. But carry's PRICE leg is a cross-sectional reversal (crowded-long,
high-funding names underperform), and reversal is price-observable on spot back to 2017. This tests a
spot-native reversal proxy for that leg:

  (1) does it CORRELATE with the real funding carry on 2020+ (i.e. is it the same effect)?
  (2) does it WORK on the deep 2017-2020 window (where funding does not exist)?

If both, spot extends carry via a proxy. If not (expected — crypto trends, and the pre-2020 cross-section
is thin), then carry stays honestly 2020+ and this documents why.

    python scripts/carry/run_carry_spot.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import REPORTS_DIR, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402

PPY, TVOL, CB = 365, VOL_TARGET_ANNUAL, 6.0
crypto = open(REPORTS_DIR / "crypto_universe.txt").read().strip().split(",")


def vt(net):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def spot_panel():
    close, vol = {}, {}
    for s in crypto:
        try:
            px = load_klines(s, "1d", "2017-08", "2026-07", market="spot")
        except Exception:
            continue
        if len(px) > 200:
            close[s], vol[s] = px["close"], px["quote_volume"]
    return pd.DataFrame(close).sort_index(), pd.DataFrame(vol).sort_index()


def reversal_book(close, lookback, crowd_vol=None):
    """Cross-sectional short-term reversal: LONG losers / SHORT winners (proxy for shorting crowded,
    recently-pumped names). If crowd_vol given, weight the signal by volume z (crowding intensity)."""
    ret = close.pct_change()
    sig = ret.rolling(lookback).sum()
    if crowd_vol is not None:                      # crowding = big move on high relative volume
        vz = (crowd_vol - crowd_vol.rolling(30).mean()) / (crowd_vol.rolling(30).std() + 1e-9)
        sig = sig * vz.clip(lower=0).reindex_like(sig).fillna(0)
    # rank by trailing return; direction=-1 -> long low-return (losers), short high-return (winners)
    bk = carry_xs.xs_book(close, ret * 0.0, sig, direction=-1.0, top_frac=0.2, cost_bps=CB,
                          weight="inv_vol", buffer=0.02)
    return bk["price"] - bk["cost"]                # spot has no funding leg


def seg(net, a, b):
    s = net[(net.index >= a) & (net.index < b)]
    return summarise(s, PPY)["sharpe_ann"] if len(s) > 60 else float("nan")


def main():
    C, V = spot_panel()
    n_names = C.notna().sum(axis=1)
    print(f"spot panel: {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}")
    print(f"  cross-section breadth: 2018 median {int(n_names.loc['2018':'2018'].median())} names, "
          f"2019 {int(n_names.loc['2019':'2019'].median())}, 2022 {int(n_names.loc['2022':'2022'].median())}\n")

    # real funding carry (perp, 2020+) for the correlation/benchmark
    fund = {}
    for s in crypto:
        f = load_funding(s, "2020-01", "2026-07")
        if len(f):
            fund[s] = f["last_funding_rate"]
    perp = pd.DataFrame({s: load_klines(s, "1d", "2020-01", "2026-07", market="um")["close"] for s in crypto}).sort_index()
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(index=perp.index, columns=perp.columns)
    bk = carry_xs.xs_book(perp, fd, carry_xs.signal_level(fd, 7), direction=-1.0, top_frac=0.2,
                          cost_bps=CB, weight="inv_vol", buffer=0.02)
    real_carry = vt(carry_xs.beta_hedge(bk["ret"], perp["BTCUSDT"].pct_change()))
    print(f"real funding carry (perp, 2020+): Sharpe {summarise(real_carry, PPY)['sharpe_ann']:+.2f}\n")

    print("=== spot reversal proxy: Sharpe by window + corr to real carry ===")
    print(f"  {'signal':22s} {'2017-2020':>10s} {'2020-2026':>10s} {'2017-2026':>10s} {'corr→carry':>11s}")
    for lb in (3, 7, 14, 30):
        net = vt(reversal_book(C, lb))
        s1, s2 = seg(net, "2017-01", "2020-01"), seg(net, "2020-01", "2027-01")
        sall = summarise(net, PPY)["sharpe_ann"]
        idx = net.index.intersection(real_carry.index)
        corr = float(pd.concat([net.reindex(idx), real_carry.reindex(idx)], axis=1).corr().iloc[0, 1]) if len(idx) > 60 else np.nan
        print(f"  reversal-{lb}d{'':13s}"[:24] + f"{s1:>10.2f} {s2:>10.2f} {sall:>10.2f} {corr:>11.2f}")
    # crowding-weighted variant at the best-ish lookback
    net = vt(reversal_book(C, 7, crowd_vol=V))
    idx = net.index.intersection(real_carry.index)
    corr = float(pd.concat([net.reindex(idx), real_carry.reindex(idx)], axis=1).corr().iloc[0, 1]) if len(idx) > 60 else np.nan
    print(f"  {'reversal-7d×volume':22s} {seg(net,'2017-01','2020-01'):>10.2f} {seg(net,'2020-01','2027-01'):>10.2f} "
          f"{summarise(net,PPY)['sharpe_ann']:>10.2f} {corr:>11.2f}")

    print("\n=== verdict ===")
    print("  funding carry is perp-only (2020+); spot's 2017-2020 history has no funding to harvest.")
    print("  the spot reversal PROXY is judged on: does it (a) correlate with real carry AND (b) work pre-2020.")
    print("\nCARRY-SPOT OK")


if __name__ == "__main__":
    main()
