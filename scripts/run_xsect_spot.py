"""Does cross-sectional MOMENTUM (x-sect) work on crypto SPOT, and does spot's deeper history (2017+)
help? Unlike carry, x-sect is a pure price signal — no funding needed — so it runs on spot, and spot
reaches 2017-08 vs the perp panel's 2020. Crypto trended hard in 2017-19, so momentum (not reversal)
has a chance there. Standalone research (does NOT touch the x-sect family's published block);
if it works, recommend extending their stream to spot.

Tests long-winners / short-losers, dollar-neutral, across momentum lookbacks, split by window, and its
correlation to the real perp x-sect. Honest checks: pre-2020 cross-section breadth, and spot-vs-perp on 2020+.

    python scripts/run_xsect_spot.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

PPY, TVOL, CB = 365, VOL_TARGET_ANNUAL, 6.0
crypto = open(REPORTS_DIR / "crypto_universe.txt").read().strip().split(",")


def vt(net):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def panel(market, start):
    close, vol = {}, {}
    for s in crypto:
        try:
            px = load_klines(s, "1d", start, "2026-07", market=market)
        except Exception:
            continue
        if len(px) > 200:
            close[s], vol[s] = px["close"], px["quote_volume"]
    return pd.DataFrame(close).sort_index(), pd.DataFrame(vol).sort_index()


def xsect_mom(close, vol, lb, skip=0, top_n=40, top_frac=0.3):
    """Cross-sectional momentum: rank by trailing lb-day return (skipping the last `skip` days), among
    the top-N most-liquid names each day; LONG winners / SHORT losers, dollar-neutral, inverse-vol."""
    ret = close.pct_change()
    sig = (close.shift(skip) / close.shift(skip + lb) - 1.0)              # momentum, skip recent
    liq = vol.rolling(30).median().shift(1)
    elig = liq.rank(axis=1, ascending=False) <= top_n
    sig = sig.where(elig)
    bk = carry_xs.xs_book(close, ret * 0.0, sig, direction=1.0, top_frac=top_frac, cost_bps=CB,
                          weight="inv_vol", buffer=0.02)                  # direction +1 = long winners
    return bk["price"] - bk["cost"]                                       # price-only (no funding on spot)


def seg(net, a, b):
    s = net[(net.index >= a) & (net.index < b)]
    return summarise(s, PPY)["sharpe_ann"] if len(s) > 60 else float("nan")


def main():
    SC, SV = panel("spot", "2017-08")
    n = SC.notna().sum(axis=1)
    print(f"spot panel: {SC.shape[1]} names, {SC.index.min().date()}..{SC.index.max().date()}")
    print(f"  breadth: 2018 {int(n.loc['2018':'2018'].median())} / 2019 {int(n.loc['2019':'2019'].median())} / "
          f"2021 {int(n.loc['2021':'2021'].median())} / 2024 {int(n.loc['2024':'2024'].median())} names\n")

    # the real perp x-sect stream from the book (for the 2020+ correlation), if available
    real = None
    p = REPORTS_DIR / "master_book_legs.parquet"
    if p.exists():
        sl = pd.read_parquet(p)
        real = sl["xs_momentum"].dropna() if "xs_momentum" in sl.columns else None

    print("=== crypto x-sect MOMENTUM on SPOT: Sharpe by window ===")
    print(f"  {'signal':16s} {'2017-2020':>10s} {'2020-2026':>10s} {'2017-2026':>10s} {'MC-P5(full)':>11s} {'corr→perp':>10s}")
    best = None
    for lb, sk in [(14, 0), (30, 0), (30, 3), (60, 3), (90, 7)]:
        net = vt(xsect_mom(SC, SV, lb, sk))
        s1, s2 = seg(net, "2017-01", "2020-01"), seg(net, "2020-01", "2027-01")
        sall = summarise(net, PPY)["sharpe_ann"]
        p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan) if sall > 0.1 else np.nan
        corr = np.nan
        if real is not None:
            idx = net.index.intersection(real.index)
            if len(idx) > 60:
                corr = float(pd.concat([net.reindex(idx), real.reindex(idx)], axis=1).corr().iloc[0, 1])
        print(f"  mom-{lb}d/sk{sk}{'':6s}"[:18] + f"{s1:>10.2f} {s2:>10.2f} {sall:>10.2f} {p5:>11.2f} {corr:>10.2f}")
        if best is None or sall > best[0]:
            best = (sall, lb, sk, net)

    # spot vs perp on 2020+ (are they the same where both exist?)
    PC, PV = panel("um", "2020-01")
    perp_mom = vt(xsect_mom(PC, PV, best[1], best[2]))
    idx = best[3].index.intersection(perp_mom.index)
    sp = seg(best[3], "2020-01", "2027-01"); pp = seg(perp_mom, "2020-01", "2027-01")
    cc = float(pd.concat([best[3].reindex(idx), perp_mom.reindex(idx)], axis=1).corr().iloc[0, 1]) if len(idx) > 60 else np.nan
    print(f"\n  spot vs perp on 2020+ (best config mom-{best[1]}d/sk{best[2]}): "
          f"spot {sp:+.2f} | perp {pp:+.2f} | corr {cc:+.2f}")
    print("\n=== verdict: does spot's deep history give x-sect momentum a usable longer track? ===")
    print(f"  best full-sample Sharpe {best[0]:+.2f} (mom-{best[1]}d/sk{best[2]}); pre-2020 vs post-2020 above tells the story.")
    print("\nXSECT-SPOT OK")


if __name__ == "__main__":
    main()
