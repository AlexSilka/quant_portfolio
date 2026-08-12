"""Leak audit for cross-sectional carry: the headline book makes most of its money on the PRICE
leg, and funding is contemporaneously correlated with returns, so before believing it we prove the
edge is not a timing artifact.

Tests (each an orthogonal angle):
  1. Leg decomposition  — funding-only vs price-only book, each vol-targeted. Which leg is the edge?
  2. Execution-lag ladder — rerun at exec_lag 2..12 days. Real predictive signal survives a longer
     delay; a contemporaneous leak collapses as the lag grows.
  3. Extra signal purge  — shift the funding signal back an extra k days so no funding settlement
     inside the traded window can inform the position. If the price edge holds, it is genuine.
  4. Long vs short leg   — is the price edge from shorting rich-funding names, longing cheap ones, or both?

    python scripts/carry/audit_carry.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

PPY, TVOL, SEED = 365, 0.15, 7
CB = 6.0
from scripts.carry.run_carry import CRYPTO, START, END  # noqa: E402


def vt(net):
    scale = vol_target_scale(net, TVOL, PPY)
    return (net * scale).dropna()


def sh(net):
    n = vt(net)
    s = summarise(n, PPY)
    p5 = bootstrap_sharpe(n, PPY, 500, SEED).get("sharpe_p5", np.nan) if s["sharpe_ann"] > 0.2 else np.nan
    return s["sharpe_ann"], p5, s["max_dd"]


def load_panel():
    close, fund = {}, {}
    for s in CRYPTO:
        px = load_klines(s, "1d", START, END, market="um")
        if len(px):
            close[s] = px["close"]
        f = load_funding(s, START, END)["last_funding_rate"]
        if len(f):
            fund[s] = f
    C = pd.DataFrame(close).sort_index()
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(C.index)
    return C, fd


def main():
    C, fd = load_panel()
    ret = C.pct_change()
    sig = carry_xs.signal_level(fd, 7)

    print("=== 1) LEG DECOMPOSITION (XScarry level-7 top20, vol-targeted each) ===")
    bk = carry_xs.xs_book(C, fd, sig, direction=-1.0, top_frac=0.2, cost_bps=CB)
    for leg in ["ret", "funding", "price"]:
        s, p5, dd = sh(bk[leg] if leg != "ret" else bk["ret"])
        print(f"  {leg:8s}-only  Sharpe {s:+.2f}  MC-P5 {p5:+.2f}  maxDD {dd:+.0%}")
    # funding-only with NO price and NO cost = the pure harvestable carry spread
    s, p5, dd = sh(bk["funding"] - bk["cost"])
    print(f"  funding-cost  Sharpe {s:+.2f}  MC-P5 {p5:+.2f}  maxDD {dd:+.0%}   <- pure carry premium")

    print("\n=== 2) EXECUTION-LAG LADDER (does the edge survive longer delay? leak dies, signal lives) ===")
    print("  lag(d)  total   funding  price")
    for lag in [2, 3, 4, 6, 8, 12]:
        b = carry_xs.xs_book(C, fd, sig, direction=-1.0, top_frac=0.2, exec_lag=lag, cost_bps=CB)
        st, _, _ = sh(b["ret"]); sf, _, _ = sh(b["funding"]); sp, _, _ = sh(b["price"])
        print(f"   {lag:3d}    {st:+.2f}   {sf:+.2f}   {sp:+.2f}")

    print("\n=== 3) EXTRA SIGNAL PURGE (shift funding signal back k extra days) ===")
    print("  purge(d)  total   price   (position uses funding strictly older than the traded window)")
    for k in [0, 1, 2, 3, 5]:
        b = carry_xs.xs_book(C, fd, sig.shift(k), direction=-1.0, top_frac=0.2, cost_bps=CB)
        st, _, _ = sh(b["ret"]); sp, _, _ = sh(b["price"])
        print(f"    {k:3d}     {st:+.2f}   {sp:+.2f}")

    print("\n=== 4) LONG-LEG vs SHORT-LEG price attribution ===")
    ranks = sig.rank(axis=1, pct=True)
    lo = (ranks <= 0.2).astype(float); hi = (ranks >= 0.8).astype(float)
    wl = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).shift(2).fillna(0.0)
    ws = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0).shift(2).fillna(0.0)
    long_px = (wl * ret).sum(axis=1)     # long cheap-funding names (price only)
    short_px = -(ws * ret).sum(axis=1)   # short rich-funding names (price only)
    for nm, series in [("long cheap-funding", long_px), ("short rich-funding", short_px)]:
        s, p5, dd = sh(series)
        print(f"  {nm:20s} price Sharpe {s:+.2f}  maxDD {dd:+.0%}")

    print("\nAUDIT OK")


if __name__ == "__main__":
    main()
