"""Test PROPER mean-reversion constructions on appropriate universes (not single trending assets):

  1. Cross-sectional short-term reversal in equities (Lehmann/Lo-MacKinlay): rank the panel by
     trailing k-day return, long losers / short winners, dollar-neutral.
  2. Pairs / spread stat-arb in crypto: z-score of a cointegration-style spread, trade the spread
     with a revert-to-mean exit (the spread reverts even though each leg trends).

If MR earns its keep anywhere, it is here — not in single-asset directional z-score on BTC/ETH/SOL.

    python scripts/meanrev/audit_mr2.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import xs_returns  # noqa: E402

PANEL = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "META", "JNJ", "XOM", "WMT",
         "V", "PG", "HD", "BAC", "KO", "DIS", "CSCO", "INTC", "CVX", "PFE"]


def _voltarget(net, ppy, tgt=0.15):
    scale = (tgt / (net.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def pos_from_z(z, entry, exit_):
    z = np.asarray(z)
    pos = np.zeros(len(z))
    st = 0
    for i in range(len(z)):
        if not np.isnan(z[i]):
            if st == 0:
                st = -1 if z[i] >= entry else (1 if z[i] <= -entry else 0)  # +1 = long a cheap spread
            elif st == 1 and z[i] >= -exit_:
                st = 0
            elif st == -1 and z[i] <= exit_:
                st = 0
        pos[i] = st
    return pos


def pairs_mr(y, x, lookback, entry, exit_, cost_bps, ppy):
    y, x = y.align(x, join="inner")
    ry, rx = y.pct_change(), x.pct_change()
    beta = (ry.rolling(lookback).cov(rx) / (rx.rolling(lookback).var() + 1e-12)).shift(1)
    spread = np.log(y) - beta * np.log(x)
    z = (spread - spread.rolling(lookback).mean()) / (spread.rolling(lookback).std() + 1e-12)
    pos = pd.Series(pos_from_z(z.to_numpy(), entry, exit_), index=y.index)
    pair_ret = ry - beta * rx                      # P&L of long-y / short-beta*x
    p = _voltarget_pos(pos, pair_ret, ppy).shift(2).fillna(0.0)   # t+2 execution
    gross = p * pair_ret
    cost = p.diff().abs().fillna(0.0) * 2 * cost_bps / 1e4        # two legs
    daily = ((1 + (gross - cost)).resample("D").prod() - 1).dropna()
    return summarise(daily, 365)["sharpe_ann"]


def _voltarget_pos(pos, ret, ppy, tgt=0.15):
    scale = (tgt / (ret.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return pos * scale


def main():
    print("=== 1. Cross-sectional short-term REVERSAL (equity panel, dollar-neutral, net costs) ===")
    panel = pd.DataFrame({s: load_equity_daily(s, start="2012-01-01")["close"]
                          for s in PANEL}).dropna(how="all").ffill()
    for lb in (1, 2, 3, 5, 10, 21):
        sig = -panel.pct_change(lb)                # losers rank high -> long losers (reversal)
        gross, turn = xs_returns(panel, sig, top_frac=0.3)
        net = gross - turn * 1.0 / 1e4
        sh = summarise(_voltarget(net, 252), 252)["sharpe_ann"]
        print(f"  reversal lookback {lb:2d}d:  Sharpe {sh:+.2f}")

    print("\n=== 2. Pairs / spread MR (crypto, revert-to-mean exit, net costs) ===")
    for tf, ppy in (("1d", 365), ("4h", 6 * 365)):
        closes = {s: load_klines(s, tf, "2020-01", market="um")["close"]
                  for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
        for a, b in (("ETHUSDT", "BTCUSDT"), ("SOLUSDT", "BTCUSDT"), ("SOLUSDT", "ETHUSDT")):
            best = (-9, None)
            for lb in (30, 60, 90):
                for entry in (1.5, 2.0, 2.5):
                    sh = pairs_mr(closes[a], closes[b], lb, entry, 0.5, 5.0, ppy)
                    if sh > best[0]:
                        best = (sh, (lb, entry))
            print(f"  {a[:3]}/{b[:3]} {tf}:  best Sharpe {best[0]:+.2f}  (lookback {best[1][0]}, entry {best[1][1]})")


if __name__ == "__main__":
    main()
