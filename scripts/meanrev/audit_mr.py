"""Audit of the mean-reversion sleeve: is its poor edge-map score a real market fact, or an
artifact of the exit logic and fixed parameters?

Compares the book's triple-barrier MR (tight, ~1-bar-vol barrier) against a proper revert-to-mean
exit, across a small parameter grid, and reports holding length + turnover so the cause is visible.

    python scripts/meanrev/audit_mr.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.labels.triple_barrier import trailing_vol, triple_barrier_labels  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.pipeline import signal_events  # noqa: E402
from src.sleeves import mean_reversion  # noqa: E402

CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=1)
TF = {"1h": (24 * 365, 24), "4h": (6 * 365, 30), "1d": (365, 10)}


def mr_revert(close, lookback, entry_z, exit_z):
    """Continuous MR: enter at |z|>=entry, hold until z reverts to within exit_z of the mean."""
    m = close.rolling(lookback).mean()
    s = close.rolling(lookback).std()
    z = ((close - m) / (s + 1e-12)).to_numpy()
    pos = np.zeros(len(z))
    st = 0
    for i in range(len(z)):
        if not np.isnan(z[i]):
            if st == 0:
                st = 1 if z[i] <= -entry_z else (-1 if z[i] >= entry_z else 0)
            elif st == 1 and z[i] >= -exit_z:
                st = 0
            elif st == -1 and z[i] <= exit_z:
                st = 0
        pos[i] = st
    return pd.Series(pos, index=close.index)


def _ev(close, pos, fund, ppy):
    bt = backtest(close, vol_target(pos, close, 0.15, ppy), capital=500_000, funding=fund, **CC)
    daily = ((1 + bt["net_ret"]).resample("D").prod() - 1).dropna()
    turn = float(bt["position"].diff().abs().sum())
    return summarise(daily, 365)["sharpe_ann"], turn


def main():
    for tf, (ppy, hz) in TF.items():
        print(f"\n===== {tf} =====")
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            px = load_klines(sym, tf, "2020-01", market="um")
            close = px["close"]
            fund = load_funding(sym, "2020-01")["last_funding_rate"]

            # book's triple-barrier MR + its average holding length in bars
            side = mean_reversion.primary_side(close)
            ev = signal_events(side)
            lab = triple_barrier_labels(close, ev, trailing_vol(close, 100), 1.0, 1.0, hz)
            held = positions_from_events(close.index, side, lab["t1"], ev)
            pos_i = {t: i for i, t in enumerate(close.index)}
            holds = [pos_i[t1] - pos_i[t0] for t0, t1 in lab["t1"].items()]
            tb_sh, tb_turn = _ev(close, held, fund, ppy)

            # revert-to-mean MR, small grid
            best = (-9, None)
            for lb in (10, 20, 50):
                for ez in (1.5, 2.0):
                    for xz in (0.0, 0.5):
                        sh, turn = _ev(close, mr_revert(close, lb, ez, xz), fund, ppy)
                        if sh > best[0]:
                            best = (sh, (lb, ez, xz, turn))
            b = best[1]
            print(f"{sym:8s} triple-barrier MR: Sharpe {tb_sh:+.2f}  avg_hold {np.mean(holds):.1f} bars  "
                  f"turnover {tb_turn:.0f}   |   revert-to-mean best: Sharpe {best[0]:+.2f} "
                  f"(lb={b[0]}, entry={b[1]}, exit={b[2]}, turnover {b[3]:.0f})")


if __name__ == "__main__":
    main()
