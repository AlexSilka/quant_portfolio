"""Walk-forward parameter selection + parameter-sensitivity (Task A §10, and the interview
set-piece: 'walk forward, and only then hyperparameter tune').

For each family we define an a-priori grid, then:
  1. Sensitivity surface — every grid point's full-sample Sharpe (is the edge a broad plateau or
     a lucky spike?).
  2. Walk-forward selection — on each train window pick the best grid point, apply it to the next
     out-of-sample block, stitch the OOS returns. This is the honest number: it pays the cost of
     choosing parameters, unlike peak-picking on the whole sample.

Demonstrated on trend (works) vs mean-reversion (dead across the whole surface).

    python scripts/run_wfo.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.config import CAPITAL_USD, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import momentum  # noqa: E402
from scripts.meanrev.audit_mr import mr_revert  # noqa: E402

CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
PPY = 6 * 365  # 4h
TREND_GRID = [(f, s) for f in (10, 20, 30, 50) for s in (100, 150, 200) if f < s]
MR_GRID = [(lb, ez, xz) for lb in (10, 20, 50) for ez in (1.5, 2.0, 2.5) for xz in (0.0, 0.5)]


def trend_pos(close, p):
    return momentum.primary_side(close, p[0], p[1])


def mr_pos(close, p):
    return mr_revert(close, p[0], p[1], p[2])


def daily_ret(close, pos, fund):
    bt = backtest(close, vol_target(pos, close, VOL_TARGET_ANNUAL, PPY), capital=CAPITAL_USD, funding=fund, **CC)
    return ((1 + bt["net_ret"]).resample("D").prod() - 1).dropna()


def sharpe_of(close, pos, fund):
    return summarise(daily_ret(close, pos, fund), 365)["sharpe_ann"]


def walk_forward(close, fund, grid, pos_fn, n_folds=5):
    """Stitch OOS: on each train block pick the best grid point, apply on the next block."""
    idx = close.index
    bounds = [idx[min(int(i * len(idx) / (n_folds + 1)), len(idx) - 1)] for i in range(n_folds + 2)]
    oos = []
    for k in range(1, n_folds + 1):
        tr = close.loc[:bounds[k]]
        te0, te1 = bounds[k], bounds[k + 1]
        best = max(grid, key=lambda p: sharpe_of(tr, pos_fn(tr, p), fund))
        seg = daily_ret(close, pos_fn(close, best), fund).loc[te0:te1]
        oos.append(seg)
    return pd.concat(oos)


def main():
    close = load_klines("BTCUSDT", "4h", "2020-01", market="um")["close"]
    fund = load_funding("BTCUSDT", "2020-01")["last_funding_rate"]

    for name, grid, pos_fn in [("TREND", TREND_GRID, trend_pos), ("MEAN-REV", MR_GRID, mr_pos)]:
        surf = np.array([sharpe_of(close, pos_fn(close, p), fund) for p in grid])
        wf = summarise(walk_forward(close, fund, grid, pos_fn), 365)["sharpe_ann"]
        print(f"\n{name}  (BTCUSDT 4h, net of costs)")
        print(f"  sensitivity surface: {len(grid)} configs  "
              f"Sharpe min {surf.min():+.2f} / median {np.median(surf):+.2f} / max {surf.max():+.2f}")
        print(f"  fraction of grid positive: {(surf > 0).mean():.0%}")
        print(f"  in-sample BEST (peak-picking, overfit): {surf.max():+.2f}")
        print(f"  WALK-FORWARD selected (honest OOS):     {wf:+.2f}")
    print("\nWFO OK")


if __name__ == "__main__":
    main()
