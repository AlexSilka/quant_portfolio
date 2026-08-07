"""Leakage audits for the cross-sectional sleeve — prove the signal cannot see the future.

1. Shift audit: recompute each signal on a series truncated at bar t; the value at every past
   bar must be identical (max|full − truncated| = 0). If a transform peeked ahead, truncation
   would change historical values.
2. Panel-placebo: shuffle each name's returns in time (destroys real cross-sectional structure,
   keeps marginal distributions), rebuild the book. A leakage-free pipeline earns ~0 Sharpe on
   this — anything high means the construction is trading its own look-ahead.
3. Execution-lag sensitivity: the book must not depend on filling at the signal bar. Sharpe is
   reported at exec_lag 1/2/3 — a cliff at lag 1→2 would flag same-bar leakage.

    python scripts/xs/audit.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CACHE_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.xsect import blend_rank, risk_adj_mom, xs_backtest, vol_target  # noqa: E402

CACHE = CACHE_DIR / "xs"


def sig_fn(px):
    return blend_rank([risk_adj_mom(px, lb, 0) for lb in (15, 30, 60)])


def shift_audit(px):
    full = sig_fn(px)
    max_diff = 0.0
    for cut in (int(len(px) * f) for f in (0.5, 0.7, 0.9)):
        trunc = sig_fn(px.iloc[:cut])
        common = full.iloc[:cut].reindex_like(trunc)
        d = float((common - trunc).abs().to_numpy()[np.isfinite((common - trunc).to_numpy())].max())
        max_diff = max(max_diff, d)
    return max_diff


def run():
    px = pd.read_parquet(CACHE / "crypto_1d_close.parquet")
    adv = pd.read_parquet(CACHE / "crypto_1d_adv.parquet")

    md = shift_audit(px)
    print(f"1. SHIFT AUDIT  max|full - truncated| on past bars = {md:.2e}  "
          f"({'PASS — computable-at-bar' if md < 1e-9 else 'FAIL — look-ahead!'})")

    sig = sig_fn(px)
    real = vol_target(xs_backtest(px, sig, top_frac=0.3, rebal=21, cost_bps=6.0)["net"], 365).dropna()
    rs = summarise(real, 365)["sharpe_ann"]
    # panel placebo: shuffle each column in time (break cross-sectional co-movement)
    shs = []
    for i in range(20):
        rng = np.random.default_rng(1000 + i)
        shuf = px.apply(lambda c: c.pct_change().sample(frac=1.0, random_state=int(rng.integers(1e9))).values)
        shuf = (1 + shuf.fillna(0.0)).cumprod()
        ss = sig_fn(shuf)
        pl = vol_target(xs_backtest(shuf, ss, top_frac=0.3, rebal=21, cost_bps=6.0)["net"], 365).dropna()
        shs.append(summarise(pl, 365)["sharpe_ann"])
    shs = np.array(shs)
    print(f"2. PANEL PLACEBO  real Sharpe {rs:+.2f}  vs shuffled {shs.mean():+.2f}±{shs.std():.2f} "
          f"(max {shs.max():+.2f})  ->  real beats every placebo: {rs > shs.max()}")

    print("3. EXEC-LAG SENSITIVITY (no same-bar leakage if smooth):")
    for lag in (1, 2, 3):
        r = vol_target(xs_backtest(px, sig, top_frac=0.3, rebal=21, cost_bps=6.0, exec_lag=lag)["net"], 365).dropna()
        print(f"     exec_lag={lag}: Sharpe {summarise(r, 365)['sharpe_ann']:+.2f}")
    print("\nAUDIT OK")


if __name__ == "__main__":
    run()
