"""Does picking the best-backtesting assets help — or is the frozen full universe better? (task §2, §7)

The tempting move is to keep only the instruments where trend worked best and hard-wire them into the
book. This tests it honestly: rank every sleeve by its IN-SAMPLE (pre-2024-07) Sharpe, form top-K books,
and measure them on the HELD-OUT block against the full frozen universe and against random-K / bottom-K.
If past winners do not beat the full book out-of-sample, cherry-picking is overfitting, not selection.

    python scripts/trend/run_trend_universe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import sh  # noqa: E402
from src.metrics import max_drawdown  # noqa: E402

OOS = T.OOS_START


def book_stats(df: pd.DataFrame, cols) -> tuple[float, float]:
    port = df[cols].mean(axis=1).dropna()
    return sh(port), max_drawdown((1 + port).cumprod())


def main():
    p = T.CACHE / "sleeves_ema_reversal_asym_lag2_1d4h1h.parquet"
    df = pd.read_parquet(p)
    is_, oos = df[df.index < OOS], df[df.index >= OOS]
    # keep sleeves with enough IS and OOS history to rank & test fairly
    cols = [c for c in df.columns if is_[c].notna().sum() > 250 and oos[c].notna().sum() > 60]
    is_sh = {c: sh(is_[c]) for c in cols}
    oos_sh = {c: sh(oos[c]) for c in cols}
    ranked = sorted(cols, key=lambda c: is_sh[c], reverse=True)
    print(f"sleeves with enough IS+OOS history: {len(cols)}  (of {df.shape[1]})\n")

    # 1) do past (IS) winners stay winners? correlation of IS vs OOS per-sleeve Sharpe
    a = np.array([is_sh[c] for c in cols]); b = np.array([oos_sh[c] for c in cols])
    rho = float(np.corrcoef(a, b)[0, 1])
    print(f"IS→OOS per-sleeve Sharpe correlation: {rho:+.2f}  "
          f"({'past winners predict future' if rho > 0.3 else 'past winners do NOT predict future'})\n")

    # 2) top-K IS winners vs full universe vs random-K vs bottom-K, measured OOS
    full_is, full_dd_is = book_stats(is_, cols)
    full_oos, full_dd_oos = book_stats(oos, cols)
    print(f"{'portfolio':22s} {'IS Sharpe':>10s} {'OOS Sharpe':>11s} {'OOS maxDD':>10s}")
    print(f"{'FULL frozen universe':22s} {full_is:>+10.2f} {full_oos:>+11.2f} {full_dd_oos:>+10.1%}")
    rng = np.random.default_rng(T.SEED)
    for k in [5, 10, 20, 40]:
        if k > len(cols):
            continue
        top = ranked[:k]
        bot = ranked[-k:]
        rnd_oos = np.mean([book_stats(oos, list(rng.choice(cols, k, replace=False)))[0] for _ in range(50)])
        t_is, _ = book_stats(is_, top)
        t_oos, t_dd = book_stats(oos, top)
        b_oos, _ = book_stats(oos, bot)
        print(f"{'top-'+str(k)+' IS winners':22s} {t_is:>+10.2f} {t_oos:>+11.2f} {t_dd:>+10.1%}"
              f"   | random-{k} OOS {rnd_oos:+.2f}   bottom-{k} OOS {b_oos:+.2f}")

    print("\nreading:")
    print("  - if top-K IS winners do NOT beat the FULL book OOS, and their OOS maxDD is WORSE,")
    print("    then selecting assets by backtest is overfitting — the frozen universe + diversification wins.")


if __name__ == "__main__":
    main()
