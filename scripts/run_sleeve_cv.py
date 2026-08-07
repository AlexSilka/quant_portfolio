"""Mean-reversion sleeve under purged CV + Monte Carlo — the honest robustness picture.

Replaces the single train/test split with a purged, embargoed K-fold: the meta-model
predicts out-of-sample across every fold, predictions are stitched into one OOS series, and
that series is backtested net of costs+funding. Reports the per-fold Sharpe distribution and
stationary-bootstrap P5/P50/P95 bands.

    python scripts/run_sleeve_cv.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402

from src.backtest.engine import backtest, positions_from_events  # noqa: E402
from src.config import CACHE_DIR, CAPITAL_USD, SEED  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.features.engine import compute_features, pit_normalize  # noqa: E402
from src.labels.triple_barrier import (  # noqa: E402
    meta_labels, trailing_vol, triple_barrier_labels)
from src.metrics import sharpe, summarise  # noqa: E402
from src.sleeves.mean_reversion import primary_side  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.validation.purged_cv import cv_oos_predictions  # noqa: E402

CAPITAL = CAPITAL_USD
PPY = 24 * 365
HORIZON = 24
EMBARGO = pd.Timedelta(hours=HORIZON)
THRESHOLD = 0.60
N_SPLITS = 6


def model_factory():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                              n_jobs=-1, verbose=-1)


def _features_cached(px, bench):
    cache = CACHE_DIR / "features_BTCUSDT_1h.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    feats = pit_normalize(compute_features(px, benchmark=bench))
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


def main() -> None:
    px = load_klines("BTCUSDT", "1h", "2021-01", market="um")
    bench = load_klines("ETHUSDT", "1h", "2021-01", market="um")["close"]
    fund = load_funding("BTCUSDT", "2021-01")["last_funding_rate"]
    close = px["close"]

    feats = _features_cached(px, bench)
    side = primary_side(close, lookback=20, entry_z=1.5)
    events = side.index[side != 0]
    sigma = trailing_vol(close, span=100)
    labels = triple_barrier_labels(close, events, sigma, pt=1.0, sl=1.0, horizon=HORIZON)
    y = meta_labels(labels, side)
    X = feats.reindex(y.index).dropna()
    y = y.reindex(X.index)
    print(f"BTCUSDT 1h  labelled events: {len(y)}  base win-rate: {y.mean():.1%}")

    oos_p, folds = cv_oos_predictions(X, y, labels["t1"], model_factory,
                                      n_splits=N_SPLITS, embargo=EMBARGO)
    oos_p = oos_p.dropna()
    print(f"purged {N_SPLITS}-fold OOS predictions: {len(oos_p)}  "
          f"(train sizes {folds['n_train'].min()}..{folds['n_train'].max()})")

    gated = oos_p[oos_p >= THRESHOLD].index
    print(f"gate @ {THRESHOLD}: precision {y.reindex(oos_p.index).mean():.1%} -> "
          f"{y.reindex(gated).mean():.1%}  ({len(gated)}/{len(oos_p)} kept)")

    t1 = labels["t1"]
    pos_base = positions_from_events(close.index, side, t1, oos_p.index)
    pos_gate = positions_from_events(close.index, side, t1, gated)
    common = dict(capital=CAPITAL, commission_bps=5.0, half_spread_bps=1.0,
                  impact_k=0.1, funding=fund, exec_lag=2)
    bt_base = backtest(close, pos_base, **common)
    bt_gate = backtest(close, pos_gate, **common)

    print("\n=== full OOS, net of costs + funding ===")
    for name, bt in [("baseline (primary)", bt_base), ("meta-gated", bt_gate)]:
        s = summarise(bt["net_ret"], PPY)
        print(f"{name:20s} Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  "
              f"months+ {s['months_in_profit']:.0%}  PSR>0 {s['psr_gt0']:.0%}  "
              f"totRet {s['total_return']:+.1%}")

    # per-fold OOS Sharpe distribution (gated sleeve)
    fold_sr = []
    for _, row in folds.iterrows():
        seg = bt_gate["net_ret"].loc[row["test_start"]:row["test_end"]]
        fold_sr.append(sharpe(seg, PPY))
    fold_sr = np.array(fold_sr)
    print(f"\nper-fold OOS Sharpe (gated): mean {fold_sr.mean():+.2f}  "
          f"std {fold_sr.std():.2f}  min {fold_sr.min():+.2f}  max {fold_sr.max():+.2f}")

    mc = bootstrap_sharpe(bt_gate["net_ret"], PPY, n_reps=1000, seed=SEED)
    if mc:
        print(f"Monte Carlo Sharpe (stationary bootstrap, block~{mc['block_len']:.0f}): "
              f"P5 {mc['sharpe_p5']:+.2f}  P50 {mc['sharpe_p50']:+.2f}  P95 {mc['sharpe_p95']:+.2f}")

    print("\nSLEEVE CV RUN OK")


if __name__ == "__main__":
    main()
