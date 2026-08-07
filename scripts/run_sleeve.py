"""End-to-end run of one sleeve: mean-reversion primary + meta-label confidence gate.

Demonstrates the "confidence factor": a LightGBM meta-model predicts P(primary wins) and
we trade only high-confidence signals. Reports the baseline (primary only) vs the gated
sleeve, net of liquidity-aware costs and funding, on a held-out test split with a purge gap.

    python scripts/run_sleeve.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import lightgbm as lgb  # noqa: E402

from src.backtest.engine import backtest, positions_from_events  # noqa: E402
from src.config import CAPITAL_USD, SEED  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.features.engine import compute_features, pit_normalize  # noqa: E402
from src.labels.triple_barrier import (  # noqa: E402
    meta_labels, trailing_vol, triple_barrier_labels)
from src.metrics import summarise  # noqa: E402
from src.sleeves.mean_reversion import primary_side  # noqa: E402

CAPITAL = CAPITAL_USD
PPY = 24 * 365          # 1h bars per year
HORIZON = 24            # vertical barrier: ~1 day on 1h
EMBARGO = HORIZON       # purge gap between train and test (>= label horizon)
THRESHOLD = 0.60        # meta-gate confidence


def main() -> None:
    px = load_klines("BTCUSDT", "1h", "2021-01", market="um")
    bench = load_klines("ETHUSDT", "1h", "2021-01", market="um")["close"]
    fund = load_funding("BTCUSDT", "2021-01")["last_funding_rate"]
    close = px["close"]
    print(f"BTCUSDT 1h: {len(px)} bars  {close.index.min().date()}..{close.index.max().date()}")

    feats = pit_normalize(compute_features(px, benchmark=bench))
    side = primary_side(close, lookback=20, entry_z=1.5)
    events = side.index[side != 0]

    sigma = trailing_vol(close, span=100)
    labels = triple_barrier_labels(close, events, sigma, pt=1.0, sl=1.0, horizon=HORIZON)
    y = meta_labels(labels, side)
    X = feats.reindex(y.index).dropna()
    y = y.reindex(X.index)
    print(f"labelled events: {len(y)}  primary base win-rate: {y.mean():.1%}")

    # chronological split with an embargo gap (no future leaking into training)
    split_i = int(len(X) * 0.7)
    train_end = X.index[split_i]
    test_start = train_end + pd.Timedelta(hours=EMBARGO)
    Xtr, ytr = X[X.index <= train_end], y[X.index <= train_end]
    Xte = X[X.index >= test_start]
    yte = y[X.index >= test_start]
    print(f"train events: {len(Xtr)}  test events: {len(Xte)}  (embargo {EMBARGO}h)")

    clf = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             n_jobs=-1, verbose=-1)
    clf.fit(Xtr, ytr)
    p = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)

    gated = p >= THRESHOLD
    base_prec = yte.mean()
    meta_prec = yte[gated].mean() if gated.any() else float("nan")
    print(f"\nmeta-gate @ {THRESHOLD}: precision {base_prec:.1%} -> {meta_prec:.1%}  "
          f"({gated.mean():.0%} of signals kept, {int(gated.sum())} trades)")

    test_close = close[close.index >= test_start]
    t1 = labels["t1"]
    pos_base = positions_from_events(test_close.index, side, t1, Xte.index)
    pos_gate = positions_from_events(test_close.index, side, t1, Xte.index[gated.values])

    common = dict(capital=CAPITAL, commission_bps=5.0, half_spread_bps=1.0,
                  impact_k=0.1, funding=fund, exec_lag=2)
    results = {
        "baseline (primary only)": backtest(test_close, pos_base, **common),
        "meta-gated (confidence)": backtest(test_close, pos_gate, **common),
    }

    print("\n=== TEST window, net of liquidity-aware costs + funding ===")
    for name, bt in results.items():
        s = summarise(bt["net_ret"], PPY)
        print(f"{name:26s} Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  "
              f"months+ {s['months_in_profit']:.0%}  PSR>0 {s['psr_gt0']:.0%}  "
              f"totRet {s['total_return']:+.1%}")

    g = results["meta-gated (confidence)"]
    print(f"\ncost drag {g['cost'].sum():.4f}  funding drag {g['funding'].sum():+.4f}  "
          f"(fractions of capital, gated sleeve)")
    print("\nSLEEVE RUN OK")


if __name__ == "__main__":
    main()
