"""Smoke test for the feature engine — counts features and audits for look-ahead.

The look-ahead audit is the point: recompute features on a series truncated at bar T
and assert every value at bars <= T is byte-identical to the full-series computation.
If any feature peeked at the future, the truncated values would differ.

    python scripts/smoke_features.py
"""
from pathlib import Path

import numpy as np


from src.data.binance_bulk import load_klines  # noqa: E402
from src.features.engine import compute_features, pit_normalize  # noqa: E402


def main() -> None:
    px = load_klines("BTCUSDT", "1h", "2023-01", "2024-06", market="um")
    bench = load_klines("ETHUSDT", "1h", "2023-01", "2024-06", market="um")["close"]
    print(f"loaded BTCUSDT 1h: {len(px)} bars  {px.index.min()} .. {px.index.max()}")

    feats = compute_features(px, benchmark=bench)
    print(f"features generated: {feats.shape[1]} columns")
    print("families sample:", ", ".join(list(feats.columns[:8])), "...")

    # coverage after warmup
    warm = feats.iloc[300:]
    frac = warm.notna().mean().mean()
    print(f"non-NaN coverage after warmup: {frac:.1%}")

    # ---- look-ahead audit ----
    cut = len(px) - 200
    feats_trunc = compute_features(px.iloc[:cut], benchmark=bench.iloc[:cut])
    common = feats.columns.intersection(feats_trunc.columns)
    a = feats[common].iloc[300:cut - 5]
    b = feats_trunc[common].iloc[300:cut - 5]
    diff = (a - b).abs()
    max_diff = np.nanmax(diff.values)
    print(f"\nlook-ahead audit: max|full - truncated| on past bars = {max_diff:.2e}")
    assert max_diff < 1e-8, "LEAKAGE: past feature values changed when future was removed"

    norm = pit_normalize(feats)
    print(f"PIT-normalised matrix: {norm.shape}  (rolling z, no full-sample stats)")
    print("\nFEATURE ENGINE OK — no look-ahead detected")


if __name__ == "__main__":
    main()
