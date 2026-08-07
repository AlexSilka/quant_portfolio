"""Feature-family analysis for the trend meta-model (task §4, §12): predictive strength and which
feature families the model actually uses to tell winning trend trades from losing ones.

Pools every EMA-50/200 chandelier trade across the core-10 × {1d,4h,1h} sleeves, trains a LightGBM
to predict P(trade wins) on the 82-feature library, and reports (a) held-out AUC (train pre-2024-07,
test on the OOS block — the honest predictive strength), and (b) gain-importance aggregated by the
brief's feature families, so we can say which families survived and which contributed nothing.

    python scripts/trend/run_trend_features.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_ml import models  # noqa: E402
from scripts.trend.run_trend_ml import CORE10, TFS_ML, trend_sleeve_data  # noqa: E402
from src.config import OOS_START  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OOS = OOS_START

# map a feature name to the brief's family (§4)
FAMILY = [
    ("trend/MA", ("px_sma", "px_ema", "sma_slope", "sma_50_200")),
    ("momentum/ROC", ("ret_", "logret_")),
    ("mean-reversion", ("zscore", "bb_pctb", "vwap_dist")),
    ("volatility", ("realvol", "parkinson", "garmanklass", "atr", "vol_of_vol", "vol_regime")),
    ("range/breakout", ("donchian_pos", "breakout_up", "breakout_dn", "range_width")),
    ("volume/flow", ("obv", "dollar_vol_z", "vol_z", "taker_imbalance")),
    ("oscillator", ("rsi", "stoch", "williams", "cci")),
    ("statistical", ("autocorr", "variance_ratio", "hurst", "entropy")),
    ("higher-moment", ("skew", "kurt", "tail_ratio")),
    ("cross-asset", ("rs_bench", "beta", "corr")),
    ("calendar", ("hour", "dayofweek", "min_since_day_open")),
]


def family_of(col: str) -> str:
    for fam, prefixes in FAMILY:
        if any(col.startswith(p) or col == p.rstrip("_") for p in prefixes):
            return fam
    return "other"


def main():
    print("=== Feature-family analysis — trend meta-model (EMA+chandelier, core-10 × 1d/4h/1h) ===\n")
    Xs, ys = [], []
    for tf in TFS_ML:
        for sym in CORE10:
            d = trend_sleeve_data(sym, tf)
            if d is None:
                continue
            _, _, X, y = d
            Xs.append(X)
            ys.append(y)
    X = pd.concat(Xs)
    y = pd.concat(ys)
    X = X.loc[:, X.notna().mean() > 0.5].dropna(axis=0)
    y = y.reindex(X.index)
    print(f"pooled trend trades: {len(X)}  features: {X.shape[1]}  base win rate: {y.mean():.0%}")

    # predictive strength: train pre-OOS, test on the held-out block
    tr, te = X.index < OOS, X.index >= OOS
    model = models()["lightgbm"]()
    model.fit(X[tr], y[tr])
    auc_oos = roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]) if y[te].nunique() > 1 else float("nan")
    auc_is = roc_auc_score(y[tr], model.predict_proba(X[tr])[:, 1])
    print(f"meta-model AUC: in-sample {auc_is:.3f}  held-out OOS {auc_oos:.3f}  "
          f"({'edge' if auc_oos > 0.53 else 'weak/none'} — 0.5 = coin flip)\n")

    # importance by family (gain), from a full-sample fit
    full = models()["lightgbm"]()
    full.fit(X, y)
    imp = pd.Series(full.booster_.feature_importance(importance_type="gain"), index=X.columns)
    imp = imp / imp.sum()
    fam_imp = imp.groupby([family_of(c) for c in imp.index]).sum().sort_values(ascending=False)
    print("feature-family importance (share of total gain):")
    for fam, v in fam_imp.items():
        top = imp[[c for c in imp.index if family_of(c) == fam]].sort_values(ascending=False)
        print(f"  {fam:16s} {v:6.1%}   top: {', '.join(top.head(3).index)}")

    survived = fam_imp[fam_imp >= 0.05].index.tolist()
    contributed_nothing = fam_imp[fam_imp < 0.02].index.tolist()
    print(f"\nsurvived (≥5% gain): {survived}")
    print(f"contributed ~nothing (<2%): {contributed_nothing}")

    (bo.REPORTS / "trend" / "trend_features.json").write_text(json.dumps({
        "n_trades": int(len(X)), "n_features": int(X.shape[1]), "base_win_rate": float(y.mean()),
        "auc_is": float(auc_is), "auc_oos": float(auc_oos),
        "family_importance": {k: float(v) for k, v in fam_imp.items()},
        "top_features": {c: float(imp[c]) for c in imp.sort_values(ascending=False).head(15).index},
        "survived": survived, "contributed_nothing": contributed_nothing,
    }, indent=2, default=float))
    print("\nwrote reports/trend/trend_features.json")


if __name__ == "__main__":
    main()
