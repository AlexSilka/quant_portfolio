"""ML layers for the cross-sectional sleeve, measured against the rule-based baseline.

Two honest experiments, both leakage-controlled (features stamped at bar t, labels purged from
training, all prediction strictly walk-forward / purged out-of-sample):

  A. Learning-to-rank — predict each name's cross-sectionally demeaned forward return from a
     multi-signal feature vector, long the top / short the bottom of the prediction. Compares
     several optimizers (Ridge, RandomForest, HistGradientBoosting, LightGBM regressor,
     LightGBM LambdaRank). ML's value here would be *combining* signals better than one momentum.

  B. Meta-label gate ("confidence factor") — predict P(the rule book wins the next period) from
     regime features and trade only high-probability periods. ML's value here is risk reduction
     (drawdown, months-in-profit), not a Sharpe boost.

    python scripts/xs/ml.py crypto_1d
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
import lightgbm as lgb  # noqa: E402

from src.config import CACHE_DIR, SEED, XS_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.xsect import mom, risk_adj_mom, xs_backtest, vol_target  # noqa: E402
from src.sleeves.xsect_ml import (expanding_predict, predictions_to_panel,  # noqa: E402
                                  rank_features, regime_features, stack_xy)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE, OUT = CACHE_DIR / "xs", XS_DIR
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
PPY = {"crypto": {"1d": 365, "4h": 6 * 365, "1h": 24 * 365},
       "stocks": {"1d": 252}, "fx": {"1d": 252}}
COST_BPS = {"crypto": 6.0, "stocks": 3.0, "fx": 1.0}
HOLD_D = 21          # holding / forward-label horizon in days (monthly)
TOP_FRAC = 0.3
APRIORI = {"crypto": dict(kind="riskadj", lb=30), "stocks": dict(kind="riskadj", lb=252, sk=7),
           "fx": dict(kind="riskadj", lb=252, sk=7)}


def apriori_signal(cfg, px, bpd):
    lb, sk = cfg["lb"] * bpd, cfg.get("sk", 0) * bpd
    return risk_adj_mom(px, lb, sk) if cfg["kind"] == "riskadj" else mom(px, lb, sk)


def lgb_reg():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.7, min_child_samples=100,
                             reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)


MODELS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "rf": lambda: RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=200,
                                        n_jobs=-1, random_state=SEED),
    "histgb": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03,
                                                    max_leaf_nodes=31, l2_regularization=1.0,
                                                    random_state=SEED),
    "lgbm": lgb_reg,
}


def backtest_signal(px, sig, adv, bpd, ppy, cost, rebal_d=HOLD_D):
    bt = xs_backtest(px, sig, top_frac=TOP_FRAC, weighting="equal", rebal=max(1, rebal_d * bpd),
                     cost_bps=cost, adv=adv, impact_k=0.1 if adv is not None else 0.0)
    netv = vol_target(bt["net"], ppy).dropna()
    s = summarise(netv, ppy)
    s["turnover"] = float(bt["turnover"].sum() / (len(px) / ppy))
    return s, netv


def run(tag: str):
    kind, tf = tag.split("_")
    px = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    advp = CACHE / f"{tag}_adv.parquet"
    adv = pd.read_parquet(advp) if advp.exists() else None
    bpd, ppy, cost = BARS_PER_DAY[tf], PPY[kind][tf], COST_BPS[kind]
    fwd = HOLD_D * bpd
    print(f"\n=== {tag}  ({px.shape[0]}×{px.shape[1]}) ===")

    # baseline: rule-based a-priori book
    base_sig = apriori_signal(APRIORI[kind], px, bpd)
    base_s, base_ret = backtest_signal(px, base_sig, adv, bpd, ppy, cost)
    print(f"  BASELINE (rule {APRIORI[kind]}): Sharpe {base_s['sharpe_ann']:+.2f}  "
          f"DD {base_s['max_dd']:+.0%}  months+ {base_s['months_in_profit']:.0%}  "
          f"turn {base_s['turnover']:.0f}x")

    # ── A. learning-to-rank ────────────────────────────────────────────────────────────────
    feats = rank_features(px, adv, bpd)
    X, y, ts = stack_xy(feats, px, fwd)
    print(f"  LTR design matrix: {X.shape[0]} rows × {X.shape[1]} features")
    # RandomForest is too slow past ~250k rows — drop it on the large intraday panels
    models = {k: v for k, v in MODELS.items() if not (k == "rf" and len(X) > 250_000)}
    rows = [{"model": "rule-baseline", "sharpe": base_s["sharpe_ann"], "max_dd": base_s["max_dd"],
             "months_in_profit": base_s["months_in_profit"], "turnover": base_s["turnover"]}]
    ml_rets = {}
    for name, factory in models.items():
        pred = expanding_predict(X, y, ts, factory, n_folds=6, embargo_bars=fwd)
        sig = predictions_to_panel(pred, px)
        s, ret = backtest_signal(px, sig, adv, bpd, ppy, cost)
        p5 = bootstrap_sharpe(ret, ppy, 400, SEED).get("sharpe_p5", np.nan) if s["sharpe_ann"] > 0.5 else np.nan
        rows.append({"model": f"LTR-{name}", "sharpe": s["sharpe_ann"], "mc_p5": p5,
                     "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
                     "turnover": s["turnover"]})
        ml_rets[name] = ret
        print(f"  LTR-{name:8s}: Sharpe {s['sharpe_ann']:+.2f}  P5 {p5:+.2f}  DD {s['max_dd']:+.0%}  "
              f"months+ {s['months_in_profit']:.0%}  turn {s['turnover']:.0f}x")

    # LTR ensemble (mean of model prediction ranks) — usually the most robust
    ens = pd.concat(ml_rets.values(), axis=1).mean(axis=1)
    es = summarise(ens.dropna(), ppy)
    print(f"  LTR-ENSEMBLE : Sharpe {es['sharpe_ann']:+.2f}  DD {es['max_dd']:+.0%}  "
          f"months+ {es['months_in_profit']:.0%}")

    # ── B. meta-label gate on the rule baseline ─────────────────────────────────────────────
    reg = regime_features(px, base_sig, base_ret.reindex(px.index).fillna(0.0), bpd)
    # label: does the book win over the NEXT fwd bars (forward sum, no overlap into the past)
    fwd_book = base_ret.reindex(px.index).fillna(0.0).rolling(fwd).sum().shift(-fwd)
    lbl = (fwd_book > 0).astype(int)
    g = reg.join(lbl.rename("y")).dropna()
    Xg, yg = g.drop(columns="y"), g["y"]
    tsg = pd.Series(Xg.index, index=Xg.index)

    def gate_predict():
        order = np.argsort(tsg.values)
        Xs, ys = Xg.iloc[order], yg.iloc[order]
        uniq = np.array(sorted(tsg.iloc[order].unique()))
        n_folds = 6
        bounds = [uniq[min(int(i * len(uniq) / (n_folds + 1)), len(uniq) - 1)] for i in range(n_folds + 2)]
        pr = pd.Series(np.nan, index=Xs.index)
        emb = (uniq[1] - uniq[0]) * fwd if len(uniq) > 1 else pd.Timedelta(0)
        tvals = pd.Series(Xs.index, index=Xs.index)
        for k in range(1, n_folds + 1):
            te0, te1 = bounds[k], bounds[k + 1]
            tr = tvals < (pd.Timestamp(te0) - emb)
            te = (tvals >= te0) & (tvals < te1)
            if tr.sum() < 200 or te.sum() == 0:
                continue
            m = LogisticRegression(max_iter=1000, C=0.5)
            m.fit(Xs[tr].to_numpy(), ys[tr].to_numpy())
            pr.iloc[np.flatnonzero(te.to_numpy())] = m.predict_proba(Xs[te].to_numpy())[:, 1]
        return pr.reindex(px.index)

    proba = gate_predict()
    for thr in (0.5, 0.55, 0.6):
        gate = (proba.shift(2) > thr).astype(float)     # gate lagged like execution
        gated = (base_ret.reindex(px.index) * gate).dropna()
        gs = summarise(gated, ppy)
        frac = float(gate.mean())
        print(f"  META-GATE p>{thr}: Sharpe {gs['sharpe_ann']:+.2f}  DD {gs['max_dd']:+.0%}  "
              f"months+ {gs['months_in_profit']:.0%}  (trades {frac:.0%} of the time)")
        rows.append({"model": f"meta-gate-{thr}", "sharpe": gs["sharpe_ann"], "max_dd": gs["max_dd"],
                     "months_in_profit": gs["months_in_profit"], "active_frac": frac})

    pd.DataFrame(rows).to_csv(OUT / f"ml_{tag}.csv", index=False)


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["crypto_1d"]):
        run(t)
    print("\nML OK")
