"""Residual-momentum (H5) — the ML layer, measured against the residual *rule* baseline.

The rule study (run_residmom.py) found residual momentum beats raw momentum on crypto and de-crashes
the equity leg. The ML question here is the same two the repo asks of every family, but pointed at H5:

  A. Learning-to-rank — can a model combining a feature vector beat the `idio_mom` *rule*? And the
     H5-specific ABLATION: does adding residual-momentum features to the raw-momentum feature set
     lift the ranker at all, or does raw momentum already span it (the corr-0.8 prediction)?
  B. Meta-label gate — does predicting P(the residual book wins next period) and trading only
     high-confidence periods cut the residual book's drawdown (risk reduction, not a Sharpe boost)?

Everything is leakage-controlled: features stamped at bar t, forward-return target cross-sectionally
demeaned (relative rank, not direction), all prediction strictly expanding/purged walk-forward. Reuses
`src/sleeves/xsect_ml.py` (the same harness as scripts/xs/ml.py) so the baseline is comparable to XSECT.

    python scripts/residmom/run_residmom_ml.py            # crypto_1d + stocks_broad_1d
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
import lightgbm as lgb  # noqa: E402

from src.config import CACHE_DIR, REPORTS_DIR, SEED  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import idio_mom, resid_mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.sleeves.xsect_ml import (expanding_predict, predictions_to_panel,  # noqa: E402
                                  rank_features, regime_features, stack_xy)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE, OUT = CACHE_DIR / "xs", REPORTS_DIR
SEED, HOLD_D = SEED, 21

# a-priori residual rule per asset (the H5 winner) — the baseline ML must beat, and the residual
# features (idio_mom at several formation windows + the single-window resid_mom) added to the ranker.
CFG = {
    "crypto_1d": dict(kind="crypto", ppy=365, cost=6.0, winsor=1.0, topn=100, imp=0.1,
                      form=30, beta=90, sk=0, tf=0.3, forms=(20, 30, 60)),
    "stocks_broad_1d": dict(kind="equity", ppy=252, cost=3.0, winsor=0.5, topn=100, imp=0.1,
                            form=252, beta=756, sk=7, tf=0.1, forms=(126, 252, 378)),
}


def lgb_reg():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8,
                             colsample_bytree=0.7, min_child_samples=100, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbose=-1)


MODELS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "rf": lambda: RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=200,
                                        n_jobs=-1, random_state=SEED),
    "histgb": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03,
                                                    max_leaf_nodes=31, l2_regularization=1.0,
                                                    random_state=SEED),
    "lgbm": lgb_reg,
}


def resid_features(px, cfg) -> dict:
    """Residual-momentum feature panels — the H5 signal, added on top of `rank_features`."""
    feats = {f"idio_{d}": idio_mom(px, d, cfg["beta"], cfg["sk"], market=None) for d in cfg["forms"]}
    feats[f"resid1w_{cfg['form']}"] = resid_mom(px, cfg["form"], cfg["sk"])
    return feats


def backtest_signal(px, sig, adv, cfg):
    bt = xs_backtest(px, sig, top_frac=cfg["tf"], weighting="equal", rebal=HOLD_D,
                     cost_bps=cfg["cost"], adv=adv, impact_k=cfg["imp"] if adv is not None else 0.0)
    net = vol_target(bt["net"], cfg["ppy"]).dropna()
    s = summarise(net, cfg["ppy"])
    s["turnover"] = float(bt["turnover"].sum() / (len(px) / cfg["ppy"]))
    return s, net


def ltr(px, adv, feats, cfg, fwd, label, rows):
    """One learning-to-rank pass over a feature dict; prints per-model OOS Sharpe, returns ensemble."""
    X, y, ts = stack_xy(feats, px, fwd)
    models = {k: v for k, v in MODELS.items() if not (k == "rf" and len(X) > 250_000)}
    print(f"  [{label}] design {X.shape[0]}×{X.shape[1]}  ({'+'.join(models)})")
    ml_rets = {}
    for name, factory in models.items():
        pred = expanding_predict(X, y, ts, factory, n_folds=6, embargo_bars=fwd)
        sig = top_n_liquid(predictions_to_panel(pred, px), adv, cfg["topn"]) if cfg["topn"] else predictions_to_panel(pred, px)
        s, ret = backtest_signal(px, sig, adv, cfg)
        p5 = bootstrap_sharpe(ret, cfg["ppy"], 400, SEED).get("sharpe_p5", np.nan) if s["sharpe_ann"] > 0.4 else np.nan
        ml_rets[name] = ret
        rows.append({"stage": f"LTR-{label}", "model": name, "sharpe": round(s["sharpe_ann"], 3),
                     "mc_p5": round(p5, 3) if p5 == p5 else None, "max_dd": round(s["max_dd"], 3),
                     "turnover": round(s["turnover"], 1)})
        print(f"    {name:8s} Sharpe {s['sharpe_ann']:+.2f}  P5 {p5:+.2f}  DD {s['max_dd']:+.0%}  turn {s['turnover']:.0f}x")
    ens = pd.concat(ml_rets.values(), axis=1).mean(axis=1)
    es = summarise(ens.dropna(), cfg["ppy"])
    rows.append({"stage": f"LTR-{label}", "model": "ENSEMBLE", "sharpe": round(es["sharpe_ann"], 3),
                 "max_dd": round(es["max_dd"], 3)})
    print(f"    ENSEMBLE Sharpe {es['sharpe_ann']:+.2f}  DD {es['max_dd']:+.0%}")
    return es["sharpe_ann"]


def run(tag):
    cfg = CFG[tag]
    px = bab.winsorize_panel(pd.read_parquet(CACHE / f"{tag}_close.parquet"), cfg["winsor"])
    advp = CACHE / f"{tag}_adv.parquet"
    adv = pd.read_parquet(advp).reindex_like(px) if advp.exists() else None
    fwd = HOLD_D
    print(f"\n{'='*78}\n{tag}  ({px.shape[0]}×{px.shape[1]})\n{'='*78}")

    # ── baseline: the residual RULE book (idio_mom a-priori) — what ML must beat ─────────────
    rule = top_n_liquid(idio_mom(px, cfg["form"], cfg["beta"], cfg["sk"], market=None), adv, cfg["topn"]) \
        if cfg["topn"] else idio_mom(px, cfg["form"], cfg["beta"], cfg["sk"], market=None)
    bs, base_ret = backtest_signal(px, rule, adv, cfg)
    print(f"  RULE idio_mom(form={cfg['form']},beta={cfg['beta']}): Sharpe {bs['sharpe_ann']:+.2f}  "
          f"DD {bs['max_dd']:+.0%}  months+ {bs['months_in_profit']:.0%}  turn {bs['turnover']:.0f}x")
    rows = [{"stage": "rule", "model": "idio_mom", "sharpe": round(bs["sharpe_ann"], 3),
             "max_dd": round(bs["max_dd"], 3), "months_in_profit": round(bs["months_in_profit"], 3)}]

    # ── A. learning-to-rank: ABLATION — raw features only vs raw + residual features ──────────
    base_feats = rank_features(px, adv, 1)
    s_raw = ltr(px, adv, base_feats, cfg, fwd, "raw-feats", rows)
    s_aug = ltr(px, adv, {**base_feats, **resid_features(px, cfg)}, cfg, fwd, "raw+resid-feats", rows)
    print(f"  ABLATION: LTR raw {s_raw:+.2f}  →  raw+residual {s_aug:+.2f}  (Δ {s_aug - s_raw:+.2f})   "
          f"vs RULE {bs['sharpe_ann']:+.2f}")

    # ── B. meta-label gate on the residual (idio) rule book ──────────────────────────────────
    reg = regime_features(px, idio_mom(px, cfg["form"], cfg["beta"], cfg["sk"], market=None),
                          base_ret.reindex(px.index).fillna(0.0), 1)
    fwd_book = base_ret.reindex(px.index).fillna(0.0).rolling(fwd).sum().shift(-fwd)
    g = reg.join((fwd_book > 0).astype(int).rename("y")).dropna()
    Xg, yg = g.drop(columns="y"), g["y"]
    tvals = pd.Series(Xg.index, index=Xg.index)
    uniq = np.array(sorted(tvals.unique()))
    proba = pd.Series(np.nan, index=Xg.index)
    if len(uniq) > 8:
        bounds = [uniq[min(int(i * len(uniq) / 7), len(uniq) - 1)] for i in range(8)]
        emb = (uniq[1] - uniq[0]) * fwd
        for k in range(1, 7):
            te0, te1 = bounds[k], bounds[k + 1]
            tr = tvals < (pd.Timestamp(te0) - emb)
            te = (tvals >= te0) & (tvals < te1)
            if tr.sum() < 200 or te.sum() == 0:
                continue
            m = LogisticRegression(max_iter=1000, C=0.5)
            m.fit(Xg[tr].to_numpy(), yg[tr].to_numpy())
            proba.iloc[np.flatnonzero(te.to_numpy())] = m.predict_proba(Xg[te].to_numpy())[:, 1]
    proba = proba.reindex(px.index)
    for thr in (0.5, 0.55, 0.6):
        gate = (proba.shift(2) > thr).astype(float)
        gated = (base_ret.reindex(px.index) * gate).dropna()
        gs = summarise(gated, cfg["ppy"])
        print(f"  META-GATE p>{thr}: Sharpe {gs['sharpe_ann']:+.2f}  DD {gs['max_dd']:+.0%}  "
              f"months+ {gs['months_in_profit']:.0%}  (trades {gate.mean():.0%})")
        rows.append({"stage": "meta-gate", "model": f"p>{thr}", "sharpe": round(gs["sharpe_ann"], 3),
                     "max_dd": round(gs["max_dd"], 3), "active_frac": round(float(gate.mean()), 3)})

    pd.DataFrame(rows).to_csv(OUT / f"residmom_ml_{tag}.csv", index=False)
    return bs["sharpe_ann"], s_raw, s_aug


def main():
    summ = {}
    for tag in ("crypto_1d", "stocks_broad_1d"):
        summ[tag] = run(tag)
    print(f"\n{'='*78}\nVERDICT (ML vs the residual rule)")
    for tag, (rule, raw, aug) in summ.items():
        print(f"  {tag:16s}: rule idio {rule:+.2f} | LTR raw {raw:+.2f} | LTR raw+resid {aug:+.2f} "
              f"| residual-features Δ {aug - raw:+.2f} | ML beats rule: {'yes' if aug > rule else 'no'}")
    print("RUN RESIDMOM ML OK")


if __name__ == "__main__":
    main()
