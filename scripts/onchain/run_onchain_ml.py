"""ML on the on-chain sleeve (H3) — does a model find alpha the linear on-chain books could not?

Two uses, each measured against the non-ML baseline (the linear on-chain VALUE book, whose in-sample
and purged walk-forward-OOS Sharpe are read from run_onchain.py's summary — run that first), each
leakage-controlled with purged/embargoed expanding CV
(train strictly before each fold minus an embargo ≥ the forward horizon — no training on the future,
no target-overlap leak). Reuses the repo's ML harness (`src/sleeves/xsect_ml.py`).

  A) ML RANKER — replace the linear on-chain rank with a model predicting each name's cross-sectionally
     demeaned forward return from the full on-chain feature panel. Grid: {6 models} × {feature set:
     on-chain / price / both} × {horizon 21d, 5d} × {top-N} × {regression, classification}. The decisive
     questions: (1) does any model beat the linear on-chain baseline? (2) does the *on-chain* feature set
     beat the *price* feature set — i.e. is there genuinely new information, or does the model just
     re-derive price momentum (read off feature importance)?

  B) META-GATE — a model predicts P(the on-chain value book is up next month) from panel regime state and
     scales exposure. Question: does an ML confidence gate cut the value book's −23% drawdown (the
     "confidence factor" story), even if it adds no Sharpe?

    python scripts/onchain/run_onchain_ml.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import ONCHAIN_DIR, REPORTS_DIR  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,  # noqa: E402
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import onchain as sig  # noqa: E402
from src.sleeves.xsect import mom  # noqa: E402
from src.sleeves.xsect_ml import (expanding_predict, predictions_to_panel,  # noqa: E402
                                  rank_features, regime_features, stack_xy)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.onchain.run_onchain import (_book, _load_onchain, _load_prices, _sh, _winsor,  # noqa: E402
                                 build_signals, PPY, SEED)

REP = REPORTS_DIR
FIG = REP / "figures"
rng = np.random.default_rng(SEED)
HEAD_N = 20   # same headline cross-section as the linear test


# ── model zoo: each a zero-arg factory (fresh estimator per fold) ───────────────────────────────
def _reg_models() -> dict:
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "rf": lambda: RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=200,
                                            max_features=0.5, n_jobs=-1, random_state=SEED),
        "extratrees": lambda: ExtraTreesRegressor(n_estimators=150, max_depth=5, min_samples_leaf=200,
                                                  max_features=0.5, n_jobs=-1, random_state=SEED),
        "histgbm": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                         l2_regularization=1.0, random_state=SEED),
        "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                              min_child_samples=200, reg_lambda=1.0, subsample=0.8,
                                              colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1),
    }


class _Proba:
    """Wrap a classifier so .predict returns P(up) — a continuous rank score for the book."""
    def __init__(self, clf):
        self.clf = clf
    def fit(self, X, y):
        self.clf.fit(X, y); return self
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


def _clf_models() -> dict:
    return {
        "logit_clf": lambda: _Proba(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))),
        "rf_clf": lambda: _Proba(RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=200,
                                                        max_features=0.5, n_jobs=-1, random_state=SEED)),
        "lgbm_clf": lambda: _Proba(lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                                      min_child_samples=200, reg_lambda=1.0, subsample=0.8,
                                                      colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)),
    }


def _clean(d: dict) -> dict:
    """±inf → NaN so stack_xy's not-NaN row mask drops them (logs of near-zero ratios make inf, which
    is not NaN and would otherwise reach the estimator and raise)."""
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in d.items()}


def _feature_sets(C, A, oc_p):
    """The three feature dicts whose comparison IS the 'new information?' test."""
    onchain = _clean(sig.ml_feature_panel(oc_p))
    price = _clean(rank_features(C, A, bpd=1))
    both = {**{f"oc_{k}": v for k, v in onchain.items()}, **{f"px_{k}": v for k, v in price.items()}}
    return {"onchain": onchain, "price": price, "both": both}


def _ml_book(C, A, feats, fwd_bars, model_factory, topn, classify=False):
    """features → purged-CV OOS predictions → wide signal → the SAME dollar-neutral book as the linear
    test. Returns (net series [OOS only], oos_coverage_frac). embargo ≥ horizon kills target overlap."""
    X, y, ts = stack_xy(feats, C, fwd_bars)
    if len(X) < 3000:
        return pd.Series(dtype=float), 0.0
    yt = (y > 0).astype(int) if classify else y
    embargo = fwd_bars + 5
    pred = expanding_predict(X, yt, ts, model_factory, n_folds=6, embargo_bars=embargo)
    panel = predictions_to_panel(pred.dropna(), C)
    net, _ = _book(C, A, panel, topn)
    cov = float(pred.notna().mean())
    return net, cov


def _linear_baseline() -> tuple[float, float]:
    """(in-sample, purged WF-OOS) Sharpe of the linear on-chain value book, from run_onchain.py's
    summary. Requires that run first — an ML comparison against a bar nobody measured is worthless."""
    p = ONCHAIN_DIR / "onchain_summary.json"
    if not p.exists():
        raise SystemExit(f"{p} missing — run scripts/onchain/run_onchain.py first (it sets the bar)")
    xs = json.loads(p.read_text())["cross_section"]
    return float(xs["headline"]["sharpe"]), float(xs["walk_forward"]["wf_oos"])


def run_ml_ranker(C, A, oc_p, S):
    fsets = _feature_sets(C, A, oc_p)
    regm, clfm = _reg_models(), _clf_models()

    # linear baselines (from the non-ML test) — the bar every ML book must clear. Read from that
    # run's artifact rather than pasted in, so the bar can never drift away from the run it names.
    lin_head, _ = _book(C, A, S["nvm_val"], HEAD_N)
    lin_is, lin_head_oos_ref = _linear_baseline()
    print(f"\n{'='*84}\nA) ML RANKER — models × feature-sets vs the linear on-chain baseline\n{'='*84}")
    print(f"  linear on-chain VALUE (nvm_val, top-{HEAD_N}): in-sample {lin_is:+.2f}, purged WF-OOS {lin_head_oos_ref:+.2f}")
    print(f"  (every ML Sharpe below is already OUT-OF-SAMPLE — expanding purged CV — so compare to {lin_head_oos_ref:+.2f})\n")

    trials, books = [], {}
    # ── main grid: 5 regressors × 3 feature sets @ 21d / top-20 ─────────────────────────────────
    print("  regression, horizon=21d, top-20 — net OOS Sharpe:")
    print("    " + "model".ljust(12) + "onchain   price     both")
    for mname, mf in regm.items():
        row = {}
        for fs in ("onchain", "price", "both"):
            net, cov = _ml_book(C, A, fsets[fs], 21, mf, HEAD_N)
            s = _sh(net); row[fs] = s
            trials.append(s)
            books[f"{mname}_{fs}_21"] = net
        print("    " + mname.ljust(12) + "".join(f"{row[fs]:<+10.2f}" for fs in ("onchain", "price", "both")))

    # ── classification twins (onchain features, 21d) ────────────────────────────────────────────
    print("\n  classification P(out-perform), onchain features, 21d, top-20:")
    for mname, mf in clfm.items():
        net, cov = _ml_book(C, A, fsets["onchain"], 21, mf, HEAD_N, classify=True)
        s = _sh(net); trials.append(s); books[f"{mname}_onchain_21"] = net
        print(f"    {mname:12s} {s:+.2f}")

    # ── robustness: best regressor family at other horizon / top-N ──────────────────────────────
    print("\n  robustness (lightgbm, onchain):")
    for (fwd, tn, lab) in [(5, HEAD_N, "horizon=5d top-20"), (21, 37, "horizon=21d top-37"),
                           (63, HEAD_N, "horizon=63d top-20")]:
        net, cov = _ml_book(C, A, fsets["onchain"], fwd, regm["lightgbm"], tn)
        s = _sh(net); trials.append(s); books[f"lightgbm_onchain_{fwd}_{tn}"] = net
        print(f"    {lab:22s} {s:+.2f}")

    return trials, books, fsets, lin_head


def feature_importance(C, A, oc_p):
    """Fit lightgbm on the FULL 'both' feature set (illustrative, in-sample) and read which features it
    leans on — does on-chain add signal, or does it re-derive price momentum?"""
    fsets = _feature_sets(C, A, oc_p)
    X, y, ts = stack_xy(fsets["both"], C, 21)
    m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=200,
                          reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(X.to_numpy(), y.to_numpy())
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    oc_share = imp[[c for c in imp.index if c.startswith("oc_")]].sum() / imp.sum()
    return imp, float(oc_share)


def meta_gate(C, A, S):
    """B) ML confidence gate on the linear value book — does it cut the −23% DD? Predict P(book up next
    21d) from panel regime features, scale exposure by the OOS probability. Purged expanding CV."""
    head, _ = _book(C, A, S["nvm_val"], HEAD_N)
    reg = regime_features(C, S["nvm_val"], head.fillna(0.0), bpd=1)
    fwd_book = head.shift(-21).rolling(21).sum()                    # next-21d book return
    df = pd.concat([reg, fwd_book.rename("fwd")], axis=1).dropna()
    X = df[reg.columns]; y = (df["fwd"] > 0).astype(int)
    ts = pd.Series(df.index, index=df.index)
    proba = expanding_predict(X, y, ts, lambda: _Proba(make_pipeline(
        StandardScaler(), LogisticRegression(C=0.3, max_iter=500))), n_folds=6, embargo_bars=26)
    gate = (proba.reindex(head.index) > 0.5).astype(float).shift(1).fillna(0.0)   # trade only when confident
    gated = (head * gate).dropna()
    base = head.reindex(gated.index)
    return {
        "base_sharpe": round(_sh(base), 2), "gated_sharpe": round(_sh(gated), 2),
        "base_maxdd": round(summarise(base.dropna(), PPY)["max_dd"], 3),
        "gated_maxdd": round(summarise(gated.dropna(), PPY)["max_dd"], 3),
        "frac_in_market": round(float(gate.reindex(gated.index).mean()), 2),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    Craw, A = _load_prices()
    C = _winsor(Craw, 1.0)
    oc_p = _load_onchain(C.index, C.columns)
    S = build_signals(C, oc_p)
    print(f"panel {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}")

    trials, books, fsets, lin_head = run_ml_ranker(C, A, oc_p, S)

    # ── DECISIVE: best ML book — does it beat the linear OOS, and does on-chain beat price? ───────
    # The on-chain side is scored over *every* on-chain-feature book, classifiers included — they
    # are the same features through the same purged-CV harness, and excluding them would quietly
    # report a weaker on-chain result than was actually achieved. The asymmetry is stated rather
    # than hidden: classifiers were run on on-chain features only, so this favours on-chain.
    grid_21 = {k: v for k, v in books.items() if k.endswith("_21")}
    best_oc = max(((k, _sh(v)) for k, v in grid_21.items() if "_onchain_" in k), key=lambda kv: kv[1])
    best_px = max(((k, _sh(v)) for k, v in grid_21.items() if "_price_" in k), key=lambda kv: kv[1])
    best_all = max(((k, _sh(v)) for k, v in books.items()), key=lambda kv: kv[1])
    print(f"\n{'='*84}\nDECISIVE")
    print(f"  best ML on-chain-features book: {best_oc[0]} {best_oc[1]:+.2f}  (regression + classifier)")
    print(f"  best ML price-features book:    {best_px[0]} {best_px[1]:+.2f}  (regression only — no clf arm was run)")
    print(f"  → on-chain features {'BEAT' if best_oc[1] > best_px[1] else 'do NOT beat'} price features "
          f"({best_oc[1]:+.2f} vs {best_px[1]:+.2f})")

    # orthogonalise the best on-chain-ML book vs price momentum + reversal (identical universe)
    b_pmom, _ = _book(C, A, mom(C, 30), HEAD_N)
    b_prev, _ = _book(C, A, -mom(C, 365, skip=30), HEAD_N)
    yb = books[best_oc[0]]
    O = pd.DataFrame({"y": yb, "pmom": b_pmom, "prev": b_prev}).dropna()
    X = np.column_stack([np.ones(len(O)), O["pmom"], O["prev"]])
    coef, *_ = np.linalg.lstsq(X, O["y"].to_numpy(), rcond=None)
    resid = O["y"].to_numpy() - X @ coef
    dof = max(len(O) - 3, 1)
    tstat = coef[0] / np.sqrt((resid @ resid / dof) * np.linalg.inv(X.T @ X)[0, 0])
    print(f"  best on-chain-ML alpha over price mom+reversal: α {coef[0]*PPY:+.3f}/yr  t={tstat:+.2f}"
          f"   → {'adds edge' if tstat > 2 else 'no edge over price'}")

    imp, oc_share = feature_importance(C, A, oc_p)
    print(f"\n  feature importance (lightgbm on both sets): on-chain features = {oc_share:.0%} of total gain")
    print("    top-8:", ", ".join(f"{k}={v}" for k, v in imp.head(8).items()))

    # ── MC + deflated Sharpe on the best book, at the TRUE trial count ───────────────────────────
    yb_c = yb.dropna()
    mc = bootstrap_sharpe(yb_c, PPY, 1000, SEED) if len(yb_c) > 100 else {}
    n_trials = len(trials)
    var_tr = float(np.var([t / np.sqrt(PPY) for t in trials if np.isfinite(t)]))
    dsr = deflated_sharpe(yb_c.mean() / yb_c.std(ddof=1), len(yb_c), yb_c.skew(), yb_c.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8))
    print(f"\n  best ML book {best_all[0]} {best_all[1]:+.2f}: MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f} "
          f"| deflated SR (N={n_trials} ML trials) {dsr:.2f}")

    gate = meta_gate(C, A, S)
    print(f"\n{'='*84}\nB) META-GATE on the linear value book (does ML cut the −23% DD?)")
    print(f"  base   Sharpe {gate['base_sharpe']:+.2f}  maxDD {gate['base_maxdd']:+.1%}")
    print(f"  gated  Sharpe {gate['gated_sharpe']:+.2f}  maxDD {gate['gated_maxdd']:+.1%}  "
          f"(in market {gate['frac_in_market']:.0%} of the time)")

    lin_is, lin_oos = _linear_baseline()
    summ = {
        "linear_baseline": {"in_sample": lin_is, "purged_wf_oos": lin_oos},
        "ml_ranker_best_onchain": {"config": best_oc[0], "oos_sharpe": round(best_oc[1], 3)},
        "ml_ranker_best_price": {"config": best_px[0], "oos_sharpe": round(best_px[1], 3)},
        "onchain_beats_price": bool(best_oc[1] > best_px[1]),
        "best_book_alpha_over_price": {"alpha_ann": round(float(coef[0] * PPY), 4), "t": round(float(tstat), 2)},
        "feature_importance_onchain_share": round(oc_share, 3),
        "feature_importance_top8": {k: int(v) for k, v in imp.head(8).items()},
        "best_book": {"config": best_all[0], "oos_sharpe": round(best_all[1], 3),
                      "mc_p5": mc.get("sharpe_p5"), "deflated_sharpe": round(dsr, 3), "n_trials": n_trials},
        "all_trials_oos_sharpe": {k: round(_sh(v), 3) for k, v in books.items()},
        "meta_gate": gate,
    }
    (ONCHAIN_DIR / "onchain_ml_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    pd.DataFrame({k: v for k, v in books.items()}).to_parquet(ONCHAIN_DIR / "onchain_ml_returns.parquet")
    _figure(summ, books, imp)

    print(f"\n{'='*84}\nVERDICT")
    print(f"  best ML (OOS) {best_all[1]:+.2f} vs linear OOS {lin_oos:+.2f} / in-sample {lin_is:+.2f}; "
          f"on-chain {'>' if summ['onchain_beats_price'] else '≤'} price features; "
          f"alpha-over-price t={tstat:+.2f}; on-chain feat share {oc_share:.0%}")
    print("RUN ONCHAIN-ML OK")


def _figure(summ, books, imp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("On-chain ML (H3) — does a model find alpha the linear books could not? (OOS, purged CV)",
                 fontsize=12, fontweight="bold")

    # (1) model × feature-set OOS Sharpe (regression, 21d)
    a = ax[0, 0]
    models = ["ridge", "rf", "extratrees", "histgbm", "lightgbm"]
    fss = ["onchain", "price", "both"]
    x = np.arange(len(models)); w = 0.26
    for i, fs in enumerate(fss):
        vals = [summ["all_trials_oos_sharpe"].get(f"{m}_{fs}_21", np.nan) for m in models]
        a.bar(x + (i - 1) * w, vals, w, label=fs)
    lin_oos = summ["linear_baseline"]["purged_wf_oos"]
    a.axhline(0, color="k", lw=0.6)
    a.axhline(lin_oos, color="r", ls="--", lw=1, label=f"linear OOS {lin_oos:+.2f}")
    a.set_xticks(x); a.set_xticklabels(models, rotation=20, fontsize=8)
    a.set_title("ML ranker OOS Sharpe: model × feature set (21d)"); a.legend(fontsize=7)

    # (2) on-chain vs price feature set, best per
    a = ax[0, 1]
    labs = ["best on-chain\nfeatures", "best price\nfeatures"]
    vals = [summ["ml_ranker_best_onchain"]["oos_sharpe"], summ["ml_ranker_best_price"]["oos_sharpe"]]
    a.bar(labs, vals, color=["#2b6", "#68a"]); a.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    a.set_title(f"New info? on-chain {'>' if summ['onchain_beats_price'] else '≤'} price")

    # (3) feature importance top-12
    a = ax[1, 0]
    top = imp.head(12)[::-1]
    a.barh(top.index, top.values, color=["#2b6" if k.startswith("oc_") else "#a86" for k in top.index])
    a.set_title(f"lightgbm importance (green=on-chain, {summ['feature_importance_onchain_share']:.0%} of gain)")
    a.tick_params(axis="y", labelsize=7)

    # (4) meta-gate DD
    a = ax[1, 1]
    g = summ["meta_gate"]
    labs = ["base", "gated"]
    a.bar(labs, [g["base_maxdd"], g["gated_maxdd"]], color=["#a33", "#3a6"])
    a.set_title(f"Meta-gate maxDD (Sharpe {g['base_sharpe']:+.2f}→{g['gated_sharpe']:+.2f})")
    for i, v in enumerate([g["base_maxdd"], g["gated_maxdd"]]):
        a.text(i, v, f"{v:.0%}", ha="center", va="top", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "onchain_ml.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
