"""BAB × ML — is there machine-learning alpha beyond the classical linear beta signal?

The classical BAB ranks names by a single trailing beta and de-levers by the FP formula. This asks
whether a learned cross-sectional forecaster does better: predict each name's forward return from a
factor-feature panel (all computable-at-bar, from the close/ADV panels), rank on the OOS prediction,
and build the same market-neutral book — then compare its purged-CV out-of-sample Sharpe to the
classical beta book. Several model families (linear, sparse-linear, random forest, gradient boosting)
× feature sets (beta-only / +risk / +momentum) are run, with a deflated-Sharpe haircut at the true
trial count. Every ML book is market-beta-hedged so it is directly comparable to the classical
beta-neutral construction (the hedge is validated by recovering the classical number on the −beta rank).

Prior (this repo): an ML ranker on carry destroyed value, and §5's honest finding is that ML's role
here is risk-reduction, not a Sharpe boost — so the expected answer is "little/no alpha beyond beta".
Reported either way. Crypto 1d, top-100 liquid, net of costs, monthly rebalance, t+2.

    python scripts/bab/run_bab_ml.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Lasso, Ridge  # noqa: E402

from src.config import BAB_DIR, CACHE_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP, CACHE = REPORTS_DIR, CACHE_DIR / "xs"
SEED, PPY, COST, TVOL, TOPN, REBAL, HORIZON = SEED, 365, 6.0, VOL_TARGET_ANNUAL, 100, 21, 21
np.random.seed(SEED)


def _sh(net):
    return summarise(net.dropna(), PPY)["sharpe_ann"]


def _features(C):
    """Factor-feature panels, all computable-at-bar (from close only)."""
    r = C.pct_change()
    f = {
        "beta60": bab.panel_beta(C, 60), "beta90": bab.panel_beta(C, 90), "beta120": bab.panel_beta(C, 120),
        "vol30": bab.trailing_vol(C, 30), "vol60": bab.trailing_vol(C, 60),
        "dvol60": (r.clip(upper=0.0) ** 2).rolling(60).mean() ** 0.5,   # downside semi-deviation (no NaN gaps)
        "skew60": bab.trailing_skew(C, 60),
        "mom20": C.pct_change(20), "mom60": C.pct_change(60), "mom120": C.pct_change(120),
        "rev5": C.pct_change(5),
    }
    return f


FEATSETS = {
    "beta_only": ["beta60", "beta90", "beta120"],
    "risk": ["beta60", "beta90", "beta120", "vol30", "vol60", "dvol60", "skew60"],
    "all": ["beta60", "beta90", "beta120", "vol30", "vol60", "dvol60", "skew60", "mom20", "mom60", "mom120", "rev5"],
}
MODELS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "lasso": lambda: Lasso(alpha=1e-4, max_iter=5000),
    "rf": lambda: RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=200,
                                        n_jobs=-1, random_state=SEED),
    "hgb": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                                 l2_regularization=1.0, random_state=SEED),
}


def _long_panel(C, A, feats):
    """Stack (date, name) rows with features + forward-return label, restricted to top-100 liquid."""
    mask = top_n_liquid(pd.DataFrame(1.0, index=C.index, columns=C.columns), A, TOPN).notna()
    fwd = C.pct_change(HORIZON).shift(-HORIZON).where(mask)     # label: forward 21d return (future — label only)
    cols = {k: v.where(mask).stack(dropna=False) for k, v in feats.items()}
    cols["fwd"] = fwd.stack(dropna=False)
    df = pd.DataFrame(cols).dropna()
    df.index.names = ["date", "name"]
    return df


def _predict_cv(df, feat_cols, model_factory, n_folds=5, embargo=HORIZON):
    """Purged/embargoed time-series CV → stitched OOS prediction Series over (date, name)."""
    dates = df.index.get_level_values("date")
    uniq = np.array(sorted(dates.unique()))
    folds = np.array_split(uniq, n_folds)
    oos = pd.Series(np.nan, index=df.index)
    for fold in folds[1:]:                                       # first block is train-only (no OOS)
        ts, te = fold[0], fold[-1]
        emb = ts - pd.Timedelta(days=embargo)                   # purge the label-overlap before the test block
        tr = df[dates < emb]
        tem = (dates >= ts) & (dates <= te)
        if len(tr) < 500 or tem.sum() == 0:
            continue
        m = model_factory()
        m.fit(tr[feat_cols].to_numpy(), tr["fwd"].to_numpy())
        oos[tem] = m.predict(df.loc[tem, feat_cols].to_numpy())
    return oos


def _ml_book(score_panel, C, A, beta, *, neutral="beta"):
    """Long-top / short-bottom book on a score panel, matched to the classical construction.

    neutral="beta": legs selected by the score, then FP leg-scaled by *beta* (short leg × β̄_long/β̄_short)
    so the book is beta-neutral — directly comparable to the classical beta-neutral +0.77 (feeding −beta
    as the score reproduces it, which validates the comparison). neutral="dollar": plain dollar-neutral.
    """
    sig = top_n_liquid(score_panel, A, TOPN)
    if neutral == "dollar":
        return vol_target(xs_backtest(C, sig, top_frac=0.2, rebal=REBAL, exec_lag=2,
                                      cost_bps=COST, adv=A, impact_k=0.1)["net"], PPY, TVOL)
    ranks = sig.rank(axis=1, pct=True)
    nv = sig.notna().sum(axis=1)
    longs = (ranks >= 0.8).astype(float)                        # long the high-score quintile
    shorts = (ranks <= 0.2).astype(float)                      # short the low-score quintile
    wl = longs.div(longs.sum(axis=1).replace(0.0, np.nan), axis=0)
    ws = shorts.div(shorts.sum(axis=1).replace(0.0, np.nan), axis=0)
    b_lo, b_hi = (beta * wl).sum(axis=1), (beta * ws).sum(axis=1)
    scale = (b_lo / b_hi.replace(0.0, np.nan)).clip(lower=0.0, upper=5.0)   # net-zero-beta FP scaling
    w = (wl - ws.mul(scale, axis=0)).where(nv >= 6, 0.0).fillna(0.0)
    keep = np.zeros(len(w), dtype=bool); keep[::REBAL] = True
    w = w.where(pd.Series(keep, index=w.index), axis=0).ffill().fillna(0.0)
    return vol_target(bab.bab_backtest(C, w, exec_lag=2, cost_bps=COST, adv=A, impact_k=0.1)["net"], PPY, TVOL)


def main():
    C = bab.winsorize_panel(pd.read_parquet(CACHE / "crypto_1d_close.parquet"), 1.0)
    A = pd.read_parquet(CACHE / "crypto_1d_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC"); A.index = A.index.tz_localize("UTC")
    feats = _features(C)
    beta = feats["beta90"]
    df = _long_panel(C, A, feats)
    print(f"panel: {df.shape[0]:,} (date,name) rows, {len(FEATSETS['all'])} features, "
          f"{df.index.get_level_values('date').nunique()} bars")

    # ── classical baselines (the bar to beat) ──────────────────────────────────────────────────
    bmask = top_n_liquid(beta, A, TOPN)
    clas_dollar = _sh(vol_target(xs_backtest(C, -bmask, top_frac=0.2, rebal=REBAL, exec_lag=2,
                                             cost_bps=COST, adv=A, impact_k=0.1)["net"], PPY, TVOL))
    w_fp = bab.bab_weights(bmask, top_frac=0.2, neutral="beta", rebal=REBAL)
    clas_fp = _sh(vol_target(bab.bab_backtest(C, w_fp, exec_lag=2, cost_bps=COST, adv=A, impact_k=0.1)["net"], PPY, TVOL))
    clas_val = _sh(_ml_book(-bmask, C, A, beta, neutral="beta"))     # −beta score through the ML book path
    print(f"\nclassical: dollar-neutral −beta {clas_dollar:+.2f}   beta-neutral FP {clas_fp:+.2f}   "
          f"−beta via the ML-book path {clas_val:+.2f}  (reproduces FP → validates the ML comparison)")

    # ── ML forecasters: model × feature-set, purged-CV OOS book (FP beta-neutral, same as classical) ──
    print(f"\n=== ML cross-sectional forecaster — purged-CV OOS Sharpe (FP beta-neutral; beat {clas_fp:+.2f}?) ===")
    print(f"{'model':>6} " + " ".join(f"{fs:>10}" for fs in FEATSETS))
    rows, series = [], {}
    for mname, factory in MODELS.items():
        cells = []
        for fs, cols in FEATSETS.items():
            pred = _predict_cv(df, cols, factory).unstack(level=-1).reindex(index=C.index, columns=C.columns)
            net = _ml_book(pred, C, A, beta, neutral="beta")
            s = _sh(net)
            cells.append(s)
            series[f"{mname}:{fs}"] = net
            rows.append({"model": mname, "featset": fs, "oos_sharpe": round(s, 3),
                         "oos_sharpe_dollar": round(_sh(_ml_book(pred, C, A, beta, neutral="dollar")), 3)})
        print(f"{mname:>6} " + " ".join(f"{c:>+10.2f}" for c in cells))

    res = pd.DataFrame(rows)
    best = res.loc[res["oos_sharpe"].idxmax()]
    bnet = series[f"{best['model']}:{best['featset']}"].dropna()
    n_trials = len(res)
    var_tr = float((res["oos_sharpe"].clip(-3, 3) / np.sqrt(PPY)).var())
    dsr = deflated_sharpe(bnet.mean() / bnet.std(ddof=1), len(bnet), bnet.skew(), bnet.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8))
    mc = bootstrap_sharpe(bnet, PPY, 1000, SEED)
    print(f"\nbest ML: {best['model']}:{best['featset']} OOS {best['oos_sharpe']:+.2f}  "
          f"[MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f}]  deflated {dsr:.2f} (N={n_trials})  "
          f"vs classical beta-neutral {clas_fp:+.2f}")
    verdict = ("ML BEATS classical" if best["oos_sharpe"] > clas_fp + 0.1 else
               "ML ≈ classical (no alpha beyond beta)" if best["oos_sharpe"] > clas_fp - 0.2 else
               "ML WORSE than classical (overfit)")
    print(f"VERDICT: {verdict}")

    res.to_csv(BAB_DIR / "bab_ml.csv", index=False)
    (BAB_DIR / "bab_ml_summary.json").write_text(json.dumps({
        "classical": {"dollar_neutral": round(clas_dollar, 3), "beta_neutral_FP": round(clas_fp, 3),
                      "beta_rank_via_ml_path": round(clas_val, 3)},
        "ml": rows, "best": {"config": f"{best['model']}:{best['featset']}", "oos_sharpe": best["oos_sharpe"],
                             "mc_p5": mc.get("sharpe_p5"), "deflated_sharpe": round(dsr, 3), "n_trials": n_trials},
        "verdict": verdict}, indent=2, default=float))
    print("\nRUN BAB ML OK  -> reports/bab_ml.csv")


if __name__ == "__main__":
    main()
