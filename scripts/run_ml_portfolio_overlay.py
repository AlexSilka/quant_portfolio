"""§5d, portfolio level — does ML *on top of the whole assembled book* lift it? (measured, it does not.)

The §5d table (`run_ml_book_contribution.py`) swaps ML into each family leg and finds the one real
portfolio-level win is a non-ML VIX rule on the short-vol leg. This asks the complementary question:
put an ML layer on the WHOLE book — three tactics, six engines, honest controls — judged on ALL five
targets (Sharpe / CAGR / max-DD / worst-month / months-in-profit / streak), full window and OOS block.

  A. whole-book regime GATE   — predict P(book up next 21d); flatten the WHOLE book in bad regimes
  B. whole-book SOFT exposure — scale gross by the predicted probability (continuous, cap 1.5x)
  C. ML ALLOCATION            — predict each family's fwd return; tilt weights away from equal (vs 1/N)
  controls: base (no overlay) · constant (same avg exposure) · random gate (20-draw placebo)

Causal walk-forward (features known at prior close, quarterly refit, 21d embargo). Verdict: nothing
beats equal-weight + the surgical volprem VIX gate. Gating the WHOLE book flattens it, and a flat month
is a non-profit month, so it makes months-in-profit and the losing streak (the binding targets) WORSE;
soft exposure is just leverage (higher CAGR, breaks max-DD/worst-month); ML allocation destroys the
book's decorrelation (Sharpe 3.8 -> 1.4). The one real timing signal (VIX) works only *because* it is
applied to the single tail leg, not the whole book.

    make ml-portfolio        # ~3-4 min; prints the table, writes reports/book/ml_portfolio_overlay.json
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,  # noqa: E402
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPClassifier  # noqa: E402
import lightgbm as lgb  # noqa: E402

from scripts.run_master_book import (  # noqa: E402
    FAMILIES, load, rescale, risk_overlay, scorecard, OOS, START_REPORT)
from src.config import RAW_DIR, SEED  # noqa: E402
from src import bo_common as bo  # noqa: E402

PPY, H = 365, 21          # book annualisation; forward regime horizon (~1 month, the streak's unit)


def _n(s):
    s = s.dropna().copy(); s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def book_and_families():
    """Replicate run_master_book's equal-weight assembly (current tree: volprem is the gated leg). Return
    the pre-overlay equal-weight book AND the per-family rescaled matrix."""
    raw = {lab: load(lab, f, c) for lab, f, c in FAMILIES}
    raw = {k: _n(v) for k, v in raw.items() if v is not None}
    df = pd.DataFrame({k: rescale(v) for k, v in raw.items()}).sort_index()
    df = df[(df.index >= pd.Timestamp(START_REPORT)) & (df.notna().sum(axis=1) >= 2)]
    return df.mean(axis=1, skipna=True).rename("ret"), df


def features(ew, df):
    """Causal book-level features — everything shifted 1 bar (known at the prior close)."""
    f = pd.DataFrame(index=ew.index)
    eq = (1 + ew).cumprod()
    f["vol20"] = ew.rolling(20).std() * np.sqrt(PPY)
    f["vol60"] = ew.rolling(60).std() * np.sqrt(PPY)
    f["mom20"] = ew.rolling(20).sum()
    f["mom60"] = ew.rolling(60).sum()
    f["dd"] = eq / eq.cummax() - 1.0
    f["disp"] = df.std(axis=1)                                       # cross-family return dispersion
    f["n_neg"] = (df < 0).sum(axis=1) / df.notna().sum(axis=1)       # fraction of families down that day
    vix = _n(pd.read_parquet(RAW_DIR / "rates" / "VIXCLS.parquet")["val"]).reindex(ew.index).ffill()
    v3m = _n(pd.read_parquet(RAW_DIR / "vol_etp" / "VIX3M_yf.parquet")["close"]).reindex(ew.index).ffill()
    f["vix"], f["vix_ts"], f["vix_chg"] = vix, v3m / vix, vix.pct_change(5)
    return f.shift(1)


def walk_forward(X, y, fit_predict, refit=63, embargo=H, min_train=756):
    """Expanding-window walk-forward: train on [0, t-embargo), predict the next block. Causal — the
    embargo purges the H-day forward-target overlap so no fold sees its own future."""
    idx = X.index
    out = pd.Series(np.nan, index=idx)
    for s in range(min_train, len(idx), refit):
        Xtr = X.iloc[:s - embargo].dropna()
        ytr = y.reindex(Xtr.index)
        ok = ytr.notna()
        if ok.sum() < 250 or ytr[ok].nunique() < 2:
            continue
        te = slice(s, min(s + refit, len(idx)))
        out.iloc[te] = fit_predict(Xtr[ok], ytr[ok], X.iloc[te].fillna(Xtr[ok].median()))
    return out


# ── engines ───────────────────────────────────────────────────────────────────────────────────────
def _clf(kind):
    def f(Xtr, ytr, Xte):
        mdl = {"logit": LogisticRegression(max_iter=1000, C=0.5),
               "rf": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50, random_state=SEED, n_jobs=-1),
               "et": ExtraTreesClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50, random_state=SEED, n_jobs=-1),
               "hgb": HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED),
               "lgbm": lgb.LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, num_leaves=7,
                                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1),
               "mlp": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=400, alpha=1e-2, random_state=SEED)}[kind]
        if kind in ("logit", "mlp"):
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            mdl.fit((Xtr - mu) / sd, ytr)
            return mdl.predict_proba((Xte - mu) / sd)[:, 1]
        mdl.fit(Xtr, ytr)
        return mdl.predict_proba(Xte)[:, 1]
    return f


def _reg(kind):
    def f(Xtr, ytr, Xte):
        mdl = {"ridge": Ridge(alpha=5.0),
               "hgb": HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED),
               "lgbm": lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, num_leaves=7,
                                         subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)}[kind]
        if kind == "ridge":
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            mdl.fit((Xtr - mu) / sd, ytr)
            return mdl.predict((Xte - mu) / sd)
        mdl.fit(Xtr, ytr)
        return mdl.predict(Xte)
    return f


# ── measurement — ALL five targets, both windows ───────────────────────────────────────────────────
def tgt(sc):
    return sum([2.5 <= sc["sharpe"] <= 4.0, sc["months_in_profit"] >= 0.80, sc["max_dd"] >= -0.15,
               sc["worst_month"] >= -0.06, sc["longest_losing_streak_mo"] <= 2])


def _cagr(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((1 + s).prod() ** (1 / yrs) - 1) if yrs > 0 else 0.0


def _card(s):
    sc = scorecard(s)
    dd = sc["max_dd"]
    return {"sharpe": round(sc["sharpe"], 2), "cagr": round(_cagr(s), 3), "max_dd": round(dd, 3),
            "worst_month": round(sc["worst_month"], 3), "months_in_profit": round(sc["months_in_profit"], 3),
            "streak": int(sc["longest_losing_streak_mo"]), "targets": tgt(sc),
            # plain-reader companions to the scorecard: total compounding over the window, and return
            # bought per unit of the deepest fall (the return/risk read that needs no Sharpe).
            "growth_x": round(float((1 + s).prod()), 1),
            "cagr_per_dd": round(_cagr(s) / abs(dd), 2) if dd else 0.0}


def _row(c):
    return (f"Sh {c['sharpe']:+.2f} CAGR {c['cagr']:+.0%} DD {c['max_dd']:+.1%} worst {c['worst_month']:+.1%} "
            f"mo {c['months_in_profit']:.0%} strk {c['streak']} [{c['targets']}/5] "
            f"x{c['growth_x']:.1f} r/dd {c['cagr_per_dd']:.2f}")


def measure(ew, exposure, tag, out):
    # Every arm runs on the UNLEVERED stack (risk_overlay's default 1.0x), not on the shipped book's
    # leverage: the question here is whether an ML exposure path beats a flat one, and holding the base
    # at 1x keeps that comparison free of the book's separate risk-budget dial (run_risk_budget.py).
    exp = exposure.reindex(ew.index).fillna(1.0).clip(0, 1.5)
    m = risk_overlay((ew * exp).dropna())[0]
    cf, co = _card(m), _card(m[m.index >= OOS])
    print(f"  {tag:32s} FULL {_row(cf)}  [avg-exp {exp.mean():.2f}, {(exp <= 1e-9).mean():.0%} flat]"
          f"\n  {'':32s}  OOS {_row(co)}")
    out[tag] = {"full": cf, "oos": co, "avg_exposure": round(float(exp.mean()), 2)}


def main():
    ew, df = book_and_families()
    X = features(ew, df)
    y_sign = (ew.shift(-1).rolling(H).sum().shift(-(H - 1)) > 0).astype(int)     # forward-21d book-up label
    out = {}
    print(f"book {ew.index.min().date()}..{ew.index.max().date()}  n={len(ew)}\n\nBASE:")
    measure(ew, pd.Series(1.0, index=ew.index), "base (no overlay)", out)

    print("\nA/B whole-book overlay — P(book up next 21d); gate {0|1} & soft {clip(2p,0,1.5)}:")
    probs = {}
    for k in ["logit", "rf", "et", "hgb", "lgbm", "mlp"]:
        probs[k] = walk_forward(X, y_sign, _clf(k))
        measure(ew, (probs[k] >= 0.50).astype(float), f"A gate:{k}", out)
    for k in ["logit", "lgbm"]:
        soft = (2.0 * probs[k]).clip(0, 1.5)
        measure(ew, soft, f"B soft:{k}", out)
        # Leverage-matched control: soft sizing averages >1x gross, so its higher CAGR must be judged
        # against flat leverage at the SAME average exposure, not against the 1.0x base. Without this
        # arm "ML raised the return" is unfalsifiable — any size dial raises return.
        lev = float(soft.reindex(ew.index).fillna(1.0).clip(0, 1.5).mean())
        measure(ew, pd.Series(lev, index=ew.index), f"B control: flat {lev:.2f}x", out)

    print("\ncontrols (isolate timing from de-risking):")
    avg = float((probs["lgbm"] >= 0.5).astype(float).reindex(ew.index).fillna(1.0).mean())
    measure(ew, pd.Series(avg, index=ew.index), f"constant {avg:.2f}", out)
    rs = []
    for d in range(20):
        rnd = pd.Series(np.random.default_rng(1000 + d).choice([0.0, 1.0], len(ew), p=[1 - avg, avg]), index=ew.index)
        rs.append(scorecard(risk_overlay((ew * rnd).dropna())[0])["sharpe"])
    out["random_gate_full_sharpe"] = {"min": round(min(rs), 2), "mean": round(float(np.mean(rs)), 2), "max": round(max(rs), 2)}
    print(f"  {'random gate (20-draw)':32s} FULL Sharpe min {min(rs):+.2f} mean {np.mean(rs):+.2f} max {max(rs):+.2f}")

    # ── D/E: predict what actually BINDS, not the direction ──────────────────────────────────────
    # A/B/C all forecast a FIRST moment (P(book up), family forward return). Direction is the lowest-
    # signal thing on this book and the scorecard does not even ask for it: what binds is the worst
    # month and the ≤2-month streak. So two arms that forecast the risk side instead.
    print("\nD volatility targeting — forecast forward 21d realised vol, size inversely (constant risk):")
    y_vol = ew.shift(-1).rolling(H).std().shift(-(H - 1)) * np.sqrt(PPY)     # forward realised vol
    tgt_vol = float(ew.std() * np.sqrt(PPY))                                # hold the book's own vol
    for k in ["ridge", "lgbm"]:
        vhat = walk_forward(X, y_vol, _reg(k)).reindex(ew.index)
        exp = (tgt_vol / vhat.clip(lower=1e-4)).clip(0, 1.5)
        measure(ew, exp, f"D voltarget:{k}", out)
        lev = float(exp.fillna(1.0).mean())                                 # leverage-matched control
        measure(ew, pd.Series(lev, index=ew.index), f"D control: flat {lev:.2f}x", out)
    # the non-ML sibling: the same idea with a trailing estimate instead of a forecast. If ML's value
    # is real it has to beat THIS, not the unmanaged base — otherwise the model is decorating a moving
    # average (Moreira-Muir volatility management).
    trail = (tgt_vol / (ew.rolling(60).std() * np.sqrt(PPY)).shift(1).clip(lower=1e-4)).clip(0, 1.5)
    measure(ew, trail, "D baseline: trailing-vol target (no ML)", out)

    print("\nE bad-month classifier — P(next 21d in the bottom decile); de-gross only that regime:")
    fwd = ew.shift(-1).rolling(H).sum().shift(-(H - 1))
    y_bad = (fwd < fwd.quantile(0.10)).astype(int)      # label uses a full-sample cut; the FIT is walk-
    for k in ["logit", "lgbm"]:                         # forward and embargoed, so no fold sees its own future
        p_bad = walk_forward(X, y_bad, _clf(k)).reindex(ew.index)
        for thr in (0.30, 0.50):
            measure(ew, (p_bad < thr).astype(float).fillna(1.0), f"E badmonth:{k}@{thr}", out)

    print("\nC ML allocation — predict family fwd returns, softmax-tilt weights (vs equal 1/N):")
    for k in ["ridge", "hgb", "lgbm"]:
        preds = {}
        for col in df.columns:
            fx = features(df[col].dropna().rename("ret"), df)
            fy = df[col].shift(-1).rolling(H).sum().shift(-(H - 1))
            common = fx.dropna().index.intersection(fy.dropna().index)
            preds[col] = walk_forward(fx.loc[common], fy.loc[common], _reg(k)).reindex(df.index)
        P = pd.DataFrame(preds).reindex(df.index)
        z = P.sub(P.mean(axis=1), axis=0).div(P.std(axis=1).replace(0, np.nan), axis=0)
        w = np.exp(0.5 * z.clip(-2, 2)); w = w.div(w.sum(axis=1), axis=0).where(df.notna()).fillna(0.0)
        m = risk_overlay((df * w).sum(axis=1, min_count=1).reindex(ew.index).dropna())[0]
        cf, co = _card(m), _card(m[m.index >= OOS])
        print(f"  {'C alloc:' + k:32s} FULL {_row(cf)}\n  {'':32s}  OOS {_row(co)}")
        out[f"C alloc:{k}"] = {"full": cf, "oos": co}

    (bo.REPORTS / "book" / "ml_portfolio_overlay.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {bo.REPORTS / 'book' / 'ml_portfolio_overlay.json'}")
    print("VERDICT: no portfolio-level ML tactic beats equal-weight + the surgical volprem VIX gate.")


if __name__ == "__main__":
    main()
