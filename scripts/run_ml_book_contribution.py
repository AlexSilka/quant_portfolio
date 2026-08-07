"""Reproduce the §5d BOOK-LEVEL ML measurements — does ML lift the *assembled* book?

Everything the §5d "Does it help the assembled book?" table and its closing paragraph cite is
regenerated here from the committed family series, non-destructively (this never overwrites the
master-book artifacts — it imports run_master_book's assembly primitives and swaps one leg at a time):

  A. breakout meta-gate isolated (1d-raw + 4h/1h ungated vs gated), leg swapped into the book
  B. carry timing gate re-fit on the book's honest carry leg (+ study-baseline validation → +1.52)
  C. trend raw-core10 vs meta-gate / conviction variants
  D. uniform confidence gate applied to ALL eight legs (the anti-cherry-pick control) + cherry-pick
  E. objective-aligned magnitude sizing (regress the forward return instead of classifying win/loss)

    make ml-contribution        # ~several minutes (per-sleeve purged-CV on trend+breakout)
    -> prints the table, writes reports/book/ml_book_contribution.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb

warnings.filterwarnings("ignore")
from scripts.run_master_book import (  # noqa: E402
    FAMILIES, load, _scale, rescale, risk_overlay, scorecard, OOS, START_REPORT, R)
from src.metrics import summarise  # noqa: E402
from src import bo_common as bo  # noqa: E402
from src.backtest.engine import positions_from_events  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from scripts.breakout.run_bo_ml import CORE10, precompute as bo_precompute, proba_cache, models  # noqa: E402
from scripts.breakout.run_bo_final import daily_ret_cost  # noqa: E402
from scripts.trend.run_trend_ml import (  # noqa: E402
    precompute as tr_precompute, proba_cache as tr_proba_cache, gated_book, sized_book)
from scripts.carry.run_carry_ml import load_panel  # noqa: E402

PPY, THR, H = 365, 0.55, 10


# ── non-destructive master-book re-assembly ───────────────────────────────────────────────────────
def _norm(s):
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def assemble(overrides=None):
    """overrides: {family_label: net_return_series}. Re-assembles the book exactly as run_master_book."""
    overrides = {k: _norm(v) for k, v in (overrides or {}).items()}
    raw = {}
    for lab, f, c in FAMILIES:
        s = overrides[lab] if lab in overrides else load(lab, f, c)
        if s is not None:
            raw[lab] = s
    df = pd.DataFrame({k: rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(START_REPORT)]
    df = df[df.notna().sum(axis=1) >= 2]
    ew = df.mean(axis=1, skipna=True).rename("ret")
    managed = risk_overlay(ew)[0]
    return {"full": scorecard(managed), "oos": scorecard(managed[managed.index >= OOS])}


def _card(r):
    return {"sharpe_full": r["full"]["sharpe"], "sharpe_oos": r["oos"]["sharpe"],
            "months_oos": r["oos"]["months_in_profit"], "worst_full": r["full"]["worst_month"],
            "streak_full": r["full"]["longest_losing_streak_mo"]}


def _sh(net):
    """Standalone (leg-level) raw Sharpe — illustrative context (full, oos). The book-level verdict
    below is convention-independent: the assembly vol-targets every leg uniformly via rescale()."""
    n = _norm(net)
    return round(summarise(n, PPY)["sharpe_ann"], 2), round(summarise(n[n.index >= OOS], PPY)["sharpe_ann"], 2)


def rescale15(net, target=0.15):
    s = (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * s).dropna()


# ── A. breakout meta-gate, isolated (1d-raw + PIT held fixed; only 4h/1h gated vs ungated) ─────────
def breakout_swap():
    sleeves = bo_precompute()
    pc = proba_cache(sleeves, models()["lightgbm"], weighted=True)
    ung, gat = {}, {}
    for key, s in sleeves.items():
        ung[key] = s["ung"]
        kept = pc[key].index[pc[key].values >= THR]
        pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], kept)
        gat[key] = daily_ret_cost(s["px"], pos, s["tf"], s["fund"], s["adv"])[0]
    oned = {}
    for sym in CORE10:
        px = bo.load_crypto(sym, "1d")
        if px is None:
            continue
        side = bl.donchian_side(px["close"], px["high"], px["low"], 55)
        pos = bl.hold_atr_trailing(px["close"], px["high"], px["low"], side, 3.0, 14)
        adv = px["quote_volume"].rolling(20).median().shift(1)
        oned[f"{sym}_1d"] = daily_ret_cost(px, pos, "1d", bo.safe_funding(sym), adv)[0]
    raw_ts = pd.DataFrame({**oned, **ung}).mean(axis=1, skipna=True)
    ml_ts = pd.DataFrame({**oned, **gat}).mean(axis=1, skipna=True)
    pit = pd.read_parquet(R / "breakout" / "bo_xs_pit_returns.parquet")[
        ["1d_PIT_top30", "4h_PIT_top30", "1h_PIT_top30"]].mean(axis=1).dropna()

    def combine(ts):
        a, b = rescale15(_norm(ts)), rescale15(_norm(pit))
        idx = a.index.intersection(b.index)
        return (0.5 * a.reindex(idx).fillna(0) + 0.5 * b.reindex(idx).fillna(0)).rename("ret")
    return {"standalone_raw": _sh(raw_ts), "standalone_ml": _sh(ml_ts),
            "book_raw": _card(assemble({"breakout": combine(raw_ts)})),
            "book_ml": _card(assemble({"breakout": combine(ml_ts)}))}


# ── B. carry timing gate — validated on the study baseline (→1.52), applied to the book's leg ──────
def _carry_gate(leg, fd, btc):
    leg = _norm(leg); leg.index = leg.index.normalize(); idx = leg.index
    mkt = pd.DataFrame({
        "mean_fund": fd.mean(axis=1).reindex(idx), "disp_fund": fd.std(axis=1).reindex(idx),
        "btc_vol": btc.pct_change().rolling(14).std().reindex(idx), "btc_mom": btc.pct_change(30).reindex(idx),
        "book_mom": leg.rolling(7).mean(), "book_vol": leg.rolling(14).std()}, index=idx).replace([np.inf, -np.inf], np.nan)
    y = leg.rolling(H // 2).sum().shift(-H // 2)
    m = mkt.notna().all(axis=1) & y.notna()
    X, yb = mkt[m], (y[m] > 0).astype(int)
    di = pd.DatetimeIndex(X.index); ud = np.array(sorted(di.unique()))
    proba = pd.Series(np.nan, index=X.index)
    for k in range(1, 7):
        tspan = pd.DatetimeIndex(np.array_split(ud, 7)[k]); t0 = tspan.min()
        tr = di < (t0 - pd.Timedelta(days=8)); te = di.isin(tspan)
        if tr.sum() < 300 or te.sum() == 0 or yb[tr].nunique() < 2:
            continue
        mdl = LogisticRegression(max_iter=500, C=1.0).fit(X[tr], yb[tr])
        proba.iloc[np.where(te)[0]] = mdl.predict_proba(X[te])[:, 1]
    proba = proba.reindex(idx).shift(1)
    return (leg * (proba > 0.5).astype(float).reindex(idx).fillna(0.0)).rename("ret")


def carry_swap():
    C, V, fd = load_panel()
    C.index = pd.to_datetime(C.index).tz_localize(None).normalize() if C.index.tz is not None else pd.to_datetime(C.index).normalize()
    fd.index = C.index; btc = C["BTCUSDT"]
    study = carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, 7), direction=-1.0, top_frac=0.2, cost_bps=6.0)["ret"]
    leg = _norm(load("carry", "carry/carry_breadth_headline.parquet", "ret"))
    # validation keeps the study's vol-target (to reproduce the published +1.21→+1.52); standalone is raw,
    # uniform with the other families
    return {"validation_study_raw": _sh(rescale15(study)), "validation_study_gated": _sh(rescale15(_carry_gate(study, fd, btc))),
            "standalone_raw": _sh(leg), "standalone_gated": _sh(_carry_gate(leg, fd, btc)),
            "book_raw": _card(assemble()), "book_gated": _card(assemble({"carry": _carry_gate(leg, fd, btc)}))}


# ── C. trend raw-core10 vs meta-gate / conviction ─────────────────────────────────────────────────
def trend_swap():
    sleeves = tr_precompute(); mdls = models()
    raw = pd.DataFrame({k: s["ung"] for k, s in sleeves.items()}).mean(axis=1, skipna=True)
    gate_lgbm = gated_book(sleeves, tr_proba_cache(sleeves, mdls["lightgbm"], True), 0.55)[0].mean(axis=1, skipna=True)
    gate_rf = gated_book(sleeves, tr_proba_cache(sleeves, mdls["randomforest"], True), 0.55)[0].mean(axis=1, skipna=True)
    sized = sized_book(sleeves, tr_proba_cache(sleeves, mdls["lightgbm"], False)).mean(axis=1, skipna=True)
    out = {"standalone_raw": _sh(raw), "book_raw": _card(assemble({"trend_momentum": raw}))}
    for lab, ser in [("lgbm_gate", gate_lgbm), ("rf_gate", gate_rf), ("conviction", sized)]:
        out[f"standalone_{lab}"] = _sh(ser); out[f"book_{lab}"] = _card(assemble({"trend_momentum": ser}))
    return out


# ── D & E: uniform confidence gate / magnitude sizing across ALL legs ─────────────────────────────
def _feat(r):
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1.0
    return pd.DataFrame({"mom5": r.rolling(5).mean(), "mom20": r.rolling(20).mean(), "mom60": r.rolling(60).mean(),
                         "vol20": r.rolling(20).std(), "vol60": r.rolling(60).std(), "dd": dd,
                         "lossfreq20": (r < 0).rolling(20).mean()}).replace([np.inf, -np.inf], np.nan)


def _purged(X, y, model_fn, clf):
    di = pd.DatetimeIndex(X.index); ud = np.array(sorted(di.unique()))
    oos = pd.Series(np.nan, index=X.index)
    for k in range(1, 7):
        tspan = pd.DatetimeIndex(np.array_split(ud, 7)[k]); t0 = tspan.min()
        tr = di < (t0 - pd.Timedelta(days=15)); te = di.isin(tspan)
        if tr.sum() < 250 or te.sum() == 0 or (clf and y[tr].nunique() < 2):
            continue
        mdl = model_fn().fit(X[tr], y[tr])
        oos.iloc[np.where(te)[0]] = (mdl.predict_proba(X[te])[:, 1] if clf else mdl.predict(X[te]))
    return oos


def _gate_all(legs, model_fn):
    hard, helped = {}, {}
    for lab, r in legs.items():
        X = _feat(r); y = (r.rolling(H).sum().shift(-H) > 0).astype(int)
        m = X.notna().all(axis=1) & r.rolling(H).sum().shift(-H).notna()
        proba = _purged(X[m], y[m], model_fn, True).reindex(r.index).shift(1)
        g = (proba > 0.5).astype(float).reindex(r.index).fillna(1.0)
        gr = (r * g).rename("ret"); hard[lab] = gr
        helped[lab] = gr if _sh(gr)[0] > _sh(r)[0] else r
    return hard, helped


def _size_all(legs, model_fn):
    out = {}
    for lab, r in legs.items():
        X = _feat(r); y = r.rolling(H).sum().shift(-H)
        m = X.notna().all(axis=1) & y.notna()
        rhat = _purged(X[m], y[m], model_fn, False).reindex(r.index)
        z = (rhat / (rhat.expanding(120).std().shift(1) + 1e-9)).clip(-3, 3)
        out[lab] = (r * (1.0 + 0.5 * z).clip(0.0, 1.5).shift(1).fillna(1.0)).rename("ret")
    return out


def uniform_and_aligned():
    legs = {lab: _norm(load(lab, f, c)) for lab, f, c in FAMILIES}
    legs = {k: v for k, v in legs.items() if v is not None}
    logit = lambda: LogisticRegression(max_iter=500, C=1.0)
    gbm_c = lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=7, n_jobs=-1, verbose=-1)
    ridge = lambda: Ridge(alpha=5.0)
    gbm_r = lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=7, n_jobs=-1, verbose=-1)
    hard_l, helped_l = _gate_all(legs, logit)
    hard_g, _ = _gate_all(legs, gbm_c)
    return {
        "uniform_hard_logit": _card(assemble(hard_l)), "uniform_hard_gbm": _card(assemble(hard_g)),
        "cherry_pick_logit": _card(assemble(helped_l)),
        "magnitude_ridge": _card(assemble(_size_all(legs, ridge))),
        "magnitude_gbm": _card(assemble(_size_all(legs, gbm_r)))}


def main():
    print("baseline:", _card(assemble()), flush=True)
    out = {"baseline": _card(assemble())}
    for name, fn in [("breakout", breakout_swap), ("carry", carry_swap), ("trend", trend_swap),
                     ("uniform_and_aligned", uniform_and_aligned)]:
        print(f"... {name}", flush=True)
        out[name] = fn()
    (R / "book" / "ml_book_contribution.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print("\nML BOOK CONTRIBUTION OK -> reports/book/ml_book_contribution.json")


if __name__ == "__main__":
    main()
