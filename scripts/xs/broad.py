"""Cross-sectional momentum on a BROAD, survivorship-free equity universe.

The narrow 78-name panel mixes mega-caps with sector/bond/commodity ETFs — a poor test of the
equity anomaly, which the literature says is strongest with *breadth*. This runs the same honest
harness on the PIT S&P 500 universe (≈812 names, delisted included), sweeping the construction plus
a **residual (idiosyncratic) momentum** signal, walk-forward, and a learning-to-rank ML layer,
against a liquidity mask (rank only names above a trailing dollar-volume floor, so the book is
tradable). The question: does breadth lift the equity sleeve, and does ML add on top?

    python scripts/xs/broad.py
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, SEED, XS_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.xsect import (blend_rank, mom, resid_mom, risk_adj_mom,  # noqa: E402
                               vol_target, xs_backtest)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE, OUT = CACHE_DIR / "xs", XS_DIR
PPY, COST, SEED = 252, 3.0, SEED
ADV_FLOOR = 10e6          # only rank names with ≥ $10M/day trailing dollar volume (tradable)


def make_sig(kind, px, lb, sk=0):
    if kind == "raw":
        return mom(px, lb, sk)
    if kind == "riskadj":
        return risk_adj_mom(px, lb, sk)
    if kind == "resid":
        return resid_mom(px, lb, sk)
    return blend_rank([risk_adj_mom(px, max(2, int(lb * f)), sk) for f in (0.5, 1.0, 2.0)])


def liq(sig, adv):
    """Mask the signal to names liquid enough to trade that bar (trailing-median $vol floor).

    min_periods is essential: the union-calendar panel has scattered NaN/0 dollar-volume days, so
    a full-window (min_periods=60) median is NaN for almost every name — the floor would then admit
    ~6 names and the cross-section would collapse. A 20-of-60 median is robust to those gaps.
    """
    if adv is None:
        return sig
    trail = adv.reindex_like(sig).replace(0, np.nan).rolling(60, min_periods=20).median().shift(1)
    return sig.where(trail >= ADV_FLOOR)


def run_cfg(px, adv, kind, lb, sk, tf, wt, rb, mc=False):
    sig = liq(make_sig(kind, px, lb, sk), adv)
    bt = xs_backtest(px, sig, top_frac=tf, weighting=wt, rebal=rb, cost_bps=COST,
                     adv=adv, impact_k=0.1, min_names=20)
    netv = vol_target(bt["net"], PPY).dropna()
    s = summarise(netv, PPY)
    s["turnover"] = float(bt["turnover"].sum() / (len(px) / PPY))
    if mc and s["sharpe_ann"] > 0.4:
        s["mc_p5"] = bootstrap_sharpe(netv, PPY, 400, SEED).get("sharpe_p5", np.nan)
    return s, netv


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "stocks_broad"   # or "stocks_midsmall"
    px = pd.read_parquet(CACHE / f"{prefix}_1d_close.parquet")
    advp = CACHE / f"{prefix}_1d_adv.parquet"
    adv = pd.read_parquet(advp) if advp.exists() else None
    # drop union-calendar junk rows (days only a handful of names have data) — keep real sessions
    live = px.notna().sum(axis=1) >= 100
    px = px[live]
    adv = adv.reindex(px.index) if adv is not None else None
    liq_names = int((adv.replace(0, np.nan).rolling(60, min_periods=20).median().iloc[-1] >= ADV_FLOOR).sum()) \
        if adv is not None else px.shape[1]
    print(f"BROAD panel {px.shape[0]}×{px.shape[1]} names  {px.index[0].date()}→{px.index[-1].date()}"
          f"  ({liq_names} currently liquid ≥ ${ADV_FLOOR/1e6:.0f}M/day)")

    # 1. construction sweep (focused) + placebo
    rows = []
    for kind in ("raw", "riskadj", "resid", "blend"):
        for lb in (60, 120, 180, 252):
            for sk in (0, 21):
                for tf in (0.1, 0.2):
                    for rb in (10, 21):
                        s, _ = run_cfg(px, adv, kind, lb, sk, tf, "equal", rb, mc=False)
                        rows.append(dict(kind=kind, lb=lb, sk=sk, tf=tf, rb=rb,
                                         sharpe=s["sharpe_ann"], max_dd=s["max_dd"], turn=s["turnover"]))
    df = pd.DataFrame(rows)
    plac = []
    for i in range(16):
        pl = pd.DataFrame(np.random.default_rng(300 + i).standard_normal(px.shape),
                          index=px.index, columns=px.columns)
        bt = xs_backtest(px, liq(pl, adv), top_frac=0.1, weighting="equal", rebal=21,
                         cost_bps=COST, min_names=20)
        plac.append(summarise(vol_target(bt["net"], PPY).dropna(), PPY)["sharpe_ann"])
    pmax = max(plac)
    df.to_csv(OUT / f"sweep_{prefix}.csv", index=False)
    print(f"\nsweep: {len(df)} configs, Sharpe med {df.sharpe.median():+.2f} / max {df.sharpe.max():+.2f} "
          f"({(df.sharpe>0).mean():.0%}+)  | placebo max {pmax:+.2f}")
    top = df.sort_values("sharpe", ascending=False).head(6)
    print(top.to_string(index=False))

    # 2. walk-forward (anchored, 3y train / 1y test, top-10 ensemble over the grid)
    cfgs = [(r.kind, r.lb, r.sk, r.tf, "equal", r.rb) for r in df.itertuples()]
    M = pd.DataFrame({i: run_cfg(px, adv, *c)[1] for i, c in enumerate(cfgs)}).dropna(how="all")
    idx = M.index
    tr_b, te_b = int(3 * PPY), int(1 * PPY)
    segs, start = [], tr_b
    while start + te_b <= len(idx):
        tr, te = M.iloc[:start], M.iloc[start:start + te_b]
        sr = (tr.mean() / tr.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        segs.append(te[list(sr.nlargest(10).index)].mean(axis=1))
        start += te_b
    wf = pd.concat(segs)
    sw = summarise(wf, PPY)
    print(f"\nWALK-FORWARD (anch 3y/1y, top-10 ensemble): Sharpe {sw['sharpe_ann']:+.2f}  "
          f"DD {sw['max_dd']:+.0%}  months+ {sw['months_in_profit']:.0%}")

    # 3. a-priori sleeve (residual 12-1, decile, monthly) + MC — the sleeve for the book
    ap_s, ap_ret = run_cfg(px, adv, "resid", 252, 21, 0.1, "equal", 21, mc=True)
    print(f"\nA-PRIORI broad sleeve (residual 12-1 decile monthly): Sharpe {ap_s['sharpe_ann']:+.2f}  "
          f"DD {ap_s['max_dd']:+.0%}  MC-P5 {ap_s.get('mc_p5', float('nan')):+.2f}  turn {ap_s['turnover']:.0f}x")

    # 3b. ML learning-to-rank on the broad panel (LightGBM, expanding walk-forward OOS)
    ml_s, ml_ret = None, None
    try:
        import lightgbm as lgb
        from src.sleeves.xsect_ml import (expanding_predict, predictions_to_panel,
                                          rank_features, stack_xy)
        # the LTR feature pipeline needs mostly-complete rows; on a gappy survivorship-free panel
        # (delisted names, union-calendar holes) restrict to names present in ≥70% of bars so the
        # design matrix isn't decimated to a biased sliver, then keep finite rows.
        dense = px.columns[px.notna().mean() >= 0.70]
        pxd = px[dense]
        feats = rank_features(pxd, adv[dense] if adv is not None else None, 1)
        X, y, ts = stack_xy(feats, pxd, 21)
        good = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
        X, y, ts = X[good], y[good], ts[good]
        y = y.clip(y.quantile(0.001), y.quantile(0.999))
        print(f"  LTR on {len(dense)} dense names, {len(X)} rows", flush=True)
        pred = expanding_predict(X, y, ts, lambda: lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
            n_jobs=-1, verbose=-1), n_folds=6, embargo_bars=21)
        sig_ml = liq(predictions_to_panel(pred, px), adv)
        bt = xs_backtest(px, sig_ml, top_frac=0.1, weighting="equal", rebal=21, cost_bps=COST,
                         adv=adv, impact_k=0.1, min_names=20)
        ml_ret = vol_target(bt["net"], PPY).dropna()
        ml_s = summarise(ml_ret, PPY)
        ml_s["mc_p5"] = bootstrap_sharpe(ml_ret, PPY, 400, SEED).get("sharpe_p5", np.nan)
        print(f"\nML learning-to-rank (LightGBM, {X.shape[0]} rows): Sharpe {ml_s['sharpe_ann']:+.2f}  "
              f"DD {ml_s['max_dd']:+.0%}  MC-P5 {ml_s['mc_p5']:+.2f}")
    except Exception as e:
        print(f"\nML LTR skipped: {type(e).__name__} {str(e)[:80]}")

    # 4. compare vs the narrow 78-name sleeve
    try:
        narrow = pd.read_parquet(OUT / "apriori_stocks_1d.parquet")["ret"]
        ns = summarise(narrow, PPY)
        print(f"\nNARROW (78 mixed names)  Sharpe {ns['sharpe_ann']:+.2f}   ->  "
              f"BROAD ({px.shape[1]} PIT names) Sharpe {ap_s['sharpe_ann']:+.2f}   "
              f"breadth lift {ap_s['sharpe_ann']-ns['sharpe_ann']:+.2f}")
    except Exception:
        pass

    # The A-PRIORI sleeve is what ships, always. This used to save "whichever is stronger (rule vs
    # ML)", compared on full-sample Sharpe — a selection made on the sample it is then scored on,
    # baked into a leg the book holds. It happens to pick the rule arm today (0.48 against the ML
    # arm's 0.26), so removing it costs nothing now; the point is that it could not have been trusted
    # if it had gone the other way. The ML arm stays in the summary as a study, where a number that
    # lost to a rule belongs.
    ap_ret.rename("ret").to_frame().to_parquet(OUT / f"{prefix}_sleeve.parquet")
    (OUT / f"{prefix}_summary.json").write_text(json.dumps({
        "panel": [px.shape[0], px.shape[1]], "sweep_median": float(df.sharpe.median()),
        "sweep_max": float(df.sharpe.max()), "placebo_max": float(pmax),
        "walk_forward": sw, "apriori_sleeve": ap_s, "ml_sleeve": ml_s,
        "chosen": "rule (a-priori; the ML arm is reported, never selected)"}, indent=2, default=float))
    print("\nBROAD OK")


if __name__ == "__main__":
    main()
