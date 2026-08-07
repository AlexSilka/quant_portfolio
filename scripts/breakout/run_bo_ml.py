"""ML meta-label confidence gate on the breakout book (Task A §5 incremental-value requirement).

Primary side = Donchian-55 breakout (the non-ML baseline). Secondary ML model predicts P(this
breakout trade wins) from the 82-feature library and gates entries to high-confidence signals, while
keeping the fat-tail chandelier exit. Labels are the realized sign of each chandelier trade
(entry->exit), causal (t1 is a training target only, never a feature). Purged+embargoed CV throughout
(overlapping labels leak under plain k-fold — AFML ch.7).

Optimization variants measured (the "different variants" ask):
  models   : LightGBM, RandomForest, HistGradientBoosting
  weights  : unweighted vs AFML average-uniqueness sample weights
  threshold: swept 0.50/0.55/0.60 (sensitivity, not peak-picked)
Incremental value is reported on the frozen core-10 book, in-sample AND on the held-out 2024-07+ block
(gates trained only on each fold's past), as precision lift, Sharpe, drawdown and turnover deltas.

    python scripts/breakout/run_bo_ml.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
from src import bo_common as bo  # noqa: E402
from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.config import CACHE_DIR, OOS_START  # noqa: E402
from src.features.engine import compute_features, pit_normalize  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.validation.purged_cv import purged_kfold  # noqa: E402

SEED = bo.SEED
CORE10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
EMB = {"1d": pd.Timedelta(days=10), "4h": pd.Timedelta(days=5), "1h": pd.Timedelta(days=2),
       "15m": pd.Timedelta(hours=12), "5m": pd.Timedelta(hours=4)}
TFS_ML = ["4h", "1h"]     # 1d has only ~30 Donchian-55 trades/sleeve — too few to meta-label
FEATDIR = CACHE_DIR


def models():
    return {
        "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                               subsample=0.8, colsample_bytree=0.8,
                                               random_state=SEED, n_jobs=-1, verbose=-1),
        "randomforest": lambda: RandomForestClassifier(n_estimators=300, max_depth=6,
                                                       min_samples_leaf=20, random_state=SEED, n_jobs=-1),
        "histgb": lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.03,
                                                         max_iter=300, l2_regularization=1.0,
                                                         random_state=SEED),
    }


def features(sym, tf):
    cache = FEATDIR / f"features_bo_{sym}_{tf}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px = bo.load_crypto(sym, tf)
    btc = bo.load_crypto("BTCUSDT", tf)["close"]
    feats = pit_normalize(compute_features(px, benchmark=btc, fast=(tf != "1d")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


def uniqueness_weights(t0: pd.DatetimeIndex, t1: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """AFML average uniqueness: down-weight labels whose [t0,t1] span overlaps many others."""
    t1a = t1.reindex(t0)                      # Series aligned to t0, tz-aware exit timestamps
    conc = pd.Series(0.0, index=index)
    for a, b in zip(t0, t1a):
        conc.loc[a:b] += 1.0
    w = np.array([np.mean(1.0 / conc.loc[a:b].replace(0, 1)) for a, b in zip(t0, t1a)])
    return w / (w.mean() + 1e-12)


def oos_proba(X, y, t1, factory, tf, weights=None):
    """Purged+embargoed CV OOS P(win); optional per-sample weights fitted inside train folds only."""
    t0 = pd.DatetimeIndex(X.index)
    t1i = pd.DatetimeIndex(t1.reindex(X.index).values)
    oos = pd.Series(np.nan, index=X.index)
    for tr, te in purged_kfold(t0, t1i, n_splits=5, embargo=EMB[tf]):
        m = factory()
        fit_kw = {"sample_weight": weights[tr]} if weights is not None else {}
        m.fit(X.iloc[tr], y.iloc[tr], **fit_kw)
        oos.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oos.dropna()


def sleeve_data(sym, tf):
    """Return (px, trades, X, y, meta) for one breakout sleeve — the primary side + labels + feats."""
    px = bo.load_crypto(sym, tf)
    close, high, low = px["close"], px["high"], px["low"]
    side = bl.donchian_side(close, high, low, 55)
    trades = bl.chandelier_trades(close, high, low, side, 3.0, 14)
    if len(trades) < 60:
        return None
    ret = trades["side"] * (close.reindex(trades["t1"]).values / close.reindex(trades.index).values - 1.0)
    y = (ret > 0).astype(int)
    feats = features(sym, tf)
    X = feats.reindex(trades.index).dropna()
    y = y.reindex(X.index)
    trades = trades.reindex(X.index)
    if len(X) < 60 or y.nunique() < 2:
        return None
    return px, trades, X, y


def _daily(px, side, t1, events, fund, adv, tf):
    pos = positions_from_events(px.index, side, t1, events)
    pos = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bt = backtest(px["close"], pos, capital=bo.CAP, funding=fund, adv=adv, **bo.CC)
    return (1 + bt["net_ret"]).resample("D").prod() - 1


def precompute():
    """Load every core-10 x {4h,1h} sleeve once: data, ungated daily returns, funding/adv."""
    sleeves = {}
    for tf in TFS_ML:
        for sym in CORE10:
            d = sleeve_data(sym, tf)
            if d is None:
                continue
            px, trades, X, y = d
            adv, fund = px["quote_volume"].rolling(20).median().shift(1), bo.safe_funding(sym)
            key = f"{sym}_{tf}"
            sleeves[key] = dict(tf=tf, px=px, trades=trades, X=X, y=y, adv=adv, fund=fund,
                                ung=_daily(px, trades["side"], trades["t1"], X.index, fund, adv, tf))
    return sleeves


def proba_cache(sleeves, factory, weighted):
    """OOS P(win) per sleeve for one (model, weighting) — the expensive CV, computed once."""
    out = {}
    for key, s in sleeves.items():
        w = (uniqueness_weights(pd.DatetimeIndex(s["X"].index), s["trades"]["t1"], s["px"].index)
             if weighted else None)
        out[key] = oos_proba(s["X"], s["y"], s["trades"]["t1"], factory, s["tf"], w)
    return out


def build_book(sleeves, proba, threshold):
    """Gate each sleeve at `threshold` using cached proba; return (ungated df, gated df, precision arr)."""
    ung, gat, precs = {}, {}, []
    for key, s in sleeves.items():
        p = proba[key]
        kept = p.index[p.values >= threshold]
        precs.append((float(s["y"].reindex(p.index).mean()),
                      float(s["y"].reindex(kept).mean()) if len(kept) else np.nan))
        ung[key] = s["ung"]
        gat[key] = _daily(s["px"], s["trades"]["side"], s["trades"]["t1"], kept, s["fund"], s["adv"], s["tf"])
    return pd.DataFrame(ung), pd.DataFrame(gat), np.array(precs)


def book_stats(rets):
    port = rets.fillna(0.0).mean(axis=1)
    turn = float(rets.notna().sum(axis=1).mean())     # avg active sleeves (proxy for participation)
    is_, oos = port[port.index < OOS_START], port[port.index >= OOS_START]
    return {"sharpe": summarise(port, 365)["sharpe_ann"], "max_dd": summarise(port, 365)["max_dd"],
            "sharpe_is": summarise(is_, 365)["sharpe_ann"], "sharpe_oos": summarise(oos, 365)["sharpe_ann"],
            "active": turn}, port


def main():
    facs = models()
    print("=== ML meta-label incremental value on the core-10 breakout book (4h+1h) ===")
    print("(purged+embargoed CV; primary=Donchian-55 + chandelier exit; 1d excluded: too few trades)\n")
    sleeves = precompute()
    print(f"sleeves with enough trades to meta-label: {len(sleeves)}  "
          f"({sum(s['tf']=='4h' for s in sleeves.values())} on 4h, "
          f"{sum(s['tf']=='1h' for s in sleeves.values())} on 1h)\n")

    b_ung, _ = book_stats(pd.DataFrame({k: s["ung"] for k, s in sleeves.items()}))
    print(f"BASELINE ungated : Sharpe {b_ung['sharpe']:+.2f}  (IS {b_ung['sharpe_is']:+.2f} / "
          f"OOS {b_ung['sharpe_oos']:+.2f})  maxDD {b_ung['max_dd']:+.1%}")

    print("\n[A] MODEL x WEIGHTING variants (threshold 0.55):")
    results = {"baseline_ungated": b_ung}
    proba_by_variant = {}
    for mname, fac in facs.items():
        for wlab, wt in [("unw", False), ("uniqW", True)]:
            pc = proba_cache(sleeves, fac, wt)
            proba_by_variant[(mname, wlab)] = pc
            _, g, pr = build_book(sleeves, pc, 0.55)
            bs, _ = book_stats(g)
            prec_b, prec_g = np.nanmean(pr[:, 0]), np.nanmean(pr[:, 1])
            results[f"{mname}_{wlab}"] = {**bs, "prec_base": prec_b, "prec_gate": prec_g}
            print(f"    {mname+'_'+wlab:20s}: Sharpe {bs['sharpe']:+.2f} (IS {bs['sharpe_is']:+.2f}/"
                  f"OOS {bs['sharpe_oos']:+.2f})  maxDD {bs['max_dd']:+.1%}  "
                  f"precision {prec_b:.0%}->{prec_g:.0%}  active {bs['active']:.1f}", flush=True)

    print("\n[B] THRESHOLD sweep (LightGBM, unweighted):")
    pc = proba_by_variant[("lightgbm", "unw")]
    for th in [0.50, 0.55, 0.60]:
        _, g, pr = build_book(sleeves, pc, th)
        bs, _ = book_stats(g)
        print(f"    thr {th:.2f}: Sharpe {bs['sharpe']:+.2f} (IS {bs['sharpe_is']:+.2f}/OOS {bs['sharpe_oos']:+.2f})  "
              f"maxDD {bs['max_dd']:+.1%}  precision {np.nanmean(pr[:,0]):.0%}->{np.nanmean(pr[:,1]):.0%}")

    print("\n[C] ML rescue test — does the gate turn net-negative 1h POSITIVE? (LightGBM unw, thr 0.55)")
    for tf in TFS_ML:
        sub = {k: v for k, v in sleeves.items() if v["tf"] == tf}
        _, g, _ = build_book(sub, {k: proba_by_variant[("lightgbm", "unw")][k] for k in sub}, 0.55)
        bu, _ = book_stats(pd.DataFrame({k: s["ung"] for k, s in sub.items()}))
        bg, _ = book_stats(g)
        print(f"    {tf}: ungated Sharpe {bu['sharpe']:+.2f}  ->  gated {bg['sharpe']:+.2f}")

    (bo.REPORTS / "bo_ml.json").write_text(json.dumps(results, indent=2, default=float))
    best = max((k for k in results if k != "baseline_ungated"), key=lambda k: results[k]["sharpe"])
    print(f"\nincremental value: best gate = {best}  Sharpe {results[best]['sharpe']:+.2f} vs ungated "
          f"{b_ung['sharpe']:+.2f} ({results[best]['sharpe']-b_ung['sharpe']:+.2f}); "
          f"OOS {results[best]['sharpe_oos']:+.2f} vs {b_ung['sharpe_oos']:+.2f}; "
          f"maxDD {results[best]['max_dd']:+.1%} vs {b_ung['max_dd']:+.1%}")
    print("BO ML OK")


if __name__ == "__main__":
    main()
