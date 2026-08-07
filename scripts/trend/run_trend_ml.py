"""Phase 4 — ML overlay on the trend book: measured incremental value vs the non-ML rule (task §5).

Primary = the canonical trend rule (EMA-50/200 cross) with a fat-tail chandelier exit, segmented
into discrete trades. A secondary ML model predicts P(this trend trade wins) from the 82-feature
library (incl. ADX / vol-regime / higher-moment context), trained under purged+embargoed CV so
overlapping labels never leak. Two ways to *use* the probability are compared against the ungated rule:

  gate    take only trades with P(win) >= threshold (the confidence filter — precision up, DD down)
  size    hold every trade but scale exposure by conviction, position ∝ clip(2·P-1) (continuous)

Optimization variants (the "different variants" ask): models {LightGBM, RandomForest, HistGB} ×
weighting {none, AFML uniqueness} × threshold {0.50, 0.55, 0.60}. Reported as precision lift, Sharpe,
maxDD and turnover deltas, in-sample AND on the held-out 2024-07+ block. Reuses the breakout ML
machinery (models/features/uniqueness_weights/oos_proba) — only the primary side differs.

    python scripts/trend/run_trend_ml.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_ml import (EMB, OOS_START, book_stats, features, models,  # noqa: E402
                       oos_proba, uniqueness_weights)
from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.sleeves import trend_lab as tl  # noqa: E402

CORE10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TFS_ML = ["1d", "4h", "1h"]      # EMA-cross+chandelier gives more trades than Donchian → 1d is usable


def trend_sleeve_data(sym, tf):
    """(px, trades, X, y) for one EMA-50/200 trend sleeve — chandelier trades + meta-label + feats."""
    px = bo.load_crypto(sym, tf)                       # perp 2020+ (matches the feature cache era)
    if px is None:
        return None
    close, high, low = px["close"], px["high"], px["low"]
    side = tl.ema_cross_side(close, 50, 200)
    trades = bl.chandelier_trades(close, high, low, side, 3.0, 14)
    if len(trades) < 50:
        return None
    ret = trades["side"] * (close.reindex(trades["t1"]).values / close.reindex(trades.index).values - 1.0)
    y = (ret > 0).astype(int)
    feats = features(sym, tf)
    X = feats.reindex(trades.index).dropna()
    y = y.reindex(X.index)
    trades = trades.reindex(X.index)
    if len(X) < 50 or y.nunique() < 2:
        return None
    return px, trades, X, y


def _daily(px, pos, fund, adv, tf):
    pos = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bt = backtest(px["close"], pos, capital=bo.CAP, funding=fund, adv=adv, **bo.CC)
    return (1 + bt["net_ret"]).resample("D").prod() - 1


def precompute():
    sleeves = {}
    for tf in TFS_ML:
        for sym in CORE10:
            d = trend_sleeve_data(sym, tf)
            if d is None:
                continue
            px, trades, X, y = d
            adv, fund = px["quote_volume"].rolling(20).median().shift(1), bo.safe_funding(sym)
            ung = _daily(px, positions_from_events(px.index, trades["side"], trades["t1"], X.index),
                         fund, adv, tf)
            sleeves[f"{sym}_{tf}"] = dict(tf=tf, px=px, trades=trades, X=X, y=y, adv=adv, fund=fund, ung=ung)
    return sleeves


def proba_cache(sleeves, factory, weighted):
    out = {}
    for key, s in sleeves.items():
        w = (uniqueness_weights(pd.DatetimeIndex(s["X"].index), s["trades"]["t1"], s["px"].index)
             if weighted else None)
        out[key] = oos_proba(s["X"], s["y"], s["trades"]["t1"], factory, s["tf"], w)
    return out


def gated_book(sleeves, proba, threshold):
    """Binary gate: keep trades with P(win) >= threshold."""
    gat, precs = {}, []
    for key, s in sleeves.items():
        p = proba[key]
        kept = p.index[p.values >= threshold]
        precs.append((float(s["y"].reindex(p.index).mean()),
                      float(s["y"].reindex(kept).mean()) if len(kept) else np.nan))
        pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], kept)
        gat[key] = _daily(s["px"], pos, s["fund"], s["adv"], s["tf"])
    return pd.DataFrame(gat), np.array(precs)


def sized_book(sleeves, proba):
    """Continuous confidence sizing: every trade held, exposure ∝ clip(2·P(win)-1, 0, 1) per trade."""
    out = {}
    for key, s in sleeves.items():
        p = proba[key]
        conf = (2.0 * p - 1.0).clip(lower=0.0, upper=1.0)
        pos = pd.Series(0.0, index=s["px"].index)
        for t0, w in conf.items():
            t1 = s["trades"]["t1"].get(t0)
            sd = s["trades"]["side"].get(t0, 0.0)
            if t1 is not None and w > 0:
                pos.loc[t0:t1] = sd * float(w)
        out[key] = _daily(s["px"], pos, s["fund"], s["adv"], s["tf"])
    return pd.DataFrame(out)


def main():
    print("=== ML incremental value on the core-10 TREND book (EMA-50/200 + chandelier) ===")
    print("(purged+embargoed CV; primary = the non-ML trend rule; 1d/4h/1h)\n")
    sleeves = precompute()
    by_tf = {tf: sum(s["tf"] == tf for s in sleeves.values()) for tf in TFS_ML}
    print(f"sleeves with enough trades to meta-label: {len(sleeves)}  {by_tf}\n")

    b_ung, _ = book_stats(pd.DataFrame({k: s["ung"] for k, s in sleeves.items()}))
    print(f"BASELINE ungated : Sharpe {b_ung['sharpe']:+.2f}  (IS {b_ung['sharpe_is']:+.2f} / "
          f"OOS {b_ung['sharpe_oos']:+.2f})  maxDD {b_ung['max_dd']:+.1%}")

    results = {"baseline_ungated": b_ung}
    proba_by = {}
    print("\n[A] MODEL × WEIGHTING (gate, threshold 0.55):")
    for mname, fac in models().items():
        for wlab, wt in [("unw", False), ("uniqW", True)]:
            pc = proba_cache(sleeves, fac, wt)
            proba_by[(mname, wlab)] = pc
            g, pr = gated_book(sleeves, pc, 0.55)
            bs, _ = book_stats(g)
            results[f"{mname}_{wlab}_gate"] = {**bs, "prec_base": float(np.nanmean(pr[:, 0])),
                                               "prec_gate": float(np.nanmean(pr[:, 1]))}
            print(f"    {mname+'_'+wlab:20s}: Sharpe {bs['sharpe']:+.2f} (IS {bs['sharpe_is']:+.2f}/"
                  f"OOS {bs['sharpe_oos']:+.2f})  maxDD {bs['max_dd']:+.1%}  "
                  f"precision {np.nanmean(pr[:,0]):.0%}->{np.nanmean(pr[:,1]):.0%}", flush=True)

    print("\n[B] THRESHOLD sweep (LightGBM, unweighted, gate):")
    pc = proba_by[("lightgbm", "unw")]
    for th in [0.50, 0.55, 0.60]:
        g, pr = gated_book(sleeves, pc, th)
        bs, _ = book_stats(g)
        results[f"lightgbm_unw_gate_thr{int(th*100)}"] = bs
        print(f"    thr {th:.2f}: Sharpe {bs['sharpe']:+.2f} (IS {bs['sharpe_is']:+.2f}/OOS {bs['sharpe_oos']:+.2f})  "
              f"maxDD {bs['max_dd']:+.1%}  precision {np.nanmean(pr[:,0]):.0%}->{np.nanmean(pr[:,1]):.0%}")

    print("\n[C] CONTINUOUS SIZING vs binary gate (LightGBM unweighted):")
    sized = sized_book(sleeves, pc)
    bs_sz, _ = book_stats(sized)
    results["lightgbm_unw_sized"] = bs_sz
    print(f"    sized : Sharpe {bs_sz['sharpe']:+.2f} (IS {bs_sz['sharpe_is']:+.2f}/OOS {bs_sz['sharpe_oos']:+.2f})  "
          f"maxDD {bs_sz['max_dd']:+.1%}")

    print("\n[D] per-TF rescue (LightGBM unw, gate thr 0.55): does ML lift the weak fast TFs?")
    for tf in TFS_ML:
        sub = {k: v for k, v in sleeves.items() if v["tf"] == tf}
        if not sub:
            continue
        g, _ = gated_book(sub, {k: pc[k] for k in sub}, 0.55)
        bg, _ = book_stats(g)
        bu, _ = book_stats(pd.DataFrame({k: s["ung"] for k, s in sub.items()}))
        print(f"    {tf}: ungated {bu['sharpe']:+.2f}  ->  gated {bg['sharpe']:+.2f}  "
              f"(OOS {bu['sharpe_oos']:+.2f} -> {bg['sharpe_oos']:+.2f})")

    (bo.REPORTS / "trend" / "trend_ml.json").write_text(json.dumps(results, indent=2, default=float))
    best = max((k for k in results if k != "baseline_ungated"), key=lambda k: results[k]["sharpe"])
    print(f"\nincremental value: best variant = {best}  Sharpe {results[best]['sharpe']:+.2f} vs ungated "
          f"{b_ung['sharpe']:+.2f} ({results[best]['sharpe']-b_ung['sharpe']:+.2f}); "
          f"OOS {results[best].get('sharpe_oos', float('nan')):+.2f} vs {b_ung['sharpe_oos']:+.2f}; "
          f"maxDD {results[best]['max_dd']:+.1%} vs {b_ung['max_dd']:+.1%}")
    print("TREND ML OK")


if __name__ == "__main__":
    main()
