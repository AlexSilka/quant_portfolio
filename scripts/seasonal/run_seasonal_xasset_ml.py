"""Calendar seasonality — cross-asset relative-value + ML extensions (follow-up to run_seasonal.py).

Two questions the headline study did not answer:

  A. **Can it be traded *between* assets, market-neutral?** The headline verdict was "the calendar
     premium is beta". A dollar-neutral long/short *across* names, live only inside the window, removes
     that beta by construction — long the names that respond MORE to the event, short those that respond
     LESS (ranked on each name's own trailing in-window history = a "seasonal-momentum" signal). If a
     cross-sectional spread survives, there is a real relative-value edge the time-series book hid; if it
     is ~0, the effect is uniform beta with no cross-sectional structure. Run on crypto (pre-FOMC + ToM)
     and US stocks (ToM), through placebo + cost + corr-to-book.

  B. **Does ML rescue it?** Two leakage-controlled variants:
     B1. *Conditional pre-FOMC* (the brief's §5 ML-forecast, as a timing gate): Lucca-Moench show the
         drift is stronger when the yield curve is flat / implied vol is high / recent drift was high.
         Predict each event's outcome from those conditioners (VIX, 10y-2y slope, trailing drift,
         momentum, realised vol) with a purged event-CV; trade only predicted-positive events; compare
         to unconditional. SPY + BTC.
     B2. *Cross-sectional ML ranker* traded in-window: reuse the sleeve's learning-to-rank stack
         (xsect_ml) — GBM on the 15-feature name panel, expanding walk-forward OOS — and ask whether a
         dollar-neutral book run ONLY on window bars beats the same book run every day (is the calendar
         window a better time to run a cross-sectional book?).

Verdict → reports/seasonal_xasset_ml_summary.json + docs/strategies/SEASONAL.md §7-8; figure → figures/seasonal_ml.png.

    python scripts/seasonal/run_seasonal_xasset_ml.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")

import scripts.seasonal.run_seasonal as rs  # noqa: E402  (reuse the headline study's loaders/config)
from src.config import FIGURES_DIR, REPORTS_DIR, SEASONAL_DIR, SEED  # noqa: E402
from src.data.fomc import announce_days  # noqa: E402
from src.data.rates import _fred_csv  # noqa: E402
from src.sleeves import seasonal as sz  # noqa: E402
from src.sleeves.xsect import xs_backtest  # noqa: E402
from src.sleeves.xsect_ml import expanding_predict, predictions_to_panel, rank_features, stack_xy  # noqa: E402

REP, FIG = REPORTS_DIR, FIGURES_DIR
rng = np.random.default_rng(SEED)
TOPFRAC, K_TRAIL = 0.3, 8      # x-sect tails; trailing #instances for the seasonal-momentum signal


def _sh(x, ppy):
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1)
    return float(np.sqrt(ppy) * x.mean() / sd) if sd > 0 and len(x) > 2 else 0.0


# ════════════════════════════════════════════════════════════════════════════════════════════
# PART A — cross-asset relative-value (dollar-neutral long/short BETWEEN names, in-window only)
# ════════════════════════════════════════════════════════════════════════════════════════════
def instance_matrix(ret: pd.DataFrame, inst: pd.Series) -> pd.DataFrame:
    """Instances × names simple-return matrix: compound each name over each window episode's bars."""
    lg = np.log1p(ret.clip(-0.99, None)).where(inst.notna(), np.nan)
    return np.expm1(lg.groupby(inst).sum(min_count=1))


def _rv_signal(R: pd.DataFrame, k_trail: int = K_TRAIL) -> pd.DataFrame:
    """Seasonal-momentum signal: each name's mean in-window return over the past `k_trail` instances
    (shifted one → strictly past, no look-ahead)."""
    return R.shift(1).rolling(k_trail, min_periods=max(3, k_trail // 2)).mean()


def _rv_book(R: pd.DataFrame, S: pd.DataFrame, cost_bps: float, top_frac: float = TOPFRAC) -> pd.Series:
    """Dollar-neutral long-top/short-bottom book on signal S earning R; full round-trip each instance."""
    ranks = S.rank(axis=1, pct=True)
    nval = S.notna().sum(axis=1)
    longs = (ranks >= 1 - top_frac) & (nval.values[:, None] >= 6)
    shorts = (ranks <= top_frac) & (nval.values[:, None] >= 6)
    wl = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0)
    ws = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).fillna(0.0)
    gross = (w * R).sum(axis=1)
    return gross - 2.0 * w.abs().sum(axis=1) * cost_bps / 1e4      # build+unwind each instance (flat between)


def seasonal_rv(R: pd.DataFrame, cost_bps: float, ppy_inst: float, k_trail: int = K_TRAIL,
                top_frac: float = TOPFRAC) -> tuple[pd.Series, pd.Series]:
    S = _rv_signal(R, k_trail)
    return _rv_book(R, S, cost_bps, top_frac), S


def _placebo_rv(R, cost_bps, ppy_inst, n_iter=200):
    """Shuffle the SIGNAL across names while keeping returns real — the honest placebo.

    A consistent column-permutation of R leaves a dollar-neutral book invariant (it only relabels names);
    the real test is to mis-assign each name's seasonal-momentum signal to a *different* name's returns,
    which destroys the cross-section but keeps every marginal. Real must beat the ~95th percentile.
    """
    S = _rv_signal(R)
    out = []
    for _ in range(n_iter):
        Sp = S.copy()
        Sp.columns = rng.permutation(S.columns)
        Sp = Sp.reindex(columns=S.columns)
        out.append(_sh(_rv_book(R, Sp, cost_bps), ppy_inst))
    return np.array(out)


def part_a() -> dict:
    print(f"\n{'='*82}\nPART A — CROSS-ASSET RELATIVE-VALUE (dollar-neutral long/short between names)\n{'='*82}")
    out, books = {}, {}
    specs = [
        ("crypto_pre_fomc", "crypto_1d", 365, rs.CR_COST, "fomc", 8),
        ("crypto_turn_of_month", "crypto_1d", 365, rs.CR_COST, "tom", 12),
        ("stocks_turn_of_month", "stocks_broad_1d", 252, rs.EQ_COST, "tom", 12),
    ]
    for label, tag, ppy, cost, kind, ppy_inst in specs:
        C, _ = rs._panel(tag)
        ret = C.pct_change(fill_method=None)
        if kind == "fomc":
            inst = sz.window_instances(C.index, announce_days("UTC"), rs.FOMC_OFFSETS)
        else:
            anch = sz.month_end_anchors(C.index)
            inst = sz.window_instances(C.index, anch, sz.turn_of_month_offsets(rs.TOM_BEFORE, rs.TOM_AFTER))
        R = instance_matrix(ret, inst)
        R = R.dropna(how="all").loc[:, R.notna().sum() >= 3 * K_TRAIL]     # names with enough history
        net, _ = seasonal_rv(R, cost, ppy_inst)
        sh = _sh(net, ppy_inst)
        placebo = _placebo_rv(R, cost, ppy_inst)
        pctile = float((sh > placebo).mean() * 100)
        # net.index is the instance-id (integer bar-position of the anchor), not a date — map it back
        idx_dates = C.index[net.index.astype(int)]
        per_year = {int(y): round(_sh(net.values[idx_dates.year == y], ppy_inst), 2)
                    for y in sorted(set(idx_dates.year))}
        out[label] = {"n_instances": int(len(net.dropna())), "n_names": int(R.shape[1]),
                      "net_sharpe": round(sh, 3), "placebo_pctile": round(pctile, 0),
                      "placebo_mean": round(float(placebo.mean()), 3),
                      "placebo_p95": round(float(np.percentile(placebo, 95)), 3), "per_year": per_year}
        s = pd.Series(net.values, index=idx_dates)      # date-indexed for corr-to-book
        books[label] = s
        print(f"  {label:22s}: {R.shape[1]} names, {len(net.dropna())} instances | RV net Sharpe {sh:+.2f} | "
              f"placebo p{pctile:.0f} (mean {placebo.mean():+.2f}, p95 {np.percentile(placebo,95):+.2f})")
    return {"summary": out, "_books": books}


# ════════════════════════════════════════════════════════════════════════════════════════════
# PART B1 — conditional pre-FOMC ML (event-level, purged CV): does conditioning rescue the timing?
# ════════════════════════════════════════════════════════════════════════════════════════════
def _event_table(px: pd.Series, ann: pd.DatetimeIndex, vix: pd.Series, slope: pd.Series) -> pd.DataFrame:
    """One row per FOMC event: target = day-before return; features known as of 2 bars before announce."""
    ret = px.pct_change(fill_method=None)
    idx = px.index
    rows = []
    locs = idx.searchsorted(ann)
    for a in locs:
        if a < 65 or a >= len(idx):
            continue
        t2 = a - 2                                    # feature cut-off: close 2 trading days before announce
        tgt = idx[a - 1]                              # day-before bar (the tradable window)
        feat_t = idx[t2]
        rows.append({
            "date": tgt, "target": float(ret.loc[tgt]),
            "vix": float(vix.asof(feat_t)) if len(vix) else np.nan,
            "vix_chg5": float(vix.asof(feat_t) - vix.asof(idx[t2 - 5])) if len(vix) else np.nan,
            "slope": float(slope.asof(feat_t)) if len(slope) else np.nan,
            "mom20": float(px.iloc[t2] / px.iloc[t2 - 20] - 1.0),
            "mom60": float(px.iloc[t2] / px.iloc[t2 - 60] - 1.0),
            "rvol20": float(ret.iloc[t2 - 20:t2].std()),
            "drift3": float(np.nanmean([ret.loc[idx[locs[j] - 1]] for j in range(max(0, list(locs).index(a) - 3),
                            list(locs).index(a))])) if list(locs).index(a) >= 3 else 0.0,
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def _event_cv(X: np.ndarray, y: np.ndarray, factory, k: int = 5, embargo: int = 1) -> np.ndarray:
    """Purged K-fold OOS predictions for time-ordered events (contiguous folds, embargo neighbours)."""
    n = len(y)
    pred = np.full(n, np.nan)
    fb = [int(round(i * n / k)) for i in range(k + 1)]
    for i in range(k):
        te = np.zeros(n, bool); te[fb[i]:fb[i + 1]] = True
        tr = ~te
        lo, hi = max(0, fb[i] - embargo), min(n, fb[i + 1] + embargo)
        tr[lo:hi] = False                              # embargo the neighbours of the test block
        if tr.sum() < 20:
            continue
        m = factory(); m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.predict(X[te])
    return pred


def part_b1() -> dict:
    print(f"\n{'='*82}\nPART B1 — CONDITIONAL PRE-FOMC ML (event-level, purged CV)\n{'='*82}")
    vix = _fred_csv("VIXCLS"); slope = _fred_csv("T10Y2Y")
    for s in (vix, slope):
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
    ann = announce_days("UTC")
    feat_cols = ["vix", "vix_chg5", "slope", "mom20", "mom60", "rvol20", "drift3"]
    out = {}
    for label, px in [("SPY", rs._etf_close("SPY")), ("BTC", rs._panel("crypto_1d")[0]["BTCUSDT"])]:
        ppy_inst = 8
        T = _event_table(px, ann, vix, slope)
        if len(T) < 40:
            continue
        X, y = T[feat_cols].to_numpy(), T["target"].to_numpy()
        uncond = _sh(y, ppy_inst)                       # always-trade the day-before window
        res = {"n_events": int(len(T)), "unconditional_sharpe": round(uncond, 3)}
        for mname, fac in [("ridge", lambda: Ridge(alpha=1.0)),
                           ("logistic", lambda: LogisticRegression(max_iter=500, C=0.5)),
                           ("lgbm", lambda: lgb.LGBMRegressor(n_estimators=120, max_depth=3,
                                                              learning_rate=0.03, min_child_samples=10, verbose=-1))]:
            yy = (y > 0).astype(int) if mname == "logistic" else y
            pred = _event_cv(X, yy, fac)
            m = np.isfinite(pred)
            thr = 0.5 if mname == "logistic" else 0.0
            traded = y[m][pred[m] > thr]                # trade only predicted-positive events
            ic = float(np.corrcoef(pred[m], y[m])[0, 1]) if m.sum() > 3 else np.nan
            res[mname] = {"conditional_sharpe": round(_sh(traded, ppy_inst), 3),
                          "n_traded": int(len(traded)), "frac_traded": round(float(len(traded) / m.sum()), 2),
                          "oos_ic": round(ic, 3)}
        out[label] = res
        best = max((res[m]["conditional_sharpe"] for m in ("ridge", "logistic", "lgbm")))
        print(f"  {label}: {len(T)} events | unconditional {uncond:+.2f} → best conditional {best:+.2f} "
              f"| ridge {res['ridge']['conditional_sharpe']:+.2f} (IC {res['ridge']['oos_ic']:+.2f}) "
              f"logit {res['logistic']['conditional_sharpe']:+.2f} lgbm {res['lgbm']['conditional_sharpe']:+.2f}")
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# PART B2 — cross-sectional ML ranker traded in-window (crypto): is the window a better time to run it?
# ════════════════════════════════════════════════════════════════════════════════════════════
def part_b2() -> dict:
    print(f"\n{'='*82}\nPART B2 — CROSS-SECTIONAL ML RANKER, all-days vs in-window (crypto)\n{'='*82}")
    C, A = rs._panel("crypto_1d")
    feats = rank_features(C, A, bpd=1)
    X, y, ts = stack_xy(feats, C, fwd_bars=1)
    print(f"  learning-to-rank: {len(y):,} name-bar rows, {X.shape[1]} features → expanding WFO (LGBM)")
    pred = expanding_predict(X, y, ts, lambda: lgb.LGBMRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.03, min_child_samples=40, subsample=0.8, verbose=-1),
        n_folds=6, embargo_bars=5)
    sig = predictions_to_panel(pred, C)
    anch = sz.month_end_anchors(C.index)
    tom = sz.window_position(C.index, anch, sz.turn_of_month_offsets(rs.TOM_BEFORE, rs.TOM_AFTER))
    fomc = sz.window_position(C.index, announce_days("UTC"), rs.FOMC_OFFSETS)
    win = ((tom + fomc) > 0)
    maskdf = pd.DataFrame(np.broadcast_to(win.values[:, None], sig.shape), index=sig.index, columns=sig.columns)
    books = {}
    for name, mask in [("ml_all_days", None), ("ml_in_window", maskdf)]:
        s = sig if mask is None else sig.where(mask)
        bt = xs_backtest(C, s, top_frac=0.3, weighting="equal", rebal=1, exec_lag=2,
                         cost_bps=rs.CR_COST, adv=A, impact_k=0.1)
        books[name] = bt["net"]
    out = {"ml_all_days_sharpe": round(_sh(books["ml_all_days"], 365), 3),
           "ml_in_window_sharpe": round(_sh(books["ml_in_window"], 365), 3),
           "frac_days_in_window": round(float(win.mean()), 3)}
    print(f"  ML cross-sectional book: all-days Sharpe {out['ml_all_days_sharpe']:+.2f} vs "
          f"in-window-only {out['ml_in_window_sharpe']:+.2f} (window = {out['frac_days_in_window']*100:.0f}% of days)")
    return out, books


def _corr_to_book(books: dict) -> dict:
    bp_path = REP / "master_book.parquet"
    if not bp_path.exists():
        return {}
    bp = pd.read_parquet(bp_path)["ret"]
    if bp.index.tz is not None:
        bp.index = bp.index.tz_localize(None)
    corr = {}
    for k, s in books.items():
        h = pd.Series(s).dropna()
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        common = h.index.intersection(bp.dropna().index)
        corr[k] = round(float(h.reindex(common).corr(bp.reindex(common))), 3) if len(common) > 30 else None
    return corr


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    a = part_a()
    b1 = part_b1()
    b2, b2_books = part_b2()
    corr = _corr_to_book({**a["_books"]})
    for k in a["summary"]:
        a["summary"][k]["corr_to_book"] = corr.get(k)

    summ = {"config": {"top_frac": TOPFRAC, "k_trailing_instances": K_TRAIL,
                       "fomc_offsets": rs.FOMC_OFFSETS, "tom": [rs.TOM_BEFORE, rs.TOM_AFTER]},
            "A_cross_asset_relative_value": a["summary"],
            "B1_conditional_pre_fomc_ml": b1, "B2_cross_sectional_ml": b2}
    (SEASONAL_DIR / "seasonal_xasset_ml_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(a["summary"], b1, b2)

    print(f"\n{'='*82}\nVERDICT (cross-asset + ML)")
    for k, v in a["summary"].items():
        print(f"  A {k:22s}: RV net {v['net_sharpe']:+.2f} (placebo p{v['placebo_pctile']:.0f}) corr-book {v.get('corr_to_book')}")
    print(f"  B1 conditional pre-FOMC: SPY uncond {b1.get('SPY',{}).get('unconditional_sharpe')} → "
          f"best cond {max((b1['SPY'][m]['conditional_sharpe'] for m in ('ridge','logistic','lgbm')), default=None) if 'SPY' in b1 else None}")
    print(f"  B2 ML x-sect: all-days {b2['ml_all_days_sharpe']:+.2f} vs in-window {b2['ml_in_window_sharpe']:+.2f}")
    print("RUN SEASONAL XASSET+ML OK")


def _figure(a, b1, b2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Calendar seasonality — cross-asset relative-value + ML (net of costs)", fontsize=13, fontweight="bold")

    # (1) RV net Sharpe vs placebo p95, by book
    aa = ax[0]
    ks = list(a.keys()); x = np.arange(len(ks)); wd = 0.38
    aa.bar(x - wd/2, [a[k]["net_sharpe"] for k in ks], wd, label="real RV", color="#2b6")
    aa.bar(x + wd/2, [a[k]["placebo_p95"] for k in ks], wd, label="placebo p95", color="#a63")
    aa.axhline(0.5, color="r", ls="--", lw=1, label="robust bar"); aa.axhline(0, color="k", lw=0.6)
    aa.set_xticks(x); aa.set_xticklabels([k.replace("_", "\n") for k in ks], fontsize=7)
    aa.set_title("A: cross-asset relative-value vs its placebo", fontsize=10); aa.legend(fontsize=7)

    # (2) conditional pre-FOMC: unconditional vs best conditional
    aa = ax[1]
    labs = list(b1.keys()); x = np.arange(len(labs)); wd = 0.28
    unc = [b1[l]["unconditional_sharpe"] for l in labs]
    ridge = [b1[l]["ridge"]["conditional_sharpe"] for l in labs]
    lgbm = [b1[l]["lgbm"]["conditional_sharpe"] for l in labs]
    aa.bar(x - wd, unc, wd, label="unconditional", color="#889")
    aa.bar(x, ridge, wd, label="ridge-gated", color="#68a")
    aa.bar(x + wd, lgbm, wd, label="lgbm-gated", color="#2b6")
    aa.axhline(0, color="k", lw=0.6); aa.set_xticks(x); aa.set_xticklabels(labs)
    aa.set_title("B1: pre-FOMC ML gate vs always-trade", fontsize=10); aa.legend(fontsize=7)

    # (3) ML cross-sectional: all-days vs in-window
    aa = ax[2]
    vals = [b2["ml_all_days_sharpe"], b2["ml_in_window_sharpe"]]
    aa.bar(["all days", "in-window\nonly"], vals, color=["#889", "#2b6"])
    aa.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(vals):
        aa.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    aa.set_title("B2: ML x-sect book — is the window a better time?", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "seasonal_ml.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
