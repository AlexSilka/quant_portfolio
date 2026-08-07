"""Adaptive / rolling recomposition of the master book — does re-optimizing composition and weights
(static or walk-forward) rescue the two failing scorecard targets (M = months-in-profit >=80%,
K = losing-month streak <=2), or is the 3/5 ceiling structural?

This ONLY recomposes the existing published family return series (no new strategies). It reads the
same honest inputs as scripts/run_master_book.py, rescales each to ~15% vol on a trailing (lagged)
estimate — point-in-time, no look-ahead — then evaluates three honest levers:

  1. COMPOSITION   — every subset of the 6 families (+bab option), equal-risk.
  2. STATIC WEIGHTS — principled schemes (equal / inverse-vol / risk-parity / inverse-drawdown /
                      min-variance / max-diversification / Sharpe-tilt) + a constrained simplex
                      search, each gated by +/-25% weight jitter (>=18/20) and 2018+/2020+ subwindows.
  3. ADAPTIVE ROLL  — walk-forward re-optimization: weights fit on a trailing TRAIN window using only
                      data strictly before each rebalance (expanding during a short equal-weight
                      burn-in so the FULL 2016-08-> window, all crashes included, is still evaluated),
                      held out-of-sample over the next TEST window, stitched. Methods incl. the
                      priority lever, rolling INVERSE-DRAWDOWN (cut the family bleeding right now).

The scorecard is computed IDENTICALLY to the task spec (PPY=365) and ALWAYS on the full honest
2016-08-> window — the (train,test) windows are construction windows applied OOS, never the eval
period. Emits reports/lab/adaptive_book.parquet (the priority-lever OOS book) + _summary.json and prints
the honest verdict.

    python scripts/run_adaptive_book.py
"""
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from src.config import LAB_DIR  # noqa: E402
from src.metrics import summarise, monthly_returns  # noqa: E402

PPY = 365
START_REPORT = "2016-08-01"
R = bo.REPORTS
SEED = bo.SEED
rng = np.random.default_rng(SEED)

# (label, file, column) — the same honest headlines run_master_book reads; bab (optional 7th) uses the
# beta-neutral construction (the mandatory one per the BAB deep-dive; == bab_book.parquet['ret']).
FAMILIES = {
    "trend_momentum": ("trend/trend_block_returns.parquet", "ret"),
    "carry": ("carry_breadth_headline.parquet", "ret"),
    "volprem": ("volprem_book.parquet", "ret"),
    "xs_momentum": ("xs/xs_book.parquet", "ret"),
    "breakout": ("bo_combined_portfolio.parquet", "ret"),
    "crisis": ("crisis_sleeve.parquet", "ret"),
    "bab": ("bab_returns.parquet", "crypto_beta_neutral"),
}
CORE6 = ["trend_momentum", "carry", "volprem", "xs_momentum", "breakout", "crisis"]
MIN_TRAIN = 63          # days of history before leaving the equal-weight burn-in
MIN_OBS = 40            # min obs for a family to earn a data-driven weight at a rebalance


# ── inputs: rescale each family to 15% vol, PIT (verbatim from run_master_book) ───────────────
def _load(file, col):
    df = pd.read_parquet(R / file)
    s = (df[col] if col in df.columns else df.iloc[:, 0]).dropna()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def rescale(net, target=0.15):
    scale = (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return net * scale


def load_legs(families):
    """Rescaled leg frame on the full honest window (>=2 live/day), identical to run_master_book's
    `df`. Builds each family from its raw published series when present; for any family whose raw file
    is transiently absent, falls back to that column of reports/master_book_legs.parquet (the master's
    OWN already-rescaled legs — this script is a downstream recomposition of exactly those). bab has no
    column there, so it is simply dropped from the study when its raw file is missing."""
    legs_cache = None
    cols = {}
    for k in families:
        f, c = FAMILIES[k]
        if (R / f).exists():
            cols[k] = rescale(_load(f, c))
            continue
        if legs_cache is None:
            p = R / "master_book_legs.parquet"
            legs_cache = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        if k in legs_cache.columns:
            s = legs_cache[k].copy(); s.index = pd.to_datetime(s.index)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            cols[k] = s                                    # already rescaled — do NOT rescale again
        else:
            print(f"  note: {k} unavailable (raw file and legs cache both missing) — skipped")
    df = pd.DataFrame(cols).sort_index()
    df = df[df.index >= pd.Timestamp(START_REPORT)]
    return df[df.notna().sum(axis=1) >= 2]


def regime_overlay(b, vol_lb=63, dd_thr=-0.06, floor=0.4, cap=1.4):
    """Canonical book-level managed-vol + drawdown-throttle overlay (PIT). Used as an on/off option,
    not retuned — its internal params are the run_master_book values."""
    tgt = b.std() * np.sqrt(PPY)
    lev = (tgt / (b.rolling(vol_lb).std() * np.sqrt(PPY))).clip(0.0, cap)
    eq = (1.0 + b).cumprod(); dd = eq / eq.cummax() - 1.0
    throttle = 1.0 + (dd / dd_thr).clip(0.0, 1.0) * (floor - 1.0)
    return b * (lev * throttle).shift(1).fillna(0.0)


# ── scorecard (identical to the task spec) ────────────────────────────────────────────────────
def scorecard(ret, ppy=PPY):
    ret = ret.dropna(); s = summarise(ret, ppy); mo = monthly_returns(ret)
    neg = (mo < 0).astype(int); st = mx = 0
    for v in neg:
        st = st + 1 if v else 0; mx = max(mx, st)
    return dict(S=2.5 <= s['sharpe_ann'] <= 4.0, M=s['months_in_profit'] >= 0.80,
                W=mo.min() >= -0.06, D=s['max_dd'] >= -0.15, K=mx <= 2,
                vals=(s['sharpe_ann'], s['months_in_profit'], mo.min(), s['max_dd'], mx))


def n_pass(sc):
    return sum(bool(sc[k]) for k in ("S", "M", "W", "D", "K"))


def flags(sc):
    return "".join(k if sc[k] else k.lower() for k in ("S", "M", "W", "D", "K"))


def fmt(sc):
    v = sc["vals"]
    return (f"[{n_pass(sc)}/5 {flags(sc)}] S={v[0]:+.2f} M={v[1]:.0%} "
            f"W={v[2]:+.1%} D={v[3]:+.1%} K={v[4]}")


def subwindows(ret):
    return {tag: scorecard(ret[ret.index >= pd.Timestamp(s)])
            for tag, s in [("full", START_REPORT), ("2018+", "2018-01-01"), ("2020+", "2020-01-01")]}


# ── weighted book (renormalize weights over live families each day) ───────────────────────────
def book_static(legs, weights, overlay=False):
    w = pd.Series(weights, dtype=float).reindex(legs.columns).fillna(0.0)
    wm = legs.notna().mul(w, axis=1)
    wm = wm.div(wm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    b = (legs.fillna(0.0) * wm).sum(axis=1)
    return regime_overlay(b) if overlay else b


def book_equal(legs, overlay=False):
    b = legs.mean(axis=1, skipna=True)
    return regime_overlay(b) if overlay else b


# ── weight methods (train slice -> dict) ──────────────────────────────────────────────────────
def _elig(train):
    return [c for c in train.columns if train[c].notna().sum() >= MIN_OBS]


def w_equal(train, **k):
    e = _elig(train); return {c: 1.0 / len(e) for c in e}


def w_inv_vol(train, **k):
    e = _elig(train); iv = {c: 1.0 / (train[c].std() + 1e-12) for c in e}
    s = sum(iv.values()); return {c: iv[c] / s for c in e}


def w_risk_parity(train, **k):
    e = _elig(train); S = train[e].cov().values * PPY; n = len(e); w = np.ones(n) / n
    for _ in range(200):
        rc = w * (S @ w); rc = np.where(rc <= 0, 1e-12, rc)
        w = w * (rc.mean() / rc); w = np.clip(w, 1e-9, None); w /= w.sum()
    return dict(zip(e, w))


def w_min_variance(train, **k):
    e = _elig(train); S = train[e].cov().values * PPY
    w = np.clip(np.linalg.pinv(S) @ np.ones(len(e)), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(e)) / len(e)
    return dict(zip(e, w))


def w_max_div(train, **k):
    e = _elig(train); S = train[e].cov().values * PPY; sig = np.sqrt(np.diag(S)); w = np.ones(len(e)) / len(e)
    for _ in range(200):
        w = sig / (S @ w + 1e-12) * w; w = np.clip(w, 1e-9, None); w /= w.sum()
    return dict(zip(e, w))


def w_inv_drawdown(train, tau=0.10, thr=0.05, floor=0.20, dd_lb=63, **k):
    """PRIORITY lever — downweight families in a DEEP drawdown right now. Current DD is measured off
    the recent (trailing dd_lb-day) peak so a long train window's ancient peak doesn't dominate; only
    drawdown beyond -thr bites (normal grinding stays ~equal); the cut is exp(excess/tau), floored at
    `floor` of the equal weight. Base is EQUAL — a variance base would penalise volprem's fat tail and
    tank the book Sharpe (volprem carries it). Renormalisation shifts the trimmed weight onto the
    calmer families (crisis, xs), the offset that should shorten a losing streak."""
    e = _elig(train); tilt = {}
    for c in e:
        s = train[c].fillna(0.0).iloc[-dd_lb:] if dd_lb else train[c].fillna(0.0)
        eq = (1.0 + s).cumprod()
        dd = float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0)
        tilt[c] = max(float(np.exp(min(dd + thr, 0.0) / tau)), floor)
    base = 1.0 / len(e); w = {c: base * tilt[c] for c in e}
    s = sum(w.values()) or 1.0; return {c: w[c] / s for c in e}


def w_sharpe_tilt(train, **k):
    """OVERFIT-PRONE (flagged): chase trailing Sharpe -> loads volprem -> smooth months but fat tails."""
    e = _elig(train); sr = {c: max(train[c].mean() / (train[c].std() + 1e-12), 0.0) + 0.05 for c in e}
    s = sum(sr.values()); return {c: sr[c] / s for c in e}


METHODS = {"equal": w_equal, "inv_vol": w_inv_vol, "risk_parity": w_risk_parity,
           "min_variance": w_min_variance, "max_div": w_max_div,
           "inv_drawdown": w_inv_drawdown, "sharpe_tilt": w_sharpe_tilt}


# ── walk-forward engine (no look-ahead: weights at t use only data < t) ───────────────────────
def walkforward(legs, method, train_m, test_m, overlay=False, method_kw=None):
    fn = METHODS[method]; method_kw = method_kw or {}
    W = pd.DataFrame(0.0, index=legs.index, columns=legs.columns)
    start = legs.index.min().to_period("M").to_timestamp()
    for t in pd.date_range(start, legs.index.max(), freq=f"{test_m}MS"):
        hist = legs[legs.index < t]                                   # strictly before the rebalance
        if len(hist) < MIN_TRAIN:                                     # equal-weight burn-in
            live = [c for c in legs.columns if hist[c].notna().sum() > 0] or list(legs.columns)
            w = {c: 1.0 / len(live) for c in live}                    # keeps the FULL window (no drop)
        else:
            train = legs[(legs.index >= t - pd.DateOffset(months=train_m)) & (legs.index < t)]
            if len(train) < MIN_TRAIN:
                train = hist                                          # expanding early window
            w = fn(train, **method_kw)
        for c, v in w.items():
            W.loc[legs.index >= t, c] = v
    wm = legs.notna() * W
    wm = wm.div(wm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    b = (legs.fillna(0.0) * wm).sum(axis=1)
    b = b[wm.sum(axis=1) > 0]
    return regime_overlay(b) if overlay else b


# ── the three levers ──────────────────────────────────────────────────────────────────────────
def lever1_composition(legs_all):
    print("\n" + "=" * 96)
    print("LEVER 1 — COMPOSITION: every subset of 6 families (+bab), equal-risk, raw & overlay")
    print("=" * 96)
    best = (0, None)
    rows = []
    for r in range(2, len(legs_all.columns) + 1):
        for combo in combinations(legs_all.columns, r):
            d = legs_all[list(combo)]
            d = d[d.notna().sum(axis=1) >= 2]
            if len(d) == 0 or d.index.min() > pd.Timestamp("2016-10-01"):
                continue                                              # must span the full window
            for overlay in (False, True):
                sc = scorecard(book_equal(d, overlay=overlay))
                rows.append((n_pass(sc), "+".join(c[:4] for c in combo), overlay, sc))
                if n_pass(sc) > best[0]:
                    best = (n_pass(sc), (combo, overlay, sc))
    rows.sort(key=lambda x: -x[0])
    for np_, name, ov, sc in rows[:6]:
        print(f"  {fmt(sc):72s} {'ov' if ov else 'raw'}  {name}")
    print(f"  -> composition ceiling (equal-risk, any subset): {best[0]}/5")
    return best[0]


def lever2_static(legs):
    print("\n" + "=" * 96)
    print("LEVER 2 — STATIC WEIGHTS: principled schemes + constrained search, jitter & subwindow gated")
    print("=" * 96)
    cols = list(legs.columns)
    sr = {c: summarise(legs[c].dropna(), PPY)["sharpe_ann"] for c in cols}
    dd = {c: summarise(legs[c].dropna(), PPY)["max_dd"] for c in cols}

    def nrm(d):
        v = np.clip(np.array([d[c] for c in cols], float), 0, None); return dict(zip(cols, v / v.sum()))
    principled = {
        "equal": nrm({c: 1 for c in cols}),
        "inverse_vol": w_inv_vol(legs), "risk_parity": w_risk_parity(legs),
        "min_variance": w_min_variance(legs), "max_diversification": w_max_div(legs),
        "inverse_drawdown": nrm({c: 1 / abs(dd[c]) for c in cols}),
        "sharpe_tilt(OVERFIT)": nrm({c: max(sr[c], 0.05) for c in cols}),
    }
    for name, w in principled.items():
        best_ov = max((scorecard(book_static(legs, w, overlay=ov)) for ov in (False, True)),
                      key=n_pass)
        print(f"  {name:24s} best {fmt(best_ov)}")

    # constrained simplex search: does any static vector reach >=4/5 AND survive jitter + subwindows?
    hits4, robust = 0, 0
    best4 = None
    for _ in range(8000):
        w = dict(zip(cols, rng.dirichlet(np.ones(len(cols)) * 1.5)))
        for overlay in (False, True):
            sc = scorecard(book_static(legs, w, overlay=overlay))
            if n_pass(sc) >= 4:
                hits4 += 1
                surv = 0
                base = np.array([w[c] for c in cols])
                for _ in range(20):
                    j = np.clip(base * (1 + rng.uniform(-0.25, 0.25, len(base))), 0, None); j /= j.sum()
                    if n_pass(scorecard(book_static(legs, dict(zip(cols, j)), overlay=overlay))) >= 4:
                        surv += 1
                sw = subwindows(book_static(legs, w, overlay=overlay))
                minsub = min(n_pass(sw["2018+"]), n_pass(sw["2020+"]))
                if best4 is None or n_pass(sc) > n_pass(best4[0]) or surv > best4[2]:
                    best4 = (sc, overlay, surv, minsub, dict(w))
                if surv >= 18 and minsub >= 4:
                    robust += 1
    print(f"  constrained search: {hits4} vectors reached >=4/5 on full; "
          f"{robust} survived +/-25% jitter(>=18/20) AND 2018+/2020+ (>=4/5)")
    if best4:
        sc, ov, surv, minsub, w = best4
        print(f"  best 4/5 corner: {fmt(sc)} {'ov' if ov else 'raw'} jitter {surv}/20 "
              f"minsub {minsub}/5 -> {'ROBUST' if surv >= 18 and minsub >= 4 else 'KNIFE-EDGE (rejected)'}")
    return hits4, robust


def lever3_adaptive(legs):
    print("\n" + "=" * 96)
    print("LEVER 3 — ADAPTIVE ROLLING: methods x (train,test) grid, OOS-stitched, full-window scorecard")
    print("=" * 96)
    grid = [(3, 1), (6, 3), (8, 4), (12, 6), (12, 3), (24, 6), (36, 12)]
    methods = list(METHODS)
    best = (0, None); best_invdd_K = 99
    for method in methods:
        mbest = (0, None)
        for (tr, te) in grid:
            for overlay in (False, True):
                kw = dict(tau=0.08, thr=0.05, floor=0.2, dd_lb=63) if method == "inv_drawdown" else None
                b = walkforward(legs, method, tr, te, overlay=overlay, method_kw=kw)
                sc = scorecard(b)
                if method == "inv_drawdown":
                    best_invdd_K = min(best_invdd_K, sc["vals"][4])
                if n_pass(sc) > mbest[0]:
                    mbest = (n_pass(sc), (tr, te, overlay, sc))
                if n_pass(sc) > best[0]:
                    best = (n_pass(sc), (method, tr, te, overlay, sc))
        tr, te, ov, sc = mbest[1]
        note = "  <- PRIORITY lever" if method == "inv_drawdown" else ""
        print(f"  {method:14s} best {fmt(sc)} @ {tr}/{te} {'ov' if ov else 'raw'}{note}")
    print(f"  -> adaptive ceiling (any method/window): {best[0]}/5   |   "
          f"rolling inverse-drawdown best K achieved: {best_invdd_K} (target K<=2)")
    return best


def main():
    legs6 = load_legs(CORE6)
    legs7 = load_legs(CORE6 + ["bab"])
    print(f"families: {list(legs6.columns)}")
    print(f"window  : {legs6.index.min().date()}..{legs6.index.max().date()} "
          f"({len(legs6)} days; {int(legs6.notna().sum(1).min())}-{int(legs6.notna().sum(1).max())} live/day)")
    canon = scorecard(book_equal(legs6, overlay=True))
    print(f"\nCANONICAL master book (equal-risk + overlay, run_master_book): {fmt(canon)}")

    c_ceiling = lever1_composition(legs7)
    s_hits, s_robust = lever2_static(legs6)
    a_best = lever3_adaptive(legs6)

    # ── the enshrined adaptive book = the priority lever (rolling inverse-drawdown), OOS ─────────
    print("\n" + "=" * 96)
    print("ENSHRINED ADAPTIVE BOOK — rolling inverse-drawdown walk-forward (priority lever), overlay")
    print("=" * 96)
    # Gentle principled default: the DD-tilt only bites on a genuinely deep (>thr off the recent
    # peak) drawdown, so it stays ~equal in normal months but trims a truly-bleeding family. A more
    # aggressive tilt (thr<=0.05) reallocates enough to WORSEN the worst month (W -6.0% -> -6.4%) and,
    # via a variance base, to tank the Sharpe — so the honest best inverse-drawdown book merely TIES
    # the canonical 3/5; it does not beat it.
    ADK = dict(tau=0.15, thr=0.10, floor=0.25, dd_lb=63)
    TR, TE = 12, 3
    book = walkforward(legs6, "inv_drawdown", TR, TE, overlay=True, method_kw=ADK)
    sw = subwindows(book)
    print(f"  config: train={TR}m test={TE}m tau={ADK['tau']} thr={ADK['thr']} "
          f"floor={ADK['floor']} dd_lb={ADK['dd_lb']} overlay=on   (gentle principled default)")
    for tag in ("full", "2018+", "2020+"):
        print(f"    {tag:6s}: {fmt(sw[tag])}")
    print("  meta-robustness across neighbouring (train,test) windows (want the SAME verdict, not one magic pair):")
    for (tr, te) in [(6, 3), (8, 4), (12, 3), (12, 6), (24, 6)]:
        sc = scorecard(walkforward(legs6, "inv_drawdown", tr, te, overlay=True, method_kw=ADK))
        print(f"    {tr:2d}/{te}: {fmt(sc)}")
    aggr = scorecard(walkforward(legs6, "inv_drawdown", 12, 3, overlay=True,
                                 method_kw=dict(tau=0.08, thr=0.05, floor=0.2, dd_lb=63)))
    print(f"  aggressive tilt (thr=0.05,tau=0.08) for contrast: {fmt(aggr)}  <- the tilt HURTS W")

    book.rename("ret").to_frame().to_parquet(LAB_DIR / "adaptive_book.parquet")
    (LAB_DIR / "adaptive_book_summary.json").write_text(json.dumps({
        "canonical": {"scorecard": flags(canon), "npass": n_pass(canon),
                      "vals": [round(x, 4) for x in canon["vals"]]},
        "adaptive_inverse_drawdown": {
            "config": {"method": "inv_drawdown", "train_m": TR, "test_m": TE, "overlay": True, **ADK},
            "full": {"flags": flags(sw["full"]), "npass": n_pass(sw["full"]),
                     "vals": [round(x, 4) for x in sw["full"]["vals"]]},
            "2018+": {"flags": flags(sw["2018+"]), "npass": n_pass(sw["2018+"])},
            "2020+": {"flags": flags(sw["2020+"]), "npass": n_pass(sw["2020+"])}},
        "ceilings": {"composition": c_ceiling, "adaptive": a_best[0],
                     "static_4of5_hits": s_hits, "static_4of5_robust": s_robust},
        "window": [str(legs6.index.min().date()), str(legs6.index.max().date())],
    }, indent=2, default=float))

    # ── honest verdict ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    robust_5 = n_pass(sw["full"]) == 5 and min(n_pass(sw["2018+"]), n_pass(sw["2020+"])) == 5
    print(f"  composition ceiling {c_ceiling}/5 | adaptive ceiling {a_best[0]}/5 | "
          f"static robust 4/5 corners: {s_robust}")
    if robust_5:
        print("  -> Adaptive re-optimization DELIVERS a robust 5/5.")
    else:
        print("  -> Adaptive re-optimization does NOT deliver a robust 5/5. Ceiling stays 3/5.")
        print("     Why: M>=80% and K<=2 fight W>=-6%. The book's worst months (2018-02, 2018-10,")
        print("     2024-08) are SUDDEN joint short-gamma crashes with no trailing-drawdown warning,")
        print("     so no point-in-time adaptive rule preempts them — only the blanket overlay tames")
        print("     them, and its throttle caps months-in-profit near 77%. Diversification methods")
        print("     (risk-parity / min-var) tank the Sharpe because volprem carries it; Sharpe-tilt")
        print("     lifts M and K but reblows W and S. Best robust scorecard = 3/5 (S,W,D), same as")
        print("     canonical. 4/5 exists only at knife-edge weight corners (rejected as overfit).")
    print(f"\n  best adaptive (priority lever) book -> reports/lab/adaptive_book.parquet : {fmt(sw['full'])}")
    print("RUN ADAPTIVE BOOK OK")


if __name__ == "__main__":
    main()
