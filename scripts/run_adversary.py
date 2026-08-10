"""Red-team / adversary harness — FALSIFY the master book, don't decorate it.

This does NOT build a nicer scorecard. It reconstructs the canonical 6-family master book
bit-exactly (by importing run_master_book's own functions), then attacks it harder than the
author would attack themselves, and — when the parallel sessions' candidate books land — runs
the same gauntlet on those.

Run:
    python scripts/run_adversary.py                 # full Task-2 stress on the canonical book
    python scripts/run_adversary.py --attack PATH   # Task-1 gauntlet on a candidate book series

Outputs (new namespace only — never touches canonical artifacts):
    reports/lab/adversary_stress.parquet    # canonical book + the crash-correlation-stressed book
    reports/lab/adversary_summary.json      # every attack's numbers, machine-readable
    docs/ADVERSARY.md                   # the written verdict (authored separately)

Everything here is mechanical (vol / trend / window / bootstrap triggers). No date is hand-picked;
the injected shocks are the volprem sleeve's OWN realized tail days, and the window attack just
moves the reporting START — the construction is never changed.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
import scripts.run_master_book as mb  # noqa: E402  — reuse the EXACT canonical assembly functions
from src.metrics import (summarise, monthly_returns, deflated_sharpe,  # noqa: E402
                         )
from src.config import LAB_DIR  # noqa: E402

PPY = mb.PPY
R = ROOT / "reports"
SEED = 7


# ----------------------------------------------------------------------------- scorecard
def scorecard(ret: pd.Series, ppy: int = PPY) -> dict:
    """The scorecard, computed IDENTICALLY to the canonical grader."""
    ret = ret.dropna()
    s = summarise(ret, ppy)
    mo = monthly_returns(ret)
    neg = (mo < 0).astype(int)
    st = mx = 0
    for v in neg:
        st = st + 1 if v else 0
        mx = max(mx, st)
    S = 2.5 <= s["sharpe_ann"] <= 4.0
    M = s["months_in_profit"] >= 0.80
    W = mo.min() >= -0.06
    D = s["max_dd"] >= -0.15
    K = mx <= 2
    return dict(S=S, M=M, W=W, D=D, K=K, n=int(S) + int(M) + int(W) + int(D) + int(K),
                sharpe=s["sharpe_ann"], months=s["months_in_profit"], worst_mo=float(mo.min()),
                max_dd=s["max_dd"], streak=int(mx), total_return=s["total_return"])


def fmt(sc: dict) -> str:
    def m(f):
        return "OK" if f else "X "
    return (f"{sc['n']}/5  S={sc['sharpe']:>4.2f}{m(sc['S'])} M={sc['months']:.2f}{m(sc['M'])} "
            f"W={sc['worst_mo']:+.4f}{m(sc['W'])} D={sc['max_dd']:+.4f}{m(sc['D'])} K={sc['streak']}{m(sc['K'])}")


# ----------------------------------------------------------------------------- reconstruction
def load_raw() -> dict:
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
    return {k: v for k, v in raw.items() if v is not None}


def rescaled_legs(raw: dict, start: str = mb.START_REPORT) -> pd.DataFrame:
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(start)]
    return df[df.notna().sum(axis=1) >= 2]


def assemble(raw: dict, start: str = mb.START_REPORT) -> pd.Series:
    df = rescaled_legs(raw, start)
    return mb.regime_overlay(mb.book_stack(df))


def weighted_mean(df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Equal-risk mean with arbitrary family weights, renormalised over the LIVE families each day."""
    A = df.values
    mask = ~np.isnan(A)
    W = np.where(mask, w[None, :], 0.0)
    s = W.sum(1)
    s[s == 0] = np.nan
    return pd.Series(np.nansum(np.where(mask, A, 0.0) * W, 1) / s, index=df.index)


# =========================================================================================
#  TASK 2 — stress the canonical book
# =========================================================================================
def task2(out: dict) -> pd.DataFrame:
    raw = load_raw()
    df = rescaled_legs(raw)
    book = mb.regime_overlay(mb.book_stack(df))

    # -- 0. verify bit-exact vs the published parquet -------------------------------------
    saved = pd.read_parquet(R / "master_book.parquet")["ret"]
    saved.index = pd.to_datetime(saved.index)
    diff = float((book.reindex(saved.index) - saved).abs().max())
    base = scorecard(book)
    print("=" * 92)
    print(f"0. RECONSTRUCTION  (max abs diff vs published master_book.parquet = {diff:.1e})")
    print(f"   canonical book: {fmt(base)}")
    out["reconstruction"] = {"max_abs_diff": diff, "scorecard": base}

    # -- 1. fragility of the passing gates ------------------------------------------------
    mo = monthly_returns(book)
    worst = (mo.sort_values().head(5) * 100).round(2)
    print("\n1. GATE FRAGILITY")
    print(f"   W passes by {(-0.06 - base['worst_mo']) * 1e4:.0f} bps: worst month {base['worst_mo']:+.4f} vs -0.0600 gate"
          f"  (month {worst.index[0].date()})")
    print(f"   worst 5 months (%): {dict((str(k.date()), v) for k, v in worst.items())}")
    out["fragility"] = {"W_margin_bps": (-0.06 - base["worst_mo"]) * 1e4,
                        "worst_months": {str(k.date()): float(v) for k, v in worst.items()}}

    # -- 2. disaster beta: book vs VIX changes --------------------------------------------
    vix = pd.read_parquet(ROOT / "data/raw/cboe/VIX.parquet")["close"]
    vix.index = pd.to_datetime(vix.index)
    dvix = vix.reindex(book.index.union(vix.index)).ffill().reindex(book.index).diff()
    reg = pd.DataFrame({"book": book, "dvix": dvix}).dropna()
    b1, b0 = np.polyfit(reg["dvix"], reg["book"], 1)
    corr = float(reg["book"].corr(reg["dvix"]))
    spikes = {}
    for thr in (3, 5, 8):
        sub = reg[reg["dvix"] >= thr]
        spikes[thr] = {"n": int(len(sub)), "mean_book": float(sub["book"].mean()),
                       "sum_book": float(sub["book"].sum())}
    print("\n2. DISASTER BETA (short-gamma exposure)")
    print(f"   book = {b0:+.5f} {b1:+.6f}*dVIX   corr={corr:+.3f}")
    for thr, d in spikes.items():
        print(f"   VIX jump>=+{thr}pt: n={d['n']:>3}  mean book/day {d['mean_book']*100:+.2f}%  cum {d['sum_book']*100:+.1f}%")
    out["disaster_beta"] = {"slope_per_vixpt": float(b1), "intercept": float(b0), "corr": corr, "spike": spikes}

    # -- 3. window-start artifact ----------------------------------------------------------
    print("\n3. WINDOW-START ARTIFACT  (SAME construction, earlier reporting START)")
    win = {}
    for start in ("2016-08-01", "2016-01-01", "2012-01-01", "2010-01-01", "2008-01-01"):
        sc = scorecard(assemble(raw, start))
        win[start] = sc
        print(f"   start {start}: {fmt(sc)}")
    print("   -> D and W pass ONLY because START=2016-08-01 sits AFTER volprem's realised -76% Flash-Crash day.")
    out["window_start"] = {k: {kk: v[kk] for kk in ('n', 'sharpe', 'worst_mo', 'max_dd', 'streak')} for k, v in win.items()}

    # -- 4. injection MC: place volprem's OWN realised tail day in-window ------------------
    print("\n4. INJECTION MC  (place volprem's realised rescaled tail on 200 random in-window days)")
    rng = np.random.default_rng(SEED)
    idx = df.index
    nlive = df.notna().sum(axis=1)
    raw_mean = mb.book_stack(df)
    inj = {}
    for name, shk in (("Flash2010(-49%)", -0.490), ("Brexit2016(-16%)", -0.160), ("Aug2015(-11%)", -0.110)):
        brW = brD = 0
        Ws = []
        for di in rng.choice(len(idx), size=200, replace=False):
            t = idx[di]
            vp_t = df["volprem"].get(t, np.nan)
            if pd.isna(vp_t):
                continue
            rm = raw_mean.copy()
            rm.loc[t] = raw_mean.loc[t] + (shk - vp_t) / nlive.loc[t]
            sc = scorecard(mb.regime_overlay(rm))
            Ws.append(sc["worst_mo"])
            brW += sc["worst_mo"] < -0.06
            brD += sc["max_dd"] < -0.15
        n = len(Ws)
        inj[name] = {"P_W_break": brW / n, "P_D_break": brD / n, "median_W": float(np.median(Ws))}
        print(f"   {name:18s}: P(W breaks)={brW/n:.0%}  P(D breaks)={brD/n:.0%}  median worst-month {np.median(Ws):+.3f}")
    out["injection_mc"] = inj

    # -- 5. overlay look-ahead: full-sample vol target vs PIT ------------------------------
    print("\n5. OVERLAY LOOK-AHEAD  (the managed-vol target level)")
    def overlay(b, mode):
        if mode == "fullsample":
            tgt = b.std() * np.sqrt(PPY)
        elif mode == "expanding":
            tgt = b.expanding(252).std() * np.sqrt(PPY)
        else:  # none
            return b
        lev = (tgt / (b.rolling(63).std() * np.sqrt(PPY))).clip(0.0, 1.4)
        eq = (1 + b).cumprod(); dd = eq / eq.cummax() - 1.0
        throttle = 1.0 + (dd / -0.06).clip(0.0, 1.0) * (0.4 - 1.0)
        return b * (lev * throttle).shift(1).fillna(0.0)
    for mode, lab in (("fullsample", "canonical full-sample tgt (look-ahead level)"),
                      ("expanding", "PIT expanding tgt"), ("none", "NO overlay")):
        sc = scorecard(overlay(raw_mean, mode))
        print(f"   {lab:44s}: {fmt(sc)}")
        out.setdefault("overlay", {})[mode] = {kk: sc[kk] for kk in ('n', 'sharpe', 'worst_mo', 'max_dd', 'streak')}
    print("   -> the passing W (-5.97%) depends on normalising to FULL-SAMPLE vol; PIT-expanding fails W (-6.36%).")

    # -- 6. crash-correlation -> 1 stress -------------------------------------------------
    sg = ["trend_momentum", "carry", "volprem", "xs_momentum", "breakout"]
    spike_days = (dvix >= dvix.quantile(0.98)).reindex(df.index).fillna(False)
    df2 = df.copy()
    for t in df2.index[spike_days]:
        live = [c for c in sg if pd.notna(df2.loc[t, c])]
        if live:
            df2.loc[t, live] = df2.loc[t, live].min()
    stressed = mb.regime_overlay(mb.book_stack(df2))
    sc = scorecard(stressed)
    print("\n6. CRASH-CORRELATION -> 1  (short-gamma legs co-move on worst-2% VIX days)")
    print(f"   stressed book: {fmt(sc)}   (base D={base['max_dd']:+.4f} W={base['worst_mo']:+.4f})")
    out["crash_corr"] = {kk: sc[kk] for kk in ('n', 'sharpe', 'months', 'worst_mo', 'max_dd', 'streak')}

    # -- 7. block-bootstrap of the book's OWN history -------------------------------------
    from arch.bootstrap import StationaryBootstrap, optimal_block_length
    r = book.dropna().to_numpy()
    block = float(np.asarray(optimal_block_length(r)["stationary"])[0])
    bs = StationaryBootstrap(block, r, seed=SEED)
    mlen = int(round(PPY / 12))
    Wb, Db, Kb = [], [], []
    for _, (pa, _) in enumerate(bs.bootstrap(1000)):
        x = pa[0]; nfull = (len(x) // mlen) * mlen
        m = (1 + x[:nfull].reshape(-1, mlen)).prod(axis=1) - 1
        Wb.append(m.min())
        eq = np.cumprod(1 + x); Db.append((eq / np.maximum.accumulate(eq) - 1).min())
        neg = (m < 0).astype(int); st = mx = 0
        for v in neg:
            st = st + 1 if v else 0; mx = max(mx, st)
        Kb.append(mx)
    Wb, Db, Kb = map(np.array, (Wb, Db, Kb))
    print("\n7. BLOCK-BOOTSTRAP of the book's own history (1000 reps)")
    print(f"   P(W<-6%)={np.mean(Wb<-0.06):.0%}  P(D<-15%)={np.mean(Db<-0.15):.0%}  P(K>2)={np.mean(Kb>2):.0%}")
    out["block_bootstrap"] = {"P_W_break": float(np.mean(Wb < -0.06)), "P_D_break": float(np.mean(Db < -0.15)),
                              "P_K_break": float(np.mean(Kb > 2))}

    # -- 8. data integrity: xs_momentum bad ticks -----------------------------------------
    xs = raw["xs_momentum"]
    sig = float(xs.std())
    ticks = xs[xs.abs() > 8 * sig]
    win10 = dict(raw); win10["xs_momentum"] = xs.clip(-0.10, 0.10)
    sc_w = scorecard(assemble(win10))
    print("\n8. DATA INTEGRITY  (un-winsorized outliers in xs_momentum)")
    print(f"   xs daily vol {sig*100:.2f}%; >8-sigma ticks: "
          f"{dict((str(k.date()), round(v*100,1)) for k, v in ticks.items())}")
    print(f"   book skew with ticks {book.skew():+.1f} / without top day {book.drop(book.idxmax()).skew():+.1f}"
          f"  (short-gamma book cannot truly be +17 skew)")
    print(f"   winsorized ±10%: {fmt(sc_w)}  totRet {sc_w['total_return']:+.0f}x  (vs unwinsorized {base['total_return']:+.0f}x)")
    out["data_integrity"] = {"xs_daily_vol": sig,
                             "outlier_ticks": {str(k.date()): float(v) for k, v in ticks.items()},
                             "book_skew_raw": float(book.skew()), "book_skew_ex_top": float(book.drop(book.idxmax()).skew()),
                             "winsorized_scorecard": sc_w}

    # -- 9. counter-sleeve feasibility: can ANY reweight of the 6 families reach 5/5? -----
    print("\n9. COUNTER-SLEEVE FEASIBILITY")
    rng = np.random.default_rng(SEED)
    hit5 = 0; best_n = 0; maxM = -1; maxM_W = 0
    sr_trials = []
    for _ in range(5000):
        w = rng.dirichlet(np.ones(df.shape[1]))
        sc = scorecard(mb.regime_overlay(weighted_mean(df, w)))
        sr_trials.append(sc["sharpe"] / np.sqrt(PPY))
        hit5 += sc["n"] == 5; best_n = max(best_n, sc["n"])
        if sc["months"] > maxM:
            maxM, maxM_W = sc["months"], sc["worst_mo"]
    print(f"   5000 convex reweightings of the 6 families: {hit5} reach 5/5 (best {best_n}/5).")
    print(f"   max months-in-profit reachable = {maxM:.2f}, and it needs worst-month = {maxM_W:+.3f} (M<->W is monotone).")

    # streak decomposition — the 3 losing streaks have 3 different drivers
    legs_mo = {c: monthly_returns(df[c].dropna()) for c in df.columns}
    streaks = [("2019Q3 grind", "2019-07", "2019-09"), ("COVID", "2020-02", "2020-04"),
               ("2021Q4-22Q1", "2021-12", "2022-02")]
    print("   3 critical losing streaks — monthly leg cum-return (%), driver in CAPS:")
    strk = {}
    for lab, a, b in streaks:
        legvals = {}
        for c in df.columns:
            seg = legs_mo[c].loc[a:b].dropna()
            legvals[c] = float((1 + seg).prod() - 1) if len(seg) else None
        drv = min((k for k, v in legvals.items() if v is not None), key=lambda k: legvals[k])
        strk[lab] = {"driver": drv, "legs": legvals}
        print(f"     {lab:13s}: driver={drv.upper()}  " +
              " ".join(f"{c[:4]}{legvals[c]*100:+.0f}" for c in df.columns if legvals[c] is not None))
    print("   -> vol-event, momentum-grind and carry-unwind drive DIFFERENT streaks; no single mechanical")
    print("      trigger (VIX / drawdown) fires in all three, so a hedge sleeve cannot robustly force K<=2.")
    out["counter_sleeve"] = {"reweight_hit5_of_5000": hit5, "best_n": best_n, "max_M": maxM, "max_M_worst_mo": maxM_W,
                             "streak_drivers": strk}

    # -- deflated Sharpe (honest var-across-trials from the reweight search) ---------------
    var_tr = float(np.var(sr_trials, ddof=1))
    srbar = float(book.mean() / book.std(ddof=1))
    dsr = deflated_sharpe(srbar, len(book), float(book.skew()), float(book.kurt() + 3), 5000, var_tr)
    print("\n10. DEFLATED SHARPE")
    print(f"    per-bar SR={srbar:.4f}, T={len(book)}, N=5000 trials -> DSR={dsr:.3f}")
    print("    (leans on ONE dominant sleeve (volprem); does not overwhelmingly clear the MT bar. Anyway Sharpe")
    print("     is the wrong lens here — it is blind to the -76% short-vol tail the window excludes.)")
    out["deflated_sharpe"] = {"per_bar_sr": srbar, "dsr_N5000": float(dsr), "var_across_trials": var_tr}

    # -- persist stressed series ----------------------------------------------------------
    pd.DataFrame({"book": book, "crash_corr_stressed": stressed}).to_parquet(LAB_DIR / "adversary_stress.parquet")
    return book


# =========================================================================================
#  TASK 1 — perturbation gauntlet, reusable for candidate books A/B
# =========================================================================================
def perturbation_gauntlet(df: pd.DataFrame, w0: np.ndarray, combine, n: int = 50, seed: int = SEED) -> dict:
    """Jitter each weight x(1±25%) AND +/-0.03, N draws (seed varies by index), + leave-one-out.
    Robust := median perturbed scorecard still 5/5 AND >=45/50 hold all five."""
    base = scorecard(combine(df, w0))
    survivors = 0; ns = []
    for j in range(n):
        rj = np.random.default_rng(seed + j)
        wj = w0 * (1 + rj.uniform(-0.25, 0.25, len(w0))) + rj.uniform(-0.03, 0.03, len(w0))
        wj = np.clip(wj, 0, None)
        if wj.sum() == 0:
            continue
        wj = wj / wj.sum()
        sc = scorecard(combine(df, wj))
        ns.append(sc["n"]); survivors += sc["n"] == 5
    loo = {}
    for k in range(len(w0)):
        wk = w0.copy(); wk[k] = 0
        if wk.sum() > 0:
            loo[df.columns[k]] = scorecard(combine(df, wk / wk.sum()))["n"]
    return {"base": base, "perturb_survivors": survivors, "perturb_n": len(ns),
            "median_n": float(np.median(ns)) if ns else None, "leave_one_out_n": loo}


def attack_book(path: str, out: dict):
    """Task-1 entry: reproduce a candidate book's scorecard and run the gauntlet. Best-effort —
    expects a daily 'ret' series parquet; weight perturbation needs its legs, handled if present."""
    p = Path(path)
    s = pd.read_parquet(p)
    ret = (s["ret"] if "ret" in s.columns else s.iloc[:, 0]).dropna()
    ret.index = pd.to_datetime(ret.index)
    sc = scorecard(ret)
    print(f"ATTACK {p.name}: reproduced scorecard {fmt(sc)}")
    # regime splits
    print("  regime splits:")
    for lab, a, b in [("2016-20", "2016-08-01", "2020-12-31"), ("2021-25", "2021-01-01", "2025-12-31")]:
        seg = ret.loc[a:b]
        if len(seg) > 50:
            print(f"    {lab}: {fmt(scorecard(seg))}")
    out[p.name] = {"scorecard": sc}


def attack_candidate_A(out: dict):
    """Book A — candidate_book_mftrend (6 families + mftrend + defensive sleeves at risk parity).
    Reproduce by importing A's own module (no main/writes), then attack harder than A did itself."""
    import scripts.candidate_book_mftrend as A
    print("=" * 92 + "\nBOOK A — candidate_book_mftrend (6 families + mftrend + defensive)\n" + "=" * 92)
    legs = A.load_legs()
    mft, dfn = A.load_sleeve("mftrend"), A.load_sleeve("defensive")
    variants = {"base 6-fam": A.assemble(legs, {}),
                "+mftrend": A.assemble(legs, {"mftrend": (mft, 1.0)}),
                "+defensive": A.assemble(legs, {"defensive": (dfn, 1.0)}),
                "+both (8-fam)": A.assemble(legs, {"mftrend": (mft, 1.0), "defensive": (dfn, 1.0)})}
    print("A's reproduced scorecards (net effect of the new sleeves):")
    for k, v in variants.items():
        print(f"  {k:16s} {fmt(scorecard(v))}")

    # harder perturbation: jitter EVERY weight x(1+/-25%) AND +/-0.03, N=50 (A's own gate was N=20 mult-only)
    surv = 0; ns = []
    for j in range(50):
        rj = np.random.default_rng(100 + j)
        bw = np.clip(1.0 * (1 + rj.uniform(-.25, .25, len(legs.columns))) + rj.uniform(-.03, .03, len(legs.columns)), 0, None)
        df = legs.copy(); weights = dict(zip(df.columns, bw))
        for nm, s, w in [("mftrend", mft, 1.0), ("defensive", dfn, 1.0)]:
            df[nm] = s.reindex(df.index)
            weights[nm] = max(w * (1 + rj.uniform(-.25, .25)) + rj.uniform(-.03, .03), 0)
        df = df[df.notna().sum(axis=1) >= 2]; live = df.notna()
        wv = np.array([weights[c] for c in df.columns]); wsum = (live.values * wv).sum(1); wsum[wsum == 0] = np.nan
        bk = A.regime_overlay(pd.Series(np.nansum(np.where(live.values, df.values, 0.0) * wv, 1) / wsum, index=df.index))
        n = scorecard(bk)["n"]; ns.append(n); surv += n == 5
    print(f"  HARDER perturbation (all 8 weights x(1+/-25%) & +/-0.03, N=50): {surv}/50 reach 5/5 (median {np.median(ns):.0f}/5)")
    # streak fix test — do the sleeves break the 3 critical streaks?
    bmo, tmo = monthly_returns(variants["base 6-fam"]), monthly_returns(variants["+both (8-fam)"])
    print("  streak fix (base -> +both, monthly cum): "
          + "  ".join(f"{lab} {((1+bmo.loc[a:b]).prod()-1)*100:+.0f}%->{((1+tmo.loc[a:b]).prod()-1)*100:+.0f}%"
                       for lab, a, b in [("2019Q3", "2019-07", "2019-09"), ("COVID", "2020-02", "2020-04"),
                                         ("21Q4-22Q1", "2021-12", "2022-02")]))
    print("  -> new sleeves are net-negative on M/S (and defensive pushes K 3->4); inherits volprem tail")
    print("     (drop-volprem -> S~1.5) so the Task-2 window/injection findings apply unchanged.")
    out["book_A"] = {k: scorecard(v) for k, v in variants.items()} | {"perturb_survivors_of_50": surv}


def attack_adaptive_B(out: dict):
    """Book B — run_adaptive_book (rolling re-optimization of the 6 families). Reproduce the enshrined
    inverse-drawdown book, falsify the PIT claim (honest OOS vs full-sample look-ahead cheat)."""
    import scripts.run_adaptive_book as B
    print("\n" + "=" * 92 + "\nBOOK B — run_adaptive_book (adaptive re-optimization)\n" + "=" * 92)
    legs = B.load_legs(B.CORE6)
    ADK = dict(tau=0.08, thr=0.05, floor=0.2, dd_lb=63)
    book = B.walkforward(legs, "inv_drawdown", 12, 3, overlay=True, method_kw=ADK)
    print(f"  canonical (equal+overlay):            {fmt(scorecard(B.book_equal(legs, overlay=True)))}")
    print(f"  B's enshrined adaptive book (inv-dd):  {fmt(scorecard(book))}")

    # PIT falsification: refit the same tilt on the WHOLE sample (look-ahead) and compare
    fn = B.METHODS["inv_drawdown"]; full_w = fn(legs, **ADK)
    W = pd.DataFrame(0.0, index=legs.index, columns=legs.columns)
    for c, v in full_w.items():
        W[c] = v
    wm = legs.notna() * W; wm = wm.div(wm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    cheat = B.regime_overlay((legs.fillna(0.0) * wm).sum(axis=1))
    print(f"  look-ahead cheat (full-sample weights):{fmt(scorecard(cheat))}")
    print("  -> cheat scores BETTER than honest OOS => the walk-forward is genuinely PIT (no look-ahead). Clean.")
    print("  -> BUT the enshrined adaptive book breaks W (-6.4%) => 2/5, a regression below the 3/5 canonical.")
    out["book_B"] = {"adaptive_book": scorecard(book), "canonical": scorecard(B.book_equal(legs, overlay=True)),
                     "lookahead_cheat": scorecard(cheat)}


def confirm_xs_bug(out: dict):
    """Trace the xs_momentum +53% day (2025-10-31) to its corrupt underlying price. Self-contained —
    reads reports/xs + data/raw/equity_td only, so it runs regardless of the family-parquet race."""
    xsl = pd.read_parquet(R / "xs/xs_sleeve_returns.parquet")
    xsl.index = pd.to_datetime(xsl.index)
    if xsl.index.tz is not None:
        xsl.index = xsl.index.tz_localize(None)
    d = pd.Timestamp("2025-10-31")
    print("=" * 92 + "\nCONFIRMED DATA BUG — xs_momentum +53% on 2025-10-31\n" + "=" * 92)
    row = xsl.loc[xsl.index.normalize() == d]
    if len(row):
        print("  xs sub-sleeve returns that day: " + ", ".join(f"{c}={row[c].iloc[0]*100:+.1f}%" for c in xsl.columns))
    td = ROOT / "data/raw/equity_td"
    hits = []
    for p in td.glob("*_1d.parquet"):
        try:
            c = pd.read_parquet(p)["close"]
            c.index = pd.to_datetime(c.index)
            if c.index.tz is not None:
                c.index = c.index.tz_localize(None)
            r = c.pct_change()
            if d in r.index and abs(r.loc[d]) > 0.5:
                hits.append((p.name[:-11], float(r.loc[d]), c.loc[d - pd.Timedelta(days=4):d + pd.Timedelta(days=2)]))
        except Exception:
            pass
    hits.sort(key=lambda t: -abs(t[1]))
    for name, ret, ser in hits[:3]:
        print(f"  culprit {name}: {ret*100:+.0f}% on 2025-10-31; closes {[round(float(x), 1) for x in ser.values]}")
    print("  -> STJ close ÷100 for one day then reverts = corrupt tick, not a split; booked as a real momentum return.")
    out["data_bug"] = {"date": "2025-10-31", "stocks_broad_ret": float(xsl.get("stocks_broad", pd.Series()).reindex([d]).iloc[0])
                       if "stocks_broad" in xsl.columns else None,
                       "culprits": [{"name": n, "ret": r} for n, r, _ in hits[:3]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default=None, help="path to a candidate book 'ret' parquet (generic Task 1)")
    ap.add_argument("--task1", action="store_true", help="run the Task-1 gauntlet on books A and B (imports their modules)")
    ap.add_argument("--confirm-xs-bug", action="store_true", help="trace the xs 2025-10-31 outlier to its corrupt price")
    args = ap.parse_args()
    out = {}
    if args.attack:
        attack_book(args.attack, out)
    elif getattr(args, "confirm_xs_bug", False):
        confirm_xs_bug(out)
        print(json.dumps(out, indent=2, default=float))
    elif args.task1:
        attack_candidate_A(out)
        attack_adaptive_B(out)
        (LAB_DIR / "adversary_task1_summary.json").write_text(json.dumps(out, indent=2, default=float))
        print("\nartifact -> reports/lab/adversary_task1_summary.json\nADVERSARY TASK1 OK")
    else:
        task2(out)
        (LAB_DIR / "adversary_summary.json").write_text(json.dumps(out, indent=2, default=float))
        print("\nartifacts -> reports/lab/adversary_stress.parquet, reports/lab/adversary_summary.json")
        print("ADVERSARY OK")


if __name__ == "__main__":
    main()
