"""Residual / idiosyncratic momentum deep-dive (H5) — run through the same funnel as every other
family (vol-target 15%, t+2-style delay, liquidity-aware costs, block-bootstrap MC, shuffled-signal
placebo, purged/embargoed walk-forward OOS, deflated Sharpe, cost sensitivity, correlation to the
deliverable book + lift curve). Crypto (300-name PIT panel), US equity (692-name PIT), FX (25 pairs).

The question this answers: residual momentum (Blitz-Huij-Martens 2011) — momentum on the market-beta
*residual* of each name's return, standardised by residual vol — is documented to be higher-Sharpe,
lower-beta and far less crash-prone than raw (total-return) momentum, especially in equities. The
book's cross-sectional momentum sleeve is its weakest equity leg (docs/strategies/XSECT.md §6 flagged residual as the
best single-stock signal but never ran it through the funnel). So the H5 acceptance test is precise:

    does residualising the return BEFORE ranking beat the raw risk-adjusted-momentum book already in
    the book — higher OOS Sharpe AND lower market beta — or is it re-labelled momentum?

The head-to-head holds execution identical (same lookback / skip / quantile / rebalance / universe)
and swaps only the signal: raw total-return `mom`, raw risk-adjusted `risk_adj_mom` (the book's
benchmark), single-window `resid_mom`, and decoupled long-beta-window `idio_mom` (the canonical BHM
construction). The honest verdict, both the Sharpe and the beta-reduction, is written to
reports/residmom_summary.json and docs/strategies/RESIDMOM.md; artifacts feed reports/figures/residmom.png.

    python scripts/residmom/run_residmom.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, CAPITAL_USD, REPORTS_DIR, RESIDMOM_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import (idio_mom, mom, resid_mom, risk_adj_mom,  # noqa: E402
                               top_n_liquid, vol_target, xs_backtest)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
FIG = REP / "figures"
CACHE = CACHE_DIR / "xs"
SEED, CAP, TVOL = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL
rng = np.random.default_rng(SEED)

# a-priori config per asset (declared before fit, surface-reported, never peak-picked). The base
# (lb=formation, sk=skip, tf=quantile, rebal) MATCHES the raw-momentum sleeve already in the book
# (docs/strategies/XSECT.md §3/§5) so the raw↔residual comparison is apples-to-apples — only the signal is swapped.
# beta_lb is the residual's separate beta-estimation window (BHM: long & stable): crypto ~90d (as
# BAB), equity ~3y (756 trading days, the classic FF 36-month), FX ~1y.
ASSETS = {
    "crypto": dict(ppy=365, cost=6.0, tag="crypto_1d", mkt_col="BTCUSDT", winsor=1.0, topn=100,
                   imp=0.1, base=dict(lb=30, sk=0, tf=0.3, rebal=21), beta_lb=90),
    "equity": dict(ppy=252, cost=3.0, tag="stocks_broad_1d", mkt_col=None, winsor=0.5, topn=100,
                   imp=0.1, base=dict(lb=252, sk=7, tf=0.1, rebal=21), beta_lb=756),
    "fx": dict(ppy=252, cost=1.0, tag="fx_1d", mkt_col=None, winsor=0.5, topn=0,
               imp=0.0, base=dict(lb=90, sk=0, tf=0.3, rebal=21), beta_lb=250),
}


def _load(tag):
    C = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    ap = CACHE / f"{tag}_adv.parquet"
    A = pd.read_parquet(ap).reindex_like(C) if ap.exists() else None
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC")
        if A is not None:
            A.index = A.index.tz_localize("UTC")
    return C, A


def _sh(net, ppy):
    n = net.dropna()
    return summarise(n, ppy)["sharpe_ann"] if len(n) > 2 else float("nan")


def _signals(C, cfg):
    """The four ranking signals on identical windows — raw total, raw risk-adj, single-window
    residual, decoupled-beta residual (BHM). Same lb/skip so only the residualisation differs.
    All residuals use the equal-weight panel market (market=None) — the repo's `resid_mom`
    convention — so resid1w↔idio isolate the beta-window decoupling, not the factor choice
    (BTC-market residualisation is a separate robustness variant in run_residmom_robust.py)."""
    b = cfg["base"]
    lb, sk = b["lb"], b["sk"]
    return {
        "raw": mom(C, lb, sk),
        "riskadj": risk_adj_mom(C, lb, sk),                        # the book's benchmark signal
        "resid1w": resid_mom(C, lb, sk),                           # single window (beta_lb==form_lb)
        "idio": idio_mom(C, lb, cfg["beta_lb"], sk, market=None),  # canonical decoupled BHM, EW market
    }


def _book(C, sig, A, cfg, *, tf=None, rebal=None, cost_mult=1.0):
    """Vol-targeted net return of a signal-swap into xs_backtest at the asset's a-priori config."""
    b = cfg["base"]
    s = top_n_liquid(sig, A, cfg["topn"]) if cfg["topn"] else sig
    bt = xs_backtest(C, s, top_frac=tf or b["tf"], weighting="equal", rebal=rebal or b["rebal"],
                     exec_lag=2, cost_bps=cfg["cost"] * cost_mult, adv=A, impact_k=cfg["imp"])
    return vol_target(bt["net"], cfg["ppy"], TVOL), bt


def _realized_beta(net: pd.Series, mkt: pd.Series) -> float:
    """Realised market beta of a book: OLS slope of net book return on market return (the honest
    'is it lower beta' measure — a momentum book is dollar-neutral, so its beta is the winners-minus-
    losers differential-beta tilt, the Daniel-Moskowitz crash channel residualisation is meant to cut)."""
    df = pd.concat([net.rename("n"), mkt.rename("m")], axis=1).dropna()
    if len(df) < 30 or df["m"].var() == 0:
        return float("nan")
    return float(np.cov(df["m"], df["n"])[0, 1] / df["m"].var())


def _ols(y: pd.Series, *xs: pd.Series):
    """OLS y ~ 1 + xs on the common index; return (coefs, t-stats, n). Intercept = alpha."""
    df = pd.concat([y.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(xs)], axis=1).dropna()
    Y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[f"x{i}"].to_numpy() for i in range(len(xs))])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    dof = max(len(Y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    t = coef / np.sqrt(np.diag(cov))
    return coef, t, len(df)


def _wf_oos(M: pd.DataFrame, ppy, train_bars, test_bars, embargo, top_k=5):
    """Purged/embargoed walk-forward: pick best-Sharpe configs on each train block (dropping the
    last `embargo` bars so the trailing windows cannot leak the test in), apply to the next block,
    stitch OOS. top_k ensembles the in-sample plateau."""
    segs, picks = [], []
    start = train_bars
    while start + test_bars <= len(M):
        train = M.iloc[max(0, start - train_bars):max(0, start - embargo)]
        test = M.iloc[start:start + test_bars]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        chosen = list(sr.nlargest(top_k).index)
        segs.append(test[chosen].mean(axis=1))
        picks.extend(chosen)
        start += test_bars
    oos = pd.concat(segs) if segs else pd.Series(dtype=float)
    return oos, len(picks) // max(top_k, 1)


def run_asset(kind: str) -> tuple[dict, dict]:
    cfg = ASSETS[kind]
    ppy, cost, winsor, b = cfg["ppy"], cfg["cost"], cfg["winsor"], cfg["base"]
    Craw, A = _load(cfg["tag"])
    C = bab.winsorize_panel(Craw, winsor)
    mkt_ext = C[cfg["mkt_col"]].pct_change() if cfg["mkt_col"] and cfg["mkt_col"] in C.columns else None
    mkt_ew = C.pct_change().mean(axis=1)                              # equal-weight market (the residual factor)
    mkt_for_beta = mkt_ext if mkt_ext is not None else mkt_ew        # book-beta measured vs the real market
    print(f"\n{'='*80}\n{kind.upper()}  panel: {C.shape[1]} names, "
          f"{C.index.min().date()}..{C.index.max().date()}, {len(C)} bars  "
          f"(base lb={b['lb']} sk={b['sk']} q={b['tf']} rebal={b['rebal']} top{cfg['topn']} beta_lb={cfg['beta_lb']})\n{'='*80}")

    # ── data integrity: artifact name-days winsorised; does cleaning move the headline? ──────────
    n_inf = int(np.isinf(Craw.pct_change().to_numpy()).sum())
    rw = Craw.pct_change().replace([np.inf, -np.inf], np.nan)
    n_big = int((rw.abs() > winsor).to_numpy().sum())
    raw_head = _sh(_book(bab.winsorize_panel(Craw, 1e9), idio_mom(Craw, b["lb"], cfg["beta_lb"], b["sk"], market=None), A, cfg)[0], ppy)
    print(f"  data integrity: {n_big} artifact name-days |ret|>{winsor:.0%} (+{n_inf} ∞) winsorised → flat; "
          f"raw-panel idio Sharpe {raw_head:+.2f} (vs clean below)")

    # ── head-to-head: swap ONLY the signal, hold execution identical (the H5 acceptance test) ────
    sigs = _signals(C, cfg)
    books = {k: _book(C, s, A, cfg)[0] for k, s in sigs.items()}
    hh = {}
    for k, net in books.items():
        s = summarise(net.dropna(), ppy)
        hh[k] = {"sharpe": round(s["sharpe_ann"], 3), "maxdd": round(s["max_dd"], 3),
                 "beta_mkt": round(_realized_beta(net, mkt_for_beta), 3),
                 "beta_ew": round(_realized_beta(net, mkt_ew), 3)}
    print("  HEAD-TO-HEAD (identical execution, signal swapped):")
    print(f"    {'signal':10s} {'Sharpe':>7s} {'maxDD':>7s} {'β(mkt)':>7s} {'β(EW)':>7s}")
    for k in ("raw", "riskadj", "resid1w", "idio"):
        h = hh[k]
        print(f"    {k:10s} {h['sharpe']:+7.2f} {h['maxdd']:+7.1%} {h['beta_mkt']:+7.3f} {h['beta_ew']:+7.3f}"
              + ("   ← book benchmark" if k == "riskadj" else "   ← residual (BHM)" if k == "idio" else ""))
    d_sharpe = hh["idio"]["sharpe"] - hh["riskadj"]["sharpe"]
    d_beta = abs(hh["idio"]["beta_mkt"]) - abs(hh["riskadj"]["beta_mkt"])
    print(f"    Δ residual−raw:  Sharpe {d_sharpe:+.2f}   |β| change {d_beta:+.3f}  "
          f"→ H5 test {'PASS' if d_sharpe > 0 and d_beta <= 0.02 else 'partial' if d_sharpe > 0 or d_beta < 0 else 'FAIL'}")

    # ── momentum-crash channel: residual momentum's selling point is smaller crashes. Return of ──
    # each book in the raw book's own worst months (raw momentum crashes when beaten-down high-β
    # names rebound; residual, beta-stripped, should bleed less there).
    def _monthly(n):
        return (1 + n.dropna()).resample("ME").prod() - 1
    raw_m = _monthly(books["riskadj"])
    worst = raw_m.nsmallest(5).index
    crash = {k: round(float(_monthly(books[k]).reindex(worst).mean()), 4) for k in books}
    print(f"  momentum-crash months (raw's worst 5): raw {crash['riskadj']:+.1%}  "
          f"residual(idio) {crash['idio']:+.1%}  (residual bleeds less if > raw)")

    # ── sign check: residual momentum is LONG high residual-mom (signal as-is). Verify + beats − ──
    s_pos = hh["idio"]["sharpe"]
    s_neg = _sh(_book(C, -sigs["idio"], A, cfg)[0], ppy)
    print(f"  sign check: +idio (long high resid-mom) {s_pos:+.2f}  vs  −idio {s_neg:+.2f}"
          f"  → {'theory sign wins' if s_pos >= s_neg else 'data prefers reversal (!)'}")

    # ── construction surface: formation × beta-window × quantile (residual, idio) ────────────────
    grid = []
    forms = ([15, 30, 45, 90] if kind == "crypto" else [126, 252, 378] if kind == "equity" else [45, 90, 180])
    betas = ([60, 90, 180] if kind == "crypto" else [504, 756, 1008] if kind == "equity" else [180, 250, 500])
    for fl in forms:
        for bl in betas:
            sig = idio_mom(C, fl, bl, b["sk"], market=None)
            for tf in (0.1, 0.2, 0.3):
                grid.append({"asset": kind, "form_lb": fl, "beta_lb": bl, "top_frac": tf,
                             "sharpe": round(_sh(_book(C, sig, A, cfg, tf=tf)[0], ppy), 3)})
    grid_df = pd.DataFrame(grid)
    pos_frac = float((grid_df.sharpe > 0).mean())
    print(f"  construction surface (idio, {len(grid_df)} cells): {pos_frac:.0%} positive, "
          f"best {grid_df.sharpe.max():+.2f}, median {grid_df.sharpe.median():+.2f}, a-priori "
          f"{hh['idio']['sharpe']:+.2f}")

    # ── placebo: column-shuffle the residual signal (kill the cross-section, keep the marginals) ─
    idio_sig = sigs["idio"]
    real = hh["idio"]["sharpe"]
    plc = []
    for _ in range(100):
        perm = idio_sig.copy()
        perm.columns = rng.permutation(idio_sig.columns)
        perm = perm.reindex(columns=idio_sig.columns)
        plc.append(_sh(_book(C, perm, A, cfg)[0], ppy))
    plc = np.array(plc); plc = plc[np.isfinite(plc)]
    pctile = float((real > plc).mean() * 100)
    print(f"  placebo: real idio {real:+.2f} at {pctile:.0f}th pctile of shuffles "
          f"(mean {plc.mean():+.2f}, p95 {np.percentile(plc, 95):+.2f})")

    # ── walk-forward OOS over the residual grid, AND over a raw-momentum grid (the incremental) ──
    def _grid_frame(signal_fn):
        M = {}
        for fl in forms:
            for tf in (0.1, 0.2, 0.3):
                M[f"{fl}_{tf}"] = _book(C, signal_fn(fl), A, cfg, tf=tf)[0]
        return pd.DataFrame(M).dropna(how="all")
    M_res = _grid_frame(lambda fl: idio_mom(C, fl, cfg["beta_lb"], b["sk"], market=None))
    M_raw = _grid_frame(lambda fl: risk_adj_mom(C, fl, b["sk"]))
    tr_b, te_b = int(2.0 * ppy), int(0.5 * ppy)
    emb = cfg["beta_lb"]
    wf_res, n_ref = _wf_oos(M_res, ppy, tr_b, te_b, emb)
    wf_raw, _ = _wf_oos(M_raw, ppy, tr_b, te_b, emb)
    s_wf_res, s_wf_raw = _sh(wf_res, ppy), _sh(wf_raw, ppy)
    n_trials = int(M_res.shape[1] + M_raw.shape[1])
    full_sr = (M_res.mean() / M_res.std(ddof=1) * np.sqrt(ppy))
    var_tr = float((full_sr.clip(-3, 3) / np.sqrt(ppy)).var())
    print(f"  walk-forward OOS (purged, embargo={emb}b, {n_ref} refits): residual {s_wf_res:+.2f}"
          f"  vs  raw-riskadj {s_wf_raw:+.2f}   → incremental {s_wf_res - s_wf_raw:+.2f}")

    # ── MC + deflated Sharpe on the chosen residual (idio) book ──────────────────────────────────
    head = books["idio"].dropna()
    mc = bootstrap_sharpe(head, ppy, 1000, SEED)
    dsr = deflated_sharpe(head.mean() / head.std(ddof=1), len(head), head.skew(), head.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8))
    per_year = {int(y): round(_sh(g, ppy), 2) for y, g in head.groupby(head.index.year)}
    print(f"  MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f} "
          f"P95 {mc.get('sharpe_p95', float('nan')):+.2f}]  deflated SR (N={n_trials}) {dsr:.2f}  "
          f"maxDD {summarise(head, ppy)['max_dd']:+.1%}")

    # ── cost sensitivity + break-even (idio book) ────────────────────────────────────────────────
    levels = {f"{m:.0f}x": round(_sh(_book(C, sigs["idio"], A, cfg, cost_mult=m)[0], ppy), 3) for m in (1, 2, 3)}
    breakeven = next((round(float(m), 2) for m in np.linspace(0.5, 12.0, 47)
                      if _sh(_book(C, sigs["idio"], A, cfg, cost_mult=m)[0], ppy) <= 0.5), None)
    print(f"  cost 1x/2x/3x: {levels}  break-even-to-0.5 ≈ {breakeven}x base")

    # ── orthogonalisation: is residual momentum anything BEYOND raw momentum? regress & correlate ─
    O = pd.DataFrame({"idio": books["idio"], "riskadj": books["riskadj"], "raw": books["raw"]}).dropna()
    corr_res_raw = float(O["idio"].corr(O["riskadj"]))
    ca, ta, na = _ols(O["idio"], O["riskadj"])      # residual-mom alpha controlling for raw-riskadj book
    ortho = {"corr_idio_riskadj": round(corr_res_raw, 3),
             "idio_on_raw_alpha_ann": round(float(ca[0] * ppy), 4),
             "idio_on_raw_alpha_t": round(float(ta[0]), 2),
             "idio_on_raw_beta": round(float(ca[1]), 3)}
    print(f"  orthogonalise vs raw momentum: corr {corr_res_raw:+.2f}  |  residual alpha ⟂ raw "
          f"{ortho['idio_on_raw_alpha_ann']:+.1%}/yr (t={ortho['idio_on_raw_alpha_t']:+.1f})  "
          f"→ {'adds alpha beyond raw' if ortho['idio_on_raw_alpha_t'] > 2 else 're-labelled momentum' if corr_res_raw > 0.85 else 'partly independent'}")

    # ── correlation to master book + does adding it lift the book? ────────────────────────────────
    corr, lift = {}, {}
    bp_path, bs_path = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]
        bs = pd.read_parquet(bs_path)
        for f in (bp, bs):
            if f.index.tz is not None:
                f.index = f.index.tz_localize(None)
        h = head.copy(); h.index = h.index.tz_localize(None)
        common = h.dropna().index.intersection(bp.dropna().index)
        if len(common) > 50:
            corr["book"] = round(float(h.reindex(common).corr(bp.reindex(common))), 3)
            for c in bs.columns:
                corr[c] = round(float(h.reindex(common).corr(bs[c].reindex(common))), 3)
            bk = bp.reindex(common).dropna()
            hm = h.reindex(bk.index).fillna(0.0)
            hm = hm * (bk.std() / hm.std())
            lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * hm, ppy), 3) for w in (0.0, 0.15, 0.3, 0.5)}
            print(f"  corr to master book {corr.get('book')}  → book-lift by weight {lift}")

    summ = {
        "config": {**{k: b[k] for k in b}, "beta_lb": cfg["beta_lb"], "top_n": cfg["topn"],
                   "cost_bps_per_side": cost, "winsor": winsor, "ppy": ppy,
                   "window": [str(C.index.min().date()), str(C.index.max().date())]},
        "data_integrity": {"artifact_name_days": n_big, "inf_name_days": n_inf,
                           "idio_sharpe_RAW": round(raw_head, 3), "idio_sharpe_CLEAN": hh["idio"]["sharpe"]},
        "head_to_head": hh,
        "delta_residual_minus_raw": {"sharpe": round(d_sharpe, 3), "abs_beta_change": round(d_beta, 3)},
        "crash_months": crash,
        "sign": {"pos_idio": round(s_pos, 3), "neg_idio": round(s_neg, 3)},
        "surface": {"n_cells": len(grid_df), "pct_positive": round(pos_frac, 3),
                    "best": round(float(grid_df.sharpe.max()), 3), "median": round(float(grid_df.sharpe.median()), 3)},
        "placebo": {"real": round(real, 3), "pctile": round(pctile, 0),
                    "shuffle_mean": round(float(plc.mean()), 3), "shuffle_p95": round(float(np.percentile(plc, 95)), 3)},
        "walk_forward": {"residual_oos": round(s_wf_res, 3), "raw_oos": round(s_wf_raw, 3),
                         "incremental": round(s_wf_res - s_wf_raw, 3), "n_refits": n_ref,
                         "n_grid_trials": n_trials, "deflated_sharpe": round(dsr, 3)},
        "mc": {k: mc.get(k) for k in ("sharpe_p5", "sharpe_p50", "sharpe_p95")},
        "per_year": per_year, "cost_sensitivity": levels, "breakeven_cost_mult_to_0.5": breakeven,
        "orthogonalisation": ortho, "corr_to_book": corr, "book_lift_by_weight": lift,
    }
    grid_df_out = grid_df
    out_books = {f"{kind}_idio": books["idio"], f"{kind}_riskadj": books["riskadj"],
                 f"{kind}_resid1w": books["resid1w"], f"{kind}_raw": books["raw"]}
    return summ, {"books": out_books, "grid": grid_df_out}


def _figure(summ, books):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Residual / idiosyncratic momentum (H5) vs raw momentum — net of costs, vol-target 15%",
                 fontsize=13, fontweight="bold")

    # (1) equity curves — residual vs raw, per asset
    a = ax[0, 0]
    for c, lab, st in [("crypto_idio", "crypto residual (idio)", "-"), ("crypto_riskadj", "crypto raw", "--"),
                       ("equity_idio", "equity residual (idio)", "-"), ("equity_riskadj", "equity raw", "--")]:
        if c in books:
            r = books[c].dropna()
            a.plot((1 + r).cumprod().index, (1 + r).cumprod().values, label=lab, lw=1.3, ls=st)
    a.axhline(1.0, color="k", lw=0.6, ls=":"); a.set_yscale("log")
    a.set_title("Equity curves — residual (solid) vs raw (dashed)"); a.legend(fontsize=8)

    # (2) head-to-head Sharpe: raw vs residual per asset
    a = ax[0, 1]
    ks = list(summ.keys()); x = np.arange(len(ks)); wd = 0.35
    a.bar(x - wd/2, [summ[k]["head_to_head"]["riskadj"]["sharpe"] for k in ks], wd, label="raw risk-adj", color="#68a")
    a.bar(x + wd/2, [summ[k]["head_to_head"]["idio"]["sharpe"] for k in ks], wd, label="residual (idio)", color="#2b6")
    a.axhline(0.5, color="r", ls="--", lw=1, label="robust bar 0.5"); a.axhline(0, color="k", lw=0.6)
    a.set_xticks(x); a.set_xticklabels(ks)
    for i, k in enumerate(ks):
        for off, key in [(-wd/2, "riskadj"), (wd/2, "idio")]:
            v = summ[k]["head_to_head"][key]["sharpe"]
            a.text(i + off, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    a.set_title("Head-to-head Sharpe (identical execution)"); a.legend(fontsize=8)

    # (3) beta reduction: |book beta| raw vs residual
    a = ax[1, 0]
    a.bar(x - wd/2, [abs(summ[k]["head_to_head"]["riskadj"]["beta_mkt"]) for k in ks], wd, label="raw |β|", color="#a63")
    a.bar(x + wd/2, [abs(summ[k]["head_to_head"]["idio"]["beta_mkt"]) for k in ks], wd, label="residual |β|", color="#3b7")
    a.axhline(0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(ks)
    a.set_title("Market-beta of the book — does residualising lower it?"); a.legend(fontsize=8)

    # (4) walk-forward incremental: residual − raw OOS per asset
    a = ax[1, 1]
    inc = [summ[k]["walk_forward"]["incremental"] for k in ks]
    a.bar(x, inc, color=["#2b6" if v > 0 else "#c33" for v in inc])
    a.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(inc):
        a.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    a.set_xticks(x); a.set_xticklabels(ks)
    a.set_title("Walk-forward OOS: residual − raw (the incremental H5 value)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "residmom.png", dpi=110)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summ, allbooks, grids = {}, {}, []
    for kind in ("crypto", "equity", "fx"):
        s, x = run_asset(kind)
        summ[kind] = s
        allbooks.update(x["books"])
        grids.append(x["grid"])

    pd.DataFrame(allbooks).to_parquet(RESIDMOM_DIR / "residmom_returns.parquet")
    pd.concat(grids, ignore_index=True).to_csv(RESIDMOM_DIR / "residmom_grid.csv", index=False)
    (RESIDMOM_DIR / "residmom_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(summ, allbooks)

    print(f"\n{'='*80}\nVERDICT")
    for k in summ:
        s = summ[k]; hh = s["head_to_head"]; wf = s["walk_forward"]
        print(f"  {k:7s}: residual {hh['idio']['sharpe']:+.2f} (raw {hh['riskadj']['sharpe']:+.2f}, "
              f"Δ{s['delta_residual_minus_raw']['sharpe']:+.2f}) | β {hh['idio']['beta_mkt']:+.2f} vs "
              f"raw {hh['riskadj']['beta_mkt']:+.2f} | WF-OOS residual {wf['residual_oos']:+.2f} raw "
              f"{wf['raw_oos']:+.2f} (Δ{wf['incremental']:+.2f}) | corr-book {s['corr_to_book'].get('book')}")
    print("RUN RESIDMOM OK")


if __name__ == "__main__":
    main()
