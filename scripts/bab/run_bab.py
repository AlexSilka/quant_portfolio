"""Betting-against-beta / low-volatility deep-dive — run through the same funnel as every other
family (vol-target 15%, t+2-style delay, liquidity-aware costs, block-bootstrap MC, shuffled-signal
placebo, purged/embargoed walk-forward OOS, deflated Sharpe, cost sensitivity, correlation to the
deliverable book + lift curve). Separately on crypto (300-name panel) and US equity (692-name PIT).

The question this answers: the leverage-constraint premium (Frazzini-Pedersen 2014) — long low-beta,
short high-beta — is one of the most robust documented factors and should be strongest in crypto
(acute retail leverage/lottery demand). Is it a tradable decorrelated sleeve net of costs, and is it
*beta* (leverage constraint) or re-labelled *lottery* (H2 skew)? The honest verdict, both signs and
both neutralisations kept, is written to reports/bab_summary.json and docs/strategies/BAB.md; artifacts feed
reports/figures/bab.png.

    python scripts/bab/run_bab.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import BAB_DIR, CACHE_DIR, CAPITAL_USD, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
FIG = REP / "figures"
CACHE = CACHE_DIR / "xs"
SEED, CAP, TVOL = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL
rng = np.random.default_rng(SEED)

# a-priori config (declared before fit, surface-reported, never peak-picked):
# 90-day trailing beta (Frazzini-Pedersen ~1y for equities; crypto is faster, 60-90d), quintile
# tails, monthly rebalance (slow signal → low turnover), top-100 liquid each bar, t+2 execution.
BETA_LB, VOL_LB, SKEW_LB = 90, 60, 60
TOPFRAC, REBAL, TOPN, EXEC_LAG, IMPACT_K = 0.2, 21, 100, 2, 0.1

ASSETS = {  # ppy, per-side cost bps, cache tag, market-proxy col for a robustness beta, winsor floor
    # winsor is a-priori: a >50% one-day move on a top-100-liquid US stock is ~always a data artifact
    # (99.99th |ret| pctile ≈ 28%); crypto's 99.99th ≈ 100%, so only >100% (liquidation/print) is clipped.
    "crypto": dict(ppy=365, cost=6.0, tag="crypto_1d", mkt_col="BTCUSDT", winsor=1.0),
    "equity": dict(ppy=252, cost=3.0, tag="stocks_broad_1d", mkt_col=None, winsor=0.5),
}


def _load(tag):
    C = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    A = pd.read_parquet(CACHE / f"{tag}_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC")
        A.index = A.index.tz_localize("UTC")
    return C, A


def _sh(net, ppy):
    return summarise(net.dropna(), ppy)["sharpe_ann"]


def _liq(sig, A, px):
    return top_n_liquid(sig, A, TOPN, px=px)


def _dollar(C, A, sig, cost, ppy):
    """Dollar-neutral book — the literal signal-swap into xs_backtest (long top of sig)."""
    bt = xs_backtest(C, sig, top_frac=TOPFRAC, weighting="equal", rebal=REBAL, exec_lag=EXEC_LAG,
                     cost_bps=cost, adv=A, impact_k=IMPACT_K)
    return vol_target(bt["net"], ppy, TVOL), bt


def _beta_neutral(C, A, beta_masked, cost, ppy, top_frac=TOPFRAC):
    """Beta-neutral (Frazzini-Pedersen leg-scaled) book on the liquidity-masked beta panel."""
    w = bab.bab_weights(beta_masked, top_frac=top_frac, neutral="beta", rebal=REBAL)
    bt = bab.bab_backtest(C, w, exec_lag=EXEC_LAG, cost_bps=cost, adv=A, impact_k=IMPACT_K)
    net_beta = bab.net_book_beta(bt["weights"], beta_masked)      # realised (post-delay) tilt
    return vol_target(bt["net"], ppy, TVOL), bt, net_beta


def _ols(y: pd.Series, *xs: pd.Series):
    """OLS y ~ 1 + xs on the common index; return (coefs, t-stats). Intercept = alpha."""
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
    last `embargo` bars so the trailing-beta window cannot leak the test in), apply to the next
    block, stitch OOS. top_k ensembles the plateau (configs are near-ties in-sample)."""
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
    return oos, len(picks) // top_k


def run_asset(kind: str) -> tuple[dict, dict]:
    cfg = ASSETS[kind]
    ppy, cost, winsor = cfg["ppy"], cfg["cost"], cfg["winsor"]
    Craw, A = _load(cfg["tag"])
    C = bab.winsorize_panel(Craw, winsor)                # clean the split/delisting artifacts first
    print(f"\n{'='*78}\n{kind.upper()}  panel: {C.shape[1]} names, "
          f"{C.index.min().date()}..{C.index.max().date()}, {len(C)} bars\n{'='*78}")

    # ── data integrity: how many artifact name-days, and does cleaning move the headline number? ──
    rliq = _liq(Craw.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan), A, Craw)
    n_inf = int(np.isinf(Craw.pct_change().to_numpy()).sum())
    n_big = int((rliq.abs() > winsor).to_numpy().sum())
    beta_raw = _liq(bab.panel_beta(Craw, BETA_LB), A, Craw)
    raw_head = _sh(_beta_neutral(Craw, A, beta_raw, cost, ppy)[0], ppy)
    print(f"  data integrity: {n_big} artifact name-days |ret|>{winsor:.0%} (+{n_inf} ∞ from a zero prior "
          f"close) winsorised → flat; raw-panel beta-neutral Sharpe {raw_head:+.2f} (vs clean below)")

    # ── signals: trailing panel beta (EW-market), plus the vol proxy and the skew (lottery) control
    beta = _liq(bab.panel_beta(C, BETA_LB), A, C)
    vol = _liq(bab.trailing_vol(C, VOL_LB), A, C)
    skew = _liq(bab.trailing_skew(C, SKEW_LB), A, C)

    # ── sign: BAB theory is long LOW beta (signal −beta). Verify −beta beats +beta on the data ──
    s_neg = _sh(_dollar(C, A, -beta, cost, ppy)[0], ppy)
    s_pos = _sh(_dollar(C, A, beta, cost, ppy)[0], ppy)
    print(f"  sign check (dollar-neutral): −beta (long low-β) {s_neg:+.2f}  vs  +beta {s_pos:+.2f}"
          f"  → funnel on {'−beta (theory)' if s_neg >= s_pos else '−beta (theory; data prefers +beta)'}")

    # ── construction surface: beta lookback × quintile fraction, both neutralisations ───────────
    grid = []
    for lb in (60, 90, 120):
        b_lb = _liq(bab.panel_beta(C, lb), A, C)
        for tf in (0.1, 0.2, 0.3):
            sd = _sh(vol_target(xs_backtest(C, -b_lb, top_frac=tf, rebal=REBAL, exec_lag=EXEC_LAG,
                                            cost_bps=cost, adv=A, impact_k=IMPACT_K)["net"], ppy, TVOL), ppy)
            sb = _sh(_beta_neutral(C, A, b_lb, cost, ppy, top_frac=tf)[0], ppy)
            grid.append({"beta_lb": lb, "top_frac": tf,
                         "sharpe_dollar": round(sd, 3), "sharpe_beta_neutral": round(sb, 3)})
    grid_df = pd.DataFrame(grid)
    print("  construction surface (net Sharpe):")
    print(grid_df.to_string(index=False).replace("\n", "\n    "))

    # ── chosen a-priori config: dollar-neutral (−beta / −vol) + beta-neutral (FP) ───────────────
    dn_beta, _ = _dollar(C, A, -beta, cost, ppy)
    dn_vol, _ = _dollar(C, A, -vol, cost, ppy)
    dn_skew, _ = _dollar(C, A, -skew, cost, ppy)          # −skew: the lottery/H2 control book
    bn_beta, _bn_bt, net_beta = _beta_neutral(C, A, beta, cost, ppy)
    dn_dollar_beta_tilt = bab.net_book_beta(
        xs_backtest(C, -beta, top_frac=TOPFRAC, rebal=REBAL, exec_lag=EXEC_LAG,
                    cost_bps=cost, adv=A, impact_k=IMPACT_K)["weights"], beta)

    s_dnb, s_dnv, s_bnb = _sh(dn_beta, ppy), _sh(dn_vol, ppy), _sh(bn_beta, ppy)
    mc = bootstrap_sharpe(bn_beta.dropna(), ppy, 1000, SEED)
    dd = summarise(bn_beta.dropna(), ppy)["max_dd"]
    per_year = {int(y): round(_sh(g, ppy), 2)
                for y, g in bn_beta.dropna().groupby(bn_beta.dropna().index.year)}
    print(f"  chosen (lb={BETA_LB}, quintile, monthly):")
    print(f"    dollar-neutral −beta {s_dnb:+.2f}  (residual net-β {dn_dollar_beta_tilt.mean():+.2f})"
          f"   −vol {s_dnv:+.2f}   −skew(lottery ctrl) {_sh(dn_skew, ppy):+.2f}")
    print(f"    BETA-NEUTRAL (FP)   {s_bnb:+.2f}  [MC P5 {mc.get('sharpe_p5', float('nan')):+.2f} "
          f"P50 {mc.get('sharpe_p50', float('nan')):+.2f}]  maxDD {dd:+.1%}  net-β {net_beta.mean():+.2f}")

    # ── placebo: column-shuffle the beta signal (kill the real cross-section, keep marginals) ───
    # Two arms: the dollar book tests the raw signal's structure; the beta-neutral book tests whether
    # the +0.77 is the signal or just the mechanical net-long-dollar tilt earning market drift under
    # the hedge (a shuffled beta-neutral book keeps that tilt but has a random cross-section).
    placebo_d, placebo_b = [], []
    for _ in range(100):
        perm = beta.copy()
        perm.columns = rng.permutation(beta.columns)
        perm = perm.reindex(columns=beta.columns)
        placebo_d.append(_sh(_dollar(C, A, -perm, cost, ppy)[0], ppy))
        placebo_b.append(_sh(_beta_neutral(C, A, perm, cost, ppy)[0], ppy))
    placebo_d = np.array(placebo_d); placebo_d = placebo_d[np.isfinite(placebo_d)]
    placebo_b = np.array(placebo_b); placebo_b = placebo_b[np.isfinite(placebo_b)]   # degenerate shuffle = no vote
    pctile_d = float((s_dnb > placebo_d).mean() * 100)
    pctile_b = float((s_bnb > placebo_b).mean() * 100)
    fdr = float((placebo_b > 0.5).mean())
    print(f"  placebo: beta-neutral real {s_bnb:+.2f} at {pctile_b:.0f}th pctile "
          f"(shuffle mean {placebo_b.mean():+.2f}, p95 {np.percentile(placebo_b, 95):+.2f}); "
          f"dollar −beta {s_dnb:+.2f} at {pctile_d:.0f}th; noise clears 0.5 in {fdr:.0%}")

    # ── purged/embargoed walk-forward OOS over the (lb × top_frac × neutral) grid ───────────────
    M = {}
    for lb in (60, 90, 120):
        b_lb = _liq(bab.panel_beta(C, lb), A, C)
        for tf in (0.1, 0.2, 0.3):
            M[f"d_{lb}_{tf}"] = vol_target(
                xs_backtest(C, -b_lb, top_frac=tf, rebal=REBAL, exec_lag=EXEC_LAG,
                            cost_bps=cost, adv=A, impact_k=IMPACT_K)["net"], ppy, TVOL)
            M[f"b_{lb}_{tf}"] = _beta_neutral(C, A, b_lb, cost, ppy, top_frac=tf)[0]
    M = pd.DataFrame(M).dropna(how="all")
    full_sr = (M.mean() / M.std(ddof=1) * np.sqrt(ppy))
    tr_b, te_b = int(2.0 * ppy), int(0.5 * ppy)
    wf_oos, n_refit = _wf_oos(M, ppy, tr_b, te_b, embargo=BETA_LB)
    s_wf = _sh(wf_oos, ppy)
    n_trials = int(M.shape[1])
    var_tr = float((full_sr.clip(-3, 3) / np.sqrt(ppy)).var())
    bn = bn_beta.dropna()
    dsr = deflated_sharpe(bn.mean() / bn.std(ddof=1), len(bn), bn.skew(), bn.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8))
    print(f"  walk-forward OOS (purged, embargo={BETA_LB}b): in-sample best {full_sr.max():+.2f}"
          f"  →  WF-OOS {s_wf:+.2f}  ({n_refit} refits)   deflated SR (N={n_trials}) {dsr:.2f}")

    # ── cost sensitivity + break-even (on the headline beta-neutral book) ───────────────────────
    def at_cost(m):
        return _beta_neutral(C, A, beta, m * cost, ppy)[0]
    levels = {f"{m:.0f}x": round(_sh(at_cost(m), ppy), 3) for m in (1, 2, 3)}
    breakeven = next((round(float(m), 2) for m in np.linspace(0.5, 8.0, 31)
                      if _sh(at_cost(m), ppy) <= 0.5), None)
    print(f"  cost 1x/2x/3x: {levels}  break-even-to-0.5 ≈ {breakeven}x base")

    # ── orthogonalisation: is it beta, or lottery? books for beta / vol / skew, corr + regress ──
    O = pd.DataFrame({"beta": bn_beta, "vol": dn_vol, "skew": dn_skew}).dropna()
    ocorr = O.corr().round(3)
    ca, ta, na = _ols(O["beta"], O["skew"])            # BAB alpha controlling for lottery(skew)
    cv, tv, nv = _ols(O["vol"], O["skew"])             # low-vol alpha controlling for lottery
    cb, tb, nb = _ols(O["beta"], O["vol"])             # beta vs vol (same low-risk effect?)
    ortho = {
        "corr_beta_vol": float(ocorr.loc["beta", "vol"]),
        "corr_beta_skew": float(ocorr.loc["beta", "skew"]),
        "corr_vol_skew": float(ocorr.loc["vol", "skew"]),
        "beta_on_skew_alpha_ann": round(float(ca[0] * ppy), 4), "beta_on_skew_alpha_t": round(float(ta[0]), 2),
        "vol_on_skew_alpha_ann": round(float(cv[0] * ppy), 4), "vol_on_skew_alpha_t": round(float(tv[0]), 2),
        "beta_on_vol_slope": round(float(cb[1]), 3), "beta_on_vol_alpha_t": round(float(tb[0]), 2),
    }
    print(f"  orthogonalise: corr(beta,vol) {ortho['corr_beta_vol']:+.2f}  corr(beta,skew) "
          f"{ortho['corr_beta_skew']:+.2f}  |  BAB alpha ⟂ skew t={ortho['beta_on_skew_alpha_t']:+.1f}"
          f"  low-vol alpha ⟂ skew t={ortho['vol_on_skew_alpha_t']:+.1f}")

    # ── crypto only: robustness beta vs BTC market (not the EW panel) ───────────────────────────
    btc_row = {}
    if cfg["mkt_col"] and cfg["mkt_col"] in C.columns:
        mkt_r = C[cfg["mkt_col"]].pct_change()
        beta_btc = _liq(bab.panel_beta(C, BETA_LB, market=mkt_r), A, C)
        s_btc = _sh(_beta_neutral(C, A, beta_btc, cost, ppy)[0], ppy)
        btc_row = {"beta_neutral_sharpe_BTC_market": round(s_btc, 3)}
        print(f"  robustness — beta vs BTC market (not EW panel): beta-neutral {s_btc:+.2f}")

    # ── correlation to the master book + does adding it lift the book? (headline series) ───
    corr, lift = {}, {}
    bp_path, bs_path = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    head = bn_beta.copy()                                 # the honest FP construction is the book candidate
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]
        bs = pd.read_parquet(bs_path)
        for f in (bp, bs):
            if f.index.tz is not None:
                f.index = f.index.tz_localize(None)
        h = head.copy()
        h.index = h.index.tz_localize(None)
        common = h.dropna().index.intersection(bp.dropna().index)
        corr["book"] = round(float(h.reindex(common).corr(bp.reindex(common))), 3)
        for c in bs.columns:
            corr[c] = round(float(h.reindex(common).corr(bs[c].reindex(common))), 3)
        bk = bp.reindex(common).dropna()
        hm = h.reindex(bk.index).fillna(0.0)
        hm = hm * (bk.std() / hm.std())                   # vol-match before blending
        lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * hm, ppy), 3)
                for w in (0.0, 0.15, 0.3, 0.5)}
        print(f"  corr to book {corr.get('book')}  → book-lift by weight {lift}")

    summ = {
        "config": {"beta_lb": BETA_LB, "vol_lb": VOL_LB, "top_frac": TOPFRAC, "rebal": REBAL,
                   "top_n": TOPN, "exec_lag": EXEC_LAG, "cost_bps_per_side": cost, "winsor": winsor,
                   "window": [str(C.index.min().date()), str(C.index.max().date())], "ppy": ppy},
        "data_integrity": {"artifact_name_days": n_big, "inf_name_days": n_inf,
                           "beta_neutral_sharpe_RAW": round(raw_head, 3),
                           "beta_neutral_sharpe_CLEAN": round(s_bnb, 3)},
        "sign": {"dollar_neg_beta": round(s_neg, 3), "dollar_pos_beta": round(s_pos, 3)},
        "sharpe": {"dollar_neutral_beta": round(s_dnb, 3), "dollar_neutral_vol": round(s_dnv, 3),
                   "dollar_neutral_skew_lottery_ctrl": round(_sh(dn_skew, ppy), 3),
                   "beta_neutral_FP": round(s_bnb, 3)},
        "dollar_book_residual_beta": round(float(dn_dollar_beta_tilt.mean()), 3),
        "beta_neutral_net_beta": round(float(net_beta.mean()), 3),
        "beta_neutral_mc": {k: mc.get(k) for k in ("sharpe_p5", "sharpe_p50", "sharpe_p95")},
        "beta_neutral_maxdd": round(dd, 3), "per_year": per_year,
        "placebo": {"beta_neutral_real": round(s_bnb, 3),
                    "beta_neutral_placebo_mean": round(float(placebo_b.mean()), 3),
                    "beta_neutral_placebo_p95": round(float(np.percentile(placebo_b, 95)), 3),
                    "beta_neutral_real_pctile": round(pctile_b, 0),
                    "dollar_real": round(s_dnb, 3), "dollar_real_pctile": round(pctile_d, 0),
                    "placebo_fdr_at_0.5": round(fdr, 3)},
        "walk_forward": {"in_sample_best": round(float(full_sr.max()), 3), "wf_oos": round(s_wf, 3),
                         "n_refits": n_refit, "n_grid_trials": n_trials, "deflated_sharpe": round(dsr, 3)},
        "cost_sensitivity": levels, "breakeven_cost_mult_to_0.5": breakeven,
        "orthogonalisation": ortho, "corr_to_book": corr, "book_lift_by_weight": lift, **btc_row,
    }
    books = {f"{kind}_dollar_beta": dn_beta, f"{kind}_beta_neutral": bn_beta,
             f"{kind}_dollar_vol": dn_vol}
    grid_df.insert(0, "asset", kind)
    return summ, {"books": books, "grid": grid_df}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summ, allbooks, grids = {}, {}, []
    for kind in ("crypto", "equity"):
        s, x = run_asset(kind)
        summ[kind] = s
        allbooks.update(x["books"])
        grids.append(x["grid"])

    rets = pd.DataFrame(allbooks)
    rets.to_parquet(BAB_DIR / "bab_returns.parquet")
    pd.concat(grids, ignore_index=True).to_csv(BAB_DIR / "bab_grid.csv", index=False)
    pd.concat([pd.DataFrame(summ[k]["orthogonalisation"], index=[k]) for k in summ]).to_csv(
        BAB_DIR / "bab_orthogonal.csv")
    (BAB_DIR / "bab_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(summ, allbooks)

    print(f"\n{'='*78}\nVERDICT")
    for k in summ:
        s = summ[k]
        print(f"  {k:7s}: beta-neutral {s['sharpe']['beta_neutral_FP']:+.2f} "
              f"[MC-P5 {s['beta_neutral_mc']['sharpe_p5']:+.2f}] WF-OOS {s['walk_forward']['wf_oos']:+.2f} "
              f"| dollar −beta {s['sharpe']['dollar_neutral_beta']:+.2f} −vol {s['sharpe']['dollar_neutral_vol']:+.2f}"
              f" | corr-book {s['corr_to_book'].get('book')}")
    print("RUN BAB OK")


def _figure(summ, books):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Betting-against-beta / low-vol — crypto & US equity (net of costs, vol-target 15%)",
                 fontsize=13, fontweight="bold")

    # (1) equity curves
    a = ax[0, 0]
    for c, lab in [("crypto_beta_neutral", "crypto beta-neutral (FP)"),
                   ("crypto_dollar_beta", "crypto dollar-neutral −β"),
                   ("equity_beta_neutral", "equity beta-neutral (FP)")]:
        if c in books:
            r = books[c].dropna()
            a.plot((1 + r).cumprod().index, (1 + r).cumprod().values, label=lab, lw=1.4)
    a.axhline(1.0, color="k", lw=0.6, ls=":")
    a.set_title("Equity curves (vol-targeted 15%)"); a.legend(fontsize=8); a.set_yscale("log")

    # (2) dollar vs beta-neutral, crypto vs equity
    a = ax[0, 1]
    ks = list(summ.keys()); x = np.arange(len(ks)); wd = 0.35
    a.bar(x - wd/2, [summ[k]["sharpe"]["dollar_neutral_beta"] for k in ks], wd, label="dollar-neutral −β", color="#68a")
    a.bar(x + wd/2, [summ[k]["sharpe"]["beta_neutral_FP"] for k in ks], wd, label="beta-neutral (FP)", color="#2b6")
    a.axhline(0.5, color="r", ls="--", lw=1, label="robust bar 0.5"); a.axhline(0, color="k", lw=0.6)
    a.set_xticks(x); a.set_xticklabels(ks)
    for i, k in enumerate(ks):
        a.text(i - wd/2, summ[k]["sharpe"]["dollar_neutral_beta"], f"{summ[k]['sharpe']['dollar_neutral_beta']:+.2f}", ha="center", va="bottom", fontsize=8)
        a.text(i + wd/2, summ[k]["sharpe"]["beta_neutral_FP"], f"{summ[k]['sharpe']['beta_neutral_FP']:+.2f}", ha="center", va="bottom", fontsize=8)
    a.set_title("Neutralising the market tilt: dollar vs beta-neutral"); a.legend(fontsize=8)

    # (3) orthogonalisation — is it beta or lottery?
    a = ax[1, 0]
    k0 = "crypto"
    o = summ[k0]["orthogonalisation"]
    labs = ["corr\nβ–vol", "corr\nβ–skew", "corr\nvol–skew"]
    vals = [o["corr_beta_vol"], o["corr_beta_skew"], o["corr_vol_skew"]]
    a.bar(labs, vals, color=["#2b6", "#a63", "#a63"])
    a.axhline(0, color="k", lw=0.6); a.set_ylim(-1, 1)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    a.set_title(f"{k0}: BAB alpha net of skew  t={o['beta_on_skew_alpha_t']:+.1f}  "
                f"(beta, not lottery, if t>~2)", fontsize=10)

    # (4) cost sensitivity (crypto beta-neutral)
    a = ax[1, 1]
    lv = summ["crypto"]["cost_sensitivity"]
    xs, ys = list(lv.keys()), list(lv.values())
    a.bar(xs, ys, color=["#3b7", "#7a3", "#a73"])
    a.axhline(0.5, color="r", ls="--", lw=1, label="robust bar 0.5"); a.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(ys):
        a.text(i, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=9)
    _be = summ['crypto']['breakeven_cost_mult_to_0.5']
    a.set_title(f"crypto beta-neutral Sharpe vs cost "
                f"(stays >0.5 to {'>8' if _be is None else _be}× base)")
    a.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "bab.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
