"""On-chain / network-signal deep-dive (H3) — run through the same funnel as every other family
(vol-target 15%, t+2 execution, liquidity-aware costs, shuffled-signal placebo, purged/embargoed
walk-forward OOS, block-bootstrap MC, deflated Sharpe, cost sensitivity, correlation-to-book + lift).

H3 is the one hypothesis whose information is *not* derived from price. Free-data reality (the
vendor catalog decides, see src/data/onchain.py): the cross-section spans 37 names, but exchange
flows are free on BTC/ETH only and adjusted-transfer-value / realized-cap / USD fees are Pro-only:
  • ADOPTION MOMENTUM  — active-address / tx-count growth (the on-chain twin of price momentum)
  • ON-CHAIN VALUE     — MVRV, NVM/Metcalfe (market cap per active user); cheap-per-network = long
  • OWNERSHIP & DILUTION — holder-count growth, cap-per-holder, issuance ÷ supply, fee yield
  • NET-vs-PRICE DIVERGENCE — activity outrunning price (orthogonal-by-construction "new info")
  • BTC/ETH TS overlays — MVRV-z, NVT, Puell, stablecoin-SSR, and **exchange net-flow / exchange
    supply**, the series the whole "coins leaving exchanges" thesis rests on. Timing rules are
    scored against buy-and-hold *and* against a HAC predictive regression, because with 14 overlays
    on 2-3 cycles the best Sharpe is a multiple-testing artifact unless the signal itself forecasts.

The decisive question (Liu-Tsyvinski-Wu JF2022; Cong-Karolyi-Tang-Zhao MgmtSci2024): does on-chain add
edge *over price*, or is value≈re-labelled value and momentum≈re-labelled momentum? Answered by
orthogonalising every on-chain book against price-momentum and price-reversal books on the *identical*
37-name universe. Honest verdict (both signs kept) → reports/onchain_summary.json + docs/strategies/ONCHAIN.md.

    python scripts/onchain/run_onchain.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, CAPITAL_USD, ONCHAIN_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data import onchain as oc  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import onchain as sig  # noqa: E402
from src.sleeves.xsect import mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
from scripts import run_master_book as mb  # noqa: E402  (scorecard + the five targets)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
FIG = REP / "figures"
CACHE = CACHE_DIR / "xs"
SEED, CAP, TVOL, PPY, COST = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL, 365, 6.0
rng = np.random.default_rng(SEED)

# a-priori config, declared before fit: value is a slow signal → monthly rebalance, 7d activity
# smoothing, tercile tails (small universe), t+2 execution, √-impact on. Headline = NVM on-chain value.
SMOOTH, REBAL, TOPFRAC, EXEC_LAG, IMPACT_K = 7, 21, 0.3, 2, 0.1
ZLB = 365  # per-asset self-relative normalisation window (≥1y, research-recommended; trailing, no look-ahead)


def _load_prices():
    C = pd.read_parquet(CACHE / "crypto_1d_close.parquet")
    A = pd.read_parquet(CACHE / "crypto_1d_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC"); A.index = A.index.tz_localize("UTC")
    keep = [s for s in oc.live_universe() if s in C.columns]
    return C[keep], A[keep]


def _load_onchain(idx, cols):
    """Load + align every on-chain panel to the price (date × name) grid. On-chain is daily UTC, so
    this is a 1:1 reindex (crypto trades every day) — no ffill gap-filling that could stale a signal."""
    out = {}
    # A live chain reporting zero active addresses / zero transactions for a day is an indexing
    # outage, not a fact about the network — and log(0) would hand the models −inf. Zeros are
    # therefore missing data for the count panels. Issuance and fees are left alone: zero there is
    # real (a fixed-supply token issues nothing; a quiet day can genuinely collect no fees).
    ZERO_IS_MISSING = ("AdrActCnt", "TxCnt", "TxTfrCnt", "AdrBalCnt")
    for m in ("AdrActCnt", "TxCnt", "CapMrktCurUSD", "CapMVRVCur", "PriceUSD",
              "AdrBalCnt", "IssTotNtv", "SplyCur", "FeeTotNtv",
              "FlowInExNtv", "FlowOutExNtv", "SplyExNtv"):
        p = oc.load(m).reindex(index=idx, columns=cols)   # names the metric misses stay all-NaN
        if m in ZERO_IS_MISSING:
            n_zero = int((p == 0).sum().sum())
            if n_zero:
                print(f"  {m}: {n_zero} zero-count days treated as missing (indexing outages)")
            p = p.mask(p == 0)
        out[m] = p
    return out


def _sh(net):
    return summarise(net.dropna(), PPY)["sharpe_ann"]


def _winsor(C, cap=1.0):
    r = C.pct_change()
    return C.mask(r.abs() > cap).ffill()   # a >100% 1d move on a liquid name = print/liquidation artifact


def per_asset_z(df, lb=ZLB):
    """Self-relative z-score: value cheap/rich vs its *own* trailing history (removes the coin
    fixed-effect so a cross-sectional rank is timing, not a permanent size/type tilt). Trailing."""
    m = df.rolling(lb, min_periods=lb // 2).mean()
    s = df.rolling(lb, min_periods=lb // 2).std()
    return (df - m) / s.replace(0.0, np.nan)


def _book(C, A, signal, topn, top_frac=TOPFRAC, rebal=REBAL, cost=COST):
    """Signal → liquidity-capped (top-N each bar) → dollar-neutral xs book → vol-targeted net."""
    s = top_n_liquid(signal, A, topn) if topn else signal
    bt = xs_backtest(C, s, top_frac=top_frac, weighting="equal", rebal=rebal, exec_lag=EXEC_LAG,
                     cost_bps=cost, adv=A, impact_k=IMPACT_K)
    return vol_target(bt["net"], PPY, TVOL), bt


def build_signals(C, oc_p):
    """The on-chain signal family — each stamped so the engine ranks LONG its top (positive = long).
    Value signals are negated (low NVM/MVRV = cheap = long); momentum/divergence are positive."""
    adr, tx = oc_p["AdrActCnt"], oc_p["TxCnt"]
    cap, mvrv, pxu = oc_p["CapMrktCurUSD"], oc_p["CapMVRVCur"], oc_p["PriceUSD"]
    hold, iss, sply, fee = oc_p["AdrBalCnt"], oc_p["IssTotNtv"], oc_p["SplyCur"], oc_p["FeeTotNtv"]
    S = {
        # adoption momentum (≈ price momentum?) — long fast-growing networks
        "adr_mom30": sig.adr_momentum(adr, 30, SMOOTH),
        "tx_mom30": sig.tx_momentum(tx, 30, SMOOTH),
        # on-chain VALUE — long cheap-per-network (the LTW/Cong "CVALUE" axis). Level & self-relative.
        "nvm_val": -sig.nvm_ratio(cap, adr, SMOOTH),
        "nvm_z_val": -per_asset_z(np.log(sig.nvm_ratio(cap, adr, SMOOTH))),
        "metcalfe_val": -sig.nvm_ratio(cap, adr, SMOOTH, metcalfe=True),
        "mvrv_val": -sig.mvrv_value(mvrv),
        "mvrv_z_val": -per_asset_z(sig.mvrv_value(mvrv)),
        # ownership stock — holders is who *owns*, active addresses is who *moved* (36 names)
        "holder_mom30": sig.holder_momentum(hold, 30, SMOOTH),
        "holder_val": -sig.mcap_per_holder(cap, hold, SMOOTH),
        # dilution: annualised issuance ÷ supply, long the low-inflation names (35)
        "low_inflation": -sig.supply_inflation(iss, sply, 90),
        # network earnings yield — fees are free on only 14 names, so this book is a 14-name subset
        "fee_yield_val": sig.fee_yield(fee, pxu, cap, 90),
        # orthogonal-by-construction: activity outrunning price
        "divergence": sig.net_vs_price_divergence(adr, pxu, 30, SMOOTH),
    }
    # a blend of the credible, less-collinear angles (value + divergence), averaged on rank
    ranks = [S[k].rank(axis=1, pct=True) for k in ("nvm_z_val", "mvrv_z_val", "divergence")]
    S["blend"] = sum(ranks) / len(ranks)
    return S


def _ols(y, *xs):
    df = pd.concat([y.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(xs)], axis=1).dropna()
    if len(df) < 50:
        return np.array([0.0]), np.array([0.0]), 0
    Y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[f"x{i}"].to_numpy() for i in range(len(xs))])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    dof = max(len(Y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    t = coef / np.sqrt(np.diag(cov))
    return coef, t, len(df)


def _wf_oos(M, train_bars, test_bars, embargo, top_k=3):
    segs, picks, start = [], [], train_bars
    while start + test_bars <= len(M):
        train = M.iloc[max(0, start - train_bars):max(0, start - embargo)]
        test = M.iloc[start:start + test_bars]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        chosen = list(sr.nlargest(top_k).index)
        if chosen:
            segs.append(test[chosen].mean(axis=1)); picks.extend(chosen)
        start += test_bars
    oos = pd.concat(segs) if segs else pd.Series(dtype=float)
    return oos, len(picks) // max(top_k, 1)


# ── cross-sectional deep-dive ─────────────────────────────────────────────────────────────────
def run_cross_section():
    Craw, A = _load_prices()
    C = _winsor(Craw, 1.0)
    oc_p = _load_onchain(C.index, C.columns)
    S = build_signals(C, oc_p)
    NAMES = C.shape[1]
    # Headline chosen a-priori on the *literature*, not the fit: the one on-chain factor with hard
    # academic backing (Liu-Tsyvinski-Wu JF2022; Cong et al MgmtSci2024) is cross-sectional on-chain
    # VALUE = market cap per active address ("CVALUE"), LOW = cheap = long — that is nvm_val (level).
    # top-20 is the round mid-cross-section (NOT the sweep-maximal N=30), so the headline is not
    # peak-picked; the full top-N × signal surface is printed above so nothing is hidden.
    HEAD_SIG, HEAD_N = "nvm_val", 20
    span = f"{C.index.min().date()}..{C.index.max().date()}"
    print(f"\n{'='*80}\nON-CHAIN CROSS-SECTION  {NAMES} names (free on-chain ∩ repo panel), {span}, {len(C)} bars\n{'='*80}")
    cov = {m: int(oc_p[m].notna().any().sum()) for m in
           ("AdrActCnt", "CapMVRVCur", "AdrBalCnt", "IssTotNtv", "FeeTotNtv", "FlowInExNtv")}
    print(f"  coverage: AdrActCnt {cov['AdrActCnt']}, MVRV {cov['CapMVRVCur']}, holders "
          f"{cov['AdrBalCnt']}, issuance {cov['IssTotNtv']}, fees {cov['FeeTotNtv']}, "
          f"exchange-flows {cov['FlowInExNtv']} names  |  free top-N ceiling = {NAMES} "
          f"(top-50/100 impossible: SOL/SUI/TON… carry market data only, no network metrics)")

    # ── sign check: verify each signal's theory direction beats its inverse on all names ─────────
    print("\n  sign check (all-names dollar book, monthly): net Sharpe of signal vs −signal")
    for k in ("adr_mom30", "nvm_val", "mvrv_val", "divergence"):
        sp = _sh(_book(C, A, S[k], NAMES)[0]); sn = _sh(_book(C, A, -S[k], NAMES)[0])
        print(f"    {k:12s} {sp:+.2f}   (inverse {sn:+.2f})")

    # ── top-N sweep × signal (THE top-10/20/30 question) ────────────────────────────────────────
    print("\n  top-N × signal (net Sharpe, monthly rebal):")
    topns = [10, 20, 30, NAMES]
    sweep = {}
    hdr = "    " + "signal".ljust(13) + "".join(f"N={n:<7}" for n in topns)
    print(hdr)
    for k in ("adr_mom30", "tx_mom30", "nvm_val", "nvm_z_val", "metcalfe_val", "mvrv_val",
              "mvrv_z_val", "holder_mom30", "holder_val", "low_inflation", "fee_yield_val",
              "divergence", "blend"):
        row = {}
        for n in topns:
            row[n] = round(_sh(_book(C, A, S[k], n)[0]), 2)
        sweep[k] = row
        print("    " + k.ljust(13) + "".join(f"{row[n]:<+8.2f}" for n in topns))

    # ── construction surface for the headline signal: lookback × rebal (timeframe) × tail ────────
    print(f"\n  construction surface — headline on-chain VALUE ({HEAD_SIG}), top-{HEAD_N}:")
    print("    rebal(=holding, bars) × top_frac, net Sharpe:")
    surf = []
    for rb in (1, 7, 21, 63):     # daily / weekly / monthly / quarterly holding == the 'timeframe' axis
        for tf in (0.2, 0.3):
            sh = _sh(_book(C, A, S[HEAD_SIG], HEAD_N, top_frac=tf, rebal=rb)[0])
            surf.append({"rebal": rb, "top_frac": tf, "sharpe": round(sh, 3)})
    surf_df = pd.DataFrame(surf)
    print(surf_df.to_string(index=False).replace("\n", "\n      "))

    # value-signal lookback sensitivity (z window) — is the edge robust to the normalisation window?
    print("\n  z-window sensitivity (nvm_z_val, top-20, monthly):")
    zrow = {}
    for zl in (180, 365, 540):
        s = -per_asset_z(np.log(sig.nvm_ratio(oc_p["CapMrktCurUSD"], oc_p["AdrActCnt"], SMOOTH)), zl)
        zrow[zl] = round(_sh(_book(C, A, s, 20)[0]), 2)
    print("    " + "  ".join(f"z{zl}={v:+.2f}" for zl, v in zrow.items()))

    # ── weekly-resampled test (on-chain is daily-native; there is no free intraday) ─────────────
    Cw = C.resample("W-MON").last()
    Aw = A.resample("W-MON").mean().reindex_like(Cw)
    nvm_w = -sig.nvm_ratio(oc_p["CapMrktCurUSD"].resample("W-MON").last(),
                           oc_p["AdrActCnt"].resample("W-MON").last(), 1)   # level, weekly already agg'd
    bt_w = xs_backtest(Cw, top_n_liquid(nvm_w, Aw, 20), top_frac=TOPFRAC, rebal=1, exec_lag=1,
                       cost_bps=COST, adv=Aw, impact_k=IMPACT_K)
    s_week = summarise(vol_target(bt_w["net"], 52, TVOL).dropna(), 52)["sharpe_ann"]
    print(f"\n  weekly-resampled book ({HEAD_SIG}, top-20): Sharpe {s_week:+.2f}   "
          f"(daily monthly-rebal {sweep[HEAD_SIG][20]:+.2f})  → on-chain edge lives at low freq")

    # ── chosen headline (defined a-priori above): nvm_val (documented CVALUE), top-20, monthly ───
    head, head_bt = _book(C, A, S[HEAD_SIG], HEAD_N)
    s_head = _sh(head)
    mc = bootstrap_sharpe(head.dropna(), PPY, 1000, SEED)
    dd = summarise(head.dropna(), PPY)["max_dd"]
    turn = head_bt["turnover"].mean()
    per_year = {int(y): round(_sh(g), 2) for y, g in head.dropna().groupby(head.dropna().index.year)}
    print(f"\n  HEADLINE {HEAD_SIG} top-{HEAD_N} monthly: Sharpe {s_head:+.2f}  "
          f"[MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]  "
          f"maxDD {dd:+.1%}  turn/bar {turn:.2f}")
    print(f"    per-year: {per_year}")

    # ── placebo: shuffle names (kill the real cross-section, keep the marginals) ─────────────────
    placebo = []
    base = S[HEAD_SIG]
    for _ in range(200):
        perm = base.copy(); perm.columns = rng.permutation(base.columns)
        perm = perm.reindex(columns=base.columns)
        placebo.append(_sh(_book(C, A, perm, HEAD_N)[0]))
    placebo = np.array(placebo); placebo = placebo[np.isfinite(placebo)]
    pctile = float((s_head > placebo).mean() * 100)
    print(f"  placebo (200 name-shuffles): real {s_head:+.2f} at {pctile:.0f}th pctile "
          f"(shuffle mean {placebo.mean():+.2f}, p95 {np.percentile(placebo, 95):+.2f})")

    # ── purged/embargoed walk-forward OOS over (signal × top-N × rebal) grid ─────────────────────
    M = {}
    for k in ("adr_mom30", "nvm_val", "nvm_z_val", "metcalfe_val", "mvrv_val", "mvrv_z_val",
              "holder_mom30", "holder_val", "low_inflation", "fee_yield_val", "divergence", "blend"):
        for n in (10, 20, NAMES):
            M[f"{k}_{n}"] = _book(C, A, S[k], n)[0]
    M = pd.DataFrame(M).dropna(how="all")
    full_sr = (M.mean() / M.std(ddof=1) * np.sqrt(PPY))
    wf_oos, n_refit = _wf_oos(M, int(2.0 * PPY), int(0.5 * PPY), embargo=ZLB)
    s_wf = _sh(wf_oos)
    n_trials = int(M.shape[1])
    var_tr = float((full_sr.clip(-3, 3) / np.sqrt(PPY)).var())
    hd = head.dropna()
    dsr = deflated_sharpe(hd.mean() / hd.std(ddof=1), len(hd), hd.skew(), hd.kurt() + 3.0, n_trials, max(var_tr, 1e-8))
    print(f"  walk-forward OOS (purged, embargo={ZLB}b): in-sample best {full_sr.max():+.2f} → "
          f"WF-OOS {s_wf:+.2f} ({n_refit} refits)   deflated SR (N={n_trials} trials) {dsr:.2f}")

    # ── cost sensitivity + break-even ───────────────────────────────────────────────────────────
    levels = {f"{m:.0f}x": round(_sh(_book(C, A, S[HEAD_SIG], HEAD_N, cost=m * COST)[0]), 3) for m in (1, 2, 3)}
    breakeven = next((round(float(m), 2) for m in np.linspace(0.5, 8.0, 31)
                      if _sh(_book(C, A, S[HEAD_SIG], HEAD_N, cost=m * COST)[0]) <= 0.5), None)
    print(f"  cost 1x/2x/3x: {levels}  break-even-to-0.5 ≈ {breakeven}x base")

    # ── THE decisive test: does on-chain add edge OVER price? orthogonalise vs price mom + reversal
    px_mom = mom(C, 30)                    # price momentum on the identical universe
    px_rev = -mom(C, 365, skip=30)         # long-term reversal = the price "value" proxy
    b_pmom = _book(C, A, px_mom, HEAD_N)[0]
    b_prev = _book(C, A, px_rev, HEAD_N)[0]
    ortho = {}
    for k in ("nvm_val", "nvm_z_val", "mvrv_val", "mvrv_z_val", "adr_mom30", "holder_mom30",
              "holder_val", "low_inflation", "fee_yield_val", "divergence", "blend"):
        bk = _book(C, A, S[k], HEAD_N)[0]
        O = pd.DataFrame({"y": bk, "pmom": b_pmom, "prev": b_prev}).dropna()
        c, t, n = _ols(O["y"], O["pmom"], O["prev"])
        ortho[k] = {"alpha_ann": round(float(c[0] * PPY), 3), "alpha_t": round(float(t[0]), 2),
                    "corr_pmom": round(float(O["y"].corr(O["pmom"])), 2),
                    "corr_prev": round(float(O["y"].corr(O["prev"])), 2)}
    print("\n  DECISIVE — on-chain book alpha net of PRICE momentum + reversal (identical universe):")
    for k, v in ortho.items():
        verdict = "adds edge" if v["alpha_t"] > 2 else ("marginal" if v["alpha_t"] > 1 else "subsumed by price")
        print(f"    {k:12s} α {v['alpha_ann']:+.3f}/yr  t={v['alpha_t']:+.2f}  "
              f"corr(pmom){v['corr_pmom']:+.2f} corr(prev){v['corr_prev']:+.2f}   → {verdict}")

    # ── the signal that survived the data fix, put through the same funnel ───────────────────────
    # nvm_val stays the headline because it was the a-priori literature pick and moving the goalposts
    # after seeing results is how families get talked into the book. But adoption momentum now clears
    # the alpha-over-price bar, so it gets measured properly and reported as what it is: a post-hoc
    # candidate, penalised for every trial in this file, not a discovery.
    ALT = "adr_mom30"
    alt, alt_bt = _book(C, A, S[ALT], HEAD_N)
    s_alt = _sh(alt)
    mc_alt = bootstrap_sharpe(alt.dropna(), PPY, 1000, SEED)
    alt_year = {int(y): round(_sh(g), 2) for y, g in alt.dropna().groupby(alt.dropna().index.year)}
    plac_alt = []
    for _ in range(200):
        perm = S[ALT].copy(); perm.columns = rng.permutation(S[ALT].columns)
        plac_alt.append(_sh(_book(C, A, perm.reindex(columns=S[ALT].columns), HEAD_N)[0]))
    plac_alt = np.array([x for x in plac_alt if np.isfinite(x)])
    pct_alt = float((s_alt > plac_alt).mean() * 100)
    # walk-forward with the construction held fixed: only the top-N is refit, so the pool cannot
    # re-discover the signal and be scored for the discovery.
    M_alt = pd.DataFrame({f"{ALT}_{n}": _book(C, A, S[ALT], n)[0] for n in (10, 20, NAMES)}).dropna(how="all")
    wf_alt, n_ref_alt = _wf_oos(M_alt, int(2.0 * PPY), int(0.5 * PPY), embargo=ZLB, top_k=1)
    ad = alt.dropna()
    dsr_alt = deflated_sharpe(ad.mean() / ad.std(ddof=1), len(ad), ad.skew(), ad.kurt() + 3.0,
                              n_trials, max(var_tr, 1e-8))
    print(f"\n  POST-HOC CANDIDATE {ALT} top-{HEAD_N} monthly: Sharpe {s_alt:+.2f} "
          f"[MC-P5 {mc_alt.get('sharpe_p5', float('nan')):+.2f}]  "
          f"placebo {pct_alt:.0f}th pctile (p95 {np.percentile(plac_alt, 95):+.2f})  "
          f"WF-OOS (construction fixed) {_sh(wf_alt):+.2f}  DSR (N={n_trials}) {dsr_alt:.2f}")
    print(f"    per-year: {alt_year}")
    alt_summ = {"signal": ALT, "topn": HEAD_N, "sharpe": round(s_alt, 3),
                "mc_p5": mc_alt.get("sharpe_p5"), "placebo_pctile": round(pct_alt, 0),
                "placebo_p95": round(float(np.percentile(plac_alt, 95)), 3),
                "wf_oos_construction_fixed": round(_sh(wf_alt), 3), "n_refits": n_ref_alt,
                "deflated_sharpe": round(dsr_alt, 3), "per_year": alt_year,
                "status": "post-hoc — surfaced after the dead-shell fix, not the a-priori pick"}

    # ── correlation to master book + does it lift the book? ─────────────────────────────────────
    corr, lift = {}, {}
    bp_path, bs_path = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]; bs = pd.read_parquet(bs_path)
        for f in (bp, bs):
            if f.index.tz is not None:
                f.index = f.index.tz_localize(None)
        h = head.copy(); h.index = h.index.tz_localize(None)
        common = h.dropna().index.intersection(bp.dropna().index)
        corr["book"] = round(float(h.reindex(common).corr(bp.reindex(common))), 3)
        for c in bs.columns:
            corr[c] = round(float(h.reindex(common).corr(bs[c].reindex(common))), 3)
        bk = bp.reindex(common).dropna()
        # Every scored target, not just Sharpe. Sharpe has never been what binds this book — the worst
        # month and the losing-month streak are — so a Sharpe-only lift asks the wrong question of an
        # addition and would call a leg that fixes the binding axis "nothing".
        lift = mb.book_lift(h, bk)
        print(f"\n  corr to master book {corr.get('book')}  (legs: "
              f"{ {k: corr[k] for k in list(corr) if k != 'book'} })")
        _print_lift(f"book-lift, {HEAD_SIG}", lift)
        a2 = alt.copy(); a2.index = a2.index.tz_localize(None)
        alt_summ["corr_to_book"] = round(float(a2.reindex(common).corr(bp.reindex(common))), 3)
        alt_summ["book_lift_by_weight"] = mb.book_lift(a2, bk)
        print(f"  {ALT}: corr to book {alt_summ['corr_to_book']}")
        _print_lift(f"book-lift, {ALT}", alt_summ["book_lift_by_weight"])

    summ = {
        "config": {"universe_names": NAMES, "window": span, "smooth": SMOOTH, "rebal": REBAL,
                   "top_frac": TOPFRAC, "exec_lag": EXEC_LAG, "cost_bps": COST, "z_lb": ZLB,
                   "headline_signal": HEAD_SIG, "headline_topn": HEAD_N, "ppy": PPY,
                   "topn_ceiling_reason": "free on-chain covers 37 names; SOL/SUI/TON/APT Pro-walled"},
        "coverage": cov, "topn_sweep": sweep, "construction_surface": surf,
        "z_window_sensitivity": zrow, "weekly_resample_sharpe": round(float(s_week), 3),
        "headline": {"signal": HEAD_SIG, "topn": HEAD_N, "sharpe": round(s_head, 3),
                     "mc_p5": mc.get("sharpe_p5"), "mc_p50": mc.get("sharpe_p50"),
                     "maxdd": round(dd, 3), "turnover": round(float(turn), 3), "per_year": per_year},
        "placebo": {"real": round(s_head, 3), "pctile": round(pctile, 0),
                    "shuffle_mean": round(float(placebo.mean()), 3),
                    "shuffle_p95": round(float(np.percentile(placebo, 95)), 3)},
        "walk_forward": {"in_sample_best": round(float(full_sr.max()), 3), "wf_oos": round(s_wf, 3),
                         "n_refits": n_refit, "n_trials": n_trials, "deflated_sharpe": round(dsr, 3)},
        "cost_sensitivity": levels, "breakeven_cost_mult_to_0.5": breakeven,
        "orthogonalisation_vs_price": ortho, "corr_to_book": corr, "book_lift_by_weight": lift,
        "post_hoc_candidate": alt_summ,
    }
    books = {"onchain_headline": head, "onchain_adr_mom": alt,
             "onchain_pmom_ctrl": b_pmom, "onchain_prev_ctrl": b_prev}
    return summ, books


def _print_lift(label, lift):
    """One row per weight, every target the book is scored on. The window is printed because the blend
    can only run where both series exist — on-chain starts 2020, so the 0% row is the book restricted to
    that overlap and will not match the headline scorecard measured over 2011-2026."""
    print(f"  {label} (overlap window {lift['window']}; the 0% row is the book on THAT window):")
    print(f"    {'w':>5} {'Sharpe':>7} {'maxDD':>7} {'worst mo':>9} {'months+':>8} {'streak':>7} {'targets':>8}")
    for w, c in lift.items():
        if w == "window":
            continue
        ctl = c.get("control")
        tail = ("" if not ctl else
                f"   | rotated control: Sh {ctl['sharpe']:+.2f} DD {ctl['max_dd']:+.1%} "
                f"worst {ctl['worst_month']:+.2%} mo {ctl['months_in_profit']:.0%} "
                f"targets {ctl['targets_median']:.1f}/5")
        print(f"    {w:>5} {c['sharpe']:>+7.2f} {c['max_dd']:>+7.1%} {c['worst_month']:>+9.2%} "
              f"{c['months_in_profit']:>8.0%} {c['longest_losing_streak_mo']:>7d} {c['targets']:>7d}/5{tail}")


def _sh_ann(x):
    return summarise(x.dropna(), PPY)["sharpe_ann"]


# ── time-series BTC/ETH overlays (near-unbacktestable; report with the caveat) ──────────────────
def run_timeseries():
    print(f"\n{'='*80}\nON-CHAIN TIME-SERIES OVERLAYS (BTC/ETH) — 2-3 cycles only; regime context, not alpha\n{'='*80}")
    Craw, A = _load_prices()
    C = _winsor(Craw, 1.0)
    oc_p = _load_onchain(C.index, C.columns)
    bc = pd.read_parquet(CACHE_DIR / "onchain/btc_blockchain.parquet")
    if bc.index.tz is None:
        bc.index = bc.index.tz_localize("UTC")
    bc = bc.reindex(C.index).ffill()
    stab = pd.read_parquet(CACHE_DIR / "onchain/stablecoin_supply.parquet")
    if stab.index.tz is None:
        stab.index = stab.index.tz_localize("UTC")
    stab_tot = stab.reindex(C.index).ffill().sum(axis=1)

    def ts_positions(timing, long_short=False):
        """Signal → executed position path (already lagged). Kept separate from the P&L so the
        same path can be re-timed for the placebo below."""
        pos = (-timing).clip(-1, 1)                      # signal already oriented (high z = fade)
        if not long_short:
            pos = pos.clip(lower=0.0)
        return pos.shift(EXEC_LAG).fillna(0.0)

    def pnl(px, pos):
        r = px.pct_change()
        net = pos * r - pos.diff().abs().fillna(0.0) * COST / 1e4
        return vol_target(net, PPY, TVOL)

    def ts_book(px, timing, long_short=False):
        """Trade one asset by an on-chain z: long/flat (z below its own median → risk-on) or
        long/short. exec_lag=2, simple bps cost on position change, vol-targeted."""
        return pnl(px, ts_positions(timing, long_short))

    def timing_placebo(px, pos, n=500):
        """Percentile of the real overlay against random *timing* of the same position path.
        Circularly rotating the path by a random offset preserves average exposure, switching
        frequency and on/off persistence exactly, and destroys only the alignment with returns —
        so this isolates timing skill from the long-only beta a long/flat rule collects by
        default. A rule that beats buy-and-hold but sits mid-distribution here is harvesting beta."""
        p = pos.to_numpy()
        real = _sh_ann(pnl(px, pos))
        draws = []
        for k in rng.integers(1, len(p) - 1, size=n):
            draws.append(_sh_ann(pnl(px, pd.Series(np.roll(p, int(k)), index=pos.index))))
        d = np.array([x for x in draws if np.isfinite(x)])
        return real, float((real > d).mean() * 100), float(d.mean()), float(np.percentile(d, 95))

    rows = []
    btc, eth = C["BTCUSDT"], C["ETHUSDT"]
    mvrv_btc = sig.mvrv_zscore(oc_p["CapMVRVCur"]["BTCUSDT"])
    mvrv_eth = sig.mvrv_zscore(oc_p["CapMVRVCur"]["ETHUSDT"])
    nvt_btc = per_asset_z(sig.nvt_signal(oc_p["CapMrktCurUSD"]["BTCUSDT"], bc["estimated-transaction-volume-usd"]))
    puell_btc = per_asset_z(sig.puell_multiple(bc["miners-revenue"]))
    ssr = sig.stablecoin_ssr_growth(stab_tot)             # >0 = supply expanding = risk-on
    # exchange flow/balance — the highest-information on-chain series and the thesis's whole point
    # ("coins leaving exchanges = accumulation"). Free on the community tier for BTC/ETH only, so it
    # is a two-asset time series, never a cross-section.
    fi, fo, sx, sc = oc_p["FlowInExNtv"], oc_p["FlowOutExNtv"], oc_p["SplyExNtv"], oc_p["SplyCur"]
    flow = {a: sig.exchange_netflow_z(fi[a], fo[a], sx[a], SMOOTH, ZLB) for a in ("BTCUSDT", "ETHUSDT")}
    exsp = {a: per_asset_z(sig.exchange_supply_trend(sx[a], sc[a], 30)) for a in ("BTCUSDT", "ETHUSDT")}
    for name, px, tim, ls in [
        ("BTC MVRV-z long/flat", btc, mvrv_btc, False),
        ("BTC MVRV-z long/short", btc, mvrv_btc, True),
        ("ETH MVRV-z long/flat", eth, mvrv_eth, False),
        ("BTC NVT long/flat", btc, nvt_btc, False),
        ("BTC Puell long/flat", btc, puell_btc, False),
        ("BTC SSR-growth long/flat", btc, -ssr, False),   # ssr already oriented (expansion=long) → pass −ssr so fade-logic long-when-expanding
        ("BTC exch-netflow long/flat", btc, flow["BTCUSDT"], False),
        ("BTC exch-netflow long/short", btc, flow["BTCUSDT"], True),
        ("ETH exch-netflow long/flat", eth, flow["ETHUSDT"], False),
        ("ETH exch-netflow long/short", eth, flow["ETHUSDT"], True),
        ("BTC exch-supply-trend long/flat", btc, exsp["BTCUSDT"], False),
        ("BTC exch-supply-trend long/short", btc, exsp["BTCUSDT"], True),
        ("ETH exch-supply-trend long/flat", eth, exsp["ETHUSDT"], False),
        ("ETH exch-supply-trend long/short", eth, exsp["ETHUSDT"], True),
    ]:
        b = ts_book(px, tim, ls)
        rows.append((name, round(_sh_ann(b), 2), round(summarise(b.dropna(), PPY)["max_dd"], 3)))
    bh, bh_eth = _sh_ann(btc.pct_change()), _sh_ann(eth.pct_change())
    print(f"  buy-and-hold Sharpe (same window, vol-targeted): BTC {bh:+.2f}  ETH {bh_eth:+.2f}")
    for n, s, d in rows:
        print(f"    {n:32s} Sharpe {s:+.2f}  maxDD {d:+.1%}")

    # ── is there information at all? predictive regression, HAC-corrected ───────────────────────
    # A trading rule can lose to buy-and-hold and the signal still be informative (or beat it by
    # luck across 14 overlays). This asks the prior question directly: does today's exchange-flow z
    # forecast the forward return? Overlapping windows ⇒ Newey-West t with lag = horizon.
    def _nw_t(y: pd.Series, x: pd.Series, lag: int) -> tuple[float, float, int]:
        d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if len(d) < 200:
            print(f"    ! predictive regression skipped — only {len(d)} overlapping obs")
            return float("nan"), float("nan"), len(d)
        X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
        b, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
        e = d["y"].to_numpy() - X @ b
        XtX_inv = np.linalg.inv(X.T @ X)
        S = (X * e[:, None]).T @ (X * e[:, None])
        for L in range(1, lag + 1):                       # Bartlett kernel
            w = 1.0 - L / (lag + 1.0)
            G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
            S += w * (G + G.T)
        se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
        return float(b[1]), float(b[1] / se[1]), len(d)

    # ── random-timing control for the exchange-flow overlays ────────────────────────────────────
    print("\n  random-timing placebo (500 rotations of the same position path — beta vs timing):")
    plac = {}
    for name, px, tim, ls in [
        ("BTC exch-netflow long/flat", btc, flow["BTCUSDT"], False),
        ("ETH exch-netflow long/flat", eth, flow["ETHUSDT"], False),
        ("BTC exch-supply-trend long/flat", btc, exsp["BTCUSDT"], False),
        ("ETH exch-supply-trend long/flat", eth, exsp["ETHUSDT"], False),
        ("BTC MVRV-z long/flat", btc, mvrv_btc, False),        # reference: an already-rejected overlay
    ]:
        pos = ts_positions(tim, ls)
        real, pct, mean, p95 = timing_placebo(px, pos)
        plac[name] = {"real": round(real, 2), "pctile": round(pct, 0),
                      "placebo_mean": round(mean, 2), "placebo_p95": round(p95, 2),
                      "avg_exposure": round(float(pos.mean()), 2)}
        print(f"    {name:32s} real {real:+.2f} at {pct:3.0f}th pctile  "
              f"(random-timing mean {mean:+.2f}, p95 {p95:+.2f}, avg exposure {pos.mean():.2f})")

    print("\n  predictive regression — forward return on exchange-flow z (NW t, overlapping):")
    pred = {}
    for asset, px in (("BTC", btc), ("ETH", eth)):
        for sname, s in (("netflow", flow[f"{asset}USDT"]), ("supply-trend", exsp[f"{asset}USDT"])):
            for h in (7, 30):
                fwd = px.shift(-h) / px - 1.0
                beta, t, n = _nw_t(fwd, s, h)
                pred[f"{asset} {sname} {h}d"] = {"beta": round(beta, 5), "t": round(t, 2), "n": n}
                sign = "bearish-as-theorised" if beta < 0 else "OPPOSITE to theory"
                print(f"    {asset} {sname:13s} {h:2d}d fwd: beta {beta:+.5f}  t={t:+.2f}  "
                      f"n={n}  → {sign if abs(t) > 2 else 'no signal'}")
    return {"buy_hold_btc": round(bh, 2), "buy_hold_eth": round(bh_eth, 2),
            "n_overlays_tried": len(rows),
            "overlays": {n: {"sharpe": s, "maxdd": d} for n, s, d in rows},
            "timing_placebo": plac, "exchange_flow_predictive": pred}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    xs, books = run_cross_section()
    ts = run_timeseries()
    summ = {"cross_section": xs, "time_series": ts}
    pd.DataFrame(books).to_parquet(ONCHAIN_DIR / "onchain_returns.parquet")
    (ONCHAIN_DIR / "onchain_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(xs, ts, books)

    print(f"\n{'='*80}\nVERDICT")
    h = xs["headline"]; o = xs["orthogonalisation_vs_price"][h["signal"]]
    print(f"  headline on-chain VALUE ({h['signal']} top-{h['topn']}): Sharpe {h['sharpe']:+.2f} "
          f"[MC-P5 {h['mc_p5']:+.2f}] WF-OOS {xs['walk_forward']['wf_oos']:+.2f} DSR {xs['walk_forward']['deflated_sharpe']:.2f}")
    print(f"  edge over price: alpha t={o['alpha_t']:+.2f}  corr-to-book {xs['corr_to_book'].get('book')}")
    print("RUN ONCHAIN OK")


def _figure(xs, ts, books):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("On-chain / network signals (H3) — crypto cross-section + BTC/ETH overlays (net, vol-target 15%)",
                 fontsize=12, fontweight="bold")

    a = ax[0, 0]
    for c, lab in [("onchain_headline", "on-chain value (nvm_z, top-20)"),
                   ("onchain_pmom_ctrl", "price momentum (same univ)"),
                   ("onchain_prev_ctrl", "price reversal (same univ)")]:
        if c in books:
            r = books[c].dropna(); a.plot((1 + r).cumprod().index, (1 + r).cumprod().values, label=lab, lw=1.3)
    a.axhline(1.0, color="k", lw=0.6, ls=":"); a.set_yscale("log")
    a.set_title("On-chain value vs price factors (identical universe)"); a.legend(fontsize=8)

    a = ax[0, 1]
    sw = xs["topn_sweep"]; sigs = ["nvm_z_val", "mvrv_z_val", "adr_mom30", "divergence", "blend"]
    ns = [10, 20, 30, xs["config"]["universe_names"]]
    for k in sigs:
        a.plot([str(n) for n in ns], [sw[k][n] for n in ns], marker="o", label=k, lw=1.2)
    a.axhline(0.5, color="r", ls="--", lw=1, label="bar 0.5"); a.axhline(0, color="k", lw=0.6)
    a.set_title("Net Sharpe vs top-N (free ceiling = 37)"); a.set_xlabel("top-N names"); a.legend(fontsize=7)

    a = ax[1, 0]
    o = xs["orthogonalisation_vs_price"]
    ks = list(o.keys()); ts_ = [o[k]["alpha_t"] for k in ks]
    a.bar(ks, ts_, color=["#2b6" if t > 2 else ("#a93" if t > 1 else "#a33") for t in ts_])
    a.axhline(2, color="g", ls="--", lw=1, label="t=2 (adds edge)"); a.axhline(0, color="k", lw=0.6)
    a.set_title("Alpha t-stat net of price momentum+reversal"); a.legend(fontsize=8)
    for i, t in enumerate(ts_):
        a.text(i, t, f"{t:+.1f}", ha="center", va="bottom" if t >= 0 else "top", fontsize=8)
    a.tick_params(axis="x", rotation=30)

    a = ax[1, 1]
    ov = ts["overlays"]; nm = list(ov.keys()); sh = [ov[n]["sharpe"] for n in nm]
    a.barh(nm, sh, color="#68a"); a.axvline(ts["buy_hold_btc"], color="r", ls="--", lw=1,
                                            label=f"BTC buy-hold {ts['buy_hold_btc']:+.2f}")
    a.axvline(0, color="k", lw=0.6); a.set_title("BTC/ETH TS overlays (2-3 cycles → context)"); a.legend(fontsize=7)
    a.tick_params(axis="y", labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "onchain.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
