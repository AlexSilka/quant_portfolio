"""On-chain / network-signal deep-dive (H3) — run through the same funnel as every other family
(vol-target 15%, t+2 execution, liquidity-aware costs, shuffled-signal placebo, purged/embargoed
walk-forward OOS, block-bootstrap MC, deflated Sharpe, cost sensitivity, correlation-to-book + lift).

H3 is the one hypothesis whose information is *not* derived from price. Free-data reality (probed
live, see src/data/onchain.py): exchange net-flows / adjusted-transfer-value / fees / realized-cap
are pay-walled, so this tests the **free** on-chain axis — network-activity & valuation:
  • ADOPTION MOMENTUM  — active-address / tx-count growth (the on-chain twin of price momentum)
  • ON-CHAIN VALUE     — MVRV, NVM/Metcalfe (market cap per active user); cheap-per-network = long
  • NET-vs-PRICE DIVERGENCE — activity outrunning price (orthogonal-by-construction "new info")
  • BTC/ETH TS overlays — MVRV-z, NVT, Puell, stablecoin-SSR (near-unbacktestable, 2-3 cycles)

The decisive question (Liu-Tsyvinski-Wu JF2022; Cong-Karolyi-Tang-Zhao MgmtSci2024): does on-chain add
edge *over price*, or is value≈re-labelled value and momentum≈re-labelled momentum? Answered by
orthogonalising every on-chain book against price-momentum and price-reversal books on the *identical*
37-name universe. Honest verdict (both signs kept) → reports/onchain_summary.json + docs/strategies/ONCHAIN.md.

    python scripts/onchain/run_onchain.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CACHE_DIR, CAPITAL_USD, ONCHAIN_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data import onchain as oc  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import onchain as sig  # noqa: E402
from src.sleeves.xsect import mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
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
    keep = [s for s in oc.UNIVERSE if s in C.columns]
    return C[keep], A[keep]


def _load_onchain(idx, cols):
    """Load + align every on-chain panel to the price (date × name) grid. On-chain is daily UTC, so
    this is a 1:1 reindex (crypto trades every day) — no ffill gap-filling that could stale a signal."""
    out = {}
    for m in ("AdrActCnt", "TxCnt", "CapMrktCurUSD", "CapMVRVCur", "PriceUSD"):
        p = oc.load(m)
        out[m] = p.reindex(index=idx, columns=cols)
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
    cov = {m: int(oc_p[m].notna().any().sum()) for m in ("AdrActCnt", "CapMVRVCur")}
    print(f"  coverage: AdrActCnt {cov['AdrActCnt']} names, MVRV {cov['CapMVRVCur']} names  |  "
          f"free top-N ceiling = {NAMES} (top-50/100 impossible: SOL/SUI/TON… Pro-walled)")

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
              "mvrv_z_val", "divergence", "blend"):
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
    for k in ("adr_mom30", "nvm_val", "nvm_z_val", "metcalfe_val", "mvrv_val", "mvrv_z_val", "divergence", "blend"):
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
    for k in ("nvm_val", "nvm_z_val", "mvrv_val", "mvrv_z_val", "adr_mom30", "divergence", "blend"):
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
        hm = h.reindex(bk.index).fillna(0.0); hm = hm * (bk.std() / hm.std())
        lift = {f"{int(w*100)}%": round(_sh_ann((1 - w) * bk + w * hm), 3) for w in (0.0, 0.15, 0.3, 0.5)}
        print(f"\n  corr to master book {corr.get('book')}  (legs: "
              f"{ {k: corr[k] for k in list(corr) if k != 'book'} })")
        print(f"  book-lift by on-chain weight: {lift}")

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
    }
    books = {"onchain_headline": head, "onchain_pmom_ctrl": b_pmom, "onchain_prev_ctrl": b_prev}
    return summ, books


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

    def ts_book(px, timing, long_short=False):
        """Trade one asset by an on-chain z: long/flat (z below its own median → risk-on) or
        long/short. exec_lag=2, simple bps cost on position change, vol-targeted."""
        r = px.pct_change()
        pos = (-timing).clip(-1, 1)                      # signal already oriented (high z = fade)
        if not long_short:
            pos = pos.clip(lower=0.0)
        pos = pos.shift(EXEC_LAG).fillna(0.0)
        net = pos * r - pos.diff().abs().fillna(0.0) * COST / 1e4
        return vol_target(net, PPY, TVOL)

    rows = []
    btc, eth = C["BTCUSDT"], C["ETHUSDT"]
    mvrv_btc = sig.mvrv_zscore(oc_p["CapMVRVCur"]["BTCUSDT"])
    mvrv_eth = sig.mvrv_zscore(oc_p["CapMVRVCur"]["ETHUSDT"])
    nvt_btc = per_asset_z(sig.nvt_signal(oc_p["CapMrktCurUSD"]["BTCUSDT"], bc["estimated-transaction-volume-usd"]))
    puell_btc = per_asset_z(sig.puell_multiple(bc["miners-revenue"]))
    ssr = sig.stablecoin_ssr_growth(stab_tot)             # >0 = supply expanding = risk-on
    for name, px, tim, ls in [
        ("BTC MVRV-z long/flat", btc, mvrv_btc, False),
        ("BTC MVRV-z long/short", btc, mvrv_btc, True),
        ("ETH MVRV-z long/flat", eth, mvrv_eth, False),
        ("BTC NVT long/flat", btc, nvt_btc, False),
        ("BTC Puell long/flat", btc, puell_btc, False),
        ("BTC SSR-growth long/flat", btc, -ssr, False),   # ssr already oriented (expansion=long) → pass −ssr so fade-logic long-when-expanding
    ]:
        b = ts_book(px, tim, ls)
        rows.append((name, round(_sh_ann(b), 2), round(summarise(b.dropna(), PPY)["max_dd"], 3)))
    bh = _sh_ann(btc.pct_change())
    print(f"  buy-and-hold BTC Sharpe (same window, vol-targeted): {bh:+.2f}")
    for n, s, d in rows:
        print(f"    {n:26s} Sharpe {s:+.2f}  maxDD {d:+.1%}")
    return {"buy_hold_btc": round(bh, 2),
            "overlays": {n: {"sharpe": s, "maxdd": d} for n, s, d in rows}}


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
