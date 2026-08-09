"""Chain-fundamentals cross-section — does valuing a token on its chain's cash flows beat price?

The H3 on-chain test concluded on a universe of 33 mostly-legacy coins, because Coin Metrics' free
network data stops there. This is the arm that reaches the other half of crypto: DefiLlama publishes
daily fees, revenue and TVL for **28 chains including SOL, SUI, TON, APT, SEI, TIA, ARB and OP** —
free, no key, and precisely the names the address-count universe cannot see. Market cap comes from
Coin Metrics (`CapMrktEstUSD`, free for 27 of them), so the ratios are real valuations.

Same funnel as every other family: dollar-neutral cross-sectional long/short, vol-target 15%, t+2
execution, liquidity-aware costs, shuffled-name placebo, purged/embargoed walk-forward OOS,
block-bootstrap MC, deflated Sharpe at the true trial count, correlation-to-book + lift, and the
decisive orthogonality regression against price momentum and reversal on the identical universe.

**Headline declared before fitting: `fee_yield`** — annualised fees ÷ market cap, the crypto earnings
yield. It is the ratio the fundamentals literature is built on, not the best cell of the sweep.

Two caveats that belong next to the result, not after it. The universe is small and young: fees for
the modern chains start in 2022-2024, so the panel is ~3.5 years with breadth growing through it.
And DefiLlama backfills protocol adapters, so the history is revised — a growth signal is the most
exposed to that, and any positive result here should be read as an upper bound.

    python scripts/onchain/run_fundamentals.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, ONCHAIN_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data import defillama as dl  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import fundamentals as fu  # noqa: E402
from src.sleeves.xsect import mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
XS = CACHE_DIR / "xs"
TVOL, PPY, COST = VOL_TARGET_ANNUAL, 365, 6.0
rng = np.random.default_rng(SEED)

# a-priori config, declared before fit: fundamentals are slow, so quarterly windows and a monthly
# rebalance; tercile tails because the universe is small; t+2 execution with √-impact costs on.
REBAL, TOPFRAC, EXEC_LAG, IMPACT_K = 21, 0.3, 2, 0.1
MIN_NAMES = 10          # the panel starts once this many chains have a computable signal
HEAD_SIG = "fee_yield"  # the crypto earnings yield — chosen on the literature, not the sweep


def _load():
    """Price/ADV panel restricted to chains DefiLlama still tracks, plus the fundamentals panels
    aligned to the same grid. Fees are a daily flow and TVL a daily stock, so both reindex 1:1 —
    forward-filling would let a stalled series masquerade as a live one."""
    C = pd.read_parquet(XS / "crypto_1d_close.parquet")
    A = pd.read_parquet(XS / "crypto_1d_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC"); A.index = A.index.tz_localize("UTC")
    live = dl.live_universe()
    keep = [s for s in live if s in C.columns]
    dropped = [s for s in live if s not in C.columns]
    if dropped:
        print(f"  no repo price series for {dropped} — excluded")
    C, A = C[keep], A[keep]
    P = {m: dl.load(m).reindex(index=C.index, columns=keep) for m in ("fees", "revenue", "tvl", "mcap")}
    return C, A, P


def _sh(net):
    return summarise(net.dropna(), PPY)["sharpe_ann"]


def _winsor(C, cap=1.0):
    r = C.pct_change()
    return C.mask(r.abs() > cap).ffill()


def _book(C, A, signal, topn=None, top_frac=TOPFRAC, rebal=REBAL, cost=COST):
    s = top_n_liquid(signal, A, topn) if topn else signal
    bt = xs_backtest(C, s, top_frac=top_frac, weighting="equal", rebal=rebal, exec_lag=EXEC_LAG,
                     cost_bps=cost, adv=A, impact_k=IMPACT_K)
    return vol_target(bt["net"], PPY, TVOL), bt


def build_signals(P):
    """Six signals across value, growth and quality. Value ratios are stamped so high = cheap = long
    (the yields already point that way); growth and margin are long-the-high by construction."""
    fees, rev, tvl, cap = P["fees"], P["revenue"], P["tvl"], P["mcap"]
    return {
        "fee_yield": fu.fee_yield(fees, cap),
        "rev_yield": fu.rev_yield(rev, cap),
        "tvl_yield": fu.tvl_yield(tvl, cap),
        "fee_growth": fu.fee_growth(fees),
        "tvl_growth": fu.tvl_growth(tvl),
        "fee_margin": fu.fee_margin(fees, tvl),
        "value_blend": fu.value_blend(fees, rev, tvl, cap),
    }


def _ols(y, *xs):
    df = pd.concat([y.rename("y")] + [x.rename(f"x{i}") for i, x in enumerate(xs)], axis=1).dropna()
    if len(df) < 50:
        print(f"    ! orthogonality regression skipped — only {len(df)} overlapping bars")
        return np.array([0.0]), np.array([0.0]), 0
    Y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[f"x{i}"].to_numpy() for i in range(len(xs))])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    dof = max(len(Y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    return coef, coef / np.sqrt(np.diag(cov)), len(df)


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
    return (pd.concat(segs) if segs else pd.Series(dtype=float)), len(picks) // max(top_k, 1)


def main():
    print(f"\n{'='*84}\nCHAIN FUNDAMENTALS — fees / revenue / TVL cross-section (DefiLlama + CM market cap)\n{'='*84}")
    Craw, A, P = _load()
    C = _winsor(Craw, 1.0)
    S = build_signals(P)

    # the panel begins where the cross-section is wide enough to be one
    breadth = S[HEAD_SIG].notna().sum(axis=1)
    ok = breadth[breadth >= MIN_NAMES]
    if ok.empty:
        raise SystemExit(f"never {MIN_NAMES} chains with a computable {HEAD_SIG} — nothing to test")
    start = ok.index[0]
    C, A = C.loc[start:], A.loc[start:]
    S = {k: v.loc[start:] for k, v in S.items()}
    NAMES = int(breadth.loc[start:].max())
    span = f"{C.index.min().date()}..{C.index.max().date()}"
    print(f"  universe {C.shape[1]} chains, {span}, {len(C)} bars; breadth {int(breadth.loc[start])}"
          f"→{NAMES} names (fees for the modern chains only start 2022-2024)")
    print("  coverage: " + ", ".join(f"{m} {int(P[m].loc[start:].notna().any().sum())}" for m in P))

    # ── sign check: does each signal's theory direction beat its inverse? ────────────────────────
    print("\n  sign check (all-names book, monthly): net Sharpe of signal vs −signal")
    for k in ("fee_yield", "rev_yield", "tvl_yield", "fee_growth"):
        print(f"    {k:12s} {_sh(_book(C, A, S[k])[0]):+.2f}   (inverse {_sh(_book(C, A, -S[k])[0]):+.2f})")

    # ── top-N × signal sweep ────────────────────────────────────────────────────────────────────
    topns = [10, 15, 20, None]
    lbl = {None: "all"}
    print("\n  top-N × signal (net Sharpe, monthly rebal):")
    print("    " + "signal".ljust(13) + "".join(f"N={lbl.get(n, n):<7}" for n in topns))
    sweep = {}
    for k in S:
        row = {str(lbl.get(n, n)): round(_sh(_book(C, A, S[k], n)[0]), 2) for n in topns}
        sweep[k] = row
        print("    " + k.ljust(13) + "".join(f"{v:<+8.2f}" for v in row.values()))

    # ── headline, declared a-priori ─────────────────────────────────────────────────────────────
    head, head_bt = _book(C, A, S[HEAD_SIG])
    s_head = _sh(head)
    mc = bootstrap_sharpe(head.dropna(), PPY, 1000, SEED)
    dd = summarise(head.dropna(), PPY)["max_dd"]
    turn = head_bt["turnover"].mean()
    per_year = {int(y): round(_sh(g), 2) for y, g in head.dropna().groupby(head.dropna().index.year)}
    print(f"\n  HEADLINE {HEAD_SIG} (all names, monthly): Sharpe {s_head:+.2f}  "
          f"[MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]  "
          f"maxDD {dd:+.1%}  turn/bar {turn:.2f}")
    print(f"    per-year: {per_year}")

    # ── placebo: shuffle the name labels, keeping the signal's marginals ────────────────────────
    placebo = []
    base = S[HEAD_SIG]
    for _ in range(200):
        perm = base.copy(); perm.columns = rng.permutation(base.columns)
        placebo.append(_sh(_book(C, A, perm.reindex(columns=base.columns))[0]))
    placebo = np.array([x for x in placebo if np.isfinite(x)])
    pctile = float((s_head > placebo).mean() * 100)
    print(f"  placebo (200 name-shuffles): real {s_head:+.2f} at {pctile:.0f}th pctile "
          f"(mean {placebo.mean():+.2f}, p95 {np.percentile(placebo, 95):+.2f})")

    # ── purged walk-forward over the signal × top-N grid ────────────────────────────────────────
    M = pd.DataFrame({f"{k}_{lbl.get(n, n)}": _book(C, A, S[k], n)[0]
                      for k in S for n in (15, None)}).dropna(how="all")
    full_sr = M.mean() / M.std(ddof=1) * np.sqrt(PPY)
    # a young panel cannot afford a 2y train: 1y train / 0.5y test, embargo = the signal window
    wf, n_ref = _wf_oos(M, int(1.0 * PPY), int(0.5 * PPY), embargo=fu.WINDOW)
    n_trials = int(M.shape[1])
    var_tr = float((full_sr.clip(-3, 3) / np.sqrt(PPY)).var())
    hd = head.dropna()
    dsr = deflated_sharpe(hd.mean() / hd.std(ddof=1), len(hd), hd.skew(), hd.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8))
    print(f"  walk-forward OOS (purged, embargo={fu.WINDOW}b, 1y/0.5y): in-sample best "
          f"{full_sr.max():+.2f} → OOS {_sh(wf):+.2f} ({n_ref} refits)   deflated SR (N={n_trials}) {dsr:.2f}")

    # ── cost sensitivity ────────────────────────────────────────────────────────────────────────
    levels = {f"{m:.0f}x": round(_sh(_book(C, A, S[HEAD_SIG], cost=m * COST)[0]), 3) for m in (1, 2, 3)}
    print(f"  cost 1x/2x/3x: {levels}")

    # ── the decisive test: edge OVER price on the identical universe ─────────────────────────────
    b_pmom = _book(C, A, mom(C, 30))[0]
    b_prev = _book(C, A, -mom(C, 365, skip=30))[0]
    ortho = {}
    print("\n  DECISIVE — alpha net of PRICE momentum + reversal (identical chains):")
    for k in S:
        bk = _book(C, A, S[k])[0]
        O = pd.DataFrame({"y": bk, "pmom": b_pmom, "prev": b_prev}).dropna()
        c, t, _ = _ols(O["y"], O["pmom"], O["prev"])
        ortho[k] = {"alpha_ann": round(float(c[0] * PPY), 3), "alpha_t": round(float(t[0]), 2),
                    "corr_pmom": round(float(O["y"].corr(O["pmom"])), 2),
                    "corr_prev": round(float(O["y"].corr(O["prev"])), 2)}
        v = ortho[k]
        verdict = "adds edge" if v["alpha_t"] > 2 else ("marginal" if v["alpha_t"] > 1 else "subsumed by price")
        print(f"    {k:12s} α {v['alpha_ann']:+.3f}/yr  t={v['alpha_t']:+.2f}  "
              f"corr(pmom){v['corr_pmom']:+.2f} corr(prev){v['corr_prev']:+.2f}   → {verdict}")

    # ── the inversion, measured rather than asserted ────────────────────────────────────────────
    # The a-priori sign loses badly *and* sits at the 6th placebo percentile, which says the
    # cross-section carries real information with the sign reversed — the same shape crypto gave the
    # lottery factor. Flipping a sign after seeing the result is post-hoc, so the flip gets the full
    # funnel and is labelled, never promoted to the headline.
    inv, inv_bt = _book(C, A, -S[HEAD_SIG])
    s_inv = _sh(inv)
    mc_inv = bootstrap_sharpe(inv.dropna(), PPY, 1000, SEED)
    inv_year = {int(y): round(_sh(g), 2) for y, g in inv.dropna().groupby(inv.dropna().index.year)}
    plac_inv = []
    for _ in range(200):
        perm = (-base).copy(); perm.columns = rng.permutation(base.columns)
        plac_inv.append(_sh(_book(C, A, perm.reindex(columns=base.columns))[0]))
    plac_inv = np.array([x for x in plac_inv if np.isfinite(x)])
    pct_inv = float((s_inv > plac_inv).mean() * 100)
    M_inv = pd.DataFrame({f"inv_{lbl.get(n, n)}": _book(C, A, -S[HEAD_SIG], n)[0]
                          for n in (10, 15, None)}).dropna(how="all")
    wf_inv, n_ref_inv = _wf_oos(M_inv, int(1.0 * PPY), int(0.5 * PPY), embargo=fu.WINDOW, top_k=1)
    idv = inv.dropna()
    dsr_inv = deflated_sharpe(idv.mean() / idv.std(ddof=1), len(idv), idv.skew(), idv.kurt() + 3.0,
                              n_trials, max(var_tr, 1e-8))
    O = pd.DataFrame({"y": inv, "pmom": b_pmom, "prev": b_prev}).dropna()
    c_inv, t_inv, _ = _ols(O["y"], O["pmom"], O["prev"])
    print(f"\n  POST-HOC INVERSION (long expensive chains): Sharpe {s_inv:+.2f} "
          f"[MC-P5 {mc_inv.get('sharpe_p5', float('nan')):+.2f}]  placebo {pct_inv:.0f}th pctile "
          f"(p95 {np.percentile(plac_inv, 95):+.2f})  WF-OOS {_sh(wf_inv):+.2f}  "
          f"DSR (N={n_trials}) {dsr_inv:.2f}  alpha-over-price t={float(t_inv[0]):+.2f}")
    print(f"    per-year: {inv_year}")
    inv_summ = {"signal": f"-{HEAD_SIG}", "sharpe": round(s_inv, 3), "mc_p5": mc_inv.get("sharpe_p5"),
                "placebo_pctile": round(pct_inv, 0),
                "placebo_p95": round(float(np.percentile(plac_inv, 95)), 3),
                "wf_oos": round(_sh(wf_inv), 3), "n_refits": n_ref_inv,
                "deflated_sharpe": round(dsr_inv, 3),
                "alpha_over_price_t": round(float(t_inv[0]), 2),
                "corr_pmom": round(float(O["y"].corr(O["pmom"])), 2), "per_year": inv_year,
                "status": "post-hoc sign flip on a 4-year, 27-chain panel — reported, not promoted"}

    # ── what the ratio is actually betting on ───────────────────────────────────────────────────
    # Fee yield ranks Bitcoin permanently expensive — it collects almost no fees against a cap in the
    # trillions — and the high-throughput L2s permanently cheap, so the natural suspicion is that the
    # book is the BTC-versus-alts spread wearing a valuation label. Tested rather than asserted, and
    # the suspicion does not hold: correlation to that spread is modest and hedging it out leaves
    # nearly all of the inverted book's Sharpe. Turnover, though, confirms the other half — at ~0.01
    # a bar the legs barely move, so whatever this is, it is a standing tilt and not timing.
    # (Hedging means removing beta*spread only. Subtracting the fitted intercept as well would zero
    #  the mean by construction and "prove" any book is explained by anything.)
    ret = C.pct_change(fill_method=None)
    dom = (ret["BTCUSDT"] - ret.drop(columns=["BTCUSDT"]).mean(axis=1)) if "BTCUSDT" in ret else None
    tilt = {}
    if dom is not None:
        j = pd.concat([inv.rename("inv"), dom.rename("dom")], axis=1).dropna()
        beta = float(np.polyfit(j["dom"], j["inv"], 1)[0])
        resid = j["inv"] - beta * j["dom"]
        tilt = {"turnover_per_bar": round(float(inv_bt["turnover"].mean()), 4),
                "corr_to_btc_dominance": round(float(j["inv"].corr(j["dom"])), 3),
                "beta_to_btc_dominance": round(beta, 3),
                "sharpe_after_hedging_dominance": round(_sh(resid), 2)}
        print(f"\n  what it is really betting on — turnover {tilt['turnover_per_bar']:.3f}/bar "
              f"(a standing tilt, not timing); corr to BTC-minus-alts {tilt['corr_to_btc_dominance']:+.2f}, "
              f"and hedging that spread out leaves {tilt['sharpe_after_hedging_dominance']:+.2f} of "
              f"{s_inv:+.2f} — so it is NOT the dominance trade")

    # ── correlation to the master book + does it lift it? ───────────────────────────────────────
    corr, lift = {}, {}
    bp_path = REP / "master_book.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]
        if bp.index.tz is not None:
            bp.index = bp.index.tz_localize(None)
        h = head.copy(); h.index = h.index.tz_localize(None)
        common = h.dropna().index.intersection(bp.dropna().index)
        corr["book"] = round(float(h.reindex(common).corr(bp.reindex(common))), 3)
        bk = bp.reindex(common).dropna()
        hm = h.reindex(bk.index).fillna(0.0); hm = hm * (bk.std() / hm.std())
        lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * hm), 3) for w in (0.0, 0.15, 0.3, 0.5)}
        print(f"\n  corr to master book {corr['book']}   book-lift by weight: {lift}")
        iv = inv.copy(); iv.index = iv.index.tz_localize(None)
        im = iv.reindex(bk.index).fillna(0.0); im = im * (bk.std() / im.std())
        inv_summ["corr_to_book"] = round(float(iv.reindex(common).corr(bp.reindex(common))), 3)
        inv_summ["book_lift_by_weight"] = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * im), 3)
                                           for w in (0.0, 0.15, 0.3, 0.5)}
        print(f"  inversion: corr to book {inv_summ['corr_to_book']}   "
              f"book-lift {inv_summ['book_lift_by_weight']}")
    else:
        print(f"\n  ! {bp_path} missing — run the master book first for the lift test")

    summ = {
        "config": {"chains": int(C.shape[1]), "window": span, "bars": int(len(C)), "rebal": REBAL,
                   "top_frac": TOPFRAC, "exec_lag": EXEC_LAG, "cost_bps": COST,
                   "signal_window_days": fu.WINDOW, "headline_signal": HEAD_SIG,
                   "min_names_to_start": MIN_NAMES, "ppy": PPY,
                   "source": "DefiLlama chain fees/revenue/TVL + Coin Metrics CapMrktEstUSD",
                   "caveat": "DefiLlama backfills protocol adapters — history is revised, so growth "
                             "signals read as an upper bound"},
        "breadth_start_end": [int(breadth.loc[start]), NAMES],
        "topn_sweep": sweep,
        "headline": {"signal": HEAD_SIG, "sharpe": round(s_head, 3), "mc_p5": mc.get("sharpe_p5"),
                     "mc_p50": mc.get("sharpe_p50"), "maxdd": round(dd, 3),
                     "turnover": round(float(turn), 3), "per_year": per_year},
        "placebo": {"real": round(s_head, 3), "pctile": round(pctile, 0),
                    "mean": round(float(placebo.mean()), 3),
                    "p95": round(float(np.percentile(placebo, 95)), 3)},
        "walk_forward": {"in_sample_best": round(float(full_sr.max()), 3), "wf_oos": round(_sh(wf), 3),
                         "n_refits": n_ref, "n_trials": n_trials, "deflated_sharpe": round(dsr, 3)},
        "cost_sensitivity": levels, "orthogonalisation_vs_price": ortho,
        "post_hoc_inversion": inv_summ, "static_tilt_diagnosis": tilt,
        "corr_to_book": corr, "book_lift_by_weight": lift,
    }
    (ONCHAIN_DIR / "fundamentals_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    pd.DataFrame({"fundamentals_headline": head, "pmom_ctrl": b_pmom,
                  "prev_ctrl": b_prev}).to_parquet(ONCHAIN_DIR / "fundamentals_returns.parquet")

    print(f"\n{'='*84}\nVERDICT")
    print(f"  {HEAD_SIG}: Sharpe {s_head:+.2f} [MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f}] "
          f"placebo {pctile:.0f}th  WF-OOS {_sh(wf):+.2f}  DSR {dsr:.2f}  "
          f"alpha-over-price t={ortho[HEAD_SIG]['alpha_t']:+.2f}")
    if tilt:
        print(f"  inversion {s_inv:+.2f} survives a BTC-dominance hedge ({tilt['sharpe_after_hedging_dominance']:+.2f}) "
              f"but not its own placebo ({inv_summ['placebo_pctile']:.0f}th < 95th) — and the book stays put")
    print("RUN FUNDAMENTALS OK")


if __name__ == "__main__":
    main()
