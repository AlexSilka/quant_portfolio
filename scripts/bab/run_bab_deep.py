"""BAB deep-dive — the two variants the robustness sweep flagged as worth a full look:

  1. CONCENTRATED top-25 crypto book through the FULL funnel (purged/embargoed WF-OOS, block-bootstrap
     MC, shuffled-ranking placebo, deflated Sharpe, cost) — §3b's split-half flagged it strongest
     (+1.51) but the a-priori fixes top-100; this asks whether the concentrated book actually clears
     the robust OOS > 0.5 bar standalone, or whether its split-half strength is concentration luck.
  2. BAB + CARRY overlay — the leverage premium is decorrelated; does it diversify the carry sleeve
     (and the master book) at a positive weight, and how does the concentrated top-25 book compare?

Writes reports/bab_deep_summary.json. Crypto 1d, net of costs, a-priori 90-day beta / quintile /
monthly rebalance / t+2 / liquidity-aware cost throughout. Reproduce: part of `make bab`.

    python scripts/bab/run_bab_deep.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import BAB_DIR, CACHE_DIR, CARRY_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import top_n_liquid, vol_target  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.bab.run_bab_portfolio import WINSOR  # noqa: E402  one winsor rule per family

REP, CACHE = REPORTS_DIR, CACHE_DIR / "xs"
SEED, PPY, COST, TVOL = SEED, 365, 6.0, VOL_TARGET_ANNUAL
BETA_LB, TOPFRAC, REBAL, EXEC_LAG, IMPACT_K = 90, 0.2, 21, 2, 0.1
rng = np.random.default_rng(SEED)


def _sh(net):
    return summarise(net.dropna(), PPY)["sharpe_ann"]


def _bn_net(C, A, top_n, tf=TOPFRAC, lb=BETA_LB, cost=COST):
    """Vol-targeted net series of the beta-neutral (FP) book at a given liquidity cut / config."""
    beta = top_n_liquid(bab.panel_beta(C, lb), A, top_n)
    w = bab.bab_weights(beta, top_frac=tf, neutral="beta", rebal=REBAL)
    return vol_target(bab.bab_backtest(C, w, exec_lag=EXEC_LAG, cost_bps=cost, adv=A, impact_k=IMPACT_K)["net"], PPY, TVOL)


def _wf_oos(M, train_bars, test_bars, embargo, top_k=5):
    segs = []
    start = train_bars
    while start + test_bars <= len(M):
        train = M.iloc[max(0, start - train_bars):max(0, start - embargo)]
        test = M.iloc[start:start + test_bars]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        segs.append(test[list(sr.nlargest(top_k).index)].mean(axis=1))
        start += test_bars
    return pd.concat(segs) if segs else pd.Series(dtype=float)


def _load_book():
    """Current master book + its legs (the parallel-session-renamed book_portfolio). Guarded."""
    bp, bl = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    book = pd.read_parquet(bp)["ret"] if bp.exists() else None
    legs = pd.read_parquet(bl) if bl.exists() else None
    for f in (book, legs):
        if f is not None and getattr(f.index, "tz", None) is not None:
            f.index = f.index.tz_localize(None)
    return book, legs


def _funnel(C, A, top_n, book):
    """Full robust-bar funnel on the beta-neutral book at a liquidity cut."""
    net = _bn_net(C, A, top_n)
    s = summarise(net.dropna(), PPY)
    mc = bootstrap_sharpe(net.dropna(), PPY, 1000, SEED)
    per_year = {int(y): round(_sh(g), 2) for y, g in net.dropna().groupby(net.dropna().index.year)}

    # placebo — shuffle the beta ranking, rebuild the beta-neutral book (keeps the mechanical tilt)
    beta = top_n_liquid(bab.panel_beta(C, BETA_LB), A, top_n)
    placebo = []
    for _ in range(100):
        perm = beta.copy()
        perm.columns = rng.permutation(beta.columns)
        placebo.append(_sh(_bn_from_beta(C, A, perm.reindex(columns=beta.columns))))
    placebo = np.array(placebo); placebo = placebo[np.isfinite(placebo)]
    pctile = float((s["sharpe_ann"] > placebo).mean() * 100)

    # purged/embargoed walk-forward over the (lb × top_frac) grid at this cut
    M = {}
    for lb in (60, 90, 120):
        for tf in (0.1, 0.2, 0.3):
            M[f"{lb}_{tf}"] = _bn_net(C, A, top_n, tf=tf, lb=lb)
    M = pd.DataFrame(M).dropna(how="all")
    full_sr = (M.mean() / M.std(ddof=1) * np.sqrt(PPY))
    wf = _wf_oos(M, int(2.0 * PPY), int(0.5 * PPY), embargo=BETA_LB)
    n_trials = int(M.shape[1])
    var_tr = float((full_sr.clip(-3, 3) / np.sqrt(PPY)).var())
    nn = net.dropna()
    dsr = deflated_sharpe(nn.mean() / nn.std(ddof=1), len(nn), nn.skew(), nn.kurt() + 3.0, n_trials, max(var_tr, 1e-8))

    levels = {f"{m:.0f}x": round(_sh(_bn_net(C, A, top_n, cost=m * COST)), 3) for m in (1, 2, 3)}

    corr, lift = None, {}
    if book is not None:
        h = net.copy(); h.index = h.index.tz_localize(None)
        common = h.dropna().index.intersection(book.dropna().index)
        if len(common) > 50:
            corr = round(float(h.reindex(common).corr(book.reindex(common))), 3)
            bk = book.reindex(common).dropna()
            hm = h.reindex(bk.index).fillna(0.0); hm = hm * (bk.std() / hm.std())
            lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * hm), 3) for w in (0.0, 0.15, 0.3, 0.5)}
    return {"top_n": top_n, "sharpe": round(s["sharpe_ann"], 3), "mc_p5": round(mc.get("sharpe_p5", float("nan")), 3),
            "mc_p50": round(mc.get("sharpe_p50", float("nan")), 3), "max_dd": round(s["max_dd"], 3),
            "placebo_pctile": round(pctile, 0), "wf_oos": round(_sh(wf), 3), "in_sample_best": round(float(full_sr.max()), 3),
            "deflated_sharpe": round(dsr, 3), "n_trials": n_trials, "cost": levels,
            "corr_to_book": corr, "book_lift": lift, "per_year": per_year}, net


def _bn_from_beta(C, A, beta):
    w = bab.bab_weights(beta, top_frac=TOPFRAC, neutral="beta", rebal=REBAL)
    return vol_target(bab.bab_backtest(C, w, exec_lag=EXEC_LAG, cost_bps=COST, adv=A, impact_k=IMPACT_K)["net"], PPY, TVOL)


def _eqvol(df):
    w = (1.0 / df.std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (df * w).sum(axis=1) / w.sum()


def main():
    C = bab.winsorize_panel(pd.read_parquet(CACHE / "crypto_1d_close.parquet"), WINSOR)
    A = pd.read_parquet(CACHE / "crypto_1d_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC"); A.index = A.index.tz_localize("UTC")
    book, legs = _load_book()

    # ── Variant 1: concentrated top-25 vs a-priori top-100, both through the full funnel ─────────
    print("=" * 78 + "\nVARIANT 1 — concentrated top-25 vs a-priori top-100 (full funnel, crypto 1d)\n" + "=" * 78)
    funnels, nets = {}, {}
    for tn in (25, 100):
        res, net = _funnel(C, A, tn, book)
        funnels[f"top{tn}"] = res; nets[tn] = net
        print(f"  top-{tn:<3}: Sharpe {res['sharpe']:+.2f}  MC-P5 {res['mc_p5']:+.2f}  WF-OOS {res['wf_oos']:+.2f}"
              f"  (in-sample {res['in_sample_best']:+.2f})  deflated {res['deflated_sharpe']:.2f}"
              f"  placebo {res['placebo_pctile']:.0f}th  maxDD {res['max_dd']:+.0%}")
        print(f"            robust bar: OOS>0.5 {'PASS' if res['wf_oos'] > 0.5 else 'FAIL'}  &  "
              f"MC-P5>0 {'PASS' if res['mc_p5'] > 0 else 'FAIL'}   corr-book {res['corr_to_book']}  lift {res['book_lift']}")

    # ── Variant 2: BAB + carry overlay (does the leverage premium diversify carry?) ──────────────
    print("\n" + "=" * 78 + "\nVARIANT 2 — BAB + carry overlay\n" + "=" * 78)
    ov = {}
    cp = CARRY_DIR / "carry_refined.parquet"
    if cp.exists():
        carry = pd.read_parquet(cp).iloc[:, 0]
        if carry.index.tz is not None:
            carry.index = carry.index.tz_localize(None)
        for tag, tn in [("a-priori top-100", 100), ("concentrated top-25", 25)]:
            b = nets[tn].copy(); b.index = b.index.tz_localize(None)
            df = pd.concat([carry.rename("carry"), b.rename("bab")], axis=1).dropna()
            df = df * (VOL_TARGET_ANNUAL / (df.std() * np.sqrt(PPY)))   # vol-match both legs to target before combining
            corr = float(df["carry"].corr(df["bab"]))
            combo = {f"bab_{int(w*100)}%": round(_sh((1 - w) * df["carry"] + w * df["bab"]), 3) for w in (0.0, 0.2, 0.35, 0.5)}
            rp = round(_sh(_eqvol(df)), 3)
            ov[tag] = {"corr_bab_carry": round(corr, 3), "carry_only": combo["bab_0%"],
                       "risk_parity_5050": rp, "weight_sweep": combo}
            print(f"  {tag:<20}: corr(BAB,carry) {corr:+.2f}  carry-only {combo['bab_0%']:+.2f}  "
                  f"risk-parity 50/50 {rp:+.2f}  weight-sweep {combo}")
    else:
        print("  carry_refined.parquet missing — skipped")

    (BAB_DIR / "bab_deep_summary.json").write_text(json.dumps(
        {"variant1_funnel": funnels, "variant2_carry_overlay": ov,
         "book": "master_book" if book is not None else None}, indent=2, default=float))
    print("\nRUN BAB DEEP OK  -> reports/bab_deep_summary.json")


if __name__ == "__main__":
    main()
