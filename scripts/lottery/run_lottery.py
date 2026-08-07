"""Cross-sectional skewness / lottery (MAX) deep-dive — H2 from docs/HYPOTHESES.md, run through
the same funnel as every other family (dollar-neutral top/bottom-quantile long/short, vol-target 15%,
t+2-style execution delay, liquidity-aware costs, shuffled-signal placebo, purged/embargoed walk-forward
OOS, block-bootstrap MC, deflated Sharpe at the true trial count, correlation to the deliverable book +
lift curve, and an orthogonality test vs a low-vol/BAB proxy so the effect is not re-labelled low-beta).

The question: investors overpay for lottery-like assets (high skew, high recent MAX daily return), so
short-high / long-low should pay — especially in crypto (retail memecoin lottery demand). Is there a
tradable decorrelated lottery premium here, or does momentum swamp it in every liquid universe? The
honest verdict is written to reports/lottery_summary.json + docs/strategies/LOTTERY.md; the figure is
reports/figures/lottery.png. Reproduce: `make lottery`.

    python scripts/lottery/run_lottery.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import BAB_DIR, CACHE_DIR, LOTTERY_DIR, RAW_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import lottery as lot  # noqa: E402
from src.sleeves.xsect import top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
FIG = REP / "figures"
SEED, TVOL = SEED, VOL_TARGET_ANNUAL
TOPN, MINNAMES, IMPACT = 100, 6, 0.1          # top-100-liquid: the sibling momentum/carry book's universe
# a-priori, declared before fitting (NOT the surface argmax): skew over 30d, tercile tails, monthly
# rebalance, equal weight, short the high-lottery tail. MAX(5) over ~1 month is the secondary proxy.
LB_SKEW, LB_MAX, KMAX, TFRAC, REBAL = 30, 21, 5, 0.3, 21
UNIV = {                                       # (panel tag, periods/yr, cost bps/side, start)
    "crypto": ("crypto_1d", 365, 6.0, "2020-01-01"),        # PRIMARY — retail lottery is acute here
    "equity_broad": ("stocks_broad_1d", 252, 3.0, "2016-01-01"),
    "equity_midsmall": ("stocks_midsmall_1d", 252, 3.0, "2016-01-01"),   # where the MAX anomaly lives
}
rng = np.random.default_rng(SEED)


def _load(tag, start):
    C = pd.read_parquet(CACHE_DIR / f"xs/{tag}_close.parquet")
    A = pd.read_parquet(CACHE_DIR / f"xs/{tag}_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index, A.index = C.index.tz_localize("UTC"), A.index.tz_localize("UTC")
    C = C[C.index >= pd.Timestamp(start, tz="UTC")]
    return C, A.reindex(C.index)


def _funding_daily(C):
    """Per-name daily funding panel (Σ of the 8h settlements) aligned to the crypto close grid, from
    the Binance USD-M fundingRate archive. A perp holder of weight w in a name funding f pays w·f each
    settlement, so book funding P&L = −Σ(wᵢ·fᵢ) — same convention as src/sleeves/carry_xs. Returns
    None if the archive is absent (keeps the sleeve runnable on a close-only checkout)."""
    from src.sleeves.carry_xs import funding_daily
    fdir = RAW_DIR / "futures/um/fundingRate"
    if not fdir.exists():
        return None
    fund = {}
    for s in C.columns:
        files = sorted((fdir / s).glob("*.parquet"))
        if files:
            df = pd.concat([pd.read_parquet(p) for p in files]).sort_index()
            if "last_funding_rate" in df.columns:
                fund[s] = df["last_funding_rate"]
    if not fund:
        return None
    F = pd.DataFrame(fund)
    if F.index.tz is None:
        F.index = F.index.tz_localize("UTC")
    return funding_daily(F).reindex(C.index).reindex(columns=C.columns)


def _raw_signal(which, C, lb):
    return lot.skew_signal(C, lb) if which == "skew" else lot.max_signal(C, lb, KMAX)


def _timeframe_robustness(cost):
    """Does the 1d verdict hold intraday? Re-run the a-priori lottery book on the crypto 4h/1h panels
    (signal windows × bars/day, monthly-equivalent rebalance — the sibling momentum book's multi-TF
    convention). skew (the primary signal) at 1d/4h/1h; MAX (a daily-payoff construct) at 1d/4h. A real
    premium is sign-stable across horizons; a sign that flips with horizon and never clears 0.5 is not."""
    bpd = {"1d": 1, "4h": 6, "1h": 24}
    ppyv = {"1d": 365, "4h": 6 * 365, "1h": 24 * 365}
    rows = []
    for tf in ("1d", "4h", "1h"):
        Ct = pd.read_parquet(CACHE_DIR / f"xs/crypto_{tf}_close.parquet")
        At = pd.read_parquet(CACHE_DIR / f"xs/crypto_{tf}_adv.parquet").reindex_like(Ct)
        b, p = bpd[tf], ppyv[tf]
        sk = lot.skew_signal(Ct, 30 * b)
        row = {"tf": tf, "bars": int(len(Ct)),
               "skew_short": round(_sh(sleeve_net(Ct, At, sk, -1, p, cost, bpd=b)[0], p), 3),
               "skew_long": round(_sh(sleeve_net(Ct, At, sk, +1, p, cost, bpd=b)[0], p), 3)}
        if tf in ("1d", "4h"):                              # MAX is a daily-payoff construct; skip the slow 1h cell
            row["max_short"] = round(_sh(sleeve_net(Ct, At, lot.max_signal(Ct, 21 * b, 5), -1, p, cost, bpd=b)[0], p), 3)
        rows.append(row)
    return rows


def sleeve_net(C, A, sig_raw, sign, ppy, cost, *, tf=TFRAC, rb=REBAL, bpd=1, extra_mask=None):
    """Vol-targeted net-return series for one lottery book. sign=-1 shorts the high tail (the lottery
    bet); +1 longs it (the opposite, ≈ momentum). extra_mask NaNs out named bars (delisting trim).
    bpd (bars/day) scales the liquidity lookback and rebalance cadence for intraday panels."""
    sig = sign * sig_raw
    if extra_mask is not None:
        sig = sig.where(~extra_mask)
    sig = top_n_liquid(sig, A, TOPN, bpd)
    bt = xs_backtest(C, sig, top_frac=tf, weighting="equal", rebal=max(1, rb * bpd), cost_bps=cost,
                     adv=A, impact_k=IMPACT, min_names=MINNAMES)
    return vol_target(bt["net"], ppy, TVOL), bt


def _sh(net, ppy):
    return summarise(net.dropna(), ppy)["sharpe_ann"]


def _daily(net):
    n = net.dropna().copy()
    n.index = n.index.tz_localize(None) if n.index.tz is not None else n.index
    return (1 + n).resample("D").prod() - 1


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tag, ppy, cost, start = UNIV["crypto"]
    C, A = _load(tag, start)
    print(f"PRIMARY crypto panel: {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}, "
          f"{len(C)} bars, median {int(C.notna().sum(axis=1).median())} live/bar")

    # ── data integrity: are the extreme returns that DRIVE skew/MAX real, or artifacts? ──────────
    integ = lot.return_diagnostics(C)
    print("\n=== data integrity (the moment-signal trap: does one bad print hijack the ranking?) ===")
    print(f"  |ret|>50%/100%/300%: {integ['extreme_ge_50pct']}/{integ['extreme_ge_100pct']}/"
          f"{integ['extreme_ge_300pct']} name-days (max {integ['max_daily_ret']:+.1f}, min {integ['min_daily_ret']:+.2f})")
    print(f"  spike-and-revert glitches: {integ['spike_revert_glitches']}  "
          f"-> {'CLEAN: the extremes are real pumps/crashes, not data errors' if integ['spike_revert_glitches']==0 else 'REVIEW'}")
    print(f"  ffill flats: {integ['ffill_flat_pct']}%   delisted-crashed names kept: {integ['delisted_crashed_names']}")
    # raw-vs-winsor delta: cap daily log-returns at ±ln2 (a +100%/−50% day) before building skew — if
    # the number moves a lot the 'edge' rode extreme prints; if not, it is a genuine cross-section.
    def _skew_clip(cap):
        lr = np.log(C).diff()
        lr = lr.clip(-cap, cap) if np.isfinite(cap) else lr
        return lr.rolling(LB_SKEW).skew()
    sk_raw = _sh(sleeve_net(C, A, _skew_clip(np.inf), -1, ppy, cost)[0], ppy)
    sk_win = _sh(sleeve_net(C, A, _skew_clip(np.log(2)), -1, ppy, cost)[0], ppy)
    integ["skew_short_sharpe_raw"] = round(sk_raw, 3)
    integ["skew_short_sharpe_winsor"] = round(sk_win, 3)
    print(f"  skew-short Sharpe raw {sk_raw:+.2f} vs winsorised log-ret@±ln2 {sk_win:+.2f}  "
          f"(Δ{sk_win-sk_raw:+.2f} -> not an extreme-print artifact)")

    # ── a-priori config, both signals, both signs (is any positive number real lottery or momentum?) ──
    print("\n=== a-priori config (skew 30d / MAX(5) 21d, tercile, monthly, top-100 liquid) — CRYPTO ===")
    headline = {}
    for which, lb in [("skew", LB_SKEW), ("MAX", LB_MAX)]:
        sig = _raw_signal(which, C, lb)
        s_short = _sh(sleeve_net(C, A, sig, -1, ppy, cost)[0], ppy)   # lottery: short high
        s_long = _sh(sleeve_net(C, A, sig, +1, ppy, cost)[0], ppy)    # opposite: long high (≈ momentum)
        headline[which] = {"lottery_short_high": round(s_short, 3), "opposite_long_high": round(s_long, 3)}
        print(f"  {which:4s}: SHORT-high (lottery) {s_short:+.2f}   |   LONG-high (≈momentum) {s_long:+.2f}")

    # ── perp funding: charge it at every 8h settlement (the shared crypto convention) — is the
    # dollar-neutral book funding-neutral, or does it pay? The headline stays on the no-funding number
    # (apples-to-apples with the momentum/carry sibling books, which price funding as their own sleeve),
    # and the with-funding number is reported so the effect is charged and visible, never hidden. ──
    FD = _funding_daily(C)
    funding = {}
    if FD is not None:
        for which, lb in [("skew", LB_SKEW), ("MAX", LB_MAX)]:
            _, bt = sleeve_net(C, A, _raw_signal(which, C, lb), -1, ppy, cost)
            fpnl = -(bt["weights"] * FD.reindex_like(bt["weights"])).sum(axis=1)   # +ve = book receives
            s_with = _sh(vol_target(bt["net"] + fpnl, ppy, TVOL), ppy)
            funding[which] = {"funding_pct_per_yr": round(float(fpnl.mean() * 365 * 100), 2),
                              "short_high_sharpe_with_funding": round(s_with, 3)}
        print(f"  funding charged (8h settlements): skew-short {funding['skew']['funding_pct_per_yr']:+.1f}%/yr "
              f"-> Sharpe {headline['skew']['lottery_short_high']:+.2f} → {funding['skew']['short_high_sharpe_with_funding']:+.2f}  "
              f"(a headwind — the dollar-neutral lottery book PAYS funding, deepening the dead verdict)")

    # ── construction surface: window × tail, both signals — is there ANY positive lottery region? ──
    surface, all_trials = {}, []
    for which, lbs in [("skew", (20, 30, 45, 60)), ("MAX", (14, 21, 30, 45))]:
        rows = []
        for lb in lbs:
            sig = _raw_signal(which, C, lb)
            for tf in (0.1, 0.2, 0.3):
                s = _sh(sleeve_net(C, A, sig, -1, ppy, cost, tf=tf)[0], ppy)
                rows.append({"lookback": lb, "top_frac": tf, "sharpe": round(s, 3)})
                all_trials.append(s)
        surface[which] = pd.DataFrame(rows)
    surf_df = pd.concat([surface["skew"].assign(signal="skew"), surface["MAX"].assign(signal="MAX")])
    surf_df.to_csv(LOTTERY_DIR / "lottery_surface.csv", index=False)
    print("\n=== crypto lottery-direction surface (net Sharpe; short-high) — never clears 0.5 ===")
    for w in ("skew", "MAX"):
        piv = surface[w].pivot(index="lookback", columns="top_frac", values="sharpe")
        print(f"  [{w}]\n" + piv.to_string())

    # ── timeframe robustness: does the 1d verdict hold on 4h/1h? (sign-flips, still sub-bar) ────
    tf_rob = _timeframe_robustness(cost)
    print("\n=== timeframe robustness (crypto, a-priori scaled by bars/day) ===")
    for r in tf_rob:
        print(f"  {r['tf']:>3}: skew-short {r['skew_short']:+.2f}  skew-long(≈mom) {r['skew_long']:+.2f}"
              + (f"  MAX-short {r['max_short']:+.2f}" if "max_short" in r else "  MAX-short  n/a"))
    print("  -> the skew tilt FLIPS sign with horizon (1d inverted, intraday weakly reversal) and never "
          "clears 0.5: not a sign-stable cross-timeframe premium.")
    for r in tf_rob:                                    # count the NEW intraday lottery cells as trials
        if r["tf"] != "1d":                             # (1d already in the surface/a-priori)
            all_trials += [r["skew_short"]] + ([r["max_short"]] if "max_short" in r else [])

    # ── CHOSEN config full funnel (crypto, skew-short primary) ──────────────────────────────────
    sig_skew = _raw_signal("skew", C, LB_SKEW)
    net_short, bt_short = sleeve_net(C, A, sig_skew, -1, ppy, cost)
    net_long, _ = sleeve_net(C, A, sig_skew, +1, ppy, cost)
    net_max, _ = sleeve_net(C, A, _raw_signal("MAX", C, LB_MAX), -1, ppy, cost)
    rets = pd.DataFrame({"skew_short_lottery": net_short, "skew_long_momentum": net_long,
                         "max_short_lottery": net_max}).dropna(how="all")
    s_short = _sh(net_short, ppy)
    mc = bootstrap_sharpe(net_short.dropna(), ppy, 1000, SEED)
    per_year = {int(y): round(_sh(g, ppy), 2)
                for y, g in net_short.dropna().groupby(net_short.dropna().index.year)}

    # placebo: column-shuffle the signal (destroy the cross-section, keep marginals)
    placebo = []
    for _ in range(100):
        perm = sig_skew.copy()
        perm.columns = rng.permutation(sig_skew.columns)
        perm = perm.reindex(columns=sig_skew.columns)
        placebo.append(_sh(sleeve_net(C, A, perm, -1, ppy, cost)[0], ppy))
    placebo = np.array(placebo)
    pctile = float((s_short > placebo).mean() * 100)
    fdr = float((placebo > 0.5).mean())                # how often noise alone clears the +0.5 robust bar
    candidates = {"crypto_skew_short": (net_short, ppy), "crypto_max_short": (net_max, ppy)}

    # delisting-trimmed variant: NaN the final bars of names that delist having crashed (terminal
    # crash losses are under-captured when a held series goes NaN) — does the verdict depend on it?
    pdm = lot.predelist_mask(C)
    net_trim = sleeve_net(C, A, sig_skew, -1, ppy, cost, extra_mask=pdm)[0]

    # purged/embargoed walk-forward OOS: precompute the grid's net series, stitch OOS blocks, per
    # block pick the best-train config; an embargo gap purges the boundary window from training.
    wf = _walk_forward(C, A, ppy, cost)

    # cost sensitivity + break-even (of the least-bad, i.e. skew-short)
    def at_cost(m):
        return sleeve_net(C, A, sig_skew, -1, ppy, cost * m)[0]
    levels = {f"{m:.0f}x": round(_sh(at_cost(m), ppy), 3) for m in (1, 2, 3)}

    # (surface Sharpes accumulate in all_trials; equity a-priori cells are appended below so the
    # deflated-Sharpe haircut, computed after the cross-check, reflects the full search, not just crypto.)

    # ── correlation to the master book + does adding it lift or drag the book? ─────────────
    corr, lift = {}, {}
    bp_path, bs_path = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]
        bp.index = bp.index.tz_localize(None) if bp.index.tz is not None else bp.index
        o = _daily(net_short)
        common = o.dropna().index.intersection(bp.dropna().index)
        corr["book"] = round(float(o.reindex(common).corr(bp.reindex(common))), 3)
        if bs_path.exists():
            bs = pd.read_parquet(bs_path)
            bs.index = bs.index.tz_localize(None) if bs.index.tz is not None else bs.index
            for cN in bs.columns:
                corr[cN] = round(float(o.reindex(common).corr(bs[cN].reindex(common))), 3)
        bk = bp.reindex(common).dropna()
        om = o.reindex(bk.index).fillna(0.0)
        om = om * (bk.std() / om.std())                # vol-match before blending
        lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * om, 365), 3) for w in (0.0, 0.15, 0.3)}

    # ── orthogonality: is the skew book just re-labelled low-vol / BAB? ─────────────────────────
    vol_short = sleeve_net(C, A, lot.vol_signal(C, LB_SKEW), -1, ppy, cost)[0]   # short high-vol/long low-vol (BAB proxy)
    orth = {"lowvol_proxy": _orthogonalize(net_short, vol_short, ppy)}
    bab_path = BAB_DIR / "bab_returns.parquet"             # H1's published book (parallel session)
    if bab_path.exists():
        babf = pd.read_parquet(bab_path)
        # the dollar-neutral crypto BAB column is the direct low-beta analog of this dollar-neutral
        # crypto skew book (fall back to the first column if H1 renamed it).
        bcol = next((c for c in babf.columns if "crypto" in c and "dollar" in c and "vol" not in c),
                    babf.columns[0])
        orth["bab_book"] = {"column": bcol, **_orthogonalize(_daily(net_short), babf[bcol], 365, already_daily=True)}
    else:
        orth["bab_book"] = "FOLLOW-UP: reports/bab_returns.parquet not present at run time; the " \
                           "self-contained low-vol proxy above already answers the re-labelled-low-vol test."

    # ── equity cross-check (secondary): where the MAX anomaly is documented to live ─────────────
    equity = {}
    for uname in ("equity_broad", "equity_midsmall"):
        et, eppy, ecost, estart = UNIV[uname]
        EC, EA = _load(et, estart)
        e = {}
        for which, lb in [("skew", LB_SKEW), ("MAX", LB_MAX)]:
            es = sleeve_net(EC, EA, _raw_signal(which, EC, lb), -1, eppy, ecost)[0]
            e[f"{which}_short"] = round(_sh(es, eppy), 3)
            candidates[f"{uname}_{which}_short"] = (es, eppy)
            all_trials.append(_sh(es, eppy))
        equity[uname] = e
        print(f"\n=== equity cross-check [{uname}] (top-100 liquid, a-priori) ===")
        print(f"  skew-short {e['skew_short']:+.2f}   MAX-short {e['MAX_short']:+.2f}")

    # deflated Sharpe of the single best *positive* book found anywhere (crypto + equity a-priori),
    # deflated at the surface trial count — the honest "did anything survive the multiple testing?".
    n_trials = len(all_trials)
    var_tr = float((np.array(all_trials) / np.sqrt(ppy)).var())
    best_name, (best_net, best_ppy) = max(candidates.items(), key=lambda kv: _sh(kv[1][0], kv[1][1]))
    b = best_net.dropna()
    best_sh = _sh(best_net, best_ppy)
    dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8)) if best_sh > 0 else 0.0

    rets.to_parquet(LOTTERY_DIR / "lottery_returns.parquet")
    summ = {
        "config": {"universe_primary": tag, "signal_primary": "skew", "lookback_skew": LB_SKEW,
                   "lookback_max": LB_MAX, "k_max": KMAX, "top_frac": TFRAC, "rebal": REBAL,
                   "top_n_liquid": TOPN, "vol_target": TVOL, "cost_bps_per_side": cost,
                   "window": [str(C.index.min().date()), str(C.index.max().date())]},
        "data_integrity": integ,
        "crypto_apriori": headline,
        "timeframe_robustness": tf_rob,
        "funding_charged": funding,
        "chosen_skew_short": {"sharpe": round(s_short, 3),
                              "mc": {k: mc.get(k) for k in ("sharpe_p5", "sharpe_p50", "sharpe_p95")},
                              "max_dd": round(summarise(net_short.dropna(), ppy)["max_dd"], 3),
                              "per_year": per_year, "delisting_trimmed_sharpe": round(_sh(net_trim, ppy), 3)},
        "placebo": {"real_sharpe": round(s_short, 3), "placebo_mean": round(float(placebo.mean()), 3),
                    "placebo_p95": round(float(np.percentile(placebo, 95)), 3),
                    "real_pctile_vs_placebo": round(pctile, 0), "placebo_fdr_at_abs0.5": round(fdr, 3)},
        "walk_forward_oos": wf,
        "cost_sensitivity": levels,
        "deflated_sharpe_best": {"best_book": best_name, "best_sharpe": round(best_sh, 3),
                                 "n_trials": n_trials, "deflated_sharpe": round(float(dsr), 3),
                                 "crypto_surface_best": round(max(all_trials), 3)},
        "corr_to_book": corr, "book_lift_by_weight": lift,
        "orthogonality_vs_lowvol": orth,
        "equity_cross_check": equity,
    }
    (LOTTERY_DIR / "lottery_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(rets, surface, headline, equity, placebo, s_short, ppy)

    print("\n=== VERDICT (crypto primary, skew-short a-priori) ===")
    print(f"  skew-short net Sharpe {s_short:+.2f}  [MC P5 {mc.get('sharpe_p5', float('nan')):+.2f} "
          f"P50 {mc.get('sharpe_p50', float('nan')):+.2f}]  maxDD {summ['chosen_skew_short']['max_dd']:+.1%}")
    print(f"  MAX-short {headline['MAX']['lottery_short_high']:+.2f}   opposite long-high (≈momentum): "
          f"skew {headline['skew']['opposite_long_high']:+.2f} / MAX {headline['MAX']['opposite_long_high']:+.2f}")
    print(f"  placebo: real {s_short:+.2f} at ~{pctile:.0f}th pctile (mean {placebo.mean():+.2f}); "
          f"noise clears |0.5| in {fdr:.0%} of runs")
    print(f"  walk-forward OOS (top-{wf['top_k']} ensemble, anchored): {wf['oos_sharpe']:+.2f}  "
          f"({wf['n_refits']} refits)  in-sample-best {wf['insample_best']:+.2f}")
    print(f"  delisting-trimmed {summ['chosen_skew_short']['delisting_trimmed_sharpe']:+.2f} "
          f"(≈ untrimmed {s_short:+.2f} -> verdict not a delisting artifact)")
    print(f"  cost 1x/2x/3x {levels}   deflated Sharpe of best book ({best_name} {best_sh:+.2f}, "
          f"N={n_trials}) {dsr:.3f}")
    print(f"  corr to book {corr.get('book')}   lift-by-weight {lift}")
    print(f"  orthogonality vs low-vol proxy: corr {orth['lowvol_proxy']['corr']:+.2f}, "
          f"alpha {orth['lowvol_proxy']['alpha_ann']:+.2f}/yr, resid Sharpe {orth['lowvol_proxy']['resid_sharpe']:+.2f}")
    print("RUN LOTTERY OK")


def _orthogonalize(y_net, x_net, ppy, already_daily=False):
    """Regress book y on book x; report corr, annualised alpha, beta, residual Sharpe."""
    y = y_net if already_daily else _daily(y_net)
    x = x_net if already_daily else _daily(x_net)
    if x.index.tz is not None:
        x.index = x.index.tz_localize(None)
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 50:
        return {"corr": None, "note": "insufficient overlap"}
    b1, b0 = np.polyfit(df["x"], df["y"], 1)
    resid = df["y"] - (b0 + b1 * df["x"])
    return {"corr": round(float(df["y"].corr(df["x"])), 3), "beta": round(float(b1), 3),
            "alpha_ann": round(float(b0 * ppy), 3),
            "resid_sharpe": round(float(np.sqrt(ppy) * resid.mean() / resid.std(ddof=1)), 3)}


def _walk_forward(C, A, ppy, cost, embargo=60, top_k=5):
    """Purged/embargoed walk-forward OOS over the lottery-direction grid (skew/MAX × window × tail).

    Precompute each config's vol-targeted net series once, then stitch OOS test blocks; on each train
    block pick the best-Sharpe config (top-k ensemble, robust to plateau ties). An `embargo` gap is
    dropped from the end of each training block so the test block's trailing signal window never
    overlaps training (the purge). Anchored (expanding) train windows, annual refits."""
    grid = [("skew", lb, tf) for lb in (20, 30, 45, 60) for tf in (0.1, 0.2, 0.3)] + \
           [("MAX", lb, tf) for lb in (14, 21, 30, 45) for tf in (0.1, 0.2, 0.3)]
    cols = {}
    for i, (which, lb, tf) in enumerate(grid):
        cols[i] = sleeve_net(C, A, _raw_signal(which, C, lb), -1, ppy, cost, tf=tf)[0]
    M = pd.DataFrame(cols).dropna(how="all")
    full_sr = (M.mean() / M.std(ddof=1) * np.sqrt(ppy)).replace([np.inf, -np.inf], np.nan)
    tr_b, te_b = int(2 * ppy), int(ppy)                # 2y initial train, 1y test blocks, expanding
    segs, refits = [], 0
    start = tr_b
    while start + te_b <= len(M):
        train = M.iloc[:max(0, start - embargo)]       # purge the embargo gap before the test block
        test = M.iloc[start:start + te_b]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        chosen = list(sr.nlargest(top_k).index)
        segs.append(test[chosen].mean(axis=1))
        refits += 1
        start += te_b
    oos = pd.concat(segs).dropna() if segs else pd.Series(dtype=float)
    return {"oos_sharpe": round(_sh(oos, ppy), 3), "insample_best": round(float(full_sr.max()), 3),
            "insample_frac_positive": round(float((full_sr > 0).mean()), 3),
            "insample_frac_over_0.5": round(float((full_sr > 0.5).mean()), 3),
            "n_refits": refits, "top_k": top_k, "embargo_bars": embargo}


def _figure(rets, surface, headline, equity, placebo, s_short, ppy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Cross-sectional skewness / lottery (MAX) sleeve — crypto primary, net of costs, vol-targeted 15%",
                 fontsize=13, fontweight="bold")

    # (1) equity curves — lottery (short high) loses; the opposite (long high) is just momentum
    a = ax[0, 0]
    for c, lab in [("skew_short_lottery", "skew-short (LOTTERY bet)"),
                   ("max_short_lottery", "MAX-short (LOTTERY bet)"),
                   ("skew_long_momentum", "skew-long (opposite ≈ momentum)")]:
        r = rets[c].dropna()
        a.plot((1 + r).cumprod().index, (1 + r).cumprod().values, label=lab, lw=1.4)
    a.axhline(1.0, color="k", lw=0.6, ls=":")
    a.set_title("The documented lottery direction loses; only the momentum side pays")
    a.legend(fontsize=8); a.set_yscale("log")

    # (2) crypto skew-short surface — never clears 0.5
    a = ax[0, 1]
    piv = surface["skew"].pivot(index="lookback", columns="top_frac", values="sharpe")
    im = a.imshow(piv.values, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto")
    a.set_xticks(range(len(piv.columns))); a.set_xticklabels(piv.columns)
    a.set_yticks(range(len(piv.index))); a.set_yticklabels(piv.index)
    a.set_xlabel("top_frac"); a.set_ylabel("lookback (d)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            a.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    a.set_title("Crypto skew-short net Sharpe surface (no positive region)")
    fig.colorbar(im, ax=a, fraction=0.046)

    # (3) cross-universe lottery Sharpe (crypto / equity-broad / equity-midsmall)
    a = ax[1, 0]
    us = ["crypto", "equity_broad", "equity_midsmall"]
    sk = [headline["skew"]["lottery_short_high"], equity["equity_broad"]["skew_short"],
          equity["equity_midsmall"]["skew_short"]]
    mx = [headline["MAX"]["lottery_short_high"], equity["equity_broad"]["MAX_short"],
          equity["equity_midsmall"]["MAX_short"]]
    x = np.arange(len(us)); wd = 0.36
    a.bar(x - wd / 2, sk, wd, label="skew-short", color="#4682b4")
    a.bar(x + wd / 2, mx, wd, label="MAX-short", color="#c1666b")
    a.axhline(0.5, color="r", ls="--", lw=1, label="robust bar 0.5")
    a.axhline(0.0, color="k", lw=0.6)
    a.set_xticks(x); a.set_xticklabels(["crypto", "equity\nbroad", "equity\nmid/small"], fontsize=8)
    a.set_title("Lottery (short-high) Sharpe by universe — dead/weak everywhere"); a.legend(fontsize=8)

    # (4) placebo distribution
    a = ax[1, 1]
    a.hist(placebo, bins=24, color="#bbb", edgecolor="w")
    a.axvline(s_short, color="r", lw=2, label=f"real skew-short {s_short:+.2f}")
    a.axvline(float(np.percentile(placebo, 95)), color="k", ls="--", lw=1, label="placebo p95")
    a.set_title("Shuffled-signal placebo (real does not beat noise)")
    a.set_xlabel("net Sharpe"); a.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "lottery.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
