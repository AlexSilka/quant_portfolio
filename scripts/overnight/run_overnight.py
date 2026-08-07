"""Calendar / session sleeve deep-dive — the overnight-vs-intraday decomposition, run through the
same funnel as every other family (vol-target 15%, t+2-style delay, liquidity-aware costs, block-
bootstrap MC, shuffled-signal placebo, cost sensitivity, correlation to the deliverable book).

The question this answers: the brief requires a "Calendar and session effects" family — is there a
tradable market-neutral sleeve in the overnight/intraday split, or is the well-documented overnight
premium just beta earned at night? The honest verdict is written to reports/overnight_summary.json
and docs/overnight.md; artifacts feed reports/figures/overnight.png.

    python scripts/overnight/run_overnight.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CACHE_DIR, CAPITAL_USD, OVERNIGHT_DIR, RAW_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import overnight as ov  # noqa: E402
from src.sleeves.xsect import vol_target  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
FIG = REP / "figures"
SEED, CAP, TVOL, PPY = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL, 252
COST_BPS = 3.0                       # US equity: 1bp commission + 2bp half-spread per side (bo_common EC)
START = "2016-01-01"                 # last ten years — the reported window
LB, TOPFRAC, TOPN = 20, 0.2, 100     # a-priori config: 1-month signal, quintile tails, top-100 liquid
ETFS = ["SPY", "QQQ", "IWM", "DIA"]
rng = np.random.default_rng(SEED)


def _nyse_sessions():
    """True NYSE trading calendar, taken from a reference ETF's raw index (SPY spans the window).

    The cached broad panel carries a *union* calendar (ffill'd rows on non-NYSE dates); left in, those
    fake rows scatter NaNs through the reindexed session panels (breaking the rolling signal) and make
    close.shift(1) cross non-sessions (a wrong overnight gap). Anchoring to real sessions fixes both.
    """
    spy = pd.read_parquet(RAW_DIR / "equity_td/SPY_1d.parquet")
    return spy.index[spy.index >= pd.Timestamp(START, tz="UTC")]


def _load_panel(tag="stocks_broad_1d"):
    C = pd.read_parquet(CACHE_DIR / f"xs/{tag}_close.parquet")
    A = pd.read_parquet(CACHE_DIR / f"xs/{tag}_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC")
        A.index = A.index.tz_localize("UTC")
    sessions = _nyse_sessions().intersection(C.index)      # real NYSE calendar only — drop union ffill rows
    C, A = C.reindex(sessions), A.reindex(sessions)
    C = ov.dense_rows(C)
    return C, A.reindex(C.index)


def _sh(net):
    return summarise(net.dropna(), PPY)["sharpe_ann"]


def _book_book(net, direction, execution, earn, C, A, cost_bps=COST_BPS, lb=LB, tf=TOPFRAC):
    """One vol-targeted net-return series for a given (execution, earn) under the reversal sign."""
    bk = ov.xs_book(C, earn, ov.trailing_session(net, lb), direction=direction, top_n=TOPN,
                    top_frac=tf, execution=execution, cost_bps=cost_bps, adv=A, impact_k=0.1)
    return vol_target(bk["net"], PPY, TVOL), bk


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    C, A = _load_panel()
    ON, ID = ov.session_returns(C)                       # clean (winsorised, ±inf dropped)
    ON_raw, _ = ov.session_returns(C, winsor=np.inf)     # unclean — only for the integrity delta
    CC = C.pct_change(fill_method=None)
    print(f"panel: {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}, {len(C)} trading days")

    # ── data integrity: the raw open/close panel manufactures a fake edge; clean it and it vanishes ──
    n_inf = int(np.isinf(ON_raw.to_numpy()).sum())
    n_big = int((ON_raw.abs() > 0.5).to_numpy().sum())
    raw_sh = _sh(_book_book(ON_raw, -1.0, "overnight_only", ON_raw, C, A)[0])
    integrity = {"artifact_name_days_over_50pct": n_big, "inf_name_days": n_inf,
                 "overnight_only_reversal_sharpe_RAW": round(raw_sh, 3)}
    print(f"\n=== data integrity ===\n  {n_big} name-days with |overnight|>50% (+{n_inf} ∞) = split-adjust artifacts")
    print(f"  overnight-only reversal Sharpe on RAW panel {raw_sh:+.2f}  ->  the 'edge' is those artifacts")

    # ── pick the honest sign on CLEAN data (fair test of the documented effect), then the surface ──
    dir_rev = _sh(_book_book(ON, -1.0, "overnight_only", ON, C, A)[0])
    dir_mom = _sh(_book_book(ON, +1.0, "overnight_only", ON, C, A)[0])
    DIR = 1.0 if dir_mom >= dir_rev else -1.0
    sign_name = "momentum (long recent overnight winners)" if DIR > 0 else "reversal (short recent overnight winners)"
    print(f"  clean-panel overnight-only: momentum {dir_mom:+.2f} vs reversal {dir_rev:+.2f}  -> funnel on {sign_name}")

    grid = []
    for lb in (10, 20, 40, 60):
        for tf in (0.1, 0.2, 0.3):
            only = _book_book(ON, DIR, "overnight_only", ON, C, A, lb=lb, tf=tf)[0]
            hold = _book_book(ON, DIR, "hold_24h", CC, C, A, lb=lb, tf=tf)[0]
            grid.append({"lookback": lb, "top_frac": tf, "sharpe_overnight_only": round(_sh(only), 3),
                         "sharpe_hold_24h": round(_sh(hold), 3)})
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(OVERNIGHT_DIR / "overnight_grid.csv", index=False)
    print(f"\n=== construction surface (net Sharpe, {sign_name}, CLEAN panel) ===")
    print(grid_df.to_string(index=False))

    # ── chosen config, both executions + the plain close-to-close reference (same sign) ────────
    only_net, only_bk = _book_book(ON, DIR, "overnight_only", ON, C, A)
    hold_net, _ = _book_book(ON, DIR, "hold_24h", CC, C, A)
    plain_net, _ = _book_book(CC, DIR, "hold_24h", CC, C, A)   # signal from close-to-close, not overnight
    rets = pd.DataFrame({"overnight_only": only_net, "hold_24h": hold_net,
                         "plain_reference": plain_net}).dropna(how="all")
    rets.to_parquet(OVERNIGHT_DIR / "overnight_returns.parquet")

    s_only, s_hold, s_plain = _sh(only_net), _sh(hold_net), _sh(plain_net)
    mc = bootstrap_sharpe(only_net.dropna(), PPY, 1000, SEED)
    per_year = {int(y): round(_sh(g), 2) for y, g in only_net.dropna().groupby(only_net.dropna().index.year)}

    # ── skip / bounce robustness (real slow signal survives a gap; bid-ask bounce dies) ────────
    skip = {}
    for sk in (0, 1, 2, 3):
        bk = ov.xs_book(C, ON, ov.trailing_session(ON, LB).shift(sk), direction=DIR, top_n=TOPN,
                        top_frac=TOPFRAC, execution="overnight_only", cost_bps=COST_BPS, adv=A, impact_k=0.1)
        skip[sk] = round(_sh(vol_target(bk["net"], PPY, TVOL)), 3)

    # ── placebo: column-shuffled signal (destroy the cross-section, keep marginals) ────────────
    sig = ov.trailing_session(ON, LB)
    placebo = []
    for _ in range(100):
        perm = sig.copy()
        perm.columns = rng.permutation(sig.columns)
        perm = perm.reindex(columns=sig.columns)
        bk = ov.xs_book(C, ON, perm, direction=DIR, top_n=TOPN, top_frac=TOPFRAC,
                        execution="overnight_only", cost_bps=COST_BPS, adv=A, impact_k=0.1)
        placebo.append(_sh(vol_target(bk["net"], PPY, TVOL)))
    placebo = np.array(placebo)
    pctile = float((s_only > placebo).mean() * 100)
    fdr = float((placebo > 0.5).mean())            # how often noise clears the robust bar

    # ── cost sensitivity (1x/2x/3x) + break-even multiple, overnight-only ──────────────────────
    def at_cost(m):
        bk = ov.xs_book(C, ON, sig, direction=DIR, top_n=TOPN, top_frac=TOPFRAC,
                        execution="overnight_only", cost_bps=COST_BPS * m, adv=A, impact_k=0.1)
        return vol_target(bk["net"], PPY, TVOL)
    levels = {f"{m:.0f}x": round(_sh(at_cost(m)), 3) for m in (1, 2, 3)}
    breakeven = next((round(float(m), 2) for m in np.linspace(0.1, 3.0, 30) if _sh(at_cost(m)) <= 0), None)

    # ── correlation to the master book + does adding it lift the book? ─────────────────────
    corr, lift = {}, {}
    bp_path, bs_path = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]
        bs = pd.read_parquet(bs_path)
        for f in (bp, bs):
            if f.index.tz is not None:
                f.index = f.index.tz_localize(None)
        o = only_net.copy()
        o.index = o.index.tz_localize(None)
        common = o.dropna().index.intersection(bp.dropna().index)
        corr["book"] = round(float(o.reindex(common).corr(bp.reindex(common))), 3)
        for c in bs.columns:
            corr[c] = round(float(o.reindex(common).corr(bs[c].reindex(common))), 3)
        bk = bp.reindex(common).dropna()
        ovm = o.reindex(bk.index).fillna(0.0)
        ovm = ovm * (bk.std() / ovm.std())          # vol-match before blending
        lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * ovm), 3) for w in (0.0, 0.15, 0.3, 0.5)}

    # ── ETF aggregate: where does the premium accrue? (overnight vs intraday vs 24h) ───────────
    etf = {}
    for t in ETFS:
        p = ov._RAW / f"{t}_1d.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[df.index >= pd.Timestamp(START, tz="UTC")]
        on = df["open"] / df["close"].shift(1) - 1.0
        idr = df["close"] / df["open"] - 1.0
        bh = df["close"].pct_change(fill_method=None)
        etf[t] = {"overnight": round(_sh(on), 2), "intraday": round(_sh(idr), 2),
                  "buy_hold": round(_sh(bh), 2), "overnight_ret_pct": round(float(on.mean() * 252 * 100), 1),
                  "intraday_ret_pct": round(float(idr.mean() * 252 * 100), 1)}

    summ = {
        "config": {"lookback": LB, "top_frac": TOPFRAC, "top_n": TOPN, "sign": sign_name,
                   "window": [str(C.index.min().date()), str(C.index.max().date())], "cost_bps_per_side": COST_BPS},
        "data_integrity": integrity,
        "sharpe": {"overnight_only": round(s_only, 3), "hold_24h": round(s_hold, 3),
                   "plain_c2c_reference": round(s_plain, 3)},
        "overnight_only_mc": {k: mc.get(k) for k in ("sharpe_p5", "sharpe_p50", "sharpe_p95")},
        "overnight_only_maxdd": round(summarise(only_net.dropna(), PPY)["max_dd"], 3),
        "per_year": per_year, "skip_robustness": skip,
        "placebo": {"real_sharpe": round(s_only, 3), "placebo_mean": round(float(placebo.mean()), 3),
                    "placebo_p95": round(float(np.percentile(placebo, 95)), 3),
                    "real_pctile_vs_placebo": round(pctile, 0), "placebo_fdr_at_0.5": round(fdr, 3)},
        "cost_sensitivity": levels, "breakeven_cost_mult": breakeven,
        "corr_to_book": corr, "book_lift_by_weight": lift, "etf_aggregate": etf,
    }
    (OVERNIGHT_DIR / "overnight_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(rets, grid_df, etf, levels, breakeven, corr)

    print(f"\n=== VERDICT (chosen config lb={LB}, tf={TOPFRAC}, {sign_name}) ===")
    print(f"  overnight-only net Sharpe {s_only:+.2f}  [MC P5 {mc.get('sharpe_p5', float('nan')):+.2f}]  "
          f"maxDD {summ['overnight_only_maxdd']:+.1%}")
    print(f"  24h-hold-tilt   net Sharpe {s_hold:+.2f}   vs plain c2c reference {s_plain:+.2f}  "
          f"(no incremental edge from the overnight framing)")
    print(f"  skip 0..3d: {skip}  (survives a gap -> not bid-ask bounce, a real slow signal)")
    print(f"  placebo: real {s_only:+.2f} at ~{pctile:.0f}th pctile of shuffles (mean {placebo.mean():+.2f}); "
          f"noise clears 0.5 in {fdr:.0%} of runs")
    print(f"  cost 1x/2x/3x: {levels}  break-even ~{breakeven}x  (daily round-trip is the killer)")
    if corr:
        print(f"  corr to book {corr.get('book')}  -> book-lift by weight {lift}")
    if etf:
        print(f"  ETF accrual (overnight/intraday/24h Sharpe): "
              f"{ {t: (v['overnight'], v['intraday'], v['buy_hold']) for t, v in etf.items()} }")
    print("RUN OVERNIGHT OK")


def _figure(rets, grid_df, etf, levels, breakeven, corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Calendar / session sleeve — overnight vs intraday (US equities, 2016→, net of costs)",
                 fontsize=13, fontweight="bold")

    # (1) equity curves
    a = ax[0, 0]
    for c, lab in [("overnight_only", "overnight-only (flat intraday)"),
                   ("hold_24h", "24h-hold, overnight-signal"), ("plain_reference", "plain c2c reference")]:
        r = rets[c].dropna()
        a.plot((1 + r).cumprod().index, (1 + r).cumprod().values, label=lab, lw=1.4)
    a.axhline(1.0, color="k", lw=0.6, ls=":")
    a.set_title("Equity curves (vol-targeted 15%)"); a.legend(fontsize=8); a.set_yscale("log")

    # (2) cost sensitivity
    a = ax[0, 1]
    xs = list(levels.keys()); ys = list(levels.values())
    a.bar(xs, ys, color=["#3b7", "#7a3", "#a73"])
    a.axhline(0.5, color="r", ls="--", lw=1, label="robust bar 0.5")
    a.axhline(0.0, color="k", lw=0.6)
    for i, v in enumerate(ys):
        a.text(i, v + 0.01, f"{v:+.2f}", ha="center", fontsize=9)
    a.set_title(f"Overnight-only Sharpe vs cost  (break-even ≈ {breakeven}× base)")
    a.legend(fontsize=8)

    # (3) ETF accrual
    a = ax[1, 0]
    ts = list(etf.keys()); x = np.arange(len(ts)); wd = 0.27
    a.bar(x - wd, [etf[t]["overnight"] for t in ts], wd, label="overnight", color="#2b6")
    a.bar(x, [etf[t]["intraday"] for t in ts], wd, label="intraday", color="#c53")
    a.bar(x + wd, [etf[t]["buy_hold"] for t in ts], wd, label="buy&hold 24h", color="#68a")
    a.axhline(0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(ts)
    a.set_title("Where the premium accrues (Sharpe): overnight is beta, not timing"); a.legend(fontsize=8)

    # (4) construction surface (overnight-only)
    a = ax[1, 1]
    piv = grid_df.pivot(index="lookback", columns="top_frac", values="sharpe_overnight_only")
    im = a.imshow(piv.values, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    a.set_xticks(range(len(piv.columns))); a.set_xticklabels(piv.columns)
    a.set_yticks(range(len(piv.index))); a.set_yticklabels(piv.index)
    a.set_xlabel("top_frac"); a.set_ylabel("lookback")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            a.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    a.set_title("Overnight-only net Sharpe surface (never clears 0.5)")
    fig.colorbar(im, ax=a, fraction=0.046)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "overnight.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
