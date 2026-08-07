"""THE canonical portfolio assembly — the single source of truth for the final book.

Combines the surviving strategy families at risk parity from each family's one honest published
return series (below). Every series is re-scaled to a common ~15% vol on a trailing (lagged) vol
estimate — point-in-time, no look-ahead — then **equal-weighted (genuine risk parity, no
performance-based selection: every family weight is 1/N)**. Each family is developed and validated in
its own deep-dive (reports/trend, reports/xs, docs/strategies/*). This script only *reads* their
published series and assembles the master.

Two books are reported and persisted:
  • gross premium stack  — the equal-weight risk-parity mean of the eight legs (the raw edge);
  • risk-managed book     — the deliverable: the §8 portfolio risk overlay applied on top of the stack —
                            a drawdown-responsive de-risking ladder (triggers −6/−9/−12% → 0.66/0.33/
                            flat, restore −4%, hysteresis) plus a daily-loss circuit breaker. On the
                            realised (benign-tail) history the overlay COSTS a little Sharpe — it is
                            tail insurance against the short-vol leg's −78% systemic tail the sample
                            does not contain, kept because that tail is real, not to lift a metric.

Metrics are reported on BOTH the full 15-year window and the frozen out-of-sample block
(OOS_START), because the brief scores targets on the final OOS block. Emits the assembled book,
per-year/quarter, the four-scheme Monte Carlo (block bootstrap / trade-order / entry jitter / random
start), the cross-family correlation matrix, the marginal-contribution curve, book exposure/turnover
series, and each family's add/remove delta.

    python scripts/run_master_book.py
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from src.config import CAPITAL_USD, OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.overlay import drawdown_ladder  # noqa: E402
from src.validation.monte_carlo import mc_all_variants  # noqa: E402

PPY = 365
START_REPORT = "2011-01-01"        # 15-year reporting window — shows the strategy holds over a long span, not
                                   # just the last decade. Pre-2016 leans on reconstructed crisis/gmacro signals
                                   # (a strategy-logic backtest for those legs; only 2020+ is fully live), flagged
                                   # in the report and dashboard. The §11 scorecard is still the frozen OOS block.
OOS = pd.Timestamp(OOS_START).tz_localize(None)   # frozen final OOS block boundary (2024-07-01)
R = bo.REPORTS

# ── §8 portfolio-level risk limits (stated triggers + step sizes), applied to the book ────────────
LADDER = ((-0.06, 0.66), (-0.09, 0.33), (-0.12, 0.0))   # drawdown → gross exposure step (flat = stop)
LADDER_RESTORE = -0.04             # re-risk only once drawdown recovers above this (hysteresis)
DAILY_LOSS_LIMIT = -0.04           # circuit breaker: flatten the day AFTER a book loss worse than this
GROSS_CAP = 2.0                    # max book gross exposure (leverage limit)
PER_FAMILY_CAP = 1.0 / 8 * 1.5     # no family may exceed 1.5× equal risk weight (equal-weight => never binds)

# (label, file, column) — each family's honest published headline (avoid the capped fake-Sharpe VRP col)
FAMILIES = [
    # trend leg = the improved deep-dive block (core-10 crypto 1d+4h + 10 US equities, EMA long-biased,
    # equal-risk; standalone Sharpe ~1.3 vs the old book_portfolio's 0.84). See docs/strategies/TREND.md.
    ("trend_momentum", "trend/trend_block_returns.parquet", "ret"),
    # carry leg = the survivorship-free POINT-IN-TIME book (top-N by trailing volume incl. delisted
    # names; standalone ~1.33). Chosen over carry_refined (1.47) on principle: refined's higher Sharpe
    # sits on a curated current-listed universe, i.e. survivorship-inflated — the same bias the whole
    # project controls for, and consistent with the PIT trend/xs legs. See docs/strategies/CARRY.md §breadth.
    ("carry", "carry/carry_breadth_headline.parquet", "ret"),
    # volprem leg = the DIVERSIFIED book across 18 Cboe underlyings with clean data (equity indices,
    # single names, international, commodities incl. gold-miners VXGDX, rates; from 2005). Crypto, FX, and
    # discontinued energy VXXLE are excluded on frozen ex-ante rules (crypto's intraday path is unhedgeable
    # for short-vol; free EURUSD OHLC is corrupt; VXXLE ended 2022), not on Sharpe — and adding free vol
    # indices lifts headline Sharpe but not the systemic -78% tail. Honest series, NET of per-leg vega
    # spreads (COST_BY_CLASS index 1.0 / single 2.5 vol-pts/roll, realistic-to-conservative; the x0->x1
    # gap in reports/volprem/volprem_cost_robustness.csv IS that charged cost). "Naked" (var_cap=1e9) =
    # no bought tail hedge (full -78% tail), NOT costless. Realised leg is OHLC (path+gap),
    # so its standalone Sharpe (~3.6) sits on a real -78% systemic-vol tail / -18 skew — it earns its slot
    # by decorrelation, and its own tail argues for sitting at or below risk parity, not above. docs/strategies/VOLPREM.md.
    # We read `ret_gated` — the deployed series with the VIX-backwardation regime gate that the strategy owns
    # and publishes (raw `ret` stays available for the validation A/B). The gate is the strategy's timing
    # signal, not a book-level risk overlay, so it lives in the volprem construction, not here.
    ("volprem", "volprem/volprem_book.parquet", "ret_gated"),
    # x-sect leg = honest survivorship-free crypto+equity top-100 liquid momentum (standalone ~0.79).
    # See docs/strategies/XSECT.md. (The BAB swap was tested and reverted — it traded smoothness for an
    # unneeded Sharpe; x-sect is smoother. BAB stays a documented standalone source, docs/strategies/BAB.md.)
    ("xs_momentum", "xs/xs_book.parquet", "ret"),
    ("breakout", "breakout/bo_combined_portfolio.parquet", "ret"),
    # crisis-alpha leg = managed-futures trend + defensive rotation on liquid ETFs (2005→). The other five
    # families are short-gamma risk premia that crash TOGETHER (2018-Q4, COVID) → correlated deep months /
    # multi-month streaks with no offset. This is the missing long-gamma leg: +6.8% in 2018-Q4, +14% in
    # COVID — it hedges exactly the months the book bleeds (Hurst-Ooi-Pedersen crisis alpha). Standalone
    # Sharpe ~0.6, ~uncorrelated. Held at FULL equal weight like every other leg (no selection). See run_crisis.py.
    ("crisis", "book/crisis_sleeve.parquet", "ret"),
    # global-macro leg = trend on EM FX + commodities — asset classes no other family trades. Only the
    # OOS-validated edges are kept (per-strategy: EM-FX trend h1/h2 +0.85/+0.89, commodity trend +0.41/+0.83;
    # xsect/reversal on these, and country-equity trend, were tested and dropped for no OOS edge). ~+0.13 to
    # the book, so it diversifies genuinely — improves the worst month and Sharpe. See scripts/run_gmacro.py.
    ("gmacro", "book/gmacro_sleeve.parquet", "ret"),
    # BAB leg = betting-against-beta / low-vol, beta-neutral concentrated top-25 crypto book (the leverage-
    # constraint premium: long low-β / short high-β with Frazzini-Pedersen leg-scaling). Crypto majors, 2020+.
    # Beta-neutral WF-OOS +1.52 top-25 (MC-P5 +0.90, deflated 1.00); standalone ~1.29 rescaled, ~uncorrelated
    # to the other legs (corr ~+0.17 to the book). See docs/strategies/BAB.md.
    ("bab", "bab/bab_book_c25.parquet", "ret"),
]


def load(label, file, col):
    p = R / file
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = (df[col] if col in df.columns else df.iloc[:, 0]).dropna()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)                   # normalise so families of mixed tz align
    return s.rename(label)


def _scale(net, target=0.15):
    """Trailing (lagged) vol-target scale factor — the leg's risk-parity weight, computable-at-bar."""
    return (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)


def rescale(net, target=0.15):
    return net * _scale(net, target)


def ppy_of(s):
    """Actual observations per calendar year — the honest Sharpe annualisation for a mixed-calendar
    series. Crypto legs trade 365 d/yr, equity/Cboe legs ~252, so the blended 2011-2026 book averages
    ~339; a flat 365 would overstate the annualised Sharpe of any sub-365 series (e.g. volprem's 252-day
    Cboe calendar). The fully-live 2020+ book and the OOS block are genuinely ~365-366 obs/yr, so their
    Sharpe is unchanged. (Vol-targeting `_scale` and turnover keep the nominal 365 — a constant factor
    there does not change any Sharpe, so the book *return series* is byte-identical; only its annualised
    Sharpe is now honest.)"""
    s = s.dropna()
    yrs = (s.index.max() - s.index.min()).days / 365.25
    return len(s) / yrs if yrs > 0 else float(PPY)


def per_period(s, freq):
    out = {}
    for k, g in s.groupby(s.index.to_period(freq) if freq != "Y" else s.index.year):
        g = g.dropna()
        out[str(k)] = round(float(np.sqrt(ppy_of(g)) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    return out


def scorecard(s, ppy=None):
    """The six task targets on a return series: Sharpe, max-DD, months-in-profit, worst month,
    longest losing streak (months). Reported for both the full window and the OOS block. Sharpe is
    annualised by the series' ACTUAL obs/yr (honest for the mixed 252/365 calendar), not a flat 365."""
    s = s.dropna()
    ss = summarise(s, ppy_of(s) if ppy is None else ppy)
    mo = (1.0 + s).resample("ME").prod() - 1.0
    neg = (mo <= 0).astype(int).to_numpy()
    streak = mx = 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"sharpe": round(ss["sharpe_ann"], 2), "max_dd": round(ss["max_dd"], 4),
            "months_in_profit": round(ss["months_in_profit"], 4),
            "worst_month": round(float(mo.min()) if len(mo) else 0.0, 4),
            "longest_losing_streak_mo": int(mx), "total_return": round(ss["total_return"], 3),
            "n_obs": int(len(s))}


def risk_overlay(raw):
    """§8 book-level risk management applied to the equal-weight premium stack, all causal (t-1 info):
      1. daily-loss circuit breaker — flat the day after a book loss worse than DAILY_LOSS_LIMIT;
      2. drawdown-responsive de-risking ladder — cut gross to the stated step as drawdown deepens
         (flat = stop trading at the deepest trigger), restore only after recovery (hysteresis);
      3. gross-exposure cap.
    Returns (managed_ret, gross_exposure, n_breaker_days)."""
    breaker = (raw.shift(1).fillna(0.0) >= DAILY_LOSS_LIMIT).astype(float)   # 0 the day after a big loss
    raw_b = raw * breaker
    _, ladder_expo = drawdown_ladder(raw_b, LADDER, LADDER_RESTORE)
    gross = (breaker * ladder_expo).clip(upper=GROSS_CAP)
    managed = raw * gross                                     # apply the combined causal exposure
    return managed.rename("ret"), gross.rename("gross"), int((breaker == 0).sum())


def book_turnover(scales):
    """Book rebalancing turnover time-series (annualised), from the daily change in each leg's
    risk-parity weight w_i = scale_i / Σ scale. Captures the re-weighting turnover the assembly adds;
    intra-sleeve turnover is reported per-family in each deep-dive (§9). Proxy, labelled as such."""
    w = scales.div(scales.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    dturn = 0.5 * w.diff().abs().sum(axis=1)                  # one-way fraction traded per bar
    return (dturn.rolling(30).mean() * PPY).rename("turnover")   # annualised, 30-bar smooth


def describe(s, mc=True):
    ppy = ppy_of(s)
    ss = summarise(s, ppy)
    out = {"sharpe": ss["sharpe_ann"], "max_dd": ss["max_dd"], "months_in_profit": ss["months_in_profit"],
           "total_return": ss["total_return"]}
    if mc:
        v = mc_all_variants(s, ppy, 2000, bo.SEED)
        bb = v["block_bootstrap"]
        out.update({"mc_p5": bb.get("sharpe_p5"), "mc_p50": bb.get("sharpe_p50"), "mc_p95": bb.get("sharpe_p95"),
                    "mc_maxdd_p5": bb.get("maxdd_p5"), "mc_maxdd_p50": bb.get("maxdd_p50"),
                    "mc_maxdd_p95": bb.get("maxdd_p95"), "mc_hit_p5": bb.get("hit_p5"),
                    "mc_hit_p50": bb.get("hit_p50"), "mc_hit_p95": bb.get("hit_p95"), "mc_variants": v})
    return out


def main():
    raw = {lab: load(lab, f, c) for lab, f, c in FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    scales = pd.DataFrame({k: _scale(v) for k, v in raw.items()}).sort_index()
    df = pd.DataFrame({k: rescale(v) for k, v in raw.items()}).sort_index()
    # UNION over the reporting window, not the intersection: crypto-perp legs (carry, breakout) only
    # exist from 2020, so `.dropna()` would collapse the 15-year book to 2020+. Average over the families
    # live each day (>=2), so 2011-2019 runs on trend/volprem/x-sect (+ reconstructed crisis/gmacro) and the
    # crypto-perp legs join in 2020.
    mask = df.index >= pd.Timestamp(START_REPORT)
    df, scales = df[mask], scales[mask]
    keep = df.notna().sum(axis=1) >= 2
    df, scales = df[keep], scales[keep]
    live = df.notna().sum(axis=1)
    print(f"families: {list(df.columns)}\nreporting window: {df.index.min().date()}..{df.index.max().date()} "
          f"({len(df)} days; {int(live.min())}-{int(live.max())} live/day)\n")

    # genuine EQUAL-WEIGHT risk parity — every family at 1/N, no performance-based selection.
    raw_ew = df.mean(axis=1, skipna=True).rename("ret")
    managed, gross, n_breaker = risk_overlay(raw_ew)
    turn = book_turnover(scales)

    print("=== GROSS PREMIUM STACK (equal-weight risk parity, no overlay) ===")
    sc_raw_full, sc_raw_oos = scorecard(raw_ew), scorecard(raw_ew[raw_ew.index >= OOS])
    print(f"  FULL {df.index.min().date()}+: {sc_raw_full}")
    print(f"  OOS  {OOS.date()}+: {sc_raw_oos}")
    print(f"\n=== RISK-MANAGED BOOK (deliverable: §8 DD-ladder + daily-loss breaker; breaker fired {n_breaker}d) ===")
    sc_full, sc_oos = scorecard(managed), scorecard(managed[managed.index >= OOS])
    print(f"  FULL {df.index.min().date()}+: {sc_full}")
    print(f"  OOS  {OOS.date()}+: {sc_oos}")
    print(f"  book gross exposure: min {gross.min():.2f} mean {gross.mean():.2f} max {gross.max():.2f} (cap {GROSS_CAP})")
    print(f"  book net exposure ~0 (legs dollar-neutral); per-family weight 1/{len(df.columns)} (cap {PER_FAMILY_CAP:.2f})")

    m = describe(managed)          # MC on the deliverable (managed) book
    per_year = per_period(managed, "Y")
    print(f"  MC block-bootstrap Sharpe[P5 {m['mc_p5']:+.2f} P50 {m['mc_p50']:+.2f} P95 {m['mc_p95']:+.2f}]")
    print(f"  per-year Sharpe: {per_year}")

    # integration delta: with vs without breakout — like-for-like (both raw equal-weight, no overlay)
    wo = df[[c for c in df.columns if c != "breakout"]].mean(axis=1, skipna=True)
    mw = describe(wo, mc=True)
    print(f"\nWITHOUT breakout (raw): Sharpe {mw['sharpe']:+.2f}  MC-P5 {mw['mc_p5']:+.2f}")
    print(f"WITH breakout    (raw): Sharpe {summarise(raw_ew, ppy_of(raw_ew))['sharpe_ann']:+.2f}")

    # correlation + marginal-contribution curve + top-removed — all on the raw equal-weight legs (consistent)
    corr = df.corr()
    solo = {c: summarise(df[c], ppy_of(df[c]))["sharpe_ann"] for c in df.columns}
    order = sorted(df.columns, key=lambda c: -solo[c])
    marg = []
    for k in range(1, len(order) + 1):
        b = df[order[:k]].mean(axis=1, skipna=True)
        sc = scorecard(b)
        marg.append({"n": k, "added": order[k - 1], "sharpe": sc["sharpe"],
                     "max_dd": sc["max_dd"], "months_in_profit": sc["months_in_profit"]})
    top = order[0]
    _notop = df[[c for c in df.columns if c != top]].mean(axis=1, skipna=True)
    notop = summarise(_notop, ppy_of(_notop))["sharpe_ann"]
    mean_corr = float(corr.values[np.triu_indices_from(corr.values, 1)].mean())
    # §7.2 correlation STABILITY — the same matrix on two halves of the window. "Diversification that exists
    # only in-sample is not diversification": if the decorrelation is real it must persist out-of-sample.
    tri = np.triu_indices_from(corr.values, 1)
    mid = df.index[len(df) // 2]
    ca = df[df.index < mid].corr().values[tri]
    cb = df[df.index >= mid].corr().values[tri]
    corr_stab = {"first_half_mean": round(float(np.nanmean(ca)), 3), "second_half_mean": round(float(np.nanmean(cb)), 3),
                 "max_pairwise_shift": round(float(np.nanmax(np.abs(ca - cb))), 3),
                 "oos_mean": round(float(df[df.index >= OOS].corr().values[tri].mean()), 3)}
    print(f"\nmean cross-family correlation: {mean_corr:+.2f}  (stability — first half {corr_stab['first_half_mean']:+.2f} "
          f"/ second half {corr_stab['second_half_mean']:+.2f} / OOS {corr_stab['oos_mean']:+.2f}, max shift {corr_stab['max_pairwise_shift']:.2f})")
    print("marginal curve (Sharpe): " + " -> ".join(f"{r['added'][:4]} {r['sharpe']:+.2f}" for r in marg))
    print(f"top contributor ({top}) removed: {notop:+.2f}  (vs managed {sc_full['sharpe']:+.2f})")

    # per-family share of book P&L (contribution of each leg to the equal-weight sum)
    contrib = (df / len(df.columns)).sum()
    pnl_share = (contrib / contrib.sum()).round(4).to_dict()
    print(f"per-family P&L share: { {k: round(v,3) for k,v in pnl_share.items()} }")

    # §13 out-of-sample trade log — the book is return-composed, so its trades ARE the daily risk-parity
    # rebalances of the family sleeves; instrument-level fills live in each family's deep-dive (e.g.
    # reports/trend/trend_oos_trade_log.csv). One row per sleeve per OOS day it is re-weighted.
    w = scales.div(scales.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w_oos = w[w.index >= OOS]
    dw_oos = w_oos.diff().fillna(0.0)
    trades = []
    for dt in w_oos.index:
        for fam in w_oos.columns:
            d = float(dw_oos.at[dt, fam])
            if abs(d) <= 1e-6:
                continue
            trades.append({"date": dt.date(), "sleeve": fam, "side": "buy" if d > 0 else "sell",
                           "delta_weight": round(d, 5), "weight_after": round(float(w_oos.at[dt, fam]), 5),
                           "notional_usd": round(abs(d) * CAPITAL_USD, 2)})
    pd.DataFrame(trades).to_csv(R / "master_book_oos_trades.csv", index=False)
    print(f"OOS trade log: {len(trades):,} sleeve-rebalances -> reports/master_book_oos_trades.csv")

    # persist
    managed.to_frame().to_parquet(R / "master_book.parquet")
    raw_ew.rename("ret").to_frame().to_parquet(R / "master_book_raw.parquet")
    df.to_parquet(R / "master_book_legs.parquet")
    pd.DataFrame({"gross": gross, "turnover": turn}).to_parquet(R / "master_book_exposure.parquet")
    corr.to_csv(R / "master_book_correlation.csv")
    pd.DataFrame(marg).to_csv(R / "master_book_marginal.csv", index=False)
    (R / "master_book_summary.json").write_text(json.dumps({
        "families": list(df.columns), "window": [str(df.index.min().date()), str(df.index.max().date())],
        "oos_start": str(OOS.date()),
        "master": {**m, **{f"full_{k}": v for k, v in sc_full.items()}},
        "scorecard_full": sc_full, "scorecard_oos": sc_oos,
        "gross_premium_full": sc_raw_full, "gross_premium_oos": sc_raw_oos,
        "without_breakout": mw, "per_year": per_year, "per_quarter": per_period(managed, "Q"),
        "standalone_sharpe": solo, "mean_correlation": mean_corr, "correlation_stability": corr_stab,
        "pnl_share": pnl_share, "marginal": marg, "top_removed": {"family": top, "sharpe": notop},
        "breakout_delta_sharpe": summarise(raw_ew, ppy_of(raw_ew))["sharpe_ann"] - mw["sharpe"],
        "risk_limits": {"ladder": LADDER, "restore": LADDER_RESTORE, "daily_loss_limit": DAILY_LOSS_LIMIT,
                        "gross_cap": GROSS_CAP, "per_family_cap": round(PER_FAMILY_CAP, 3),
                        "breaker_days": n_breaker, "max_gross": round(float(gross.max()), 2)},
        "mc_variants": m["mc_variants"]}, indent=2, default=float))

    _figure(managed, df, corr, marg, per_year)
    print("\nartifacts -> reports/master_book*  |  figure -> reports/figures/master_book.png")
    print("MASTER BOOK OK")


def _figure(master, df, corr, marg, per_year):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    (1 + master).cumprod().plot(ax=a, color="#1f77b4", lw=1.6, label="risk-managed book")
    (1 + df[[c for c in df.columns if c != "breakout"]].mean(axis=1)).cumprod().plot(
        ax=a, color="#b0b0b0", lw=1.2, label="without breakout")
    a.set_yscale("log"); a.legend(); a.set_title("1) Master book equity (with vs without breakout)")
    a = ax[0, 1]
    im = a.imshow(corr.values, cmap="coolwarm", vmin=-0.5, vmax=1.0)
    a.set_xticks(range(len(corr))); a.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    a.set_yticks(range(len(corr))); a.set_yticklabels(corr.columns, fontsize=7)
    for (i, j), v in np.ndenumerate(corr.values):
        a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    a.set_title("2) Cross-family correlation"); a.grid(False); fig.colorbar(im, ax=a, fraction=0.046)
    a = ax[1, 0]
    a.plot([r["n"] for r in marg], [r["sharpe"] for r in marg], "o-", color="#2ca02c")
    for r in marg:
        a.annotate(r["added"][:4], (r["n"], r["sharpe"]), fontsize=7, xytext=(0, 6), textcoords="offset points")
    a.set_title("3) Marginal-contribution curve"); a.set_xlabel("# families"); a.set_ylabel("Sharpe")
    a = ax[1, 1]
    yrs = list(per_year)
    a.bar(range(len(yrs)), [per_year[y] for y in yrs],
          color=["#2ca02c" if per_year[y] > 0 else "#d62728" for y in yrs])
    a.set_xticks(range(len(yrs))); a.set_xticklabels([y[2:] for y in yrs]); a.axhline(0, color="k", lw=0.5)
    a.set_title("4) Master book per-year Sharpe"); a.set_ylabel("Sharpe")
    fig.tight_layout()
    (R / "figures").mkdir(exist_ok=True)
    fig.savefig(R / "figures" / "master_book.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
