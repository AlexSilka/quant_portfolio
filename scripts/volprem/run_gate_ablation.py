"""What does the SHARED VIX gate still buy, now that every sleeve gates on its own curve?

The deployed short-vol leg ANDs two regime gates: the shared VIX term structure (both curve segments in
contango) and each sleeve's own implied vol against its own 63-day mean. The second one was added because
the first speaks only for the five equity-index sleeves. That raises the obvious follow-up: once every
sleeve has a gate that reads its own volatility, is the shared one still earning its slot, or is it now
just standing thirteen non-equity sleeves down on a market they do not trade?

Four arms, each built by the SHIPPED `gated_leg`/`book_of` construction with nothing swapped but the gate,
so the numbers are comparable to the published series (the `both` arm is asserted to reproduce
`volprem_book.parquet:ret_gated` exactly):

  ungated     no gate at all — the always-short baseline
  vix         shared VIX term structure only            (what shipped before the coverage fix)
  own         per-sleeve own curve only, NO shared gate (the arm this question is about)
  both        own AND vix                               (SHIPPED)

Judged on the whole metric set, not Sharpe: Sharpe is blind to the money a gate costs, because every leg
is vol-targeted — halving time-in-market can leave the ratio flat while cutting compounded return. So CAGR,
volatility, drawdown depth AND duration, monthly distribution, tails, and the five brief targets at book
level all sit in the panel, plus the direct attribution: the days the shared gate overrides (own says live,
VIX says flat) and what the leg earned on exactly those days.

    python -m scripts.volprem.run_gate_ablation     # ~2 min; writes reports/lab/volprem_gate_ablation*.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import LAB_DIR, OOS_START, SEED, VOLPREM_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.vol_regime import own_curve_gate, short_vol_gate  # noqa: E402

from .run_gate_coverage import book_of  # noqa: E402
from .run_vol_premium_book import (PPY_BOOK, UNIVERSE, gated_leg, implied,  # noqa: E402
                                   naive_dt, underlying_bars)

WINDOW = "2011-01-01"       # the master book's reporting window; also where VIX9D starts, so the shared
                            # gate is only a live signal from here (before it abstains and every arm agrees)
OOS = OOS_START.tz_localize(None) if OOS_START.tz is not None else OOS_START
ALWAYS = pd.Series(1.0, index=pd.date_range("2004-01-01", "2027-01-01", freq="D"))   # the "no shared gate" gate
EQ, US, INTL = {"eq_index"}, {"eq_index", "single"}, {"eq_index", "single", "intl"}
ALL = {"eq_index", "single", "intl", "commodity", "rates"}

# (asset classes the SHARED VIX gate reaches, per-sleeve own-curve gate on?). The shipped leg points the
# VIX at all eighteen sleeves, so metals, oil and duration stand down on a market they do not trade —
# these arms walk the shared gate's reach back class by class to find where it stops being informative.
ARMS = {"ungated": (set(), False), "vix": (ALL, False), "own": (set(), True),
        "vix_eq": (EQ, True), "vix_us": (US, True), "vix_equity": (INTL, True), "both": (ALL, True)}


def cagr(s: pd.Series) -> float:
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((1 + s).prod() ** (1 / yrs) - 1) if yrs > 0 else 0.0


def underwater(s: pd.Series) -> tuple[float, float]:
    """(longest stretch below the prior peak in calendar days, share of days spent below it)."""
    eq = (1 + s).cumprod()
    below = (eq / eq.cummax() - 1.0) < -1e-12
    longest, start = 0.0, None
    for ts, flag in below.items():
        if not flag:
            start = None
            continue
        start = ts if start is None else start
        longest = max(longest, (ts - start).days + 1)
    return float(longest), float(below.mean())


def panel(s: pd.Series, gates: dict[str, pd.Series] | None = None) -> dict:
    """Every metric that can separate two gates, not just the ratio ones."""
    s = s.dropna()
    ss = summarise(s, PPY_BOOK)
    mo = (1 + s).resample("ME").prod() - 1.0
    yr = (1 + s).resample("YE").prod() - 1.0
    neg = (mo <= 0).astype(int).to_numpy()
    streak = mx = 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    uw_days, uw_share = underwater(s)
    vol = float(s.std(ddof=1) * np.sqrt(PPY_BOOK))
    g = round(float(np.mean([x.mean() for x in gates.values()])), 4) if gates else 1.0
    sw = (round(float(np.mean([(x.diff().abs() > 0).sum() for x in gates.values()])
                      / ((s.index[-1] - s.index[0]).days / 365.25)), 1) if gates else 0.0)
    return {"start": s.index.min().date(), "end": s.index.max().date(), "n_days": len(s),
            "cagr": cagr(s), "total_return": ss["total_return"], "vol_ann": vol,
            "sharpe": ss["sharpe_ann"], "sortino": ss["sortino_ann"],
            "calmar": cagr(s) / abs(ss["max_dd"]) if ss["max_dd"] else np.nan,
            "max_dd": ss["max_dd"], "underwater_days": uw_days, "underwater_share": uw_share,
            "worst_month": float(mo.min()), "best_month": float(mo.max()),
            "months_in_profit": ss["months_in_profit"], "losing_streak_mo": mx,
            "worst_year": float(yr.min()), "years_positive": float((yr > 0).mean()),
            "pos_days": float((s > 0).mean()), "worst_day": float(s.min()), "best_day": float(s.max()),
            "skew": float(s.skew()), "excess_kurt": float(s.kurt()),
            "var95_daily": float(s.quantile(0.05)), "cvar95_daily": float(s[s <= s.quantile(0.05)].mean()),
            "cvar99_daily": float(s[s <= s.quantile(0.01)].mean()),
            "psr_gt0": ss["psr_gt0"], "duty_cycle": g, "switches_per_yr": sw}


ROWS = [("cagr", "CAGR", "{:+7.2%}"), ("total_return", "total return", "{:+7.1%}"),
        ("vol_ann", "vol (ann)", "{:7.1%}"), ("sharpe", "Sharpe", "{:+7.2f}"),
        ("sortino", "Sortino", "{:+7.2f}"), ("calmar", "Calmar (CAGR/DD)", "{:+7.2f}"),
        ("max_dd", "max drawdown", "{:+7.1%}"), ("underwater_days", "longest underwater (d)", "{:7.0f}"),
        ("underwater_share", "time underwater", "{:7.0%}"), ("worst_month", "worst month", "{:+7.1%}"),
        ("best_month", "best month", "{:+7.1%}"), ("months_in_profit", "months in profit", "{:7.0%}"),
        ("losing_streak_mo", "longest losing streak", "{:7.0f}"), ("worst_year", "worst year", "{:+7.1%}"),
        ("years_positive", "years positive", "{:7.0%}"), ("pos_days", "positive days", "{:7.1%}"),
        ("worst_day", "worst day", "{:+7.2%}"), ("best_day", "best day", "{:+7.2%}"),
        ("skew", "daily skew", "{:+7.2f}"), ("excess_kurt", "excess kurtosis", "{:7.1f}"),
        ("var95_daily", "VaR 95% (daily)", "{:+7.2%}"), ("cvar95_daily", "CVaR 95% (daily)", "{:+7.2%}"),
        ("cvar99_daily", "CVaR 99% (daily)", "{:+7.2%}"), ("psr_gt0", "PSR(SR>0)", "{:7.3f}"),
        ("duty_cycle", "gate live (mean sleeve)", "{:7.1%}"), ("switches_per_yr", "switches/yr/sleeve", "{:7.1f}")]


def table(title: str, cards: dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'':26s}" + "".join(f"{k:>12s}" for k in cards))
    for key, label, fmt in ROWS:
        cells = "".join(f"{fmt.format(c[key]):>12s}" if np.isfinite(c[key]) else f"{'—':>12s}" for c in cards.values())
        print(f"  {label:26s}{cells}")


# ── master-book re-assembly with one leg swapped, mirroring run_ml_book_contribution.assemble ──────
def book_card(volprem: pd.Series) -> dict:
    from src.config import BOOK_LEVERAGE  # noqa: PLC0415
    from scripts.run_master_book import (FAMILIES, OOS as MB_OOS, START_REPORT, load,  # noqa: PLC0415
                                         rescale, risk_overlay, scorecard)
    raw = {}
    for lab, f, c in FAMILIES:
        s = volprem.rename(lab) if lab == "volprem" else load(lab, f, c)
        if s is not None:
            raw[lab] = s
    df = pd.DataFrame({k: rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(START_REPORT)]
    df = df[df.notna().sum(axis=1) >= 2]
    managed = risk_overlay(df.mean(axis=1, skipna=True).rename("ret"), leverage=BOOK_LEVERAGE)[0]
    full, oos = scorecard(managed), scorecard(managed[managed.index >= MB_OOS])
    return {"full": full, "oos": oos, "cagr_full": cagr(managed),
            "cagr_oos": cagr(managed[managed.index >= MB_OOS])}


def main() -> None:
    vix = short_vol_gate(pd.DatetimeIndex(sorted(set(underlying_bars("SPY", "eq_index").index))))
    print(f"building {len(ARMS)} gate arms x {len(UNIVERSE)} sleeves (shipped construction, gate is the only swap)")

    ungated, own_g, vix_g = {}, {}, {}
    for src, sym, und, cls, ppy in UNIVERSE:
        ungated[sym] = gated_leg(src, sym, und, cls, ppy, ALWAYS, own_curve=False)
        idx = underlying_bars(und, cls).index
        own_g[sym] = own_curve_gate(naive_dt(implied(src, sym)), idx)
        vix_g[sym] = vix.reindex(idx).ffill().fillna(0.0)

    legs, books, gate_of = {}, {}, {}
    for arm, (reach, own) in ARMS.items():
        legs[arm] = {sym: (ungated[sym] if arm == "ungated" else
                           gated_leg(src, sym, und, cls, ppy, vix if cls in reach else ALWAYS, own_curve=own))
                     for src, sym, und, cls, ppy in UNIVERSE}
        books[arm] = book_of(legs[arm], ungated)
        one = {sym: pd.Series(1.0, index=g.index) for sym, g in vix_g.items()}    # neutral gate factor
        gate_of[arm] = None if arm == "ungated" else {
            sym: (own_g[sym] if own else one[sym]) * (vix_g[sym] if cls in reach else one[sym])
            for _, sym, _, cls, _ in UNIVERSE}
        print(f"  {arm:10s} built ({len(legs[arm])} sleeves, shared gate on "
              f"{sum(cls in reach for *_, cls, _ in UNIVERSE)}/{len(UNIVERSE)})")

    # The `both` arm must BE the published series, or this whole ablation is measuring something else.
    published = pd.read_parquet(VOLPREM_DIR / "volprem_book.parquet")["ret_gated"].dropna()
    pix = pd.DatetimeIndex(published.index)
    published.index = (pix.tz_convert("UTC").tz_localize(None) if pix.tz is not None else pix).normalize()
    common = books["both"].index.intersection(published.index)
    err = float((books["both"].reindex(common) - published.reindex(common)).abs().max())
    print(f"\n  reconstruction check: max|both - published ret_gated| = {err:.2e} over {len(common)} days"
          f"{'  OK' if err < 1e-12 else '  *** MISMATCH — numbers below are not the shipped leg ***'}")

    def since(gates, start):
        """Duty cycle and switch count have to be measured on the same window as the returns beside them —
        before 2011 the VIX gate has no VIX9D to read and abstains, which would read as 100% live."""
        return None if gates is None else {k: v[v.index >= start] for k, v in gates.items()}

    w = {a: b[b.index >= WINDOW] for a, b in books.items()}
    table(f"SHORT-VOL LEG, {WINDOW}+ (18 sleeves, net of vega spread, vol-targeted 15%, switching charged)",
          {a: panel(w[a], since(gate_of[a], WINDOW)) for a in ARMS})
    table(f"SHORT-VOL LEG, OOS block {OOS.date()}+",
          {a: panel(w[a][w[a].index >= OOS], since(gate_of[a], OOS)) for a in ARMS})

    # --- what the shared gate actually does: the days it overrides the per-sleeve one -----------------
    print("\n=== WHAT THE SHARED GATE OVERRIDES (own says live, VIX says flat) ===")
    rows, tot_days, tot_pnl = [], [], 0.0
    for src, sym, und, cls, ppy in UNIVERSE:
        o, v = own_g[sym], vix_g[sym].reindex(own_g[sym].index).fillna(0.0)
        over = (o > 0) & (v == 0)
        r = legs["own"][sym].reindex(o.index)
        held, cut = float(r[o > 0].sum()), float(r[over].sum())
        rows.append({"sleeve": sym, "class": cls, "own_live": float(o.mean()), "vix_live": float(v.mean()),
                     "override_days": int(over.sum()), "override_share": float(over.mean()),
                     "pnl_on_override": round(cut, 4), "pnl_own_live": round(held, 4)})
        tot_days.append(int(over.sum()))
        tot_pnl += cut
    ov = pd.DataFrame(rows)
    print(f"  {'sleeve':8s}{'class':11s}{'own live':>10s}{'VIX live':>10s}{'override d':>12s}{'% of days':>11s}"
          f"{'P&L cut':>10s}")
    for _, r in ov.iterrows():
        print(f"  {r['sleeve']:8s}{r['class']:11s}{r['own_live']:>9.0%}{r['vix_live']:>10.0%}"
              f"{r['override_days']:>12d}{r['override_share']:>10.1%}{r['pnl_on_override']:>+10.1%}")
    by_cls = ov.groupby("class")[["override_days", "pnl_on_override"]].agg({"override_days": "mean",
                                                                           "pnl_on_override": "sum"})
    print("\n  by asset class (mean override days / summed P&L the shared gate removed):")
    for cls, r in by_cls.iterrows():
        print(f"    {cls:11s}{r['override_days']:>8.0f} d   {r['pnl_on_override']:>+8.1%}")
    print(f"  -> the shared gate stands sleeves down on {np.mean(tot_days):.0f} days each on average and "
          f"removes {tot_pnl:+.1%} of summed sleeve return (sum over sleeves, before book re-scaling)")

    # --- tail coverage: a gate is bought for the worst days, so score it on exactly those ---------------
    print("\n=== TAIL COVERAGE — the UNGATED leg's worst days, and what each arm earned on them ===")
    u = w["ungated"]
    for n in (10, 20, 50):
        worst = u.nsmallest(n).index
        cells = "  ".join(f"{a} {float(w[a].reindex(worst).sum()):+7.1%}" for a in ARMS)
        flat = "  ".join(f"{a} {float((w[a].reindex(worst).abs() < 1e-9).mean()):4.0%}" for a in ARMS)
        print(f"  worst {n:2d} days   summed return: {cells}")
        print(f"  {'':15s}stepped out: {flat}")

    # --- decorrelation is why the leg is in the book at all, so a gate must not spend it ----------------
    print("\n=== CORRELATION ===")
    from scripts.run_master_book import FAMILIES, load, rescale  # noqa: PLC0415
    legs_rest = {lab: load(lab, f, c) for lab, f, c in FAMILIES if lab != "volprem"}
    rest = pd.DataFrame({k: rescale(v) for k, v in legs_rest.items() if v is not None}).sort_index()
    rest = rest[rest.index >= WINDOW].mean(axis=1, skipna=True).dropna()
    for a in ARMS:
        j = w[a].index.intersection(rest.index)
        print(f"  {a:8s} corr to the other 7 families {float(w[a].reindex(j).corr(rest.reindex(j))):+.3f}"
              f"   corr to shipped 'both' {float(w[a].corr(w['both'])):+.3f}")

    # --- is own-vs-both a real difference or a coin flip? paired moving-block bootstrap -----------------
    print("\n=== IS own-vs-both REAL? (moving-block bootstrap, 21d blocks, 1000 draws, paired days) ===")
    rng = np.random.default_rng(SEED)
    j = w["own"].index.intersection(w["both"].index)
    A, B = w["own"].reindex(j).to_numpy(), w["both"].reindex(j).to_numpy()
    nb, blk = len(j) // 21, 21
    d_sh, d_mu = [], []
    for _ in range(1000):
        starts = rng.integers(0, len(j) - blk, nb)
        pick = np.concatenate([np.arange(s, s + blk) for s in starts])
        a, b = A[pick], B[pick]
        d_sh.append(np.sqrt(PPY_BOOK) * (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)))
        d_mu.append((a.mean() - b.mean()) * PPY_BOOK)
    for nm, arr, f in (("Sharpe(own)-Sharpe(both)", np.array(d_sh), "{:+.2f}"),
                       ("ann mean(own)-mean(both)", np.array(d_mu), "{:+.1%}")):
        print(f"  {nm:26s} p5 {f.format(np.percentile(arr, 5))}  median {f.format(np.median(arr))}  "
              f"p95 {f.format(np.percentile(arr, 95))}   P(own>both) {float((arr > 0).mean()):.0%}")

    # --- per-year, because a gate that pays in one crisis and bleeds every other year is not a gate ----
    print(f"\n=== PER-YEAR RETURN ({WINDOW}+) ===")
    yrs = pd.DataFrame({a: (1 + w[a]).resample("YE").prod() - 1 for a in ARMS})
    yrs.index = yrs.index.year
    print(f"  {'year':6s}" + "".join(f"{a:>12s}" for a in ARMS) + f"{'own - both':>14s}")
    for y, r in yrs.iterrows():
        print(f"  {y:<6d}" + "".join(f"{r[a]:>+11.1%} " for a in ARMS) + f"{r['own'] - r['both']:>+13.1%}")

    # --- book level: the five targets are what the leg is actually judged on -------------------------
    print("\n=== MASTER BOOK with each volprem arm (8 families, book risk overlay, shipped leverage) ===")
    cards = {}
    for arm in ARMS:
        c = cards[arm] = book_card(books[arm])
        for win in ("full", "oos"):
            s = c[win]
            print(f"  {arm:8s} {win.upper():4s} Sharpe {s['sharpe']:+5.2f}  maxDD {s['max_dd']:+7.2%}  "
                  f"months+ {s['months_in_profit']:.0%}  worst mo {s['worst_month']:+6.2%}  "
                  f"streak {s['longest_losing_streak_mo']}  CAGR {c[f'cagr_{win}']:+6.1%}")

    from scripts.run_master_book import n_targets  # noqa: PLC0415
    print("\n  brief targets cleared (Sharpe 2.5-4.0, DD >= -15%, months >= 80%, worst mo >= -6%, streak <= 2):")
    for arm in ARMS:
        print(f"    {arm:8s} full {n_targets(cards[arm]['full'])}/5   OOS {n_targets(cards[arm]['oos'])}/5")

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({a: panel(w[a], since(gate_of[a], WINDOW)) for a in ARMS}).to_csv(
        LAB_DIR / "volprem_gate_ablation.csv")
    ov.to_csv(LAB_DIR / "volprem_gate_ablation_override.csv", index=False)
    pd.DataFrame(books).to_parquet(LAB_DIR / "volprem_gate_ablation_series.parquet")
    pd.DataFrame({f"{a}_{k}": v for a, c in cards.items() for k, v in
                  [("full", c["full"]), ("oos", c["oos"])]}).to_csv(LAB_DIR / "volprem_gate_ablation_book.csv")
    print(f"\nwrote {LAB_DIR / 'volprem_gate_ablation.csv'} (+ override, book, daily series parquet)")


if __name__ == "__main__":
    main()
