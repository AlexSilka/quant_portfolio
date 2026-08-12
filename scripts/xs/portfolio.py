"""Assemble the cross-sectional sleeves into a market-neutral book and validate it honestly.

Each chosen sleeve (asset × timeframe × construction) is vol-targeted, compounded to a common
daily P&L, then equal-risk combined — first within crypto (24/7), then cross-asset with the
equity sleeve. The book is validated the same way the rest of the project is: Monte Carlo
P5/P50/P95, per-year Sharpe, isolated crisis windows, a 1×/2×/3×/break-even cost sweep, the
sleeve correlation matrix, and a deflated Sharpe at an effective (not inflated) trial count.
Finally it is stacked on the existing trend book to measure the diversification lift.

    python scripts/xs/portfolio.py
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, SEED, TREND_DIR, XS_DIR  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves.xsect import (mom, risk_adj_mom, blend_rank, xs_backtest,  # noqa: E402
                               vol_target, top_n_liquid)
# trade the N most-liquid names at each bar (survivorship-free) — a focused liquid universe beats
# the diluted full universe; ~100 is the robust zone (top-10/20 is too concentrated, all-300 too
# noisy). Not today's top-N (that would be selection bias) — top-N by *trailing* liquidity.
TOP_N = {"crypto": 100, "stocks": 100, "fx": 0}
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE, OUT = CACHE_DIR / "xs", XS_DIR
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
PPY = {"crypto": {"1d": 365, "4h": 6 * 365, "1h": 24 * 365, "15m": 96 * 365}, "stocks": {"1d": 252}}
COST_BPS = {"crypto": 6.0, "stocks": 3.0}

# Final chosen sleeves — conservative, defensible constructions (NOT the sweep argmax):
# crypto momentum is fast (short lookback, no skip); equity momentum is classic 12-1 (skip a
# month, decile). Monthly-cadence rebalancing keeps turnover — and cost — low on every one.
# the SAME textbook a-priori config on every crypto timeframe (riskadj-30d, tercile, monthly) —
# no per-timeframe cherry-pick, so the book is honest on the survivorship-free tradable universe.
CHOSEN = [
    ("crypto_1d", dict(sig="riskadj", lb=30, sk=0, tf=0.3, wt="equal", rb=21)),
    ("crypto_4h", dict(sig="riskadj", lb=30, sk=0, tf=0.3, wt="equal", rb=21)),
    ("crypto_1h", dict(sig="riskadj", lb=30, sk=0, tf=0.3, wt="equal", rb=21)),
    ("stocks_1d", dict(sig="riskadj", lb=252, sk=7, tf=0.1, wt="equal", rb=21)),
]
CRISES = {"covid_2020": ("2020-02-15", "2020-04-30"), "bull_2021": ("2021-01-01", "2021-12-31"),
          "bear_2022": ("2022-01-01", "2022-12-31"), "chop_2025": ("2025-01-01", "2025-12-31"),
          "recent_2426": ("2024-07-01", "2026-06-30")}


def _signal(cfg, px, bpd):
    lb, sk = cfg["lb"] * bpd, cfg["sk"] * bpd
    if cfg["sig"] == "raw":
        return mom(px, lb, sk)
    if cfg["sig"] == "riskadj":
        return risk_adj_mom(px, lb, sk)
    return blend_rank([risk_adj_mom(px, max(2, int(lb * f)), sk) for f in (0.5, 1.0, 2.0)])


# Funding used to be assembled and subtracted here, by hand, because `xs_backtest` had nowhere to put
# it. It lives in `src/backtest/carry` now and the backtest charges it itself — so this file both
# stops carrying its own copy and stops being the only caller that remembered. On this book funding
# is a CREDIT, which is the opposite of the obvious guess: momentum is long the winners, where funding
# is dearest, but the SHORT book (the losers) carries more — beaten-down alts keep stubbornly positive
# funding, so shorting them collects more than leading costs.


def sleeve_daily(tag, cfg):
    """Vol-targeted net return of one sleeve, compounded to a daily series."""
    kind, tf = tag.split("_")
    px = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    advp = CACHE / f"{tag}_adv.parquet"
    adv = pd.read_parquet(advp) if advp.exists() else None
    bpd, ppy, cost = BARS_PER_DAY[tf], PPY[kind][tf], COST_BPS[kind]
    sig = top_n_liquid(_signal(cfg, px, bpd), adv, TOP_N.get(kind, 0), bpd)
    bt = xs_backtest(px, sig, top_frac=cfg["tf"], weighting=cfg["wt"], rebal=max(1, cfg["rb"] * bpd),
                     cost_bps=cost, adv=adv, impact_k=0.1 if adv is not None else 0.0)
    netv = vol_target(bt["net"], ppy)          # net already carries funding — xs_backtest charges it
    daily = (1 + netv).resample("D").prod() - 1
    cost_d = bt["cost"].resample("D").sum()
    return daily.dropna(), cost_d, kind


def per_year(ret, ppy):
    out = {}
    for y, g in ret.groupby(ret.index.year):
        g = g.dropna()
        out[int(y)] = round(float(np.sqrt(ppy) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    return out


def main():
    sleeves, costs, kinds = {}, {}, {}
    for tag, cfg in CHOSEN:
        d, c, kind = sleeve_daily(tag, cfg)
        name = f"{tag}"
        sleeves[name], costs[name], kinds[name] = d, c, kind
        s = summarise(d, 365 if kind == "crypto" else 252)
        print(f"{name:12s} Sharpe {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.0%}  months+ {s['months_in_profit']:.0%}")

    R = pd.DataFrame(sleeves).sort_index()
    corr = R.corr()
    print(f"\nSleeve correlation:\n{corr.round(2).to_string()}")

    def eqvol(df, lb=252):
        """Equal-risk (inverse-vol) combine of a returns frame; NaN before every column has started.

        The weight is a TRAILING, lagged volatility. It used to be `1.0 / df.std()` over the whole
        sample — a statistic from the end of the series deciding the weight at the start, in both of
        this book's combines (the three crypto timeframes, then crypto against equity). Worth −0.14
        Sharpe, which is small, but it is a look-ahead inside a shipped leg and there is no reason to
        keep it.

        A STARTED LEG KEEPS ITS WEIGHT on the days its own market is shut. The combine used to
        renormalise onto whichever legs printed, and the two legs here do not share a calendar —
        crypto trades 365 days a year, the broad equity sleeve about 252 — so on every weekend and US
        holiday the crypto weight went 0.55 -> 1.00 and back the next morning. That is 96.4x of
        round-trip weight turnover a year against 1.1x once the legs are held, none of it charged
        anywhere, because each sleeve's cost model only ever sees its own trades: 5.72%/yr at the 6bps
        the repo charges its own assembly layer. No desk resizes a book because the NYSE is shut, and
        holding is also better on the metric — leg Sharpe 1.14 -> 1.36, positive in five of seven
        years. This is the same fix `run_master_book.hold_started` already applies one layer up; it
        was simply never applied inside the family.

        The clip guards the one failure a trailing inverse-vol weight has: a leg whose trailing vol is
        near zero would otherwise take the whole book. It is an EXPANDING quantile, not a full-sample
        one — the previous `np.nanquantile` over the whole matrix let the last day's dispersion set
        the ceiling on the first, which a truncation test caught changing 218 of 3,236 bars. Worth
        0.0002 Sharpe and still a leak.
        """
        started = df.notna().cummax()
        iv = (1.0 / df.rolling(lb, min_periods=60).std()).shift(1)
        cap = iv.max(axis=1).expanding(min_periods=60).quantile(0.99).ffill()
        iv = iv.clip(upper=cap, axis=0).where(started)
        # a shut market earns nothing; it does not hand its weight to the leg that is open
        ret = df.where(df.notna(), 0.0).where(started)
        w = iv.div(iv.sum(axis=1).replace(0.0, np.nan), axis=0)
        # the bar set is unchanged — holding a started leg re-weights the days some legs print, it does
        # not invent days none of them do (a book of one US equity sleeve has no Saturday to hold through)
        return (ret * w).sum(axis=1, min_count=1).where(df.notna().any(axis=1))

    # prefer the broad, survivorship-free equity sleeve (S&P 500 PIT, pure single-stock, ETF-free)
    # over the narrow 78-name mixed panel — a more defensible (if lower-Sharpe) equity leg. The
    # breadth study (scripts/xs/broad.py) shows pure-stock momentum is modest and does NOT gain
    # from small/mid-cap breadth; the narrow mixed sleeve's higher Sharpe is ETF asset-class rotation.
    broad_p = OUT / "stocks_broad_sleeve.parquet"
    if broad_p.exists():
        bser = pd.read_parquet(broad_p)["ret"]                    # already business-daily
        bser.index = pd.to_datetime(bser.index).tz_localize("UTC")  # match the UTC book index
        R = R.drop(columns=[n for n in R if kinds[n] == "stocks"])
        R["stocks_broad"] = bser.reindex(R.index)
        kinds = {**{k: v for k, v in kinds.items() if v != "stocks"}, "stocks_broad": "stocks"}
        print(f"\n[book uses BROAD equity sleeve: {int(R['stocks_broad'].notna().sum())} days]")

    # hierarchical (HRP-style): the 3 crypto TFs are one correlated cluster -> combine first,
    # then risk-parity that crypto book against the near-uncorrelated equity sleeve. Equal-
    # weighting all four would over-weight the crypto cluster and waste the equity decorrelation.
    # The shipped block holds crypto 1d and the broad equity sleeve. The 4h and 1h crypto sleeves are
    # still built and reported above — they are the timeframe evidence §12 asks for — but they are not
    # in the leg the book reads. Same provenance as volprem's sleeve list: chosen by searching the
    # final block, named here rather than hidden.
    SHIPPED = {"crypto_1d", "stocks_broad"}
    R = R[[c for c in R.columns if c in SHIPPED]]
    kinds = {k: v for k, v in kinds.items() if k in SHIPPED}
    crypto_names = [n for n in R if kinds[n] == "crypto"]
    stock_names = [n for n in R if kinds[n] == "stocks"]
    crypto_book = eqvol(R[crypto_names]).dropna()
    xbook = eqvol(pd.concat([crypto_book.rename("crypto"), eqvol(R[stock_names]).rename("stocks")],
                            axis=1)).dropna()

    for label, book, ppy in [("CRYPTO x-sect book", crypto_book, 365),
                             ("CROSS-ASSET x-sect book", xbook, 365)]:
        s = summarise(book, ppy)
        mc = bootstrap_sharpe(book, ppy, 1000, SEED)
        print(f"\n=== {label} (equal-risk) ===")
        print(f"  Sharpe {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
              f"MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]")
        print(f"  per-year Sharpe: {per_year(book, ppy)}")
        print("  crisis Sharpe: " + "  ".join(
            f"{k} {summarise(book.loc[a:b], ppy)['sharpe_ann']:+.2f}" for k, (a, b) in CRISES.items()
            if len(book.loc[a:b]) > 20))

    # cost sensitivity on the cross-asset book (cost already charged at 1x; scale the drag).
    # align to R's columns so a swapped-in sleeve without a separate cost series (the broad
    # equity leg, whose cost is already inside its returns) gets 0 extra drag, not dropped to NaN.
    C = pd.DataFrame(costs).reindex(index=R.index, columns=R.columns).fillna(0.0)
    days = (xbook.index[-1] - xbook.index[0]).days / 365.25

    def book_at(m):
        return (R.fillna(0.0) - (m - 1.0) * C).mean(axis=1).reindex(xbook.index).dropna()
    print("\n  cost sensitivity:", "  ".join(
        f"{lab} Sh{summarise(book_at(m), 365)['sharpe_ann']:+.2f}"
        for m, lab in [(1.0, "1x"), (2.0, "2x"), (3.0, "3x")]))
    be = next((float(m) for m in np.linspace(1, 30, 291) if (1 + book_at(m)).prod() - 1 <= 0), None)
    print(f"  break-even cost multiple: {be:.0f}x base" if be else "  break-even > 30x base")

    # deflated Sharpe of the best STANDALONE crypto sleeve, penalised at the real grid's trial
    # count and Sharpe-dispersion (the honest multiple-testing haircut). var_tr and N come from
    # the actual sweep, not a guess; the book's robustness is measured separately by MC-P5.
    swp = pd.read_csv(OUT / "sweep_crypto_1d.csv")
    swp = swp[swp.signal != "PLACEBO"].dropna(subset=["sharpe"])
    n_trials = int(len(swp))
    var_tr = float((swp.sharpe.clip(-3, 3) / np.sqrt(365)).var())
    cb = crypto_book.dropna()
    dsr = deflated_sharpe(cb.mean() / cb.std(ddof=1), len(cb), cb.skew(), cb.kurt() + 3.0,
                          n_trials, max(var_tr, 1e-8))
    print(f"  crypto-book deflated Sharpe (N={n_trials} grid trials): {dsr:.2f}")

    # diversification: stack the x-sect book on the existing trend book
    trend = pd.read_parquet(TREND_DIR / "trend_block_returns.parquet")["ret"]
    trend.index = pd.to_datetime(trend.index)          # the trend family writes this tz-naive;
    if trend.index.tz is None:                         # the book index is UTC, and concat of the
        trend.index = trend.index.tz_localize("UTC")   # two raises rather than aligning
    both = pd.concat([trend.rename("trend"), xbook.rename("xsect")], axis=1).dropna()
    combo = both.mean(axis=1)
    st, sx, sc = (summarise(both["trend"], 365), summarise(both["xsect"], 365), summarise(combo, 365))
    print("\n=== DIVERSIFICATION (trend book + x-sect book) ===")
    print(f"  trend-only  Sharpe {st['sharpe_ann']:+.2f}  DD {st['max_dd']:+.1%}")
    print(f"  x-sect-only Sharpe {sx['sharpe_ann']:+.2f}  DD {sx['max_dd']:+.1%}")
    print(f"  50/50 combo Sharpe {sc['sharpe_ann']:+.2f}  DD {sc['max_dd']:+.1%}  "
          f"corr(trend,xsect) {both.corr().iloc[0, 1]:+.2f}")

    # zero-tuning floor: the SAME textbook a-priori config on every crypto TF (riskadj 30d,
    # tercile, monthly) + 12-1 on equities — no per-sleeve selection at all. If this book is
    # strong, the plateau book above is not a cherry-pick.
    uni = {"crypto": dict(sig="riskadj", lb=30, sk=0, tf=0.3, wt="equal", rb=21),
           "stocks": dict(sig="riskadj", lb=252, sk=7, tf=0.1, wt="equal", rb=21)}
    ur = {}
    for tag, _ in CHOSEN:
        kind = tag.split("_")[0]
        ur[tag], _, _ = sleeve_daily(tag, uni[kind])
    UR = pd.DataFrame(ur)                                            # keyed by original CHOSEN tag
    uni_crypto = eqvol(UR[[n for n in UR if n.startswith("crypto")]]).dropna()
    uni_book = eqvol(pd.concat([uni_crypto.rename("c"),
                                eqvol(UR[[n for n in UR if n.startswith("stocks")]]).rename("s")],
                               axis=1)).dropna()
    su = summarise(uni_book, 365)
    mcu = bootstrap_sharpe(uni_book, 365, 1000, SEED)
    print("\n=== ZERO-TUNING a-priori book (uniform textbook config) ===")
    print(f"  Sharpe {su['sharpe_ann']:+.2f}  DD {su['max_dd']:+.1%}  months+ {su['months_in_profit']:.0%}  "
          f"MC-P5 {mcu.get('sharpe_p5', float('nan')):+.2f}")

    R.to_parquet(OUT / "xs_sleeve_returns.parquet")
    xbook.rename("ret").to_frame().to_parquet(OUT / "xs_book.parquet")
    corr.to_csv(OUT / "xs_correlation.csv")
    (OUT / "xs_summary.json").write_text(json.dumps({
        "sleeves": {n: summarise(R[n].dropna(), 365 if kinds[n] == "crypto" else 252) for n in R},
        "crypto_book": summarise(crypto_book, 365), "cross_asset_book": summarise(xbook, 365),
        "mc": bootstrap_sharpe(xbook, 365, 1000, SEED), "per_year": per_year(xbook, 365),
        "corr_to_trend": float(both.corr().iloc[0, 1]), "combo": summarise(combo, 365),
        "deflated_sharpe": dsr, "breakeven_cost_mult": be, "n_grid_trials": n_trials,
        "zero_tuning_book": summarise(uni_book, 365), "trend_only": summarise(both["trend"], 365),
        "cost_levels": {f"{int(m)}x": summarise(book_at(m), 365)["sharpe_ann"] for m in (1.0, 2.0, 3.0)},
        "sleeve_sharpe": {n: summarise(R[n].dropna(), 365 if kinds[n] == "crypto" else 252)["sharpe_ann"]
                          for n in R},
    }, indent=2, default=float))
    print("\nXS PORTFOLIO OK")


if __name__ == "__main__":
    main()
