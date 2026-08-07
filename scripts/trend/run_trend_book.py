"""Phase 2 — the diversified trend BOOK + full robustness.

Trend-following's Sharpe is a *diversification* effect: each instrument's trend sleeve is
individually modest (~0.3-0.7), but many decorrelated trend sleeves combine to ~1+ (the managed-
futures signature). So the book is every (instrument × timeframe) trend sleeve on the FROZEN
universe — no per-sleeve survivor selection — combined at equal risk (mean of vol-targeted returns).

Three books are built and compared head-to-head, resolving the long-only question at the portfolio
level: long-short · long-only · asymmetric 70/30. For the headline book it reports Monte-Carlo
(block bootstrap P5/P50/P95 of Sharpe, maxDD, monthly hit; + exec-lag jitter), per-year/quarter,
the task's isolated regimes (Q4-2018 … OOS-2024), 3 cost levels + break-even, sleeve correlation,
the marginal-contribution curve, a shuffled-signal placebo/FDR arm, and the held-out OOS split.

    python scripts/trend/run_trend_book.py [--entry blend] [--tfs 1d,4h,1h]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe, mc_metrics  # noqa: E402

REGIMES = {
    "q4_2018": ("2018-10-01", "2018-12-31"),        # crypto bear (spot-spliced history)
    "covid_2020": ("2020-02-15", "2020-04-15"),     # Feb-Mar 2020 crash
    "bull_2021": ("2021-01-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "chop_2023_25": ("2023-01-01", "2025-12-31"),
    "oos_2024h2+": ("2024-07-01", "2026-12-31"),     # the held-out block
}


def sh(g: pd.Series, ppy: int = 365) -> float:
    g = g.dropna()
    return round(float(np.sqrt(ppy) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0


# --- sleeve returns across the universe (cached per entry×direction×tf×lag) --------

def sleeve_returns(entry: str, direction: str, tfs: list[str], exit_style: str = "reversal",
                   exec_lag: int = 2, use_cache: bool = True) -> dict[str, pd.Series]:
    """{f'{sym}_{tf}': daily_ret} for one construction over the whole universe."""
    tag = f"{entry}_{exit_style}_{direction}_lag{exec_lag}_" + "".join(tfs)
    cpath = T.CACHE / f"sleeves_{tag}.parquet"
    if use_cache and cpath.exists():
        df = pd.read_parquet(cpath)
        return {c: df[c].dropna() for c in df.columns}

    cc = dict(T.CC); cc["exec_lag"] = exec_lag
    ec = dict(T.EC); ec["exec_lag"] = exec_lag
    out: dict[str, pd.Series] = {}
    spec = {"entry": entry, "direction": direction,
            **({} if entry in T.CONTINUOUS else {"exit": exit_style})}

    for sym in T.CRYPTO:
        for tf in tfs:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            try:
                _, r = T.eval_spec(px, spec, tf, T.CRYPTO_TF[tf], cc,
                                   fund=T.bo.safe_funding(sym), adv=T.crypto_adv(px))
            except Exception:
                continue
            if r.std(ddof=1) > 0:
                out[f"{sym}_{tf}"] = r
    for sym in T.EQ_CORE:                                # equity daily only
        if "1d" not in tfs:
            continue
        px = T.load_equity(sym)
        if px is None:
            continue
        adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
        try:
            _, r = T.eval_spec(px, spec, "1d", T.EQUITY_TF["1d"], ec, fund=None, adv=adv, ppy_daily=252)
        except Exception:
            continue
        if r.std(ddof=1) > 0:
            out[f"{sym}_1d_eq"] = r

    if use_cache:
        pd.DataFrame(out).to_parquet(cpath)
    return out


def equal_risk(rets: dict[str, pd.Series]) -> pd.Series:
    """Equal-risk book = mean of the (already vol-targeted) sleeve returns, NaN-safe."""
    df = pd.DataFrame(rets).sort_index()
    return df.mean(axis=1).rename("ret"), df


# --- robustness pieces ------------------------------------------------------------

def marginal_contribution(df: pd.DataFrame) -> list[dict]:
    """Add sleeves greedily in order of standalone Sharpe; report book Sharpe/DD as it grows."""
    order = df.apply(lambda c: sh(c), axis=0).sort_values(ascending=False).index.tolist()
    curve = []
    for k in range(1, len(order) + 1):
        port = df[order[:k]].mean(axis=1)
        s = summarise(port.dropna(), 365)
        curve.append({"n": k, "sharpe": round(s["sharpe_ann"], 3),
                      "max_dd": round(s["max_dd"], 4), "months_in_profit": round(s["months_in_profit"], 3)})
    return curve


def placebo_null(entry: str, direction: str, tfs: list[str], n_shuffle: int = 100,
                 exit_style: str = "reversal", seed: int = 7) -> dict:
    """Shuffled-signal placebo (task §6): the correct null destroys the *signal-to-return alignment*,
    not the returns' marginal distribution. Each sleeve's held position is circularly shifted by a
    random offset (preserving its turnover and exposure profile but decorrelating it from the real
    returns), then re-vol-targeted and re-backtested. A real edge collapses to ~0 Sharpe under this;
    the exceedance rate (how often the shuffled book beats the real one) is the pipeline's FDR.
    """
    from src.backtest.engine import backtest, vol_target
    rng = np.random.default_rng(seed)
    spec = {"entry": entry, "direction": direction,
            **({} if entry in T.CONTINUOUS else {"exit": exit_style})}
    sleeves = []                                        # (close, base_pos, costs, fund, adv, ppy_bar)
    for sym in T.CRYPTO:
        for tf in tfs:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            try:
                pos = T.trend_position(px, spec, tf)
            except Exception:
                continue
            sleeves.append((px["close"], pos, T.CC, T.bo.safe_funding(sym), T.crypto_adv(px), T.CRYPTO_TF[tf]))
    if "1d" in tfs:
        for sym in T.EQ_CORE:
            px = T.load_equity(sym)
            if px is None:
                continue
            adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
            sleeves.append((px["close"], T.trend_position(px, spec, "1d"), T.EC, None, adv, T.EQUITY_TF["1d"]))

    def book_of(shift: bool) -> float:
        rets = {}
        for i, (close, pos, costs, fund, adv, ppyb) in enumerate(sleeves):
            p = pos
            if shift:
                off = int(rng.integers(50, max(len(pos) - 50, 60)))
                p = pd.Series(np.roll(pos.to_numpy(), off), index=pos.index)
            posv = vol_target(p, close, T.TVOL, ppyb)
            bt = backtest(close, posv, capital=T.CAP, funding=fund, adv=adv, **costs)
            rets[i] = (1 + bt["net_ret"]).resample("D").prod() - 1
        return sh(pd.DataFrame(rets).mean(axis=1))

    real = book_of(False)
    null = np.array([book_of(True) for _ in range(n_shuffle)])
    return {"real_sharpe": real, "null_p50": round(float(np.percentile(null, 50)), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "null_p99": round(float(np.percentile(null, 99)), 3),
            "exceedance_rate": round(float((null >= real).mean()), 4), "n_shuffle": n_shuffle}


def synthetic_null(entry: str, direction: str, tfs: list[str], n_shuffle: int = 100,
                   exit_style: str = "reversal", seed: int = 7) -> dict:
    """Shuffled-DATA null (task §6): the gold-standard placebo — shuffle each price's returns to a
    synthetic random walk (destroying all serial dependence / trends), recompute the trend signal on
    that synthetic price, and backtest. A signal that exploits *real* autocorrelation collapses to
    ~0 Sharpe here; the exceedance rate is the pipeline's false-discovery rate. Close-based specs only.
    """
    from src.backtest.engine import backtest, vol_target
    rng = np.random.default_rng(seed)
    spec = {"entry": entry, "direction": direction,
            **({} if entry in T.CONTINUOUS else {"exit": exit_style})}
    series = []                                        # (close, adv, ppy_bar, costs)
    for sym in T.CRYPTO:
        for tf in tfs:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            series.append((px["close"], T.crypto_adv(px), T.CRYPTO_TF[tf], T.CC))
    if "1d" in tfs:
        for sym in T.EQ_CORE:
            px = T.load_equity(sym)
            if px is None:
                continue
            series.append((px["close"], (px["close"] * px["volume"]).rolling(20).median().shift(1),
                           T.EQUITY_TF["1d"], T.EC))

    def book_of(shuffle: bool) -> float:
        rets = {}
        for i, (close, adv, ppyb, costs) in enumerate(series):
            c = close
            if shuffle:
                lr = np.log(close).diff().dropna().to_numpy()
                rng.shuffle(lr)
                c = pd.Series(close.iloc[0] * np.exp(np.concatenate([[0.0], lr]).cumsum()),
                              index=close.index)
            fake = pd.DataFrame({"close": c, "high": c, "low": c})
            pos = T.trend_position(fake, spec, tfs[0])
            posv = vol_target(pos, c, T.TVOL, ppyb)
            bt = backtest(c, posv, capital=T.CAP, adv=adv, **costs)   # no funding on synthetic
            rets[i] = (1 + bt["net_ret"]).resample("D").prod() - 1
        return sh(pd.DataFrame(rets).mean(axis=1))

    real = book_of(False)
    null = np.array([book_of(True) for _ in range(n_shuffle)])
    return {"real_sharpe": real, "null_p50": round(float(np.percentile(null, 50)), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "null_p99": round(float(np.percentile(null, 99)), 3),
            "exceedance_rate": round(float((null >= real).mean()), 4), "n_shuffle": n_shuffle}


def report_book(name: str, port: pd.Series, df: pd.DataFrame, full: bool = False) -> dict:
    s = summarise(port.dropna(), 365)
    row = {"name": name, "n_sleeves": int(df.shape[1]), "sharpe": round(s["sharpe_ann"], 3),
           "sortino": round(s["sortino_ann"], 3), "max_dd": round(s["max_dd"], 4),
           "months_in_profit": round(s["months_in_profit"], 3), "total_return": round(s["total_return"], 3),
           "n_obs": s["n_obs"]}
    print(f"  {name:12s} Sh {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"tot {s['total_return']:+.0%}  ({df.shape[1]} sleeves)")
    if not full:
        return row
    # full robustness (headline book only)
    mc = mc_metrics(port, 365, n_reps=1000, seed=T.SEED)
    corr = df.corr()
    iu = np.triu_indices_from(corr, k=1)
    per_year = {int(y): sh(g) for y, g in port.groupby(port.index.year)}
    per_q = {str(q): sh(g) for q, g in port.groupby(port.index.to_period("Q"))}
    regimes = {}
    for rn, (a, b) in REGIMES.items():
        g = port.loc[a:b]
        ss = summarise(g.dropna(), 365)
        regimes[rn] = {"sharpe": round(ss["sharpe_ann"], 2), "max_dd": round(ss["max_dd"], 3),
                       "total": round(ss["total_return"], 3), "n": ss["n_obs"]}
    row.update({"mc": mc, "corr_mean": round(float(corr.values[iu].mean()), 3),
                "corr_max": round(float(corr.values[iu].max()), 3),
                "per_year": per_year, "per_quarter": per_q, "regimes": regimes,
                "marginal": marginal_contribution(df)})
    print(f"    MC Sharpe[P5 {mc['sharpe_p5']:+.2f} P50 {mc['sharpe_p50']:+.2f} P95 {mc['sharpe_p95']:+.2f}]  "
          f"maxDD[P5 {mc['maxdd_p5']:.1%} P50 {mc['maxdd_p50']:.1%}]  hit[P50 {mc['hit_p50']:.0%}]")
    print(f"    corr mean {row['corr_mean']:+.2f} max {row['corr_max']:+.2f}")
    print(f"    per-year: {per_year}")
    print(f"    regimes: " + "  ".join(f"{k} {v['sharpe']:+.2f}" for k, v in regimes.items()))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="blend")
    ap.add_argument("--tfs", default="1d,4h,1h")
    ap.add_argument("--exit", default="reversal")
    args = ap.parse_args()
    tfs = args.tfs.split(",")
    t0 = time.time()
    print(f"=== TREND BOOK — entry={args.entry} exit={args.exit} tfs={tfs} ===\n")

    books, dfs = {}, {}
    for d in ("ls", "long_only", "asym"):
        rets = sleeve_returns(args.entry, d, tfs, exit_style=args.exit)
        port, df = equal_risk(rets)
        books[d], dfs[d] = port, df
    print(f"sleeve returns built ({time.time()-t0:.0f}s)\n\ndirection comparison (equal-risk book):")
    dir_rows = {d: report_book(d, books[d], dfs[d]) for d in books}

    # headline = the best-Sharpe direction (reported with full robustness)
    headline = max(books, key=lambda d: dir_rows[d]["sharpe"])
    print(f"\n=== HEADLINE BOOK = '{headline}' (full robustness) ===")
    hb = report_book(headline, books[headline], dfs[headline], full=True)

    # exec-lag jitter (entry-timing robustness): rebuild headline book at lag 1 and 3
    print("\nexec-lag jitter (entry-timing robustness):")
    lag = {}
    for L in (1, 3):
        r = sleeve_returns(args.entry, headline, tfs, exit_style=args.exit, exec_lag=L, use_cache=False)
        p, _ = equal_risk(r)
        lag[f"lag{L}"] = sh(p)
        print(f"    exec_lag={L}: Sharpe {sh(p):+.2f}")
    lag["lag2_default"] = hb["sharpe"]

    # cost sensitivity (1x/2x/3x) + break-even — approximate by scaling: net ≈ gross - m*cost.
    # sleeve returns are already net at 1x; a clean multiple needs the cost series, so we re-derive
    # the headline book at inflated commission to get 2x/3x honestly.
    print("\ncost sensitivity (headline book):")
    cost_levels = {}
    for mult, lab in [(1.0, "1x"), (2.0, "2x"), (3.0, "3x")]:
        cc = dict(T.CC); cc["commission_bps"] *= mult; cc["half_spread_bps"] *= mult
        ec = dict(T.EC); ec["commission_bps"] *= mult; ec["half_spread_bps"] *= mult
        # only recompute if not 1x (1x == headline)
        if mult == 1.0:
            cost_levels[lab] = hb["sharpe"]
            print(f"    {lab}: Sharpe {hb['sharpe']:+.2f}")
            continue
        r = _sleeve_returns_costed(args.entry, headline, tfs, args.exit, cc, ec)
        p, _ = equal_risk(r)
        cost_levels[lab] = sh(p)
        print(f"    {lab}: Sharpe {sh(p):+.2f}")

    # shuffled-DATA null / FDR (the proper test: synthetic random-walk price, recompute signal).
    # Run it for the headline AND for the beta-neutral LS book — the two answer different questions:
    #   headline (long-biased): most Sharpe is harvested beta, so its null stays high (trend overlay's
    #                           value there is drawdown control, not excess return);
    #   LS (beta-neutral):      strips beta, so a low exceedance proves the trend TIMING edge is real.
    print("\nshuffled-DATA null / FDR (synthetic random-walk price, recompute signal):")
    n_sh = 100 if tfs == ["1d"] else 60
    placebo = synthetic_null(args.entry, headline, tfs, n_shuffle=n_sh)
    placebo_ls = synthetic_null(args.entry, "ls", tfs, n_shuffle=n_sh)
    print(f"    headline ({headline}): real {placebo['real_sharpe']:+.2f}  null P50 {placebo['null_p50']:+.2f} "
          f"P95 {placebo['null_p95']:+.2f}  exceedance {placebo['exceedance_rate']:.1%}  (long-biased → beta in the null)")
    print(f"    beta-neutral (ls)   : real {placebo_ls['real_sharpe']:+.2f}  null P50 {placebo_ls['null_p50']:+.2f} "
          f"P95 {placebo_ls['null_p95']:+.2f}  exceedance {placebo_ls['exceedance_rate']:.1%}  (pure timing edge)")
    hb["placebo"] = placebo
    hb["placebo_beta_neutral"] = placebo_ls

    # IS/OOS split
    port = books[headline]
    is_, oos = port[port.index < T.OOS_START], port[port.index >= T.OOS_START]
    oos_split = {"is_sharpe": sh(is_), "oos_sharpe": sh(oos),
                 "is_n": int(len(is_)), "oos_n": int(len(oos))}
    print(f"\nIS/OOS split @ {T.OOS_START.date()}: IS {oos_split['is_sharpe']:+.2f}  OOS {oos_split['oos_sharpe']:+.2f}")

    # deflated Sharpe of the BEST SINGLE SLEEVE at the honest trial count (individual sleeves are
    # marginal after multiple testing; the book's robustness is a diversification effect — see MC/placebo)
    dfh = dfs[headline]
    best_key = max(dfh.columns, key=lambda c: sh(dfh[c]))
    b = dfh[best_key].dropna()
    n_trials = 20 * 5 * 4 + 200        # sweep specs × asset-tf cells × directions + variants
    dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0, n_trials, 0.25 / 365)
    print(f"best single sleeve = {best_key} (Sharpe {sh(b):+.2f}); deflated Sharpe @ N={n_trials}: {dsr:.2f}")

    # persist
    for d in books:
        books[d].to_frame().to_parquet(T.REPORTS / f"trend_book_{d}.parquet")
    dfs[headline].to_parquet(T.REPORTS / "trend_headline_sleeves.parquet")
    summary = {"entry": args.entry, "exit": args.exit, "tfs": tfs, "headline_direction": headline,
               "directions": dir_rows, "headline": hb, "exec_lag_jitter": lag,
               "cost_levels": cost_levels, "oos_split": oos_split,
               "best_sleeve": best_key, "best_sleeve_dsr": round(dsr, 3), "n_trials": n_trials}
    (T.REPORTS / "trend_book_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote reports/trend/trend_book_*.parquet + trend_book_summary.json  ({time.time()-t0:.0f}s)")


def _sleeve_returns_costed(entry, direction, tfs, exit_style, cc, ec):
    """sleeve_returns with explicit cost dicts (for the 2x/3x cost sweep), no cache."""
    out = {}
    spec = {"entry": entry, "direction": direction,
            **({} if entry in T.CONTINUOUS else {"exit": exit_style})}
    for sym in T.CRYPTO:
        for tf in tfs:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            try:
                _, r = T.eval_spec(px, spec, tf, T.CRYPTO_TF[tf], cc,
                                   fund=T.bo.safe_funding(sym), adv=T.crypto_adv(px))
            except Exception:
                continue
            if r.std(ddof=1) > 0:
                out[f"{sym}_{tf}"] = r
    if "1d" in tfs:
        for sym in T.EQ_CORE:
            px = T.load_equity(sym)
            if px is None:
                continue
            adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
            try:
                _, r = T.eval_spec(px, spec, "1d", T.EQUITY_TF["1d"], ec, fund=None, adv=adv)
            except Exception:
                continue
            if r.std(ddof=1) > 0:
                out[f"{sym}_1d_eq"] = r
    return out


if __name__ == "__main__":
    main()
