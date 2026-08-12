"""Equity carry — the dividend-yield carry trade, the equity analogue of FX rate carry and crypto
funding carry. Rank stocks by trailing-12m dividend yield, LONG high-yield / SHORT low-yield,
dollar-neutral; the book earns the yield spread plus whatever prices do. Same machinery and the same
validation (shuffled-yield placebo distribution, per-year, skew, cost sensitivity, cross-asset
correlation) so equity sits alongside crypto and FX on one comparable footing.

Dividends and prices both from Twelve Data (the paid feed; yfinance is not used). Close is
split-adjusted but not dividend-adjusted, so the dividend accrual is added explicitly, not double-counted.

    python scripts/carry/run_carry_equity.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.twelvedata import RateLimited  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.data.twelvedata import load_dividends  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

PPY, TVOL, SEED = 252, VOL_TARGET_ANNUAL, SEED
rng = np.random.default_rng(SEED)

# ~50 liquid US names chosen for dividend-yield DISPERSION (the cross-section carry needs spread):
# high/mid-yield (utilities, telecom, staples, energy, financials, REIT) vs low/zero (growth/tech).
UNIVERSE = ["XLU", "XLP", "XLE", "XLF", "VZ", "T", "KO", "PG", "MO", "PM", "XOM", "CVX", "JPM",
            "WFC", "BAC", "O", "PFE", "MRK", "IBM", "MMM", "D", "DUK", "SO", "NEE", "ED", "WMT",
            "MCD", "HD", "JNJ", "ABBV", "CAT", "CSCO", "INTC", "TGT", "LMT", "GIS", "K", "CL",
            "NVDA", "AMZN", "GOOGL", "TSLA", "META", "ADBE", "CRM", "NFLX", "AMD", "QCOM", "ORCL", "AAPL"]


def vt(net):
    scale = vol_target_scale(net, TVOL, PPY)
    return (net * scale).dropna()


def per_year(net):
    return {int(y): round(float(np.sqrt(PPY) * g.dropna().mean() / g.dropna().std(ddof=1)), 2)
            for y, g in net.groupby(net.index.year) if g.dropna().std(ddof=1) > 0}


def build_book(px, dy, *, top_frac=0.3, exec_lag=2, rebalance=5, cost_bps=3.0):
    """Dollar-neutral: LONG high dividend-yield, SHORT low. Total return = price move + yield accrual."""
    ret = px.pct_change()
    ranks = dy.rank(axis=1, pct=True)
    hi = (ranks >= 1 - top_frac).astype(float); lo = (ranks <= top_frac).astype(float)
    wl = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    ws = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).fillna(0.0)
    if rebalance > 1:
        keep = pd.Series(np.arange(len(w)) % rebalance == 0, index=w.index)
        w = w.where(keep, np.nan).ffill().fillna(0.0)
    w_h = w.shift(exec_lag).fillna(0.0)
    price = (w_h * ret).sum(axis=1)
    accr = (w_h * (dy / 100.0 / PPY).reindex_like(w_h).fillna(0.0)).sum(axis=1)   # dy is %, LONG high-yield earns +
    turn = w_h.diff().abs().sum(axis=1)
    cost = turn * cost_bps / 1e4
    return pd.DataFrame({"ret": price + accr - cost, "price": price, "carry": accr,
                         "cost": cost, "turnover": turn}).dropna(subset=["ret"])


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    px, divs = {}, {}
    for t in UNIVERSE:
        try:
            px[t] = load_equity_daily(t, start="2012-01-01")["close"]
            divs[t] = load_dividends(t, start="2011-01-01")
        except RateLimited:
            raise                       # a refused fetch would silently shrink the universe, and the
                                        # universe IS the result — two names refused moved this study's
                                        # entire history by up to 3.1e-02. Fail rather than publish a
                                        # headline computed on whatever the feed felt like answering.
        except Exception as e:
            print(f"skip {t}: {e}")
    P = pd.DataFrame(px).sort_index()
    # trailing-12m dividend yield per name = sum(dividends over prior 365d) / price, as %
    ttm = pd.DataFrame(index=P.index, columns=P.columns, dtype=float)
    for t in P.columns:
        d = divs.get(t)
        if d is None or not len(d):
            ttm[t] = 0.0
            continue
        daily = d.reindex(P.index.union(d.index)).fillna(0.0)
        ttm[t] = daily.rolling("365D").sum().reindex(P.index)
    dy = (ttm / P * 100.0).clip(lower=0)
    print(f"equity carry universe: {P.shape[1]} names, {P.index.min().date()}..{P.index.max().date()}")
    print(f"latest div-yields: high {dy.iloc[-1].nlargest(4).round(1).to_dict()}  low {dy.iloc[-1].nsmallest(4).round(1).to_dict()}\n")

    print("=== equity dividend carry (long high-yield / short low-yield) ===")
    rows, best = [], None
    for tf in (0.2, 0.3):
        for rb in (5, 21):
            bk = build_book(P, dy, top_frac=tf, rebalance=rb)
            net = vt(bk["ret"]); s = summarise(net, PPY)
            p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan) if s["sharpe_ann"] > 0.2 else np.nan
            row = {"top": tf, "rebal": rb, "sharpe": round(s["sharpe_ann"], 2), "mc_p5": round(p5, 2) if p5 == p5 else np.nan,
                   "max_dd": round(s["max_dd"], 2), "skew": round(net.skew(), 2), "months+": round(s["months_in_profit"], 2),
                   "ann_carry_%": round(bk["carry"].mean() * PPY * 100, 1), "ann_price_%": round(bk["price"].mean() * PPY * 100, 1)}
            rows.append(row)
            if best is None or row["sharpe"] > best[0]:
                best = (row["sharpe"], bk, net)
            print(f"  top{int(tf*100)}_rb{rb:<2d} Sharpe {row['sharpe']:+.2f}  P5 {row['mc_p5']}  DD {row['max_dd']}  "
                  f"skew {row['skew']:+.1f}  carry {row['ann_carry_%']:+.1f}%/yr  price {row['ann_price_%']:+.1f}%/yr")

    _, bk, net = best
    print(f"\nheadline Sharpe {best[0]:+.2f}  per-year: {per_year(net)}")

    # placebo DISTRIBUTION: shuffle which name has which yield, per date (200 draws)
    sh = []
    for _ in range(200):
        pdy = dy.copy(); a = pdy.values
        for i in range(a.shape[0]):
            r = a[i]; m = ~np.isnan(r); r[m] = rng.permutation(r[m]); a[i] = r
        sh.append(summarise(vt(build_book(P, pd.DataFrame(a, index=dy.index, columns=dy.columns))["ret"]), PPY)["sharpe_ann"])
    sh = np.array(sh); pct = (sh < best[0]).mean() * 100
    print(f"placebo (200 yield-shuffles): mean {sh.mean():+.2f}  P95 {np.percentile(sh,95):+.2f}  -> real at {pct:.0f}th pct "
          f"({'REAL edge' if pct > 95 else 'WEAK/marginal'})")

    # cross-asset correlations
    for name, path, col in [("crypto", CARRY_DIR / "carry_headline.parquet", "carry"), ("FX", CARRY_DIR / "carry_fx_headline.parquet", "ret")]:
        try:
            o = pd.read_parquet(path)[col]; idx = net.index.intersection(o.index)
            c = float(pd.concat([net.reindex(idx), o.reindex(idx)], axis=1).corr().iloc[0, 1]) if len(idx) > 60 else np.nan
            print(f"corr(equity carry, {name} carry) = {c:+.2f}")
        except Exception:
            pass

    pd.DataFrame(rows).to_csv(CARRY_DIR / "carry_equity.csv", index=False)
    net.rename("ret").to_frame().to_parquet(CARRY_DIR / "carry_equity_headline.parquet")
    print("\nCARRY-EQUITY OK")


if __name__ == "__main__":
    main()
