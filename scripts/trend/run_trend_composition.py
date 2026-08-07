"""Trend-block composition + concentration (task §7): exactly what gets wired into the portfolio.

For the headline trend book it lists every (instrument × timeframe) sleeve with its standalone Sharpe,
share of book P&L, and risk contribution (covariance-based, sums to 100%), then aggregates by asset
class and timeframe and stress-tests concentration (portfolio with the top contributor removed;
crypto-only vs equity-only vs combined). The trend block's own frozen universe is crypto perps +
US equities at 1d/4h/1h — a per-FAMILY rule (breakout is crypto-only, vol-premium is equity/FX-led;
each family's asset set follows its own edge map, not a shared one).

    python scripts/trend/run_trend_composition.py [--book asym|blend_long_only|ensemble]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import sh  # noqa: E402
from src.metrics import summarise  # noqa: E402


def load_sleeves(book: str) -> pd.DataFrame:
    if book == "ensemble":
        e = pd.read_parquet(T.CACHE / "sleeves_ema_reversal_long_only_lag2_1d4h1h.parquet")
        b = pd.read_parquet(T.CACHE / "sleeves_blend_reversal_long_only_lag2_1d4h1h.parquet")
        return pd.concat([e.add_suffix("|ema"), b.add_suffix("|blend")], axis=1)
    tag = {"asym": "ema_reversal_asym", "blend_long_only": "blend_reversal_long_only"}[book]
    return pd.read_parquet(T.CACHE / f"sleeves_{tag}_lag2_1d4h1h.parquet")


def meta(col: str) -> tuple[str, str, str]:
    """(asset_class, symbol, tf) from a sleeve key like 'BTCUSDT_4h' or 'SPY_1d_eq' or '...|ema'."""
    key = col.split("|")[0]
    if key.endswith("_eq"):
        return "equity", key[:-6], "1d"
    parts = key.rsplit("_", 1)
    return "crypto", parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="asym", choices=["asym", "blend_long_only", "ensemble"])
    args = ap.parse_args()
    df = load_sleeves(args.book)
    book = df.mean(axis=1).dropna()
    var = book.var()

    rows = []
    tot_pnl = df.sum().sum()
    for c in df.columns:
        r = df[c].dropna()
        ac, sym, tf = meta(c)
        aligned = pd.concat([df[c], book], axis=1).dropna()
        rc = (aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / var / df.shape[1]) if var > 0 else 0.0
        rows.append({"asset_class": ac, "symbol": sym, "tf": tf, "entry": c.split("|")[1] if "|" in c else "ema",
                     "sharpe": round(sh(r), 2), "pnl_share": df[c].sum() / tot_pnl,
                     "risk_contrib": rc, "n_obs": len(r)})
    comp = pd.DataFrame(rows)
    comp["risk_contrib"] = comp["risk_contrib"] / comp["risk_contrib"].sum()   # normalise to 100%
    comp = comp.sort_values("risk_contrib", ascending=False).reset_index(drop=True)
    comp.to_csv(T.REPORTS / f"trend_composition_{args.book}.csv", index=False)

    s = summarise(book, 365)
    print(f"=== TREND BLOCK COMPOSITION — {args.book} ({df.shape[1]} sleeves) ===")
    print(f"book Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}\n")

    print("by ASSET CLASS:")
    for ac, g in comp.groupby("asset_class"):
        print(f"  {ac:8s}: {len(g):3d} sleeves  P&L share {g['pnl_share'].sum():5.0%}  "
              f"risk share {g['risk_contrib'].sum():5.0%}  median sleeve Sharpe {g['sharpe'].median():+.2f}")
    print("by TIMEFRAME:")
    for tf, g in comp.groupby("tf"):
        print(f"  {tf:8s}: {len(g):3d} sleeves  P&L share {g['pnl_share'].sum():5.0%}  "
              f"risk share {g['risk_contrib'].sum():5.0%}  median Sharpe {g['sharpe'].median():+.2f}")

    print("\ntop 10 risk contributors:")
    for _, r in comp.head(10).iterrows():
        print(f"  {r['symbol']:12s} {r['tf']:3s} {r['entry']:5s}  Sharpe {r['sharpe']:+.2f}  "
              f"P&L {r['pnl_share']:+.1%}  risk {r['risk_contrib']:.1%}")

    # concentration stress (task §7): remove the single top contributor; class-only books
    def bsh(cols):
        return sh(df[cols].mean(axis=1))
    top = comp.iloc[0]
    top_col = [c for c in df.columns if meta(c)[1] == top["symbol"] and meta(c)[2] == top["tf"]][0]
    without_top = bsh([c for c in df.columns if c != top_col])
    crypto_only = bsh([c for c in df.columns if meta(c)[0] == "crypto"])
    equity_only = bsh([c for c in df.columns if meta(c)[0] == "equity"])
    print(f"\nconcentration (task §7):")
    print(f"  full book Sharpe {s['sharpe_ann']:+.2f}  |  minus top contributor ({top['symbol']} {top['tf']}) {without_top:+.2f}")
    print(f"  crypto-only {crypto_only:+.2f}  |  equity-only {equity_only:+.2f}  |  combined {s['sharpe_ann']:+.2f} "
          f"(cross-asset diversification)")
    print(f"\nwrote reports/trend/trend_composition_{args.book}.csv")


if __name__ == "__main__":
    main()
