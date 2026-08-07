"""Breadth test: does the trend book get better with MORE crypto instruments? (managed-futures breadth)

We have ~243 perps with 1d/4h/1h (not just the harness's 50). This ranks the crypto universe by
LIQUIDITY (median pre-2024 daily dollar-volume — an a-priori rule, never by backtest performance, which
§5 proved is overfitting), builds trend sleeves for the top-200 ONCE, then measures the equal-risk book
at N = 20/50/100/150/200 crypto (+ the 10 equities). The question: where does adding instruments stop paying?

    python scripts/trend/run_trend_breadth.py [--entry ema] [--direction long_only]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import sh  # noqa: E402
from src.config import RAW_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import mc_metrics  # noqa: E402

TFS = ["1d", "4h", "1h"]
UM = RAW_DIR / "futures/um/klines"
MIN_PRE_OOS_BARS = 250          # ≥ ~1y of pre-OOS 1d history to be rankable/established
MAX_N = 200


def liquidity_ranked() -> list[str]:
    """All perps with 1d+4h+1h data, ranked by median pre-2024 daily dollar-volume (frozen rule)."""
    rows = []
    for d in sorted(UM.iterdir()):
        if not d.is_dir():
            continue
        sym = d.name
        if not all((d / tf).exists() and any((d / tf).glob("[0-9]*.parquet")) for tf in TFS):
            continue
        px = T.bo.load_crypto(sym, "1d")
        if px is None:
            continue
        pre = px[px.index < T.OOS_START]
        if len(pre) < MIN_PRE_OOS_BARS or "quote_volume" not in pre:
            continue
        rows.append((sym, float(pre["quote_volume"].median())))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [s for s, _ in rows]


def build_all_sleeves(symbols: list[str], entry: str, direction: str) -> pd.DataFrame:
    """Sleeve daily returns for every symbol×TF (+ equity), cached — subset later per N."""
    tag = f"breadth_{entry}_{direction}_{len(symbols)}"
    cpath = T.CACHE / f"{tag}.parquet"
    if cpath.exists():
        return pd.read_parquet(cpath)
    spec = {"entry": entry, "direction": direction,
            **({} if entry in T.CONTINUOUS else {"exit": "reversal"})}
    out, t0 = {}, time.time()
    for i, sym in enumerate(symbols):
        for tf in TFS:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            try:
                _, r = T.eval_spec(px, spec, tf, T.CRYPTO_TF[tf], T.CC,
                                   fund=T.bo.safe_funding(sym), adv=T.crypto_adv(px))
                if r.std(ddof=1) > 0:
                    out[f"{sym}_{tf}"] = r
            except Exception:
                pass
        if (i + 1) % 25 == 0:
            print(f"  built {i+1}/{len(symbols)} symbols ({time.time()-t0:.0f}s)")
    for sym in T.EQ_CORE:
        px = T.load_equity(sym)
        if px is None:
            continue
        adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
        try:
            _, r = T.eval_spec(px, spec, "1d", T.EQUITY_TF["1d"], T.EC, fund=None, adv=adv, ppy_daily=252)
            if r.std(ddof=1) > 0:
                out[f"{sym}_1d_eq"] = r
        except Exception:
            pass
    df = pd.DataFrame(out)
    df.to_parquet(cpath)
    return df


def cols_for(df: pd.DataFrame, symbols: list[str]) -> list[str]:
    keep = set(symbols)
    return [c for c in df.columns if c.endswith("_eq") or c.rsplit("_", 1)[0] in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="ema")
    ap.add_argument("--direction", default="long_only")
    args = ap.parse_args()

    ranked = liquidity_ranked()
    print(f"crypto perps with 1d+4h+1h and ≥{MIN_PRE_OOS_BARS} pre-OOS bars: {len(ranked)}")
    print(f"top-10 by liquidity: {ranked[:10]}\n")
    universe = ranked[:MAX_N]
    df = build_all_sleeves(universe, args.entry, args.direction)
    print(f"sleeve panel: {df.shape[1]} sleeves\n")

    print(f"=== breadth scaling — {args.entry} {args.direction} book (top-N crypto + 10 equity) ===")
    print(f"  {'N crypto':9s} {'sleeves':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'OOS':>6s} {'MC-P5':>6s} {'corr':>6s}")
    results = {}
    for n in [20, 50, 100, 150, 200]:
        if n > len(universe):
            continue
        cols = cols_for(df, universe[:n])
        sub = df[cols]
        port = sub.mean(axis=1).dropna()
        s = summarise(port, 365)
        oos = port[port.index >= T.OOS_START]
        mc = mc_metrics(port, 365, 500, T.SEED)
        corr = sub.corr().values
        cm = float(np.nanmean(corr[np.triu_indices_from(corr, 1)]))
        results[n] = {"sleeves": sub.shape[1], "sharpe": round(s["sharpe_ann"], 3),
                      "max_dd": round(s["max_dd"], 4), "oos": sh(oos),
                      "mc_p5": mc.get("sharpe_p5"), "corr_mean": round(cm, 3)}
        print(f"  {n:>9d} {sub.shape[1]:>8d} {s['sharpe_ann']:>+7.2f} {s['max_dd']:>+7.1%} "
              f"{sh(oos):>+6.2f} {(mc.get('sharpe_p5') or float('nan')):>+6.2f} {cm:>+6.2f}")

    (T.REPORTS / f"trend_breadth_{args.entry}_{args.direction}.json").write_text(
        json.dumps({"universe_size": len(ranked), "top10": ranked[:10], "scaling": results}, indent=2, default=float))
    print(f"\nwrote reports/trend/trend_breadth_{args.entry}_{args.direction}.json")


if __name__ == "__main__":
    main()
