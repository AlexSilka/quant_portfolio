"""Phase 1 — single-instrument trend sweep across the whole universe × every timeframe.

Answers, per (asset class × timeframe), four controlled questions with one spec grid:
  entry     which trend entry captures the premium (EMA/SMA/TSMOM/MACD/Donchian/blend/strength)
  direction long-short vs long-only vs short-only vs asymmetric 70/30
  exit      held-to-reversal vs chandelier ATR-trail vs Donchian channel
  gate      no regime filter vs ADX vs long-term-EMA vs vol-band

Every sleeve is net of liquidity-aware costs + funding, t+2 execution, vol-targeted to 15%.
Writes reports/trend/trend_sweep.csv (one row per asset×tf×spec) — the raw edge map.

    python scripts/trend/run_trend_sweep.py [--tfs 1d,4h,1h] [--max-crypto N] [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402


# --- the spec grid (each dict is one sleeve construction) -------------------------

def build_specs() -> list[dict]:
    specs = []
    # (A) entry comparison — exit=reversal, dir=ls, no gate
    for entry, params in [
        ("ema", {"fast": 20, "slow": 100}), ("ema", {"fast": 50, "slow": 200}),
        ("ema", {"fast": 100, "slow": 300}), ("sma", {"fast": 50, "slow": 200}),
        ("tsmom", {"lookback": 30}), ("tsmom", {"lookback": 90}),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}), ("donchian", {"lookback": 55}),
    ]:
        specs.append({"entry": entry, "params": params, "exit": "reversal",
                      "direction": "ls", "group": "entry"})
    for entry in ("blend", "strength"):
        specs.append({"entry": entry, "direction": "ls", "group": "entry"})
    # (B) direction comparison — ema 50/200 reversal
    for d in ("long_only", "short_only", "asym"):
        specs.append({"entry": "ema", "params": {"fast": 50, "slow": 200},
                      "exit": "reversal", "direction": d, "group": "direction"})
    # (C) exit comparison — ema 50/200 ls
    for ex in ("atr_trailing", "channel"):
        specs.append({"entry": "ema", "params": {"fast": 50, "slow": 200},
                      "exit": ex, "direction": "ls", "group": "exit"})
    # (D) gate comparison — ema 50/200 reversal ls
    for g in ("adx", "ltf", "vol"):
        specs.append({"entry": "ema", "params": {"fast": 50, "slow": 200},
                      "exit": "reversal", "direction": "ls", "gate": g, "group": "gate"})
    # (E) best-guess long-only blend (continuous, no short drag) — a headline candidate
    specs.append({"entry": "blend", "direction": "long_only", "group": "headline"})
    specs.append({"entry": "blend", "direction": "asym", "group": "headline"})
    return specs


def run_symbol(px, tf, ppy_bar, costs, fund, adv, meta, specs, rows, ppy_daily=365):
    for spec in specs:
        try:
            s, _ = T.eval_spec(px, spec, tf, ppy_bar, costs, fund=fund, adv=adv,
                               with_mc=False, ppy_daily=ppy_daily)
        except Exception as e:                       # a construction may fail on a short series
            print(f"    [skip] {T.spec_label(spec)}: {type(e).__name__} {e}")
            continue
        rows.append({**meta, "group": spec.get("group", ""), "spec": T.spec_label(spec),
                     "entry": spec["entry"], "direction": spec.get("direction", "ls"),
                     "exit": spec.get("exit", "-") if spec["entry"] not in T.CONTINUOUS else "-",
                     "gate": spec.get("gate", "-"),
                     "sharpe": round(s["sharpe_ann"], 3), "sortino": round(s["sortino_ann"], 3),
                     "max_dd": round(s["max_dd"], 4), "months_in_profit": round(s["months_in_profit"], 3),
                     "ann_turnover": round(s["ann_turnover"], 1) if s["ann_turnover"] == s["ann_turnover"] else None,
                     "total_return": round(s["total_return"], 4), "n_obs": s["n_obs"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="1d,4h,1h,15m,5m")
    ap.add_argument("--max-crypto", type=int, default=len(T.CRYPTO))
    ap.add_argument("--quick", action="store_true", help="6 crypto + 1d/4h only (fast sanity)")
    args = ap.parse_args()

    tfs = ["1d", "4h"] if args.quick else args.tfs.split(",")
    crypto = T.CRYPTO[:6] if args.quick else T.CRYPTO[:args.max_crypto]
    specs = build_specs()
    print(f"trend sweep: {len(crypto)} crypto + {len(T.STOCKS)} equity × {tfs} × {len(specs)} specs")

    rows: list[dict] = []
    t0 = time.time()

    # crypto (perp klines + funding)
    for i, sym in enumerate(crypto):
        for tf in tfs:
            px = T.load_crypto_long(sym, tf)   # spot(2017+)+perp splice on 1d/4h; perp-only intraday
            if px is None:
                continue
            fund = T.bo.safe_funding(sym)
            adv = T.crypto_adv(px)
            meta = {"asset_class": "crypto", "symbol": sym, "tf": tf}
            run_symbol(px, tf, T.CRYPTO_TF[tf], T.CC, fund, adv, meta, specs, rows)
        print(f"  crypto {i+1}/{len(crypto)} {sym} done ({time.time()-t0:.0f}s, {len(rows)} rows)")

    # equity (config core-10, daily only — intraday equity feed is shallow; trend needs long history)
    if "1d" in tfs:
        for sym in T.EQ_CORE:
            px = T.load_equity(sym)
            if px is None:
                continue
            adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
            meta = {"asset_class": "equity", "symbol": sym, "tf": "1d"}
            run_symbol(px, "1d", T.EQUITY_TF["1d"], T.EC, None, adv, meta, specs, rows, ppy_daily=252)
    print(f"  equity done ({time.time()-t0:.0f}s, {len(rows)} rows)")

    df = pd.DataFrame(rows)
    out = T.REPORTS / ("trend_sweep_quick.csv" if args.quick else "trend_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows, {time.time()-t0:.0f}s)")

    # quick headline: median Sharpe by asset_class × tf × entry, dir=ls
    ls = df[(df.direction == "ls") & (df.group == "entry")]
    piv = ls.pivot_table(index=["asset_class", "tf"], columns="entry", values="sharpe", aggfunc="median")
    print("\nmedian Sharpe by asset_class × tf × entry (dir=ls, exit=reversal):")
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
