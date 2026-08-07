"""Regime-filter study at the BOOK level: does gating trend entries to a trending regime (ADX,
Kaufman efficiency ratio, vol band) improve the book — especially its weak OOS and its drawdown —
or does it just remove good trend bars? The single-instrument sweep suggested gates are marginal on
the core 1d/4h TFs; this checks the book-level Sharpe / maxDD / held-out-OOS picture directly.

Each (instrument × tf) is loaded once and every gate evaluated on it, so the comparison is like-for-like.

    python scripts/trend/run_trend_regime.py [--entry ema] [--direction asym]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import equal_risk, sh  # noqa: E402
from src.metrics import summarise  # noqa: E402

GATES = {
    "none": {},
    "adx25": {"gate": "adx", "params": {"adx_thr": 25.0}},
    "adx20": {"gate": "adx", "params": {"adx_thr": 20.0}},
    "eff0.3": {"gate": "eff", "params": {"eff_n": 20, "eff_thr": 0.3}},
    "eff0.4": {"gate": "eff", "params": {"eff_n": 20, "eff_thr": 0.4}},
    "vol": {"gate": "vol", "params": {"vol_hi_q": 0.9}},
}


def gated_books(entry: str, direction: str, tfs: list[str]) -> dict[str, pd.DataFrame]:
    """Return {gate_name: sleeve-returns DataFrame} — each (sym,tf) loaded once, all gates on it."""
    cols = {g: {} for g in GATES}
    def one(px, tf, ppy, costs, fund, adv, key, ppy_daily=365):
        for g, gc in GATES.items():
            spec = {"entry": entry, "direction": direction,
                    **({} if entry in T.CONTINUOUS else {"exit": "reversal"}),
                    **({"gate": gc["gate"]} if gc else {}), "params": gc.get("params", {})}
            try:
                _, r = T.eval_spec(px, spec, tf, ppy, costs, fund=fund, adv=adv, ppy_daily=ppy_daily)
                if r.std(ddof=1) > 0:
                    cols[g][key] = r
            except Exception:
                pass
    for sym in T.CRYPTO:
        for tf in tfs:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            one(px, tf, T.CRYPTO_TF[tf], T.CC, T.bo.safe_funding(sym), T.crypto_adv(px), f"{sym}_{tf}")
    if "1d" in tfs:
        for sym in T.EQ_CORE:
            px = T.load_equity(sym)
            if px is None:
                continue
            adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
            one(px, "1d", T.EQUITY_TF["1d"], T.EC, None, adv, f"{sym}_1d_eq", ppy_daily=252)
    return {g: pd.DataFrame(c) for g, c in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="ema")
    ap.add_argument("--direction", default="asym")
    ap.add_argument("--tfs", default="1d,4h,1h")
    args = ap.parse_args()
    tfs = args.tfs.split(",")
    print(f"=== Regime-filter book study — entry={args.entry} dir={args.direction} tfs={tfs} ===\n")
    books = gated_books(args.entry, args.direction, tfs)

    print(f"  {'gate':8s} {'Sharpe':>7s} {'maxDD':>7s} {'OOS':>6s} {'2022':>6s} {'2025':>6s} {'exposure%':>9s}")
    out = {}
    base = None
    for g, df in books.items():
        if df.empty:
            continue
        port = df.mean(axis=1)
        s = summarise(port.dropna(), 365)
        oos = port[port.index >= T.OOS_START]
        y = {int(yr): sh(gg) for yr, gg in port.groupby(port.index.year)}
        exposure = float(df.notna().mean(axis=1).mean())     # avg fraction of sleeves active (proxy)
        out[g] = {"sharpe": round(s["sharpe_ann"], 3), "max_dd": round(s["max_dd"], 4),
                  "oos": sh(oos), "y2022": y.get(2022), "y2025": y.get(2025), "exposure": round(exposure, 3)}
        if g == "none":
            base = out[g]
        print(f"  {g:8s} {s['sharpe_ann']:>+7.2f} {s['max_dd']:>+7.1%} {sh(oos):>+6.2f} "
              f"{y.get(2022, float('nan')):>+6.2f} {y.get(2025, float('nan')):>+6.2f} {exposure*100:>8.0f}%")

    (T.REPORTS / "trend_regime.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nverdict (Δ vs no-gate):")
    for g, s in out.items():
        if g == "none" or base is None:
            continue
        print(f"  {g:8s}: Sharpe {s['sharpe']-base['sharpe']:+.2f}  maxDD {s['max_dd']-base['max_dd']:+.1%}  "
              f"OOS {s['oos']-base['oos']:+.2f}")
    print("\nwrote reports/trend/trend_regime.json")


if __name__ == "__main__":
    main()
