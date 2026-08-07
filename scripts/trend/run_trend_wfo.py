"""Phase 3 — walk-forward & parameter sensitivity (task §10).

Three honesty checks, each attacking a different overfit vector:

  1. PARAMETER SENSITIVITY SURFACE — evaluate a whole trend-config grid on representative assets;
     report min/median/max Sharpe and % of the grid positive. A broad positive plateau (not a lone
     spike) is the signature of a real premium, not a fitted parameter.
  2. PARAMETER WALK-FORWARD — on each train window pick the best config, apply it OOS to the next
     window, stitch. The honest cost of *choosing* the construction. Compared to the in-sample peak.
  3. SELECTION WALK-FORWARD — rebuild the book keeping only sleeves with trailing Sharpe > 0, across
     {anchored, rolling-2y, rolling-3y} × {annual, semiannual, quarterly} refits, to show the OOS
     result does NOT depend on the window/cadence choice (task §10).

    python scripts/trend/run_trend_wfo.py [--entry blend] [--tfs 1d,4h,1h]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import equal_risk, sh, sleeve_returns  # noqa: E402

# a broad trend-config grid for the sensitivity surface (all held-to-reversal, LS)
GRID = (
    [{"entry": "ema", "params": {"fast": f, "slow": s}} for f, s in
     [(10, 50), (20, 100), (20, 200), (50, 150), (50, 200), (50, 300), (100, 300), (100, 400)]]
    + [{"entry": "sma", "params": {"fast": f, "slow": s}} for f, s in [(20, 100), (50, 200), (100, 300)]]
    + [{"entry": "tsmom", "params": {"lookback": lb}} for lb in [20, 30, 60, 90, 120, 180]]
    + [{"entry": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}]
    + [{"entry": "donchian", "params": {"lookback": lb}} for lb in [20, 55, 100]]
    + [{"entry": "blend"}, {"entry": "strength", "params": {"lookback": 90}}]
)


def cfg_returns(px, cfg, tf, ppy, costs, fund, adv, direction="ls"):
    spec = {**cfg, "direction": direction,
            **({} if cfg["entry"] in T.CONTINUOUS else {"exit": "reversal"})}
    try:
        _, r = T.eval_spec(px, spec, tf, ppy, costs, fund=fund, adv=adv)
        return r
    except Exception:
        return None


# --- 1. parameter sensitivity surface ---------------------------------------------

def sensitivity(tf="1d"):
    """Grid Sharpe surface per asset class (crypto core-6 + equity core-5), net of costs."""
    rows = []
    crypto = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
    for sym in crypto:
        px = T.load_crypto_long(sym, tf)
        if px is None:
            continue
        fund, adv = T.bo.safe_funding(sym), T.crypto_adv(px)
        for cfg in GRID:
            r = cfg_returns(px, cfg, tf, T.CRYPTO_TF[tf], T.CC, fund, adv)
            if r is not None:
                rows.append({"asset_class": "crypto", "symbol": sym, "cfg": T.spec_label({**cfg, "direction": "ls"}),
                             "sharpe": sh(r)})
    for sym in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]:
        px = T.load_equity(sym)
        if px is None:
            continue
        adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
        for cfg in GRID:
            r = cfg_returns(px, cfg, "1d", T.EQUITY_TF["1d"], T.EC, None, adv)
            if r is not None:
                rows.append({"asset_class": "equity", "symbol": sym, "cfg": T.spec_label({**cfg, "direction": "ls"}),
                             "sharpe": sh(r)})
    df = pd.DataFrame(rows)
    surf = {}
    for ac, g in df.groupby("asset_class"):
        by_cfg = g.groupby("cfg")["sharpe"].median()
        surf[ac] = {"min": round(float(by_cfg.min()), 2), "median": round(float(by_cfg.median()), 2),
                    "max": round(float(by_cfg.max()), 2), "pct_positive": round(float((by_cfg > 0).mean()), 2),
                    "n_cfgs": int(len(by_cfg))}
    return df, surf


# --- 2. parameter walk-forward ----------------------------------------------------

def parameter_wf(sym, loader, tf, ppy, costs, is_crypto):
    """Per calendar year: pick the best-of-GRID config on all data strictly before, apply OOS to the
    year. Stitch the OOS pieces. Reports in-sample-peak vs walk-forward Sharpe (the overfit gap)."""
    px = loader(sym, tf) if is_crypto else loader(sym)
    if px is None:
        return None
    fund = T.bo.safe_funding(sym) if is_crypto else None
    adv = T.crypto_adv(px) if is_crypto else (px["close"] * px["volume"]).rolling(20).median().shift(1)
    allret = {T.spec_label({**c, "direction": "ls"}): cfg_returns(px, c, tf, ppy, costs, fund, adv) for c in GRID}
    allret = {k: v for k, v in allret.items() if v is not None}
    if not allret:
        return None
    R = pd.DataFrame(allret)
    years = sorted(set(R.index.year))
    oos_pieces, best_is = [], []
    for y in years:
        train = R[R.index.year < y]
        if len(train) < 252:
            continue
        best = train.apply(lambda c: sh(c)).idxmax()
        oos_pieces.append(R[R.index.year == y][best].rename("ret"))
        best_is.append(sh(train[best]))
    if not oos_pieces:
        return None
    wf = pd.concat(oos_pieces).sort_index()
    peak = R.apply(lambda c: sh(c)).max()             # in-sample best over the whole sample (overfit)
    return {"symbol": sym, "wf_sharpe": sh(wf), "insample_peak": round(float(peak), 2),
            "mean_best_is": round(float(np.mean(best_is)), 2), "n_oos": int(len(wf))}


# --- 3. selection walk-forward (window/cadence robustness) ------------------------

def selection_wf(df: pd.DataFrame, window: str, cadence: str) -> pd.Series:
    """Rebuild the book across rebalances: keep sleeves with trailing Sharpe>0 over the lookback
    window (anchored=all history, else rolling N years), hold equal-risk to the next rebalance."""
    dates = df.index
    step = {"annual": 12, "semiannual": 6, "quarterly": 3}[cadence]
    rebal = pd.date_range(dates.min(), dates.max(), freq=f"{step}MS")   # bounds already tz-aware
    out = pd.Series(index=dates, dtype=float)
    win_years = {"anchored": None, "rolling2y": 2, "rolling3y": 3}[window]
    for i, d in enumerate(rebal):
        lo = dates.min() if win_years is None else d - pd.DateOffset(years=win_years)
        train = df[(df.index >= lo) & (df.index < d)]
        if len(train) < 126:
            continue
        keep = [c for c in df.columns if train[c].notna().sum() > 60 and sh(train[c]) > 0]
        if not keep:
            continue
        nxt = rebal[i + 1] if i + 1 < len(rebal) else dates.max() + pd.Timedelta(days=1)
        seg = df.loc[(df.index >= d) & (df.index < nxt), keep]
        out.loc[seg.index] = seg.mean(axis=1)
    return out.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="blend")
    ap.add_argument("--tfs", default="1d,4h,1h")
    args = ap.parse_args()
    tfs = args.tfs.split(",")
    t0 = time.time()

    print("=== 1. PARAMETER SENSITIVITY SURFACE (grid Sharpe, net, dir=ls) ===")
    surf_df, surf = sensitivity("1d")
    surf_df.to_csv(T.REPORTS / "trend_sensitivity.csv", index=False)
    for ac, s in surf.items():
        print(f"  {ac:7s}: min {s['min']:+.2f}  median {s['median']:+.2f}  max {s['max']:+.2f}  "
              f"%positive {s['pct_positive']:.0%}  ({s['n_cfgs']} cfgs)")

    print(f"\n=== 2. PARAMETER WALK-FORWARD (pick best config on train, apply OOS) ({time.time()-t0:.0f}s) ===")
    pwf = []
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]:
        r = parameter_wf(sym, T.load_crypto_long, "1d", T.CRYPTO_TF["1d"], T.CC, True)
        if r:
            pwf.append({**r, "asset_class": "crypto"})
    for sym in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]:
        r = parameter_wf(sym, lambda s, _tf=None: T.load_equity(s), "1d", T.EQUITY_TF["1d"], T.EC, False)
        if r:
            pwf.append({**r, "asset_class": "equity"})
    pwf_df = pd.DataFrame(pwf)
    if len(pwf_df):
        for ac, g in pwf_df.groupby("asset_class"):
            print(f"  {ac:7s}: WF Sharpe {g['wf_sharpe'].median():+.2f}  vs in-sample peak {g['insample_peak'].median():+.2f}  "
                  f"(gap {g['insample_peak'].median()-g['wf_sharpe'].median():+.2f})")
        pwf_df.to_csv(T.REPORTS / "trend_parameter_wf.csv", index=False)

    print(f"\n=== 3. SELECTION WALK-FORWARD — window×cadence robustness ({time.time()-t0:.0f}s) ===")
    rets = sleeve_returns(args.entry, "asym", tfs)     # asym book sleeves (headline candidate)
    _, df = equal_risk(rets)
    full = sh(df.mean(axis=1))
    print(f"  full-sample equal-risk book Sharpe (no selection): {full:+.2f}")
    grid = {}
    for window in ("anchored", "rolling2y", "rolling3y"):
        for cadence in ("annual", "semiannual", "quarterly"):
            wf = selection_wf(df, window, cadence)
            grid[f"{window}/{cadence}"] = sh(wf)
    for k, v in grid.items():
        print(f"    {k:24s} OOS Sharpe {v:+.2f}")
    vals = np.array(list(grid.values()))
    print(f"  selection-WF Sharpe across 9 policies: min {vals.min():+.2f}  median {np.median(vals):+.2f}  "
          f"max {vals.max():+.2f}  std {vals.std():.2f}")

    (T.REPORTS / "trend_wfo_summary.json").write_text(json.dumps({
        "sensitivity": surf, "parameter_wf": pwf,
        "selection_wf": grid, "selection_wf_fullsample": full,
        "selection_wf_stats": {"min": round(float(vals.min()), 2), "median": round(float(np.median(vals)), 2),
                               "max": round(float(vals.max()), 2), "std": round(float(vals.std()), 2)},
    }, indent=2, default=float))
    print(f"\nwrote reports/trend/trend_wfo_summary.json + sensitivity/parameter_wf CSVs  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
