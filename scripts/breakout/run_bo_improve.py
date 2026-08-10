"""Candidate improvements to the breakout construction, each measured against the same baseline.

The baseline is the shipped construction on the honest venue: Donchian-55 entry, chandelier ATR(3)
exit, long-short, core-10, 1d+4h, spot prices. Every candidate changes exactly one thing, and each
is scored on BOTH the matched perp-era window (2020+) and the full spot history (2017+) — a variant
that only helps on one of the two is a window artifact, not an improvement.

Candidates, and where each comes from:

  fresh       re-entry suppressed: a chandelier stop-out is final until price makes a *new* breakout.
              The shipped entry is a persistent side, so a stop-out is bought straight back on the
              next bar while the channel condition still holds — a defect found reading the code,
              not from the literature.
  adx20/adx25 arm entries only when Wilder's ADX clears 20 / 25. The standard CTA regime gate: below
              ~20 the market is ranging and breakouts fail more than they follow through.
  volwgt      position scaled by the bar's dollar-volume z-score. Volume-weighted time-series
              momentum is the one recent crypto TSMOM result that reports a large Sharpe lift
              (Huang, Sangiorgi & Urquhart 2024) — this is its cheapest breakout analogue.
  lb-*        Donchian lookback 20/34/55/89/144 — is 55 a plateau or a peak? A peak is a fitted
              parameter, and the shipped book has no evidence either way.
  k-*         chandelier width 2/3/4/5 ATR — same question for the exit.

    python scripts/breakout/run_bo_improve.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import bo_common as bo  # noqa: E402
from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.config import BREAKOUT_DIR, CRYPTO_PPY, OOS_START  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from scripts.breakout.run_bo_spot import CORE10, VENUE, _borrow, load, stats  # noqa: E402

TFS = ["1d", "4h"]
WINDOWS = {"2020+ (perp era)": ("2020-01-01", "2026-07-31"),
           "2017+ (full spot)": ("2017-08-01", "2026-07-31")}
VOLZ_CAP = 2.0          # clip the volume scaler so one print cannot dominate the position


def position(px: pd.DataFrame, variant: str) -> pd.Series | None:
    """Held +1/-1/0 (or scaled) position. `variant` is one knob, or knobs joined by '+'."""
    close, high, low = px["close"], px["high"], px["low"]
    knobs = variant.split("+")
    lookback = next((int(t[3:]) for t in knobs if t.startswith("lb-")), 55)
    k = next((float(t[2:]) for t in knobs if t.startswith("k-")), 3.0)

    side = bl.donchian_side(close, high, low, lookback)
    if "fresh" in knobs:
        side = bl.fresh_side(side)
    for t in knobs:
        if t.startswith("adx"):
            side = side.where(bl.adx(high, low, close, 14) >= float(t[3:]), 0.0)

    pos = bl.hold_atr_trailing(close, high, low, side, k, 14)
    if "volwgt" in knobs:
        qv = px["quote_volume"]
        z = ((qv - qv.rolling(20).mean()) / (qv.rolling(20).std() + 1e-12)).shift(1)
        pos = pos * (1.0 + z.clip(-VOLZ_CAP, VOLZ_CAP).fillna(0.0) / VOLZ_CAP).clip(0.0, 2.0)
    return pos if pos.abs().sum() > 0 else None


def sleeve(sym: str, tf: str, variant: str, lo: str, hi: str) -> pd.Series | None:
    px = load("spot", sym, tf, lo, hi)
    if px is None:
        return None
    pos = position(px, variant)
    if pos is None:
        return None
    v = VENUE["spot"]
    posv = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bt = backtest(px["close"], posv, capital=bo.CAP, funding=None,
                  adv=px["quote_volume"].rolling(20).median().shift(1), **v["costs"])
    net = bt["net_ret"] - _borrow(bt["position"], v["borrow_bps"], bo.CRYPTO_TF[tf])
    turn = float(bt["position"].diff().abs().resample("D").sum().mean() * CRYPTO_PPY)
    out = ((1 + net).resample("D").prod() - 1).rename("ret")
    out.attrs["turnover"] = turn
    return out


def book(variant: str, lo: str, hi: str) -> tuple[pd.Series, float]:
    cols, turns = {}, []
    for tf in TFS:
        for sym in CORE10:
            s = sleeve(sym, tf, variant, lo, hi)
            if s is not None:
                cols[f"{sym}_{tf}"] = s
                turns.append(s.attrs["turnover"])
    if not cols:
        return pd.Series(dtype=float), float("nan")
    return pd.DataFrame(cols).sort_index().mean(axis=1), float(np.mean(turns))


def main():
    variants = ["baseline", "fresh", "adx20", "adx25", "volwgt",
                "lb-20", "lb-34", "lb-89", "lb-144", "k-2.0", "k-4.0", "k-5.0",
                "lb-89+adx20"]
    print("=== BREAKOUT CONSTRUCTION — one-knob candidates vs the shipped baseline ===")
    print("spot, long-short, core-10, 1d+4h, Donchian-55 -> chandelier ATR(3) unless the knob says otherwise\n")

    rows, series = [], {}
    for wname, (lo, hi) in WINDOWS.items():
        base_sharpe = None
        print(f"--- {wname} ---")
        print(f"{'variant':<12}{'Sharpe':>8}{'MC-P5':>8}{'CAGR':>9}{'vol':>8}{'maxDD':>9}"
              f"{'mo+':>6}{'turn/yr':>9}{'vs base':>9}   OOS")
        for v in variants:
            port, turn = book(v, lo, hi)
            if port.dropna().empty:
                continue
            s = stats(port, v)
            o = stats(port[port.index >= OOS_START], v)
            if v == "baseline":
                base_sharpe = s["sharpe"]
            delta = s["sharpe"] - base_sharpe if base_sharpe is not None else 0.0
            mc = f"{s['mc_p5']:+.2f}" if np.isfinite(s.get("mc_p5", np.nan)) else "  — "
            print(f"{v:<12}{s['sharpe']:+8.2f}{mc:>8}{s['cagr']:+9.1%}{s['vol']:8.1%}"
                  f"{s['max_dd']:+9.1%}{s['months_in_profit']:6.0%}{turn:9.1f}{delta:+9.2f}"
                  f"   {o['sharpe']:+.2f}")
            rows.append({**s, "window": wname, "turnover": turn, "delta_sharpe": delta,
                         "oos_sharpe": o["sharpe"], "oos_cagr": o["cagr"]})
            series[f"{v} | {wname}"] = port
        print()

    # a candidate only counts if it lifts BOTH windows — one-window winners are window artifacts
    df = pd.DataFrame(rows).pivot(index="label", columns="window", values="delta_sharpe")
    both = df[(df > 0).all(axis=1)].sort_values(df.columns[-1], ascending=False)
    print("=== candidates that lift BOTH windows ===")
    print(both.to_string(float_format=lambda v: f"{v:+.2f}") if len(both)
          else "   none — every candidate helps on at most one window")

    pd.DataFrame(series).to_parquet(BREAKOUT_DIR / "bo_improve_series.parquet")
    (BREAKOUT_DIR / "bo_improve.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\nBO IMPROVE OK")


if __name__ == "__main__":
    main()
