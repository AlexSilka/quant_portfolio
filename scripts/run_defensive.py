"""Defensive / flight-to-safety sleeve (defensive) — a stress-timed multi-haven basket.

The complement to mftrend: instead of trend-following, hold the assets that RALLY in risk-off — gold,
duration (TLT/IEF) and the safe-haven currencies (long JPY, long CHF vs USD) — and scale the whole
basket UP when equity implied vol is elevated (VIX trailing-percentile), toward flat in calm. The
vol-timing is the point: a static long-haven basket bleeds carry every calm month (and gold/bonds
carry is what fails the master book's months-in-profit target); concentrating the exposure in stress
raises the crisis payoff per unit of calm-month drag, giving the positive-skew profile a hedge needs.

Construction (point-in-time, 2-bar exec lag, ~2 bps turnover cost):
  havens     = GLD, TLT, IEF, long-JPY (short USDJPY), long-CHF (short USDCHF); inverse-vol to 15%,
               equal-risk averaged
  stress mult= VIX 2y trailing percentile mapped linearly to [0.15, 2.5] (calm -> 0.15x, panic -> 2.5x)
  sleeve     = basket * stress, vol-targeted to 15% annualised (so 15% is the FULL-SAMPLE vol; the
               risk is concentrated WHEN taken, not spread evenly)

    python scripts/run_defensive.py   ->  reports/lab/defensive_sleeve.parquet  (+ crash-window diagnostics)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
from src.metrics import summarise  # noqa: E402
from src.risk.sizing import vol_target_scale
from src.sleeves.xsect import held_turnover  # noqa: E402

RAW = ROOT / "data/raw/equity_td"
PPY = 252
HAVENS = {"GLD": +1, "TLT": +1, "IEF": +1, "USDJPY=X": -1, "USDCHF=X": -1}   # sign = long the safe asset
STRESS_FLOOR, STRESS_CAP, STRESS_LB = 0.15, 2.5, 504                          # VIX-percentile -> multiplier


def _px(t: str) -> pd.Series:
    df = pd.read_parquet(RAW / f"{t}_1d.parquet")
    s = (df["close"] if "close" in df.columns else df.iloc[:, 0]).astype(float)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _vix_stress(index) -> pd.Series:
    vix = pd.read_parquet(ROOT / "data/raw/cboe/VIX.parquet")
    v = (vix["close"] if "close" in vix.columns else vix.iloc[:, 0]).astype(float)
    v.index = pd.to_datetime(v.index)
    if v.index.tz is not None:
        v.index = v.index.tz_localize(None)
    pct = v.rolling(STRESS_LB).apply(lambda w: (w.iloc[-1] >= w).mean(), raw=False)   # PIT percentile
    mult = STRESS_FLOOR + (STRESS_CAP - STRESS_FLOOR) * pct
    return mult.shift(1).reindex(index).ffill().fillna(STRESS_FLOOR)


def build_defensive(cost_bps=2.0, exec_lag=2, target=0.15) -> pd.Series:
    legs, turns = [], []
    for a, sgn in HAVENS.items():
        r = _px(a).pct_change() * sgn
        vt = (target / np.sqrt(PPY) / r.rolling(40).std()).clip(upper=3.0).shift(exec_lag)
        legs.append((vt * r).rename(a))
        # a single levered leg drifts too: hold p of NAV through r and it becomes p(1+r)/(1+p*r)
        turns.append(held_turnover(vt.fillna(0.0).to_frame(a), r.fillna(0.0).to_frame(a))[a])
    basket = pd.concat(legs, axis=1).mean(axis=1)
    turn = pd.concat(turns, axis=1).mean(axis=1)
    basket = basket - turn.fillna(0.0) * cost_bps / 1e4
    timed = basket * _vix_stress(basket.index)
    lev = vol_target_scale(timed, target, PPY)
    return (timed * lev).dropna().rename("ret")


def main():
    d = build_defensive()
    d.to_frame().to_parquet(ROOT / "reports/lab/defensive_sleeve.parquet")
    s = summarise(d, PPY)
    mo = (1 + d).resample("ME").prod() - 1
    print("defensive sleeve (stress-timed haven: gold + duration + long JPY/CHF):")
    print(f"  Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"daily-skew {d.skew():+.2f}  monthly-skew {mo.skew():+.2f}  "
          f"{d.index.min().date()}..{d.index.max().date()}")
    print("  return in the book's worst crash windows (want POSITIVE):")
    for lab, a, b in [("2018-Q4", "2018-10", "2018-12"), ("2019-Q3", "2019-07", "2019-09"),
                      ("COVID 2020Q1", "2020-02", "2020-03"), ("2022 bear", "2022-01", "2022-12"),
                      ("2024-08 carry", "2024-08", "2024-08"), ("2026 chop", "2026-01", "2026-07")]:
        print(f"    {lab:14s}: {(1 + d.loc[a:b]).prod() - 1:+.1%}")
    print("RUN DEFENSIVE OK -> reports/lab/defensive_sleeve.parquet")


if __name__ == "__main__":
    main()
