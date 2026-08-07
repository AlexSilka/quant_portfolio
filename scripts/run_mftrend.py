"""Broad managed-futures / CTA trend sleeve (mftrend) — the canonical crisis-alpha / long-gamma
diversifier, built on the widest liquid proxy universe with pre-2015 history.

Motivation: the master book is six short-gamma risk-premia (short-vol / carry / momentum / breakout /
trend) that crash TOGETHER in risk-off, so its worst months and multi-month losing streaks are
correlated crashes. Time-series momentum across many markets is the textbook long-gamma offset
(Hurst-Ooi-Pedersen "A Century of Evidence on Trend Following"; Moskowitz-Ooi-Pedersen TSMOM): it is
*up* in sustained crashes (shorts equities, rides the bond/gold flight, shorts a rate-hike bond
selloff) and carries positive skew. This is the bigger, class-balanced version of the existing 11-ETF
crisis sleeve — 32 markets across four asset classes.

Construction (all point-in-time, signals executed with a 2-bar lag, ~2 bps turnover cost):
  universe   = 8 equity-index ETFs + 8 rate/bond ETFs + 8 commodity ETFs + 8 FX pairs (all from <=2011)
  signal     = multi-horizon TSMOM sign blend over 63/126/252 trading days (1/3/12-month, AQR-standard)
  sizing     = per-asset inverse-vol to 15% ann, equal-risk average WITHIN each class,
               equal weight ACROSS the four classes (balanced risk, so no class dominates by count)
  sleeve     = the 4-class blend, vol-targeted to 15% annualised

    python scripts/run_mftrend.py   ->  reports/lab/mftrend_sleeve.parquet  (+ crash-window diagnostics)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from src.metrics import summarise  # noqa: E402

RAW = ROOT / "data/raw/equity_td"
PPY = 252
HORIZONS = (63, 126, 252)          # 1/3/12-month TSMOM blend (a-priori literature horizons, not tuned)
# Class-balanced universe — every ticker has history from <=2011 (warmup before the 2016-08 window).
CLASSES = {
    "eq":     ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWZ", "FXI"],           # equity indices + intl
    "rates":  ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "TIP"],           # duration + credit
    "commod": ["GLD", "SLV", "USO", "DBC", "DBA", "UNG", "GDX", "CPER"],          # metals/energy/ags
    "fx":     ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X",         # G10 + EM vs USD
               "USDCHF=X", "USDCAD=X", "USDMXN=X"],
}


def _px(t: str) -> pd.Series:
    df = pd.read_parquet(RAW / f"{t}_1d.parquet")
    s = (df["close"] if "close" in df.columns else df.iloc[:, 0]).astype(float)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _class_trend(tickers, cost_bps=2.0, exec_lag=2, target=0.15) -> pd.Series:
    """Equal-risk TSMOM within one asset class: per-asset vol-targeted sign blend, cost, average."""
    px = pd.DataFrame({t: _px(t) for t in tickers}).sort_index()
    r = px.pct_change()
    vol = r.rolling(40).std()
    sig = sum(np.sign(px / px.shift(h) - 1.0) for h in HORIZONS) / len(HORIZONS)
    pos = (sig * (target / np.sqrt(PPY) / vol).clip(upper=3.0)).shift(exec_lag)
    gross = (pos * r).mean(axis=1)
    turn = pos.diff().abs().mean(axis=1)
    return gross - turn.fillna(0.0) * cost_bps / 1e4


def build_mftrend(target=0.15) -> pd.Series:
    class_rets = {cls: _class_trend(tk) for cls, tk in CLASSES.items()}
    book = pd.DataFrame(class_rets).mean(axis=1)                 # equal weight across the 4 classes
    lev = (target / (book.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (book * lev).dropna().rename("ret")


def main():
    mft = build_mftrend()
    mft.to_frame().to_parquet(ROOT / "reports/lab/mftrend_sleeve.parquet")
    s = summarise(mft, PPY)
    mo = (1 + mft).resample("ME").prod() - 1
    print(f"mftrend sleeve (broad managed-futures, 32 markets / 4 classes, 63/126/252 TSMOM):")
    print(f"  Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"daily-skew {mft.skew():+.2f}  monthly-skew {mo.skew():+.2f}  "
          f"{mft.index.min().date()}..{mft.index.max().date()}")
    print("  return in the book's worst crash windows (the hedge value — want POSITIVE):")
    for lab, a, b in [("2018-Q4", "2018-10", "2018-12"), ("2019-Q3", "2019-07", "2019-09"),
                      ("COVID 2020Q1", "2020-02", "2020-03"), ("2022 bear", "2022-01", "2022-12")]:
        print(f"    {lab:14s}: {(1 + mft.loc[a:b]).prod() - 1:+.1%}")
    print("RUN MFTREND OK -> reports/lab/mftrend_sleeve.parquet")


if __name__ == "__main__":
    main()
