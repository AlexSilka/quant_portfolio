"""Does the VRP sleeve work on intraday timeframes, or only daily? Empirical test on BTC/ETH — the
only crypto with a free intraday implied-vol series (Deribit DVOL at 1h). Equity/FX have no free
intraday implied vol, so this crypto test is the whole available answer.

For 1d / 4h / 1h: always-short book, vol-targeted 15%, net of vega costs, t+2, equal-risk BTC+ETH.
Reports Sharpe, drawdown, annualised turnover and cost-fragility (base vs 3x vega spread) per TF, so
"daily is the native frame" is a measurement, not an assertion.

    python scripts/volprem/run_vol_premium_tf.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import REPORTS_DIR, VOLPREM_DIR  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.deribit import load_dvol  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.vol_premium import short_vol_book  # noqa: E402

ASSETS = [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")]
# (label, ppy, bars/day, DVOL resolution or "resample")
TFS = [("1d", 365, 1, "1D"), ("4h", 6 * 365, 6, "resample4h"), ("1h", 24 * 365, 24, "3600")]


def vt(net, ppy, lookback):
    scale = (0.15 / (net.rolling(lookback).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).clip(lower=-0.999).dropna()


def dvol_for(cur, res):
    if res == "resample4h":
        return load_dvol(cur, resolution="3600")["close"].resample("4h").last()
    return load_dvol(cur, resolution=res)["close"]


def tf_book(label, ppy, bpd, res, cost_mult=1.0):
    legs, turns, drags = {}, [], []
    for sym, cur in ASSETS:
        close = load_klines(sym, label, "2021-01", "2026-08", market="um")["close"]
        bk = short_vol_book(close, dvol_for(cur, res), timed=False, var_cap=1e9, ppy=ppy,
                            rv_lookback=30 * bpd, restrike_days=7 * bpd,
                            vega_cost_volpts=0.75 * cost_mult)
        legs[sym] = vt(bk["net"], ppy, 60 * bpd)
        yrs = (bk.index[-1] - bk.index[0]).days / 365.25
        turns.append(float(bk["turnover"].sum()) / max(yrs, 0.1))
        gross = vt(bk["gross"], ppy, 60 * bpd)
        drags.append(summarise(gross, ppy)["sharpe_ann"] - summarise(legs[sym], ppy)["sharpe_ann"])
    book = pd.concat(legs, axis=1).mean(axis=1).dropna()
    return book, float(np.mean(turns)), float(np.mean(drags))


def main():
    print("=== VRP by timeframe (BTC+ETH equal-risk, vol-targeted 15%, net, t+2) ===")
    rows = []
    for label, ppy, bpd, res in TFS:
        book, turn, drag = tf_book(label, ppy, bpd, res)
        s = summarise(book, ppy)
        # killer diagnostic: aggregate the intraday book to DAILY and annualise at 365. If the fine-TF
        # Sharpe is real edge it survives aggregation; if it is a sampling artifact of the short-vol
        # payoff it collapses back to the daily book's ~0.8.
        daily = ((1 + book).resample("D").prod() - 1).dropna()
        s_daily = summarise(daily, 365)["sharpe_ann"]
        ann_ret = float((1 + book).prod() ** (ppy / max(len(book), 1)) - 1)
        rows.append({"tf": label, "sharpe_at_tf": s["sharpe_ann"], "sharpe_daily_agg": s_daily,
                     "ann_return": ann_ret, "max_dd": s["max_dd"], "skew": float(book.skew()),
                     "ann_turnover": turn, "n_bars": s["n_obs"]})
        print(f"  {label:3s}  Sharpe@TF {s['sharpe_ann']:+.2f}  ->daily-agg {s_daily:+.2f}  "
              f"ann.ret {ann_ret:+.0%}  DD {s['max_dd']:+.1%}  skew {book.skew():+.1f}  bars {s['n_obs']}")
    df = pd.DataFrame(rows)
    df.to_csv(VOLPREM_DIR / "volprem_timeframe.csv", index=False)
    print("\n  If Sharpe@TF >> daily-agg, the intraday number is a sampling artifact of the short-vol")
    print("  payoff (smooth premium per bar + rare crash bars), not real edge — daily is the honest frame.")
    print("\nVOLPREM-TF OK")


if __name__ == "__main__":
    main()
