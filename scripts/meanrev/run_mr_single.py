"""Single-asset mean-reversion across the FULL downloaded universes (30 crypto coins AND 40
individual equities), done properly: revert-to-mean exit + walk-forward parameter selection, net
of costs. Also reports each asset's price stationarity (ADF p on log-price) — if the price itself
is not mean-reverting (p > 0.05, a random walk / trend), single-asset MR is doomed regardless of
parameters.

    python scripts/meanrev/run_mr_single.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller  # noqa: E402

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from scripts.meanrev.audit_mr import mr_revert  # noqa: E402
from scripts.meanrev.run_mr_proper import wf_select  # noqa: E402

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
         "LINKUSDT", "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT",
         "BCHUSDT", "FILUSDT", "TRXUSDT", "NEARUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
         "GRTUSDT", "FTMUSDT", "EOSUSDT", "THETAUSDT", "ALGOUSDT", "EGLDUSDT"]
EQUITY = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CSCO", "QCOM",
          "JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "COP", "SLB",
          "JNJ", "PFE", "MRK", "ABBV", "UNH", "KO", "PEP", "PG", "WMT", "COST",
          "HD", "LOW", "MCD", "SBUX", "V", "MA", "T", "VZ", "DIS", "NFLX"]
CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
EC = dict(commission_bps=1.0, half_spread_bps=2.0, impact_k=0.1, exec_lag=2)


def mr_daily(close, lb, ez, xz, ppy, costs):
    pos = mr_revert(close, lb, ez, xz)
    bt = backtest(close, vol_target(pos, close, 0.15, ppy), capital=500_000, funding=None, **costs)
    return ((1 + bt["net_ret"]).resample("D").prod() - 1).dropna()


def run_universe(title, load_fn, symbols, ppy, costs):
    print(f"\n===== {title} (single-asset revert-to-mean MR, walk-forward OOS) =====")
    rows, series = [], {}
    for s in symbols:
        close = load_fn(s)
        if close is None or len(close) < 300:
            continue
        adf_p = adfuller(np.log(close).dropna().to_numpy(), maxlag=1, autolag=None)[1]
        grid = {(lb, ez, xz): mr_daily(close, lb, ez, xz, ppy, costs)
                for lb in (10, 20, 50) for ez in (1.5, 2.0, 2.5) for xz in (0.0, 0.5)}
        wfs = wf_select(grid)
        series[s] = wfs
        wf = summarise(wfs, ppy)["sharpe_ann"]
        rows.append((s, adf_p, wf))
        print(f"  {s:10s}  ADF p={adf_p:.2f} ({'stationary' if adf_p < 0.05 else 'rw/trend'})"
              f"  ->  MR OOS Sharpe {wf:+.2f}")
    df = pd.DataFrame(rows, columns=["sym", "adf_p", "wf"])
    print(f"  --> {len(df)} assets | stationary price (ADF<0.05): {int((df.adf_p < 0.05).sum())} | "
          f"positive MR: {int((df.wf > 0).sum())} | median {df.wf.median():+.2f} | "
          f"best {df.wf.max():+.2f} ({df.loc[df.wf.idxmax(), 'sym']})")
    basket = pd.DataFrame(series).mean(axis=1).dropna()
    bs = summarise(basket, ppy)
    mean_corr = pd.DataFrame(series).corr().values
    mc = mean_corr[np.triu_indices_from(mean_corr, 1)]
    print(f"  EQUAL-WEIGHT BASKET of {len(series)} single-asset MR sleeves: Sharpe {bs['sharpe_ann']:+.2f}"
          f"  DD {bs['max_dd']:+.1%}  months+ {bs['months_in_profit']:.0%}  (mean pairwise corr {np.nanmean(mc):+.2f})")
    return df


def _eq(s):
    px = load_equity_daily(s, start="2012-01-01")
    return px["close"] if len(px) > 300 else None


def _cr(s):
    px = load_klines(s, "1d", "2021-01", market="um")
    return px["close"] if len(px) > 300 else None


def main():
    run_universe("CRYPTO 30 coins", _cr, COINS, 365, CC)
    run_universe("EQUITY 40 names", _eq, EQUITY, 252, EC)


if __name__ == "__main__":
    main()
