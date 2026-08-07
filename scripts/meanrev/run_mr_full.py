"""FULL mean-reversion sweep on EXPANDED universes (crypto 50, equities 50, FX 25), every way:
single-asset MR (individual + equal-weight basket), cross-sectional reversal, and cointegration
pairs basket. Proper throughout: revert-to-mean exit, walk-forward params, formation-window pair
selection, placebo controls, net of costs.

    python scripts/meanrev/run_mr_full.py
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from scripts.meanrev.run_mr_single import run_universe  # noqa: E402
from scripts.meanrev.run_mr_universe import pairs_basket, reversal  # noqa: E402

CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
          "LINKUSDT", "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT",
          "BCHUSDT", "FILUSDT", "TRXUSDT", "NEARUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
          "GRTUSDT", "FTMUSDT", "EOSUSDT", "THETAUSDT", "ALGOUSDT", "EGLDUSDT", "APEUSDT", "GALAUSDT",
          "CHZUSDT", "CRVUSDT", "SNXUSDT", "COMPUSDT", "MKRUSDT", "ENJUSDT", "ZILUSDT", "ONEUSDT",
          "IOTAUSDT", "KAVAUSDT", "RUNEUSDT", "DYDXUSDT", "APTUSDT", "OPUSDT", "INJUSDT", "LDOUSDT",
          "IMXUSDT", "STXUSDT"]
EQUITY = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CSCO", "QCOM",
          "JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "COP", "SLB",
          "JNJ", "PFE", "MRK", "ABBV", "UNH", "KO", "PEP", "PG", "WMT", "COST",
          "HD", "LOW", "MCD", "SBUX", "V", "MA", "T", "VZ", "DIS", "NFLX",
          "ORCL", "IBM", "TXN", "ADBE", "CRM", "NKE", "CAT", "BA", "GE", "HON"]
FX = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X", "USDCAD=X", "USDCHF=X",
      "AUDNZD=X", "EURGBP=X", "EURCHF=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "GBPCHF=X",
      "USDNOK=X", "USDSEK=X", "USDSGD=X", "USDMXN=X", "USDZAR=X", "NZDJPY=X", "CADJPY=X", "CHFJPY=X",
      "EURCAD=X", "GBPAUD=X"]

CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
EC = dict(commission_bps=1.0, half_spread_bps=2.0, impact_k=0.1, exec_lag=2)
FXC = dict(commission_bps=0.3, half_spread_bps=0.6, impact_k=0.05, exec_lag=2)


def cr(s):
    px = load_klines(s, "1d", "2020-01", market="um")
    return px["close"] if len(px) > 300 else None


def eq(s):
    px = load_equity_daily(s, start="2012-01-01")
    return px["close"] if len(px) > 300 else None


def fx(s):
    px = load_equity_daily(s, start="2010-01-01")
    return px["close"] if len(px) > 300 else None


def do(name, load_fn, symbols, ppy, cdict, cbps):
    print(f"\n############## {name} ({len(symbols)} assets) ##############")
    run_universe(name, load_fn, symbols, ppy, cdict)
    panel = pd.DataFrame({s: load_fn(s) for s in symbols}).dropna(how="all").ffill()
    wf, plac = reversal(panel, ppy, cbps)
    print(f"  CROSS-SECTIONAL REVERSAL: walk-forward {wf['sharpe_ann']:+.2f}  (placebo {plac:+.2f})")
    res, tested, n = pairs_basket(panel, ppy, cbps)
    if res is None:
        print(f"  PAIRS BASKET: {n}/{tested} cointegrated — none tradeable")
    else:
        print(f"  PAIRS BASKET: {n}/{tested} cointegrated -> Sharpe {res['sharpe_ann']:+.2f}  DD {res['max_dd']:+.1%}")


def main():
    do("CRYPTO", cr, CRYPTO, 365, CC, 5.0)
    do("EQUITY", eq, EQUITY, 252, EC, 1.0)
    do("FX", fx, FX, 252, FXC, 0.9)


if __name__ == "__main__":
    main()
