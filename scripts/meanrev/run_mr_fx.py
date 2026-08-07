"""Mean-reversion on FX — a class genuinely more mean-reverting than stocks/crypto (currencies
range around fundamentals). FX data is freely available (yfinance FX tickers, deep daily history).

Tests single-asset MR on majors AND crosses (crosses like AUDNZD, EURGBP, EURCHF are the classic
cointegrated FX pairs), an equal-weight basket, and a cointegration pairs basket over the USD
majors. Proper: revert-to-mean exit, walk-forward params, net of (small, liquid) costs.

    python scripts/meanrev/run_mr_fx.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.data.equity import load_equity_daily  # noqa: E402
from scripts.meanrev.run_mr_single import run_universe  # noqa: E402
from scripts.meanrev.run_mr_universe import pairs_basket  # noqa: E402

FX = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X", "USDCAD=X", "USDCHF=X",
      "AUDNZD=X", "EURGBP=X", "EURCHF=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "GBPCHF=X"]
MAJORS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X", "USDCAD=X", "USDCHF=X",
          "USDNOK=X", "USDSEK=X", "USDSGD=X", "USDMXN=X", "USDZAR=X", "USDPLN=X", "USDHUF=X"]
FXCOST = dict(commission_bps=0.3, half_spread_bps=0.6, impact_k=0.05, exec_lag=2)  # majors: ~1bp round-trip


def load_fx(sym):
    px = load_equity_daily(sym, start="2010-01-01")
    return px["close"] if len(px) > 300 else None


def main():
    run_universe("FX 15 majors + crosses", load_fx, FX, 252, FXCOST)

    print("\n=== FX cointegration pairs BASKET (USD majors, formation-selected, traded OOS) ===")
    panel = pd.DataFrame({s: load_fx(s) for s in MAJORS}).dropna(how="all").ffill()
    res, tested, n = pairs_basket(panel, 252, 0.9)
    if res is None:
        print(f"  {n}/{tested} cointegrated — none tradeable")
    else:
        print(f"  {n}/{tested} pairs cointegrated -> basket walk-forward OOS Sharpe "
              f"{res['sharpe_ann']:+.2f}  DD {res['max_dd']:+.1%}  months+ {res['months_in_profit']:.0%}")


if __name__ == "__main__":
    main()
