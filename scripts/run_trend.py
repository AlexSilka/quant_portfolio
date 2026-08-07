"""Diagnostic: continuous trend-following held TO REVERSAL (not a fixed barrier).

Trend edge lives in the fat tail of big moves; a fixed-horizon exit throws it away. Here the
position is simply the sign of the EMA cross, held until it flips, net of costs + funding.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from src.backtest.engine import backtest  # noqa: E402
from src.config import CAPITAL_USD  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import momentum  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
EC = dict(commission_bps=1.0, half_spread_bps=2.0, impact_k=0.1, exec_lag=2)


def test(close, side, fund, ppy, costs, name):
    bt = backtest(close, side, capital=CAPITAL_USD, funding=fund, **costs)
    s = summarise(bt["net_ret"], ppy)
    mc = bootstrap_sharpe(bt["net_ret"], ppy, 500)
    to = bt["position"].diff().abs().sum()
    print(f"{name:24s} Sh {s['sharpe_ann']:+.2f}  MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} "
          f"P50 {mc.get('sharpe_p50', float('nan')):+.2f}]  DD {s['max_dd']:+.0%}  "
          f"months+ {s['months_in_profit']:.0%}  turnover {to:.0f}")


def main():
    print("=== CRYPTO trend held-to-reversal, net of costs + funding ===")
    for tf, ppy in [("4h", 6 * 365), ("1d", 365)]:
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            px = load_klines(sym, tf, "2020-01", market="um")
            fund = load_funding(sym, "2020-01")["last_funding_rate"]
            for f, sl in [(20, 100), (50, 200)]:
                test(px["close"], momentum.primary_side(px["close"], f, sl),
                     fund, ppy, CC, f"{sym}_{tf}_ema{f}/{sl}")
    print("\n=== EQUITY trend held-to-reversal, net of costs ===")
    for sym in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]:
        px = load_equity_daily(sym, start="2012-01-01")
        for f, sl in [(50, 200)]:
            test(px["close"], momentum.primary_side(px["close"], f, sl),
                 None, 252, EC, f"{sym}_1d_ema{f}/{sl}")


if __name__ == "__main__":
    main()
