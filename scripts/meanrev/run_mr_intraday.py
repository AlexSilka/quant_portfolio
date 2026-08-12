"""Intraday mean-reversion on EQUITIES using Twelve Data Pro (15m / 1h from ~2020) — the data
yfinance cannot reach and where mean-reversion tends to be stronger than on daily bars.

Single-asset MR (revert-to-mean, walk-forward) individual + basket, and intraday cross-sectional
reversal, net of costs. P&L is aggregated to daily and annualised at 252.

    python scripts/meanrev/run_mr_intraday.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.data.twelvedata import load_bars  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import xs_returns  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from scripts.meanrev.audit_mr import mr_revert  # noqa: E402
from scripts.meanrev.run_mr_proper import wf_select  # noqa: E402

EQ = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "JNJ", "KO",
      "WMT", "V", "HD", "UNH", "PG", "BAC", "PFE", "CSCO", "INTC", "T"]
TF = [("1h", 6.5 * 252), ("15min", 26 * 252)]
EC = dict(commission_bps=1.0, half_spread_bps=1.5, impact_k=0.1, exec_lag=2)


def load(sym, tf):
    df = load_bars(sym, tf, "2020-01-01")
    return df["close"] if len(df) > 800 else None


def mr_daily(close, lb, ez, xz, bar_ppy):
    pos = mr_revert(close, lb, ez, xz)
    bt = backtest(close, vol_target(pos, close, 0.15, bar_ppy), capital=500_000, funding=None, **EC)
    return ((1 + bt["net_ret"]).resample("D").prod() - 1).dropna()


def main():
    for tf, bar_ppy in TF:
        print(f"\n===== EQUITY {tf} (Twelve Data Pro, ~2020+) =====")
        series, panel = {}, {}
        for s in EQ:
            close = load(s, tf)
            if close is None:
                continue
            panel[s] = close
            grid = {(lb, ez, xz): mr_daily(close, lb, ez, xz, bar_ppy)
                    for lb in (10, 20, 50) for ez in (1.5, 2.0, 2.5) for xz in (0.0, 0.5)}
            series[s] = wf_select(grid)
            print(f"  {s:6s}  single-asset MR OOS Sharpe {summarise(series[s], 252)['sharpe_ann']:+.2f}")
        if not series:
            continue
        wfs = pd.DataFrame(series)
        pos_ct = int((wfs.apply(lambda c: summarise(c, 252)["sharpe_ann"]) > 0).sum())
        basket = summarise(wfs.mean(axis=1).dropna(), 252)
        print(f"  --> {len(series)} names | positive: {pos_ct} | "
              f"EQUAL-WEIGHT BASKET Sharpe {basket['sharpe_ann']:+.2f}  DD {basket['max_dd']:+.1%}")

        pan = pd.DataFrame(panel).dropna(how="all").ffill()

        def rev(lb):
            g, t = xs_returns(pan, -pan.pct_change(lb), top_frac=0.3)
            net = g - t * EC["commission_bps"] / 1e4
            sc = vol_target_scale(net, 0.15, bar_ppy, lookback=500)
            return ((1 + net * sc).resample("D").prod() - 1).dropna()

        wf = summarise(wf_select({lb: rev(lb) for lb in (2, 4, 8, 16)}), 252)
        print(f"  CROSS-SECTIONAL REVERSAL (intraday): walk-forward Sharpe {wf['sharpe_ann']:+.2f}  "
              f"DD {wf['max_dd']:+.1%}")


if __name__ == "__main__":
    main()
