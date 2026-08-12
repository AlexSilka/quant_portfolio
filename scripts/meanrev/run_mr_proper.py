"""Proper mean-reversion, done to the literature: cross-sectional short-term reversal (large-cap,
dollar-neutral) and cointegration-selected pairs (spread MR with exit-to-mean + OU half-life),
each with WALK-FORWARD parameter selection and a placebo control. Rolling hedge ratios and z-score
stats use only past data (no look-ahead).

    python scripts/meanrev/run_mr_proper.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from statsmodels.tsa.stattools import adfuller  # noqa: E402

from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import xs_returns  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from scripts.meanrev.audit_mr2 import pos_from_z  # noqa: E402

PANEL = ["AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "META", "JNJ", "XOM", "WMT",
         "V", "PG", "HD", "BAC", "KO", "DIS", "CSCO", "INTC", "CVX", "PFE"]
SEED = 7


def wf_select(daily_by_param, n_folds=5):
    """Stitch OOS: on each expanding train block pick the best-Sharpe param, apply on the next block."""
    dates = next(iter(daily_by_param.values())).index
    bnd = [dates[min(int(i * len(dates) / (n_folds + 1)), len(dates) - 1)] for i in range(n_folds + 2)]
    oos = []
    for k in range(1, n_folds + 1):
        tr_end, te0, te1 = bnd[k], bnd[k], bnd[k + 1]
        best = max(daily_by_param,
                   key=lambda p: summarise(daily_by_param[p].loc[:tr_end], 252)["sharpe_ann"])
        oos.append(daily_by_param[best].loc[te0:te1])
    out = pd.concat(oos)
    return out[~out.index.duplicated(keep="first")]     # drop overlapping fold-boundary days


def _vt(net, ppy):
    return (vol_target_scale(net, 0.15, ppy) * net).dropna()


def reversal_daily(panel, lb, sign=-1.0, cost_bps=1.0):
    sig = sign * panel.pct_change(lb)          # sign=-1 -> long losers (reversal)
    gross, turn = xs_returns(panel, sig, top_frac=0.3)
    return _vt(gross - turn * cost_bps / 1e4, 252)


def coint(y, x):
    """Engle-Granger: OLS hedge ratio, ADF p-value on the residual, and OU half-life of the spread.

    Robust to degenerate pairs (near-identical series -> ~constant residual -> LAPACK error): such
    pairs return p=1 (not cointegrated in a tradeable sense; the spread has no variance to trade).
    """
    df = pd.concat([np.log(y), np.log(x)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 100:
        return 1.0, np.nan
    ly, lx = df.iloc[:, 0], df.iloc[:, 1]
    beta = np.polyfit(lx, ly, 1)[0]
    res = ly - beta * lx
    if res.std() < 1e-8:
        return 1.0, np.nan
    try:
        p = float(adfuller(res.to_numpy(), maxlag=1, autolag=None)[1])
    except Exception:
        return 1.0, np.nan
    dr, lag = res.diff().dropna(), res.shift(1).dropna()
    lam = np.polyfit(lag.loc[dr.index], dr, 1)[0]
    hl = -np.log(2) / lam if lam < 0 else np.nan
    return p, float(hl) if np.isfinite(hl) else np.nan


def pairs_daily(y, x, lookback, entry, ppy, cost_bps=5.0, exit_=0.5):
    y, x = y.align(x, join="inner")
    ry, rx = y.pct_change(), x.pct_change()
    beta = (ry.rolling(lookback).cov(rx) / (rx.rolling(lookback).var() + 1e-12)).shift(1)  # past-only
    spread = np.log(y) - beta * np.log(x)
    z = (spread - spread.rolling(lookback).mean()) / (spread.rolling(lookback).std() + 1e-12)
    pos = pd.Series(pos_from_z(z.to_numpy(), entry, exit_), index=y.index)
    pair_ret = ry - beta * rx
    p = (_voltarget_pos(pos, pair_ret, ppy)).shift(2).fillna(0.0)                 # t+2 execution
    net = p * pair_ret - p.diff().abs().fillna(0.0) * 2 * cost_bps / 1e4          # two legs
    return ((1 + net).resample("D").prod() - 1).dropna()


def _voltarget_pos(pos, ret, ppy):
    return pos * vol_target_scale(ret, 0.15, ppy)


def main():
    rng = np.random.default_rng(SEED)
    panel = pd.DataFrame({s: load_equity_daily(s, start="2012-01-01")["close"]
                          for s in PANEL}).dropna(how="all").ffill()

    print("=== Cross-sectional short-term reversal (large-cap, dollar-neutral) ===")
    rev = {lb: reversal_daily(panel, lb) for lb in (1, 2, 3, 5, 10)}
    wf = summarise(wf_select(rev), 252)
    # placebo: shuffle each day's ranking (destroy the reversal signal)
    noise = pd.DataFrame(rng.standard_normal(panel.shape), index=panel.index, columns=panel.columns)
    pg, pt = xs_returns(panel, noise, 0.3)
    plac = summarise(_vt(pg - pt / 1e4, 252), 252)
    print(f"  walk-forward OOS (honest):  Sharpe {wf['sharpe_ann']:+.2f}  DD {wf['max_dd']:+.1%}  months+ {wf['months_in_profit']:.0%}")
    print(f"  placebo (random ranking):   Sharpe {plac['sharpe_ann']:+.2f}")

    print("\n=== Cointegration-selected pairs (crypto spread MR) ===")
    for tf, ppy in (("1d", 365), ("4h", 6 * 365)):
        cl = {s: load_klines(s, tf, "2020-01", market="um")["close"]
              for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
        for a, b in (("ETHUSDT", "BTCUSDT"), ("SOLUSDT", "BTCUSDT"), ("SOLUSDT", "ETHUSDT")):
            p, hl = coint(cl[a], cl[b])
            grid = {(lb, ez): pairs_daily(cl[a], cl[b], lb, ez, ppy)
                    for lb in (30, 60, 90) for ez in (1.5, 2.0, 2.5)}
            wf = summarise(wf_select(grid), 365)["sharpe_ann"]
            flag = "cointegrated" if p < 0.05 else "not cointegrated"
            print(f"  {a[:3]}/{b[:3]} {tf}: ADF p={p:.3f} ({flag}), half-life~{hl:.0f} bars  "
                  f"-> walk-forward OOS Sharpe {wf:+.2f}")


if __name__ == "__main__":
    main()
