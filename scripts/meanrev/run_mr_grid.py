"""Close the mean-reversion grid: every asset class x every intraday timeframe x every method.

Daily (1d) is already covered by run_mr_full.py (crypto 50 / equity 50 / FX 25). This fills the
intraday rungs 5m / 15m / 1h / 4h for crypto (Binance bulk), equities and FX (Twelve Data Pro),
each tested three ways: single-asset revert-to-mean (individual + equal-weight basket), cross-sectional
reversal (dollar-neutral, vs a random-ranking placebo), and cointegration pairs basket.

Bounds (stated, not silent): intraday universes are 15 liquid names per class and pairs use the first
10 (45 pairs) — enough to judge the effect while keeping intraday cointegration tractable; the daily
sweep already ran the full 50/50/25. Vol-target sizes on the bar frequency (bar_ppy, measured from the
data); single-asset and cross-sectional Sharpe annualise per-bar at bar_ppy, pairs annualise the
daily-aggregated basket at 252/365. Revert-to-mean exit, walk-forward params, t+2, net of costs.

    python scripts/meanrev/run_mr_grid.py
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.data.twelvedata import load_bars  # noqa: E402
from src.metrics import summarise  # noqa: E402
from scripts.meanrev.audit_mr import mr_revert  # noqa: E402
from scripts.meanrev.run_mr_proper import coint, pairs_daily, wf_select  # noqa: E402
from scripts.meanrev.run_mr_universe import reversal  # noqa: E402

CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
          "LINKUSDT", "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT"]
FX = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X", "USDCAD=X", "USDCHF=X", "AUDNZD=X",
      "EURGBP=X", "EURCHF=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "GBPCHF=X"]
EQ = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM", "JNJ", "KO",
      "WMT", "V", "HD", "UNH", "PG"]

CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
EC = dict(commission_bps=1.0, half_spread_bps=2.0, impact_k=0.1, exec_lag=2)
FXC = dict(commission_bps=0.3, half_spread_bps=0.6, impact_k=0.05, exec_lag=2)
TD_TF = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}
# per-TF full available depth (probed): 1h/4h reach ~2019, 5m/15m ~2020, daily 2005
TD_START = {"5m": "2020-01-01", "15m": "2020-01-01", "1h": "2018-01-01", "4h": "2018-01-01"}
GRID = [(lb, ez, xz) for lb in (20, 50) for ez in (1.5, 2.5) for xz in (0.0, 0.5)]  # 8 params
TFS = ("1d", "4h", "1h", "15m", "5m")  # coarse -> fine: coarse cells return fast, heavy 5m last
N_FOLDS = 5
MIN_BARS = (N_FOLDS + 2) * 90   # 5-fold walk-forward -> 7 blocks; each must hold >= one rolling-90 window


def _fx_td(sym):
    b = sym[:-2]
    return f"{b[:3]}/{b[3:]}"


def bars_per_year(idx):
    span = (idx[-1] - idx[0]).total_seconds() / 86400 / 365.25
    return len(idx) / span if span > 0 else 252.0


def crypto_panel(tf):
    d = {}
    for s in CRYPTO:
        px = load_klines(s, tf, "2020-01", market="um")   # end defaults to now (all data)
        if len(px) > MIN_BARS:
            d[s] = px["close"]
    return d


def td_panel(tf, names, is_fx):
    d = {}
    for s in names:
        if tf == "1d":                                   # full daily depth (2005+), symbol-scoped cache
            px = load_equity_daily(s, start="2005-01-01")
        else:                                            # intraday from each TF's real floor; end -> now
            px = load_bars(_fx_td(s) if is_fx else s, TD_TF[tf], TD_START[tf])
        if len(px) > MIN_BARS:
            d[s] = px["close"]
    return d


def _mr(close, lb, ez, xz, bar_ppy, cost):
    pos = mr_revert(close, lb, ez, xz)
    bt = backtest(close, vol_target(pos, close, 0.15, bar_ppy), capital=500_000, funding=None, **cost)
    return bt["net_ret"].dropna()


def single_basket(pdict, bar_ppy, cost):
    series = {}
    for s, close in pdict.items():
        grid = {k: _mr(close, *k, bar_ppy, cost) for k in GRID}
        series[s] = wf_select(grid)
    wfs = pd.DataFrame(series)
    pos = int((wfs.apply(lambda c: summarise(c.dropna(), bar_ppy)["sharpe_ann"]) > 0).sum())
    return summarise(wfs.mean(axis=1).dropna(), bar_ppy), pos, len(series)


def pairs_basket(pdict, subset, bar_ppy, daily_ppy, cost_bps, form_frac=0.4):
    names = [s for s in subset if s in pdict]
    aligned = pd.DataFrame({s: pdict[s] for s in names}).dropna(how="all").ffill().dropna()
    if len(aligned) < MIN_BARS:
        return None, 0, 0
    form_end = aligned.index[int(len(aligned) * form_frac)]
    sel, tested = [], 0
    for a, b in itertools.combinations(names, 2):
        tested += 1
        sub = aligned[[a, b]].loc[:form_end]
        if len(sub) < 100:      # ADF needs ~100 points (coint returns p=1 below this)
            continue
        p, hl = coint(sub[a], sub[b])
        if p < 0.05 and 3 < hl < 500:
            sel.append((a, b))
    legs = []
    for a, b in sel:
        grid = {(lb, ez): pairs_daily(aligned[a], aligned[b], lb, ez, bar_ppy, cost_bps)
                for lb in (60, 90) for ez in (1.5, 2.0)}
        legs.append(wf_select(grid).loc[form_end:])
    if not legs:
        return None, tested, 0
    basket = pd.concat(legs, axis=1).mean(axis=1).dropna()
    return summarise(basket, daily_ppy), tested, len(sel)


def cell(cls, tf, pdict, cost, cbps, daily_ppy):
    panel = pd.DataFrame(pdict).dropna(how="all").ffill()
    bar_ppy = bars_per_year(panel.index)
    sb, pos, n = single_basket(pdict, bar_ppy, cost)
    wf, plac = reversal(panel, bar_ppy, cbps)
    pr, tested, nc = pairs_basket(pdict, list(pdict)[:10], bar_ppy, daily_ppy, cbps)
    pr_s = f"{pr['sharpe_ann']:+.2f} ({nc}/{tested} coint)" if pr is not None else f"none ({nc}/{tested})"
    print(f"  {cls:7s} {tf:4s} [~{bar_ppy:.0f} bars/yr, {n} names] | "
          f"single-basket {sb['sharpe_ann']:+.2f} ({pos}/{n}+, DD {sb['max_dd']:+.0%}) | "
          f"cross-sec {wf['sharpe_ann']:+.2f} (plac {plac:+.2f}) | pairs {pr_s}", flush=True)


def main():
    for tf in TFS:
        print(f"\n===== INTRADAY {tf} =====", flush=True)
        cp = crypto_panel(tf)
        if len(cp) >= 5:
            cell("crypto", tf, cp, CC, 5.0, 365)
        ep = td_panel(tf, EQ, False)
        if len(ep) >= 5:
            cell("equity", tf, ep, EC, 1.0, 252)
        fp = td_panel(tf, FX, True)
        if len(fp) >= 5:
            cell("fx", tf, fp, FXC, 0.9, 252)


if __name__ == "__main__":
    main()
