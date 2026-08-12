"""Proper mean-reversion on BROAD universes (you cannot judge MR from 3 assets):
  - Cross-sectional reversal on a wide cross-section (crypto ~30 perps, equities ~40 names).
  - Pairs stat-arb as a BASKET: scan all pairs, select cointegrated ones on a formation window
    only (no selection look-ahead), trade the basket out-of-sample with walk-forward params.

    python scripts/meanrev/run_mr_universe.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import xs_returns  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from scripts.meanrev.run_mr_proper import coint, pairs_daily, wf_select  # noqa: E402

SEED = 7
rng = np.random.default_rng(SEED)

CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
          "LINKUSDT", "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT",
          "BCHUSDT", "FILUSDT", "TRXUSDT", "NEARUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
          "GRTUSDT", "FTMUSDT", "EOSUSDT", "THETAUSDT", "ALGOUSDT", "EGLDUSDT"]
EQUITY = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CSCO", "QCOM",
          "JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "COP", "SLB",
          "JNJ", "PFE", "MRK", "ABBV", "UNH", "KO", "PEP", "PG", "WMT", "COST",
          "HD", "LOW", "MCD", "SBUX", "V", "MA", "T", "VZ", "DIS", "NFLX"]


def crypto_panel(tf):
    d = {}
    for s in CRYPTO:
        px = load_klines(s, tf, "2021-01", market="um")
        if len(px) > 300:
            d[s] = px["close"]
    return pd.DataFrame(d).dropna(how="all").ffill()


def equity_panel():
    d = {}
    for s in EQUITY:
        px = load_equity_daily(s, start="2014-01-01")
        if len(px) > 300:
            d[s] = px["close"]
    return pd.DataFrame(d).dropna(how="all").ffill()


def reversal(panel, ppy, cost_bps):
    def daily(lb):
        g, t = xs_returns(panel, -panel.pct_change(lb), top_frac=0.3)
        net = g - t * cost_bps / 1e4
        sc = vol_target_scale(net, 0.15, ppy)
        return (net * sc).dropna()
    wf = summarise(wf_select({lb: daily(lb) for lb in (1, 2, 3, 5)}), ppy)
    g, t = xs_returns(panel, pd.DataFrame(rng.standard_normal(panel.shape),
                                          index=panel.index, columns=panel.columns), 0.3)
    pn = g - t * cost_bps / 1e4
    plac = (pn * vol_target_scale(pn, 0.15, ppy)).dropna()
    return wf, summarise(plac, ppy)["sharpe_ann"]


def pairs_basket(panel, ppy, cost_bps, form_frac=0.4):
    syms = list(panel.columns)
    form_end = panel.index[int(len(panel) * form_frac)]
    coint_pairs, tested = [], 0
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            tested += 1
            sub = panel[[a, b]].loc[:form_end].dropna()
            if len(sub) < 100:
                continue
            p, hl = coint(sub[a], sub[b])
            if p < 0.05 and 3 < hl < 250:                 # cointegrated & tradeably fast
                coint_pairs.append((a, b))
    legs = []
    for a, b in coint_pairs:
        grid = {(lb, ez): pairs_daily(panel[a], panel[b], lb, ez, ppy, cost_bps)
                for lb in (60, 90) for ez in (1.5, 2.0)}
        legs.append(wf_select(grid).loc[form_end:])       # trade only after formation
    if not legs:
        return None, tested, 0
    basket = pd.concat(legs, axis=1).mean(axis=1).dropna()
    return summarise(basket, ppy), tested, len(coint_pairs)


def main():
    print("=== CROSS-SECTIONAL REVERSAL on broad universes (walk-forward OOS) ===")
    ce = equity_panel()
    we, pe = reversal(ce, 252, 1.0)
    print(f"  equities ({ce.shape[1]} names): walk-forward Sharpe {we['sharpe_ann']:+.2f}  "
          f"DD {we['max_dd']:+.1%}  (placebo {pe:+.2f})")
    cc = crypto_panel("1d")
    wc, pc = reversal(cc, 365, 5.0)
    print(f"  crypto   ({cc.shape[1]} perps): walk-forward Sharpe {wc['sharpe_ann']:+.2f}  "
          f"DD {wc['max_dd']:+.1%}  (placebo {pc:+.2f})")

    print("\n=== PAIRS STAT-ARB BASKET (cointegration selected on formation window, traded OOS) ===")
    for name, panel, ppy, cost in [("equities", ce, 252, 1.0), ("crypto 1d", cc, 365, 5.0)]:
        res, tested, n = pairs_basket(panel, ppy, cost)
        if res is None:
            print(f"  {name}: {n}/{tested} pairs cointegrated — none tradeable")
        else:
            print(f"  {name}: {n}/{tested} pairs cointegrated -> basket walk-forward OOS "
                  f"Sharpe {res['sharpe_ann']:+.2f}  DD {res['max_dd']:+.1%}  months+ {res['months_in_profit']:.0%}")


if __name__ == "__main__":
    main()
