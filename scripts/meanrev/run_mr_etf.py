"""Mean-reversion on the universe most likely to work: ETF / treasury / sector / dual-class pairs,
where cointegration is economically grounded (shared driver) and more stable than single stocks.

Tests specific textbook pairs individually, then the full ETF-universe cointegration basket, with
formation-window pair selection (no look-ahead) and walk-forward parameter selection, net of costs.

    python scripts/meanrev/run_mr_etf.py
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from scripts.meanrev.run_mr_proper import coint, pairs_daily, wf_select  # noqa: E402
from scripts.meanrev.run_mr_universe import pairs_basket  # noqa: E402

ETFS = ["TLT", "IEF", "IEI", "SHY", "TLH", "GOVT",            # treasuries (duration ladder)
        "GLD", "IAU", "SLV", "GDX", "GDXJ",                   # precious metals + miners
        "XLE", "XOP", "USO", "VDE", "OIH",                    # energy / oil
        "XLF", "KRE", "KBE", "KIE",                           # financials
        "XLK", "VGT", "SMH", "SOXX", "QQQ",                   # tech / semis
        "SPY", "IVV", "VOO", "DIA",                           # broad S&P near-substitutes
        "XLU", "XLP", "XLV", "XLI", "XLB", "XLY", "XLRE",     # sectors
        "EEM", "VWO", "EFA", "VEA", "EWZ", "FXI",             # country / regional
        "GOOG", "GOOGL", "IWM", "IWD", "IWF"]                 # dual-class + style

TEXTBOOK = [("TLT", "IEF"), ("TLT", "IEI"), ("IEF", "IEI"), ("GLD", "GDX"), ("GLD", "IAU"),
            ("GLD", "SLV"), ("XLE", "XOP"), ("GOOG", "GOOGL"), ("SPY", "IVV"), ("SPY", "VOO"),
            ("SMH", "SOXX"), ("XLF", "KRE"), ("VWO", "EEM"), ("EFA", "VEA")]


def main():
    d = {}
    for s in ETFS:
        px = load_equity_daily(s, start="2007-01-01")
        if len(px) > 300:
            d[s] = px["close"]
    panel = pd.DataFrame(d).dropna(how="all").ffill()
    print(f"loaded {panel.shape[1]} ETFs, {panel.index.min().date()}..{panel.index.max().date()}")

    print("\n=== Textbook pairs (individual, cointegration + walk-forward OOS, net costs) ===")
    for a, b in TEXTBOOK:
        if a not in panel or b not in panel:
            continue
        p, hl = coint(panel[a], panel[b])
        grid = {(lb, ez): pairs_daily(panel[a], panel[b], lb, ez, 252, cost_bps=1.5)
                for lb in (30, 60, 90) for ez in (1.5, 2.0)}
        wf = summarise(wf_select(grid), 252)["sharpe_ann"]
        tag = "coint" if p < 0.05 else "  -  "
        print(f"  {a:5s}/{b:5s}: ADF p={p:.3f} [{tag}]  half-life~{hl:6.0f}d  ->  OOS Sharpe {wf:+.2f}")

    print("\n=== Full ETF-universe pairs BASKET (cointegration selected on formation, traded OOS) ===")
    res, tested, n = pairs_basket(panel, 252, 1.5)
    if res is None:
        print(f"  {n}/{tested} cointegrated — none tradeable")
    else:
        print(f"  {n}/{tested} pairs cointegrated -> basket walk-forward OOS Sharpe "
              f"{res['sharpe_ann']:+.2f}  DD {res['max_dd']:+.1%}  months+ {res['months_in_profit']:.0%}  "
              f"PSR>0 {res['psr_gt0']:.0%}")


if __name__ == "__main__":
    main()
