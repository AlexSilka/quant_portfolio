"""Sector-ETF pairs sleeve: standalone profile, then its contribution to the portfolio.

    python scripts/pairs/run_sector_pairs.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import REPORTS_DIR  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.sector_pairs import SECTOR_ETFS, pairs_basket  # noqa: E402


def main():
    panel = pd.DataFrame({s: load_equity_daily(s, start="2016-01-01")["close"]
                          for s in SECTOR_ETFS}).dropna(how="all").ffill()
    sleeve, sel, tested = pairs_basket(panel, ppy=252, cost_bps=2.0)   # 2 bp/leg — realistic for liquid ETFs
    s = summarise(sleeve, 252)
    print("=== sector-ETF pairs sleeve (walk-forward re-selection, net 2bp/leg, OOS) ===")
    print(f"  {sel:.1f} pairs/period across {tested} re-selections | {sleeve.index.min().date()} -> {sleeve.index.max().date()}")
    print(f"  Sharpe {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}")

    bp = pd.read_parquet(REPORTS_DIR / "master_book.parquet")
    book = bp["ret"] if "ret" in bp.columns else bp.iloc[:, 0]
    for x in (book, sleeve):
        x.index = pd.to_datetime(x.index, utc=True).tz_convert(None)
    d = pd.concat([(1 + book).resample("W").prod() - 1, (1 + sleeve).resample("W").prod() - 1],
                  axis=1, keys=["book", "pairs"]).dropna()

    def sh(x):
        return float(x.mean() / x.std(ddof=1) * np.sqrt(52))

    mu, cov = d.mean().values, d.cov().values
    w = np.linalg.solve(cov, mu)
    w = w / w.sum()                                                    # max-Sharpe (tangency) weights
    tan = sh((d * w).sum(axis=1))
    print(f"\n=== contribution to the portfolio (weekly, {len(d)}w overlap {d.index.min().date()}->{d.index.max().date()}) ===")
    print(f"  correlation book / pairs : {d['book'].corr(d['pairs']):+.2f}")
    print(f"  book alone               : Sharpe {sh(d['book']):+.2f}")
    print(f"  book + sector-pairs      : Sharpe {tan:+.2f}   (pairs weight {w[1]:.0%}, uplift {tan - sh(d['book']):+.2f})")


if __name__ == "__main__":
    main()
