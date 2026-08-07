"""US equity / ETF / FX daily bars via Twelve Data Pro (professional feed).

Yahoo scraping (yfinance) proved unreliable for a reproducible deliverable — besides
shallow intraday depth, `.history()` transiently 500s/404s while fetching complementary
fundamentals (e.g. TRAILING_PEG_RATIO), which crashes an unattended `make reproduce`.
Twelve Data Pro serves the whole daily book instead: deep history (2006+), split-adjusted,
no scraping fragility. Intraday bars come from the sibling loader in `twelvedata.py`.

Bars follow the shared contract: UTC DatetimeIndex, columns open/high/low/close/volume,
split-adjusted. FX carries no volume (column present, zero-filled).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR  # noqa: E402
from src.data.twelvedata import load_bars

_START = "2005-01-01"   # history start; loader end defaults to now (all data through today)


def _to_td_symbol(ticker: str) -> str:
    """Map a yfinance-style ticker to a Twelve Data symbol.

    FX uses the `=X` suffix in yfinance (`EURUSD=X`) and a slash pair in Twelve Data
    (`EUR/USD`); equities/ETFs are identical (`AAPL`, `SPY`).
    """
    if ticker.endswith("=X"):
        base = ticker[:-2]
        return f"{base[:3]}/{base[3:]}"
    return ticker


def load_equity_daily(ticker: str, start: str | None = None, end: str | None = None,
                      cache_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return split-adjusted daily OHLCV (UTC index, lowercase columns), cached as parquet."""
    cpath = Path(cache_dir) / "equity_td" / f"{ticker}_1d.parquet"
    if cpath.exists():
        df = pd.read_parquet(cpath)
    else:
        df = load_bars(_to_td_symbol(ticker), "1day", _START, cache_dir=cache_dir)
        if df.empty:
            return df
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df
