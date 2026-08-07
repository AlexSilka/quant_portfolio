"""Cboe official volatility-index history (free, authoritative) — the implied side of the equity
and FX variance risk premium, straight from the index owner.

Source choice. Twelve Data Pro (the project's paid feed) does not carry Cboe index data on this
plan — VIX/VXN resolve as invalid symbols and EVZ returns "not authorized to access CXAC data" —
and yfinance is the scraping source this repo deliberately avoids for reproducibility. Cboe itself
publishes the definitive daily history as CSV, which is both free and canonical:

  https://cdn.cboe.com/api/global/us_indices/daily_prices/<SYMBOL>_History.csv

Coverage: VIX (1990+), VXN (2009+), EVZ (2009 - 2025-03, index discontinued). Close is in annualised
vol points (VIX 15.8 = 15.8%), the same units as Deribit DVOL. Cached as parquet, fetched once.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR  # noqa: E402

_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def load_cboe_vol(symbol: str, start: str | None = None, end: str | None = None,
                  cache_dir: str = RAW_DIR) -> pd.Series:
    """Return the daily Cboe vol-index close (tz-naive DatetimeIndex, vol points), cached as parquet.

    The CSV's first column is the date; the last column is the level (CLOSE for VIX/VXN, the single
    value column for EVZ), so `iloc[:, -1]` reads the close for every Cboe vol-index layout.
    """
    symbol = symbol.upper()
    cpath = Path(cache_dir) / "cboe" / f"{symbol}.parquet"
    if cpath.exists():
        s = pd.read_parquet(cpath)["close"]
    else:
        req = urllib.request.Request(_URL.format(symbol), headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=30).read().decode()
        df = pd.read_csv(io.StringIO(raw))
        idx = pd.to_datetime(df.iloc[:, 0]).dt.normalize()          # Cboe uses US month-first dates
        s = (pd.Series(df.iloc[:, -1].astype(float).to_numpy(), index=idx, name="close")
             .sort_index().dropna())
        cpath.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame("close").to_parquet(cpath)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    return s
