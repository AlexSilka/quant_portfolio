"""Deribit DVOL implied-volatility index (free public API) — the implied side of the crypto
variance risk premium.

DVOL is Deribit's 30-day forward implied-vol index (their VIX analog), published for BTC and ETH
from 2021-03-24. Paired with realised vol from the perp bars, it gives the variance-risk-premium
signal for the short-vol sleeve. No API key required.

  GET /api/v2/public/get_volatility_index_data
      ?currency=BTC&start_timestamp=<ms>&end_timestamp=<ms>&resolution=1D
  -> result.data = [[ts_ms, open, high, low, close], ...]   close in annualised vol points
     (34.66 = 34.66% implied vol). The endpoint caps each response at 1000 rows, so history is
     fetched in <1000-row windows and de-duplicated.

Bars follow the shared contract: UTC DatetimeIndex, OHLC in vol points. Cached as parquet, fetched
once (delete the cache to refresh) so `make` stays reproducible and offline after the first run.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR  # noqa: E402

_BASE = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_START = "2021-03-24"          # first published DVOL day (BTC & ETH)
_DAY_MS = 86_400_000
# resolution -> bars/day, for sizing each request under the 1000-row response cap
_BARS_PER_DAY = {"1D": 1, "43200": 2, "3600": 24, "60": 1440, "1": 86400}


def _fetch_window(currency: str, start_ms: int, end_ms: int, resolution: str) -> list:
    url = (f"{_BASE}?currency={currency}&start_timestamp={start_ms}"
           f"&end_timestamp={end_ms}&resolution={resolution}")
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["result"]["data"]


def load_dvol(currency: str = "BTC", start: str | None = None, end: str | None = None,
              resolution: str = "1D", cache_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return DVOL bars (UTC index, columns open/high/low/close in vol points), cached as parquet.

    resolution: "1D" (daily), "3600" (1h), "43200" (12h), "60" (1min). Intraday history is fetched in
    windows sized to stay under the endpoint's 1000-row response cap.
    """
    currency = currency.upper()
    tag = "1d" if resolution == "1D" else resolution
    cpath = Path(cache_dir) / "deribit" / f"DVOL_{currency}_{tag}.parquet"
    if cpath.exists():
        df = pd.read_parquet(cpath)
    else:
        window_days = max(1, 900 // _BARS_PER_DAY.get(resolution, 24))
        start_ms = int(pd.Timestamp(_START, tz="UTC").timestamp() * 1000)
        end_ms = int(time.time() * 1000)
        rows: list = []
        w = start_ms
        while w < end_ms:
            we = min(w + window_days * _DAY_MS, end_ms)
            rows += _fetch_window(currency, w, we, resolution)
            w = we + (_DAY_MS if resolution == "1D" else 3_600_000)
        df = (pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
              .drop_duplicates("ts").sort_values("ts"))
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["open", "high", "low", "close"]].astype(float)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df
