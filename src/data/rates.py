"""Short-term interest rates from FRED (keyless CSV via the fredgraph endpoint — no API key).

Used for FX carry: the 3-month interbank rate is the currency's "carry", the FX analogue of a
crypto perp's funding rate. Monthly series, cached to parquet, lagged one month at use to stay
point-in-time (published with a delay; never trade on a month-average not yet observable).
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR  # noqa: E402

# 3-month or 90-day interbank rates (OECD MEI: IR3TIB01<ISO>M156N), per currency
FRED_3M = {
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N", "CHF": "IR3TIB01CHM156N", "MXN": "IR3TIB01MXM156N",
    "NOK": "IR3TIB01NOM156N", "SEK": "IR3TIB01SEM156N", "ZAR": "IR3TIB01ZAM156N",
}


def _fred_csv(series_id: str, cache_dir: str = RAW_DIR / "rates") -> pd.Series:
    cache = Path(cache_dir) / f"{series_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["val"]
    raw = urllib.request.urlopen(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "val"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna().set_index("date")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df["val"]


def short_rates(currencies: list[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame [date x currency] of 3-month rates in percent (monthly, UTC index)."""
    ccys = currencies or list(FRED_3M)
    out = {c: _fred_csv(FRED_3M[c]) for c in ccys if c in FRED_3M}
    return pd.DataFrame(out).sort_index()
