"""Binance public bulk-data loader (data.binance.vision).

Downloads monthly kline / funding-rate archives, handles the real-world quirks of
the dumps, and returns clean pandas frames cached as parquet per month.

Quirks handled (verified against live archives, 2026-08):
- CSV header row is inconsistent: spot dumps are headerless (even in 2026), some
  USD-M futures dumps carry a header. The first line is sniffed.
- Spot timestamps switched to microseconds from 2025-01; futures stay in
  milliseconds. The epoch unit is detected from magnitude.
- USD-M kline archives begin 2019-12-31 platform-wide, so BTC/ETH have a post-listing
  gap only `trades` fill; missing months are skipped with a warning rather than failing.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR  # noqa: E402

BASE = "https://data.binance.vision/data"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
SEGMENTS = {"spot": "spot", "um": "futures/um", "cm": "futures/cm"}

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
_NUMERIC = ["open", "high", "low", "close", "volume", "quote_volume",
            "taker_buy_volume", "taker_buy_quote_volume"]


def _months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start=start, end=end, freq="M")]


def _recent(tag: str) -> bool:
    """True if a monthly dump for `tag` may simply not be published yet (last ~2 months)."""
    return tag >= (pd.Timestamp.now() - pd.DateOffset(months=2)).strftime("%Y-%m")


def _earliest_month(prefix: str, cache_dir: str) -> str | None:
    """Earliest YYYY-MM actually available under a data.binance.vision prefix, via the S3 listing.

    Queried once per symbol/interval and cached to disk, so we request only months that exist
    (a symbol listed 2020-09 is never probed back to 2020-01). Returns None if the listing fails,
    in which case the caller falls back to probing.
    """
    cache = Path(cache_dir) / "_earliest" / (prefix.strip("/").replace("/", "_") + ".txt")
    if cache.exists():
        return cache.read_text().strip() or None
    result = ""
    try:
        q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "delimiter": "/",
                                    "max-keys": "1000"})
        raw = urllib.request.urlopen(f"{S3}?{q}", timeout=20).read()
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        months = sorted("-".join(k.text.split("/")[-1][:-4].split("-")[-2:])
                        for k in ET.fromstring(raw).iter(ns + "Key")
                        if k.text and k.text.endswith(".zip"))
        result = months[0] if months else ""
    except Exception:
        result = ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(result)
    return result or None


def _fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _read_zip_csv(raw: bytes, names: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    data = z.read(z.namelist()[0])
    first_char = data.lstrip()[:1].decode("ascii", "ignore")
    header = None if first_char.isdigit() else 0  # sniff: data row starts with a digit
    return pd.read_csv(io.BytesIO(data), header=header, names=names)


def _epoch_to_utc(s: pd.Series) -> pd.Series:
    unit = "us" if s.max() >= 1e15 else "ms"  # spot >= 2025-01 is microseconds
    return pd.to_datetime(s, unit=unit, utc=True)


def load_klines(symbol: str, interval: str, start: str, end: str | None = None,
                market: str = "um", cache_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return OHLCV+flow bars indexed by UTC open_time for [start, end] (monthly granularity).

    `end` defaults to the current month, i.e. all available data through now — never a fixed cutoff.
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m")
    seg = SEGMENTS[market]
    earliest = _earliest_month(f"data/{seg}/monthly/klines/{symbol}/{interval}/", cache_dir)
    frames = []
    for tag in _months(start, end):
        if earliest and tag < earliest:
            continue  # symbol not listed yet this month — not in the archive, so never requested
        cpath = Path(cache_dir) / seg / "klines" / symbol / interval / f"{tag}.parquet"
        mpath = cpath.with_suffix(".missing")
        if cpath.exists():
            frames.append(pd.read_parquet(cpath))
            continue
        if mpath.exists():
            continue  # month recorded as absent (pre-listing) — skip without re-probing the network
        url = f"{BASE}/{seg}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{tag}.zip"
        raw = _fetch(url)
        if raw is None:
            mpath.parent.mkdir(parents=True, exist_ok=True)
            if _recent(tag):
                print(f"[warn] {market} klines {symbol} {interval} {tag}: not published yet (404)")
            else:
                mpath.touch()  # pre-listing / permanently absent — record and stop re-probing
            continue
        df = _read_zip_csv(raw, KLINE_COLS)
        df["open_time"] = _epoch_to_utc(df["open_time"])
        for c in _NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop(columns=["ignore", "close_time"]).set_index("open_time")
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]


def load_funding(symbol: str, start: str, end: str | None = None,
                 cache_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return USD-M funding rates indexed by UTC settlement time (monthly-only archives).

    `end` defaults to the current month (all data through now).
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m")
    earliest = _earliest_month(f"data/futures/um/monthly/fundingRate/{symbol}/", cache_dir)
    frames = []
    for tag in _months(start, end):
        if earliest and tag < earliest:
            continue
        cpath = Path(cache_dir) / "futures/um" / "fundingRate" / symbol / f"{tag}.parquet"
        mpath = cpath.with_suffix(".missing")
        if cpath.exists():
            frames.append(pd.read_parquet(cpath))
            continue
        if mpath.exists():
            continue
        url = f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{tag}.zip"
        raw = _fetch(url)
        if raw is None:
            mpath.parent.mkdir(parents=True, exist_ok=True)
            if _recent(tag):
                print(f"[warn] funding {symbol} {tag}: not published yet (404)")
            else:
                mpath.touch()  # pre-listing / permanently absent — record and stop re-probing
            continue
        df = _read_zip_csv(raw, FUNDING_COLS)
        df["calc_time"] = _epoch_to_utc(df["calc_time"])
        df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
        df = df.set_index("calc_time")
        cpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]
