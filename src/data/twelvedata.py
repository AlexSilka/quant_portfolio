"""Twelve Data loader — professional equity bars (intraday + daily).

Provisioned via a Pro API key (config `equity.intraday_source: twelvedata`; key read
from env `TWELVEDATA_API_KEY` or the gitignored `.env`). Probed depth (2026-08, this Pro
key — actually queried, not assumed): intraday is shallow and interval-dependent — US-equity
1h reaches ~2019, equity 5m/15m and FX intraday ~2020, nothing intraday before that; daily
reaches 2005. Crypto is available too, but Binance bulk is preferred there (deeper history,
funding, no credit limits), so this loader serves equities and FX.

Bars follow the shared contract: UTC DatetimeIndex, columns open/high/low/close/volume.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR  # noqa: E402

_BASE = "https://api.twelvedata.com/time_series"
_DIV = "https://api.twelvedata.com/dividends"
_MAX = 5000  # max bars per request


def _api_key() -> str:
    key = os.environ.get("TWELVEDATA_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TWELVEDATA_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("TWELVEDATA_API_KEY not set (env or .env)")


def _request(**params) -> dict:
    return _request_ep(_BASE, **params)


def _request_ep(base: str, **params) -> dict:
    params["apikey"] = _api_key()
    url = base + "?" + urllib.parse.urlencode(params)
    for _ in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            data = json.load(e)
        if data.get("status") == "error":
            msg = data.get("message", "")
            if data.get("code") == 429:            # rate limited: back off, retry
                time.sleep(8)
                continue
            if "No data is available" in msg or "Data not found" in msg:  # before history start: empty, not fatal
                return {"values": []}
            raise RuntimeError(f"twelvedata: {msg}")
        return data
    raise RuntimeError("twelvedata: rate-limited after retries")


def load_bars(symbol: str, interval: str, start: str, end: str | None = None,
              cache_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return UTC-indexed OHLCV for [start, end], paging backward through history.

    `end` defaults to today: all available data through now, never a fixed cutoff.
    """
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
    itag = interval.replace(" ", "")
    fname = f"{symbol.replace('/', '-')}_{itag}_{start}_{end}.parquet"
    cpath = Path(cache_dir) / "twelvedata" / fname
    if cpath.exists():
        return pd.read_parquet(cpath)

    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    frames, cursor = [], end
    while True:
        r = _request(symbol=symbol, interval=interval, end_date=cursor,
                     outputsize=_MAX, order="DESC", timezone="UTC")
        vals = r.get("values") or []
        if not vals:
            break
        df = pd.DataFrame(vals)
        df.index = pd.to_datetime(df["datetime"], utc=True)
        if "volume" not in df.columns:      # FX / indices carry no volume
            df["volume"] = 0.0
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        frames.append(df)
        earliest = df.index.min()
        if earliest <= lo or len(vals) < _MAX:      # reached target start or history start
            break
        cursor = (earliest - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out[(out.index >= lo) & (out.index <= hi)]
    cpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cpath)
    return out


def load_dividends(symbol: str, start: str = "2010-01-01", end: str | None = None,
                   cache_dir: str = RAW_DIR) -> pd.Series:
    """Return cash dividends per share indexed by ex-date (UTC), cached per symbol. Used to build a
    trailing-12m dividend yield — the equity analogue of a currency's rate or a perp's funding."""
    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
    cpath = Path(cache_dir) / "twelvedata" / f"{symbol.replace('/', '-')}_div.parquet"
    if cpath.exists():
        return pd.read_parquet(cpath)["amount"]
    r = _request_ep(_DIV, symbol=symbol, start_date=start, end_date=end)
    vals = r.get("dividends") or []
    if not vals:
        s = pd.Series(dtype=float, name="amount")
    else:
        df = pd.DataFrame(vals)
        df.index = pd.to_datetime(df["ex_date"], utc=True)
        s = pd.to_numeric(df["amount"], errors="coerce").dropna().sort_index().rename("amount")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(cpath)
    return s
