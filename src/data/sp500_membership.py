"""Point-in-time S&P 500 membership — and the reason the broad equity panel needs it.

The broad panel is built from every ticker that was in the index at any point since 2012, which is
what makes it survivorship-free. But a ticker outlives the company: once a US listing is gone, Twelve
Data resolves the bare symbol to whatever else carries it, and the panel silently continues the
series with a *different company on a foreign exchange*. Probed live (2026-08): of 71 panel names
with no current US listing, **67 return a non-USD instrument** — ANTM is Aneka Tambang on the
Indonesia exchange in rupiah, FRC is a Mexican listing, RHT and MON are Canadian venture names.
Left alone they contribute years of unrelated foreign returns to a US cross-section, and the rupiah
price level makes ANTM the single most "liquid" name in the panel, ahead of NVDA.

The fix is to end each ticker's series when its US index life ends. Membership dates come from the
same point-in-time source the universe list was built from (`fja05680/sp500`), cached locally.

Only the *trailing* edge is masked. History before a name joined the index is still that company's
genuine US tape, so it stays — the defect is specific to what happens after the listing dies.

    python -m src.data.sp500_membership       # refresh the cached membership file
"""
from __future__ import annotations

import pandas as pd
import requests

from src.config import CACHE_DIR  # noqa: E402

SRC = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
CACHE = CACHE_DIR / "xs" / "_sp500_ticker_start_end.csv"


def fetch(force: bool = False) -> pd.DataFrame:
    """Cached ticker → (start_date, end_date) index membership spans."""
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE, parse_dates=["start_date", "end_date"])
    r = requests.get(SRC, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"S&P membership fetch failed: HTTP {r.status_code}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(r.content)
    return pd.read_csv(CACHE, parse_dates=["start_date", "end_date"])


def last_index_day() -> dict[str, pd.Timestamp]:
    """Ticker → the last day it was an index member. A ticker with several spells (dropped and
    re-added) keeps the latest, since it traded in the US throughout the gap. Tickers still in the
    index are absent — nothing to truncate."""
    d = fetch()
    out: dict[str, pd.Timestamp] = {}
    for t, g in d.groupby("ticker"):
        if g["end_date"].isna().any():          # still a member
            continue
        out[str(t)] = g["end_date"].max()
    return out


US_LISTINGS = CACHE_DIR / "xs" / "_us_listed_symbols.csv"


def us_listed(force: bool = False) -> set[str]:
    """Symbols currently listed on a US exchange, from Twelve Data's reference endpoint (cached).

    This is the half of the test that keeps the fix from over-reaching. Leaving the S&P 500 is not
    the same as dying: a name dropped for size keeps trading in the US, and its post-exit tape is
    its own. Only a ticker with *no* US listing left can have been recycled onto a foreign company,
    so only those get truncated."""
    if US_LISTINGS.exists() and not force:
        return set(pd.read_csv(US_LISTINGS)["symbol"].astype(str))
    from src.data.twelvedata import _api_key  # local import: only this path needs a key
    r = requests.get("https://api.twelvedata.com/stocks",
                     params={"country": "United States", "apikey": _api_key()}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"US listing reference fetch failed: HTTP {r.status_code}")
    syms = sorted({row["symbol"] for row in r.json().get("data", [])})
    if not syms:
        raise RuntimeError("US listing reference came back empty — refusing to cache it")
    US_LISTINGS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": syms}).to_csv(US_LISTINGS, index=False)
    return set(syms)


def truncate_after_exit(panel: pd.DataFrame, grace_days: int = 5,
                        label: str = "panel") -> pd.DataFrame:
    """Blank a column after its ticker left the index **when the ticker no longer has a US listing**,
    so a recycled symbol cannot keep printing a foreign company's prices into a US cross-section.
    `grace_days` keeps the delisting bar itself and the few days around it, which is real trading."""
    last = last_index_day()
    live = us_listed()
    known = set(fetch()["ticker"])
    idx, tz = panel.index, panel.index.tz
    out = panel.copy()
    cut, kept_live = [], 0
    for c in out.columns:
        d = last.get(c)
        if d is None:
            continue                                   # still an index member
        if c in live:
            kept_live += 1                             # dropped from the index, still trading here
            continue
        stop = pd.Timestamp(d, tz=tz) + pd.Timedelta(days=grace_days)
        if stop >= idx.max() or not (out.index > stop).any():
            continue
        if out.loc[idx > stop, c].notna().any():
            out.loc[idx > stop, c] = pd.NA
            cut.append(c)
    unknown = [c for c in out.columns if c not in known]
    if unknown:
        print(f"  {label}: {len(unknown)} tickers absent from the membership file, left untouched: "
              f"{unknown[:8]}{'…' if len(unknown) > 8 else ''}")
    print(f"  {label}: truncated {len(cut)} de-listed tickers at their index exit; "
          f"{kept_live} index leavers kept (still US-listed)")
    return out.astype(float)


if __name__ == "__main__":
    df = fetch(force=True)
    print(f"S&P membership: {len(df)} spans, {df['ticker'].nunique()} tickers → {CACHE}")
