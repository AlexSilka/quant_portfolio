"""Global-macro diversifier sleeve — time-series momentum on EM FX + extended commodities.

A seventh, genuinely-new decorrelated leg built from asset classes the other families never trade.
Only the edges that survive a per-strategy out-of-sample test are kept: trend (TSMOM) on EM FX
(USD/TRY, BRL, INR, ZAR, PLN, CNH) and on a broad commodity set (metals, energy, agri, uranium).
Cross-sectional momentum and reversal were tested on the same universes and DROPPED — no OOS edge;
country-equity trend was tested and DROPPED — no standalone edge. So this is trend-only, and only on
the two classes where the edge holds in both halves of history (EM FX Sharpe ~0.9 h1/h2 +0.85/+0.89;
commodities ~0.6 h1/h2 +0.41/+0.83).

Value to the book: correlation ~+0.13 to the master, so it diversifies genuinely and improves the
worst month and Sharpe (the book's passing targets get more margin). It is a positive-skew trend leg,
so it does NOT lift months-in-profit — a hedge shrinks losing months, it cannot flip them to gains.

Construction per class: TSMOM sign-blend over three lookbacks (fast 10/20/40, medium 20/40/63, slow
40/63/120), per-asset vol-targeted, the three tranches averaged, the class vol-targeted to 15%; the two
class books combined at equal risk and vol-targeted to 15%. Signals lagged one bar; ~2 bps turnover cost.

    python scripts/run_gmacro.py   ->  <BOOK_DIR>/gmacro_sleeve.parquet
"""
from __future__ import annotations

import glob
import json
import os
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from src.config import BOOK_DIR  # noqa: E402
from src.data.twelvedata import _api_key  # noqa: E402
from src.metrics import summarise  # noqa: E402

EQ_STORE = ROOT / "data/raw/equity_td"
TD_DIR = ROOT / "data/raw/twelvedata"
EMFX = ["USD/TRY", "USD/BRL", "USD/INR", "USD/ZAR", "USD/PLN", "USD/CNH"]     # EM FX (risk-off diversifier)
COMMOD_TD = ["XAG/USD", "XPT/USD", "WTI/USD", "URA", "CORN", "WEAT", "SOYB"]  # metals/energy/agri (TwelveData)
COMMOD_LOCAL = ["GLD", "SLV", "USO", "DBC", "DBA"]                            # commodity ETFs (local store)
LOOKBACKS = [(10, 20, 40), (20, 40, 63), (40, 63, 120)]
PPY = 252


def _fetch_td(sym: str) -> pd.Series | None:
    """TwelveData daily close, cached under data/raw/twelvedata/<sym>_1day_..._gmacro.parquet."""
    cache = sorted(glob.glob(str(TD_DIR / f"{sym.replace('/', '-')}_1day_*_gmacro.parquet")))
    if cache:
        s = pd.read_parquet(cache[0])["close"]
    else:
        p = {"symbol": sym, "interval": "1day", "start_date": "2005-01-01", "outputsize": 5000,
             "order": "ASC", "apikey": _api_key()}
        url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(p)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.load(r)
        except Exception:
            return None
        v = d.get("values")
        if not v:
            return None
        df = pd.DataFrame(v).astype({"close": float}); df["datetime"] = pd.to_datetime(df["datetime"])
        s = df.set_index("datetime").sort_index()["close"]
        os.makedirs(TD_DIR, exist_ok=True)
        s.to_frame().to_parquet(TD_DIR / f"{sym.replace('/', '-')}_1day_2005-01-01_gmacro.parquet")
    s.index = pd.to_datetime(s.index)
    s = (s.tz_localize(None) if s.index.tz is not None else s).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _local(sym: str) -> pd.Series | None:
    p = EQ_STORE / f"{sym}_1d.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["close"]; s.index = pd.to_datetime(s.index)
    return (s.tz_localize(None) if s.index.tz is not None else s).sort_index()


def _panel(td_syms, local_syms=()):
    cols = {s: _fetch_td(s) for s in td_syms}
    cols.update({s: _local(s) for s in local_syms})
    return pd.DataFrame({k: v for k, v in cols.items() if v is not None}).sort_index()


def _vol_target(x, target=0.15, lb=60):
    lev = (target / (x.rolling(lb).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (x * lev).dropna()


def _tsmom(close, lookbacks):
    r = close.pct_change(); vol = r.rolling(40).std()
    sig = sum(np.sign(close / close.shift(h) - 1.0) for h in lookbacks) / len(lookbacks)
    pos = sig.shift(1) * (0.15 / np.sqrt(PPY) / vol).clip(upper=3.0)
    n = close.shape[1]
    return _vol_target((pos * r).sum(axis=1) / n - (pos.diff().abs().sum(axis=1) / n) * 2 / 1e4)


def _class_book(close):
    if close.empty or close.shape[1] == 0:
        return None
    tranches = [_vol_target(_tsmom(close, lb)) for lb in LOOKBACKS]
    return _vol_target(pd.concat(tranches, axis=1).mean(axis=1).dropna())


def build_gmacro() -> pd.Series:
    books = {"emfx": _class_book(_panel(EMFX)),
             "commod": _class_book(_panel(COMMOD_TD, COMMOD_LOCAL))}
    live = {k: v for k, v in books.items() if v is not None and len(v) > 100}
    df = pd.DataFrame(live).sort_index()
    return _vol_target(df.mean(axis=1, skipna=True).dropna()).rename("ret")


def main():
    g = build_gmacro()
    g.to_frame().to_parquet(BOOK_DIR / "gmacro_sleeve.parquet")
    s = summarise(g, PPY)
    print(f"global-macro sleeve (EM-FX + commodities TSMOM): Sharpe {s['sharpe_ann']:+.2f}  "
          f"maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  {g.index.min().date()}..{g.index.max().date()}")
    print(f"RUN GMACRO OK -> {BOOK_DIR}/gmacro_sleeve.parquet")


if __name__ == "__main__":
    main()
