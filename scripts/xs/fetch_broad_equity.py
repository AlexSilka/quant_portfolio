"""Fetch a broad, survivorship-free US equity daily panel via Twelve Data (the project's feed).

Universe = every S&P 500 member at any point during 2012-2026 (503 still in the index + 325 that
left since 2012), from `fja05680/sp500`'s point-in-time membership file — so the panel includes the
names that were dropped, not only today's survivors (the honest momentum survivorship correction).
Twelve Data Pro (key from .env) gives deep, adjusted daily history (from ~2010); yfinance is not
used (incomplete on delisted tickers). A modest thread pool + the loader's 429-backoff keeps it
fast without tripping the rate limit.

    python scripts/xs/fetch_broad_equity.py
"""
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import CACHE_DIR  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402

OUT = CACHE_DIR / "xs"
# argv: [universe_json_basename] [output_prefix]  (defaults = large-cap S&P 500 panel)
_UNI_FILE = sys.argv[1] if len(sys.argv) > 1 else "_equity_universe.json"
_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "stocks_broad"
UNI = json.load(open(OUT / _UNI_FILE))
START = "2010-01-01"


def one(sym):
    try:
        px = load_equity_daily(sym, start=START)
        if len(px) < 500:
            return sym, None, None
        return sym, px["close"], px["close"] * px["volume"]
    except Exception:
        return sym, None, None


def main():
    close, dollar, ok, fail = {}, {}, 0, 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, s): s for s in UNI}
        for i, f in enumerate(as_completed(futs), 1):
            sym, c, d = f.result()
            if c is not None:
                close[sym], dollar[sym] = c, d
                ok += 1
            else:
                fail += 1
            if i % 50 == 0:
                print(f"  {i}/{len(UNI)}  kept {ok}  missing {fail}", flush=True)

    cp = pd.DataFrame(close).sort_index()
    cp.index = pd.to_datetime(cp.index).tz_localize(None)
    cp = cp.dropna(how="all").ffill(limit=3)
    cp.to_parquet(OUT / f"{_PREFIX}_1d_close.parquet")
    ap = pd.DataFrame(dollar).sort_index()
    ap.index = pd.to_datetime(ap.index).tz_localize(None)
    ap.reindex(cp.index).to_parquet(OUT / f"{_PREFIX}_1d_adv.parquet")
    print(f"BROAD EQUITY PANEL: {cp.shape[0]} bars × {cp.shape[1]} names "
          f"({ok} fetched, {fail} unavailable)  {cp.index[0].date()}→{cp.index[-1].date()}")


if __name__ == "__main__":
    main()
