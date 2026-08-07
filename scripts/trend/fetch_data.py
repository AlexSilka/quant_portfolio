"""Fill the data gaps the trend deep-dive benefits from (user-authorised network fetch).

  1. Spot 1d + 4h back to 2017 for the whole crypto set — extends the trend backtest to the
     2017 bull and 2018 bear (Q4-2018 isolation the task requires; perp klines only start 2019-12).
  2. MKR perp intraday (4h/1h/15m/5m) + funding — the one universe name missing intraday.

Idempotent: `load_klines` skips months already cached and records pre-listing gaps, so re-running
only fetches what is genuinely missing. Progress is printed per symbol.
"""
from __future__ import annotations

import time

from src import bo_common as bo  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402

SPOT_START = "2017-01"


def main():
    t0 = time.time()
    # 1. spot long history (1d then 4h) for the full universe
    for tf in ("1d", "4h"):
        for i, sym in enumerate(bo.CRYPTO):
            try:
                df = load_klines(sym, tf, SPOT_START, market="spot")
                n = len(df)
                first = df.index[0].date() if n else "-"
                print(f"[spot {tf}] {i+1:2d}/{len(bo.CRYPTO)} {sym:12s} {n:>7d} bars from {first}"
                      f"  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"[spot {tf}] {sym}: {type(e).__name__} {e}")

    # 2. MKR perp intraday + funding
    for tf in ("4h", "1h", "15m", "5m"):
        try:
            df = load_klines("MKRUSDT", tf, "2020-01", market="um")
            print(f"[um {tf}] MKRUSDT {len(df):>7d} bars  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"[um {tf}] MKRUSDT: {type(e).__name__} {e}")
    try:
        f = load_funding("MKRUSDT", "2020-01")
        print(f"[funding] MKRUSDT {len(f)} settlements")
    except Exception as e:
        print(f"[funding] MKRUSDT: {type(e).__name__} {e}")

    print(f"\nfetch done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
