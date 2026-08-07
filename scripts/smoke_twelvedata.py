"""Smoke test for the Twelve Data loader (equity intraday). Uses a few API credits.

    python scripts/smoke_twelvedata.py
"""


from src.data.twelvedata import load_bars  # noqa: E402


def main() -> None:
    print("== Twelve Data: AAPL 15min, 2020-03-01 .. 2020-06-30 (COVID crash + recovery) ==")
    df = load_bars("AAPL", "15min", "2020-03-01", "2020-06-30")
    print(f"rows={len(df)}  {df.index.min()} .. {df.index.max()}  tz={df.index.tz}")
    print(df.head(2).to_string())
    print(df.tail(2).to_string())

    assert str(df.index.tz) == "UTC", "index must be UTC"
    assert df.index.is_monotonic_increasing, "time index not sorted"
    assert df[["open", "high", "low", "close"]].notna().all().all(), "NaN in OHLC"
    tod = df.index.strftime("%H:%M")
    print(f"\nsession window (UTC): {tod.min()} .. {tod.max()}  (9:30-16:00 ET)")

    print("\nTWELVE DATA LOADER OK")


if __name__ == "__main__":
    main()
