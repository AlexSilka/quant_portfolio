"""Smoke test for the data layer — proves ingestion end to end.

    python scripts/smoke_test.py
"""


from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402


def main() -> None:
    print("== Binance USD-M perp: BTCUSDT 1h ==")
    k = load_klines("BTCUSDT", "1h", "2024-01", "2024-03", market="um")
    print(f"rows={len(k)}  {k.index.min()} .. {k.index.max()}")
    print(k[["open", "high", "low", "close", "volume"]].head(2).to_string())
    assert k[["open", "high", "low", "close"]].notna().all().all(), "NaN in OHLC"
    assert k.index.is_monotonic_increasing, "time index not sorted"

    print("\n== Binance funding: BTCUSDT ==")
    f = load_funding("BTCUSDT", "2024-01", "2024-01")
    cadence = int(f["funding_interval_hours"].mode().iloc[0])
    mean_rate = f["last_funding_rate"].mean()
    print(f"rows={len(f)}  cadence={cadence}h  "
          f"mean={mean_rate:.5%}  ~{mean_rate * 3 * 365:.1%}/yr carry")
    assert cadence == 8, "expected 8h funding cadence for BTCUSDT"

    print("\n== Equity daily: AAPL ==")
    a = load_equity_daily("AAPL", start="2018-01-01")
    print(f"rows={len(a)}  {a.index.min().date()} .. {a.index.max().date()}")
    assert len(a) > 1500, "expected deep daily history"

    print("\nDATA LAYER OK")


if __name__ == "__main__":
    main()
