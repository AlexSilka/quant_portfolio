"""Portfolio-level out-of-sample ledger (§13 trade log) for the assembled master book.

The book is assembled from per-family return series (risk parity), so its portfolio-level record is a
daily contribution/position ledger over the frozen OOS block (OOS_START→): for each day, every family's
risk-parity return contribution, the book's gross exposure (from the §8 drawdown-ladder overlay), the
book return, and the dollar P&L on the $500k book. Where a family also keeps a discrete per-trade log
(the trend sleeve), it is concatenated into a combined per-trade OOS log.

Outputs: reports/master_book_oos_ledger.csv (daily), reports/master_book_oos_trades.csv (per-trade,
families that have one).

    python scripts/make_oos_ledger.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import CAPITAL_USD, OOS_START  # noqa: E402

R = Path("reports")
OOS = pd.Timestamp(OOS_START).tz_localize(None)


def main():
    legs = pd.read_parquet(R / "master_book_legs.parquet")
    book = pd.read_parquet(R / "master_book.parquet")["ret"]
    ex = pd.read_parquet(R / "master_book_exposure.parquet") if (R / "master_book_exposure.parquet").exists() else None
    legs.index = pd.to_datetime(legs.index)
    book.index = pd.to_datetime(book.index)

    oos = legs[legs.index >= OOS].copy()
    n = legs.shape[1]
    contrib = oos.fillna(0.0) / n                                # each family's equal-weight contribution
    contrib.columns = [f"contrib_{c}" for c in contrib.columns]
    led = contrib.copy()
    led["book_ret"] = book.reindex(oos.index)
    if ex is not None:
        led["gross_exposure"] = ex["gross"].reindex(oos.index)
    led["book_pnl_usd"] = led["book_ret"] * CAPITAL_USD
    led["equity_usd"] = CAPITAL_USD * (1.0 + led["book_ret"]).cumprod()
    led.index.name = "date"
    led.round(6).to_csv(R / "master_book_oos_ledger.csv")

    sh = float(np.sqrt(365) * led["book_ret"].mean() / led["book_ret"].std(ddof=1))
    print(f"OOS ledger {oos.index.min().date()}..{oos.index.max().date()}: {len(led)} days, "
          f"book Sharpe {sh:+.2f}, net P&L ${led['book_pnl_usd'].sum():,.0f} on ${CAPITAL_USD:,}")

    # combined per-trade OOS log from families that keep one
    # A per-trade log belongs in the BOOK's record only if the book holds that family. Trend keeps the
    # repo's only instrument-level log and the book dropped trend, so this list is derived from the
    # assembler rather than typed — otherwise the next composition change republishes another family's
    # fills as the book's, which is exactly what happened here.
    import scripts.run_master_book as mb
    held = {lab for lab, _, _ in mb.FAMILIES}
    trade_logs = []
    for name, path in [("trend", R / "trend" / "trend_oos_trade_log.csv")]:
        if name not in held:
            print(f"  - {name}: has a per-trade log but the book does not hold it — not the book's record")
            continue
        if not path.exists():
            print(f"  - {name}: held by the book but publishes no per-trade log ({path.name} missing)")
            continue
        t = pd.read_csv(path)
        t.insert(0, "family", name)
        trade_logs.append(t)
        print(f"  + {name}: {len(t)} trades from {path.relative_to(R.parent)}")
    if trade_logs:
        combined = pd.concat(trade_logs, ignore_index=True)
        combined.to_csv(R / "master_book_oos_trades.csv", index=False)
        print(f"per-trade OOS log -> reports/master_book_oos_trades.csv ({len(combined)} trades)")
    print("artifacts -> reports/master_book_oos_ledger.csv" + (" · master_book_oos_trades.csv" if trade_logs else ""))


if __name__ == "__main__":
    main()
