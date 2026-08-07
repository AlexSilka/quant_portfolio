"""Walk-forward sleeve selection — the honest out-of-sample test (Task A §10).

The book's headline picks its survivors using the whole history, which look-aheads the selection.
Here, at each rebalance date the portfolio holds ONLY sleeves that were robust on data strictly
BEFORE that date, so the concatenated return series is genuinely out-of-sample. Run under rolling
(trailing-window) and anchored (expanding-window) policies at two cadences to show the result does
not depend on that choice (§10: "show results do not depend on that choice").

    python scripts/walk_forward.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import BOOK_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402

PPY = 365
SEL_SHARPE, MIN_OBS = 0.5, 252   # pick a sleeve if trailing Sharpe clears the bar on >= 1y of data


def walk_forward(rets, dates, window_years):
    """Concatenate held returns: at each date pick sleeves robust on prior data, hold to the next."""
    port, picks = [], []
    for i in range(len(dates) - 1):
        T, Tn = dates[i], dates[i + 1]
        lo = T - pd.DateOffset(years=window_years) if window_years else rets.index[0]
        win = rets.loc[lo:T]
        win = win[win.index < T]                                   # strictly past data — no look-ahead
        sh = np.sqrt(PPY) * win.mean() / win.std(ddof=1)
        keep = sh.index[(sh > SEL_SHARPE) & (win.count() >= MIN_OBS)]
        held = rets.loc[T:Tn, keep]
        held = held[held.index < Tn]
        port.append(held.mean(axis=1) if len(keep) else pd.Series(0.0, index=held.index))
        picks.append(len(keep))
    return pd.concat(port).sort_index().dropna(), picks


def main():
    rets = pd.read_parquet(BOOK_DIR / "all_returns.parquet")
    tz = rets.index.tz
    rets = rets[rets.index >= pd.Timestamp("2012-01-01", tz=tz)]
    print(f"candidates: {rets.shape[1]}  span {rets.index.min().date()}..{rets.index.max().date()}")

    results = {}
    for wlab, wy in [("anchored", None), ("rolling-2y", 2)]:
        for clab, freq in [("annual", "YS"), ("semiannual", "6MS")]:
            dates = pd.date_range("2016-01-01", "2026-07-01", freq=freq, tz=tz)
            wf, picks = walk_forward(rets, dates, wy)
            s = summarise(wf, PPY)
            results[(wlab, clab)] = s
            print(f"  {wlab:10s} {clab:11s}: Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  "
                  f"months+ {s['months_in_profit']:.0%}  avg picks {np.mean(picks):.0f}")

    # primary series (anchored, annual) -> dashboard/report
    dates = pd.date_range("2016-01-01", "2026-07-01", freq="YS", tz=tz)
    wf, _ = walk_forward(rets, dates, None)
    wf.rename("ret").to_frame().to_parquet(BOOK_DIR / "walk_forward.parquet")
    s = summarise(wf, PPY)
    shs = [v["sharpe_ann"] for v in results.values()]
    print("\nPRIMARY walk-forward (anchored, annual), 2016-2026 OUT-OF-SAMPLE:")
    print(f"  Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"total {s['total_return']:+.0%}")
    print(f"Sharpe across the 4 (window x cadence) configs: {min(shs):+.2f}..{max(shs):+.2f}  "
          + ("-> robust to the choice (§10)" if max(shs) - min(shs) < 0.6 else "-> sensitive"))
    print("WALK FORWARD OK")


if __name__ == "__main__":
    main()
