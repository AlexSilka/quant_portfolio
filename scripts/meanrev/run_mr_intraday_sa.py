"""Session-aware intraday mean-reversion on equities (Twelve Data Pro 1h / 15m): the canonical
day-trading construction — the position is FLATTENED at each session close (no overnight hold), so
overnight-gap risk is removed and the z-score reverts within the session. This is the strongest
variant for intraday reversal; if it does not survive costs, naive continuous MR will not either.

Reuses the cached 1h/15m bars from run_mr_intraday.py (no new API calls).

    python scripts/meanrev/run_mr_intraday_sa.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.metrics import summarise  # noqa: E402
from scripts.meanrev.audit_mr import mr_revert  # noqa: E402
from scripts.meanrev.run_mr_intraday import EC, EQ, TF, load  # noqa: E402
from scripts.meanrev.run_mr_proper import wf_select  # noqa: E402


def session_flatten(pos):
    """Zero the position on the last bar of each UTC date -> no overnight hold."""
    last = pos.groupby(pos.index.date).tail(1).index
    p = pos.copy()
    p.loc[last] = 0.0
    return p


def mr_sa_daily(close, lb, ez, xz, bar_ppy):
    pos = session_flatten(mr_revert(close, lb, ez, xz))
    bt = backtest(close, vol_target(pos, close, 0.15, bar_ppy), capital=500_000, funding=None, **EC)
    return ((1 + bt["net_ret"]).resample("D").prod() - 1).dropna()


def main():
    for tf, bar_ppy in TF:
        print(f"\n===== EQUITY {tf} — SESSION-AWARE MR (flatten at close) =====")
        series = {}
        for s in EQ:
            close = load(s, tf)
            if close is None:
                continue
            grid = {(lb, ez, xz): mr_sa_daily(close, lb, ez, xz, bar_ppy)
                    for lb in (10, 20, 50) for ez in (1.5, 2.0, 2.5) for xz in (0.0, 0.5)}
            series[s] = wf_select(grid)
            print(f"  {s:6s}  session-aware MR OOS Sharpe {summarise(series[s], 252)['sharpe_ann']:+.2f}")
        if not series:
            continue
        wfs = pd.DataFrame(series)
        pos_ct = int((wfs.apply(lambda c: summarise(c, 252)["sharpe_ann"]) > 0).sum())
        basket = summarise(wfs.mean(axis=1).dropna(), 252)
        print(f"  --> {len(series)} names | positive: {pos_ct} | "
              f"EQUAL-WEIGHT BASKET Sharpe {basket['sharpe_ann']:+.2f}  DD {basket['max_dd']:+.1%}")


if __name__ == "__main__":
    main()
