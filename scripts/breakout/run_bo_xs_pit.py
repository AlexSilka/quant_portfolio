"""Honest cross-sectional breakout: a POINT-IN-TIME top-30 universe (no look-ahead in selection).

The static top-30 (`run_bo_xs_liq.py`) ranks coins by full-history volume, so it pre-loads the coins
that *became* liquid and survived — a look-ahead + survivorship boost. Here, at each bar the universe
is the top-30 by *trailing* median dollar-volume (63-day window, lagged), so membership only knows the
past: a coin enters when it has actually become liquid, and drops out when it fades. The cross-section
(52-week-high nearness, long-top / short-bottom, dollar-neutral, ~daily rebalance) ranks only the
current members. Universe churn is real turnover and is charged.

Reports STATIC vs PIT side by side so the selection bias is quantified, across timeframes.

    python scripts/breakout/run_bo_xs_pit.py
"""

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_xs_big import NONCRYPTO, symbols_with_tf  # noqa: E402
from scripts.breakout.run_bo_xs_tf import BPD, PPY, xs_daily  # noqa: E402
from src.config import OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import breakout_signal  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

N = 30


def panels(tf, min_days=150):
    """Aligned close and quote-volume panels for all crypto perps with >= min_days history."""
    min_obs = min_days * BPD[tf]
    close, qv = {}, {}
    for s in symbols_with_tf(tf):
        if s[:-4] in NONCRYPTO:
            continue
        px = bo.load_crypto(s, tf)
        if px is None or "quote_volume" not in px or px["close"].notna().sum() < min_obs:
            continue
        close[s], qv[s] = px["close"], px["quote_volume"]
    idx = sorted(set().union(*[c.index for c in close.values()]))
    C = pd.DataFrame(close).reindex(idx).sort_index().ffill(limit=5)
    Q = pd.DataFrame(qv).reindex(idx).sort_index()
    return C, Q


def pit_members(qv, n, win):
    """Boolean membership: top-n coins by TRAILING median dollar-volume (lagged, PIT)."""
    tv = qv.rolling(win, min_periods=win // 2).median().shift(1)
    return tv.rank(axis=1, ascending=False) <= n


def metrics(net, label):
    s = summarise(net, 365)
    mc = bootstrap_sharpe(net, 365, 1000, bo.SEED) if s["sharpe_ann"] > 0.2 else {}
    oos = net[net.index >= OOS_START]
    return {"universe": label, "sharpe": s["sharpe_ann"], "oos": summarise(oos, 365)["sharpe_ann"],
            "mc_p5": mc.get("sharpe_p5", np.nan), "max_dd": s["max_dd"],
            "months_in_profit": s["months_in_profit"], "net": net}


def main():
    print(f"=== Cross-sectional breakout on a POINT-IN-TIME top-{N} universe (no look-ahead) ===")
    print("(52w-high nearness, ~daily rebalance, daily-resampled, net 6bps/side)\n")
    rows, series = [], {}
    for tf in ["1d", "4h", "1h"]:
        C, Q = panels(tf)
        near = breakout_signal(C, "nearness", 126 * BPD[tf])

        # STATIC top-N: the N coins with the highest FULL-history volume (look-ahead selection)
        top = Q.median().sort_values(ascending=False).index[:N]
        net_s = xs_daily(C[top], near[top], PPY[tf], rebal=BPD[tf])

        # PIT top-N: membership from trailing volume only; non-members masked out of the ranking
        mem = pit_members(Q, N, 63 * BPD[tf])
        avg_n = float(mem.reindex(C.index).sum(axis=1).replace(0, np.nan).mean())
        net_p = xs_daily(C, near.where(mem), PPY[tf], rebal=BPD[tf])

        print(f"--- {tf}  ({C.shape[1]} coins in pool, PIT universe avg {avg_n:.0f} names) ---")
        for label, net in [("static_top30", net_s), ("PIT_top30", net_p)]:
            m = metrics(net, label)
            rows.append({"tf": tf, **{k: v for k, v in m.items() if k != "net"}})
            series[f"{tf}_{label}"] = m["net"]
            print(f"    {label:12s}: Sharpe {m['sharpe']:+.2f}  OOS {m['oos']:+.2f}  "
                  f"MC-P5 {m['mc_p5']:+.2f}  DD {m['max_dd']:+.1%}  months+ {m['months_in_profit']:.0%}",
                  flush=True)
        print()

    df = pd.DataFrame(rows)
    df.to_csv(bo.REPORTS / "bo_xs_pit.csv", index=False)
    pd.DataFrame(series).to_parquet(bo.REPORTS / "bo_xs_pit_returns.parquet")

    # per-year of the PIT 4h book (the headline honest cross-sectional sleeve)
    net = series.get("4h_PIT_top30")
    if net is not None:
        py = {int(y): round(float(np.sqrt(365) * g.dropna().mean() / g.dropna().std(ddof=1)), 2)
              for y, g in net.groupby(net.index.year) if g.dropna().std(ddof=1) > 0}
        print(f"PIT top-30 4h per-year Sharpe: {py}")
    print("\nBO XS-PIT OK")


if __name__ == "__main__":
    main()
