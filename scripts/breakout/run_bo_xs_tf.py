"""Cross-sectional crypto breakout ACROSS timeframes — the §7b sleeve was only tested on 1d bars;
this asks whether the dispersion edge also lives on 4h/1h/15m. Same construction (52-week-high
nearness, long top / short bottom 30%, dollar-neutral), lookback and rebalance scaled to each TF's
bars-per-day, returns resampled to daily so all timeframes are Sharpe-comparable.

    python scripts/breakout/run_bo_xs_tf.py
"""

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from src.config import OOS_START, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import breakout_signal  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

TVOL, COST = VOL_TARGET_ANNUAL, 6.0
BPD = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}          # bars per day
PPY = {"1d": 365, "4h": 6 * 365, "1h": 24 * 365, "15m": 96 * 365}


def crypto_panel(tf):
    cols = {}
    for s in bo.CRYPTO:
        px = bo.load_crypto(s, tf)
        if px is not None:
            cols[s] = px["close"]
    return pd.DataFrame(cols).sort_index()


def xs_daily(pnl, sig, ppy_bar, rebal):
    """Dollar-neutral long-top/short-bottom at bar frequency, vol-targeted, resampled to daily."""
    rets = pnl.pct_change().clip(-0.5, 0.5)
    ranks = sig.rank(axis=1, pct=True)
    wl = (ranks >= 0.7).astype(float)
    ws = (ranks <= 0.3).astype(float)
    w = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0) - ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0)
    hold = pd.Series(False, index=w.index)
    hold.iloc[::rebal] = True
    w = w.where(hold, np.nan).ffill().shift(2).fillna(0.0)
    gross = (w * rets.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1)
    net_bar = gross - turn * COST / 1e4
    scale = (TVOL / (net_bar.rolling(60).std() * np.sqrt(ppy_bar))).clip(upper=3.0).shift(1).fillna(0.0)
    net_bar = net_bar * scale
    return ((1 + net_bar).resample("D").prod() - 1).dropna()


def main():
    print("=== Cross-sectional crypto breakout (52w-high nearness) across timeframes ===")
    print("(lookback ~126d-equiv, rebalanced ~daily, returns resampled to daily, net of 6bps/side)\n")
    rows = {}
    for tf in ["1d", "4h", "1h", "15m"]:
        pnl = crypto_panel(tf)
        if pnl.shape[1] < 10:
            print(f"  {tf}: panel too small"); continue
        lb = 126 * BPD[tf]                       # ~6-month nearness window in bars
        sig = breakout_signal(pnl, "nearness", lb)
        net = xs_daily(pnl, sig, PPY[tf], rebal=BPD[tf])
        plac = pd.DataFrame(bo.rng.standard_normal(pnl.shape), index=pnl.index, columns=pnl.columns)
        netp = xs_daily(pnl, plac, PPY[tf], rebal=BPD[tf])
        s = summarise(net, 365)
        mc = bootstrap_sharpe(net, 365, 500, bo.SEED) if s["sharpe_ann"] > 0.3 else {}
        oos = net[net.index >= OOS_START]
        rows[tf] = net
        print(f"  {tf:3s} ({pnl.shape[1]} coins): Sharpe {s['sharpe_ann']:+.2f}  "
              f"OOS {summarise(oos,365)['sharpe_ann']:+.2f}  MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f}  "
              f"DD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  placebo {summarise(netp,365)['sharpe_ann']:+.2f}",
              flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(bo.REPORTS / "bo_xs_tf_returns.parquet")
    print("\nBO XS-TF OK")


if __name__ == "__main__":
    main()
