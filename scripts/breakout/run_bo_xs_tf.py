"""Cross-sectional crypto breakout ACROSS timeframes — the §7b sleeve was only tested on 1d bars;
this asks whether the dispersion edge also lives on 4h/1h/15m. Same construction (52-week-high
nearness, long top / short bottom 30%, dollar-neutral), lookback and rebalance scaled to each TF's
bars-per-day, returns resampled to daily so all timeframes are Sharpe-comparable.

    python scripts/breakout/run_bo_xs_tf.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from src.backtest.costs import panel_impact_cost  # noqa: E402
from src.config import IMPACT_K, OOS_START, VOL_TARGET_ANNUAL  # noqa: E402
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


def adv_panel(cols, tf, win=20):
    """Per-bar dollar volume for the √-impact term — trailing median, lagged one bar."""
    qv = {}
    for s in cols:
        px = bo.load_crypto(s, tf)
        if px is not None and "quote_volume" in px:
            qv[s] = px["quote_volume"]
    return pd.DataFrame(qv).sort_index().rolling(win).median().shift(1) if qv else None


def funding_panel(cols, index):
    """Perp funding accrued per bar. Settlements are on an 8h grid, so each one is binned into the
    bar it falls in and summed — a plain reindex would silently drop two of the three in a 1d bar."""
    bar = index.to_series().diff().dropna().median()
    f = {}
    for s in cols:
        fr = bo.safe_funding(s)
        if len(fr):
            f[s] = fr.sort_index().resample(bar, origin=index[0]).sum()
    return pd.DataFrame(f).reindex(index).fillna(0.0) if f else None


def xs_daily(pnl, sig, ppy_bar, rebal, adv=None, funding=None):
    """Dollar-neutral long-top/short-bottom at bar frequency, vol-targeted, resampled to daily.

    Costs are the book's own model rather than one flat constant: commission + half-spread on
    turnover, Almgren √-impact per name when an ADV panel is given, and perp funding at every
    settlement when a funding panel is given. `src/config.py` fixes the rule (never a flat constant,
    or the illiquid tail of a panel trades at the majors' price) and `xsect.xs_backtest` states that
    crypto callers must charge funding separately — this driver did neither until 2026-08.

    Funding is not a rounding error here and it is not a cost: measured on the point-in-time top-30,
    the SHORT book (names far below their high) carries more funding than the LONG book (names at
    their high) — +10.2%/yr against +6.0%/yr on 1d — so the leg collects a net ~+4%/yr for shorting
    laggards. The sign was the opposite of what was assumed before it was measured.
    """
    rets = pnl.pct_change().clip(-0.5, 0.5)
    ranks = sig.rank(axis=1, pct=True)
    wl = (ranks >= 0.7).astype(float)
    ws = (ranks <= 0.3).astype(float)
    w = wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0) - ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0)
    hold = pd.Series(False, index=w.index)
    hold.iloc[::rebal] = True
    w = w.where(hold, np.nan).ffill().shift(2).fillna(0.0)
    dw = w.diff().abs()
    gross = (w * rets.fillna(0.0)).sum(axis=1)
    net_bar = gross - dw.sum(axis=1) * COST / 1e4
    if adv is not None:
        net_bar = net_bar - panel_impact_cost(dw, rets.rolling(20).std(),
                                              adv.reindex_like(w).ffill(), bo.CAP, IMPACT_K)
    if funding is not None:
        net_bar = net_bar - (w * funding.reindex_like(w).fillna(0.0)).sum(axis=1)
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
        adv, fund = adv_panel(pnl.columns, tf), funding_panel(pnl.columns, pnl.index)
        net = xs_daily(pnl, sig, PPY[tf], rebal=BPD[tf], adv=adv, funding=fund)
        plac = pd.DataFrame(bo.rng.standard_normal(pnl.shape), index=pnl.index, columns=pnl.columns)
        netp = xs_daily(pnl, plac, PPY[tf], rebal=BPD[tf], adv=adv, funding=fund)
        s = summarise(net, 365)
        mc = bootstrap_sharpe(net, 365, 500, bo.SEED) if s["sharpe_ann"] > 0.3 else {}
        oos = net[net.index >= OOS_START]
        rows[tf] = net
        print(f"  {tf:3s} ({pnl.shape[1]} coins): Sharpe {s['sharpe_ann']:+.2f}  "
              f"OOS {summarise(oos,365)['sharpe_ann']:+.2f}  MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f}  "
              f"DD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  placebo {summarise(netp,365)['sharpe_ann']:+.2f}",
              flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(bo.BREAKOUT / "bo_xs_tf_returns.parquet")
    print("\nBO XS-TF OK")


if __name__ == "__main__":
    main()
