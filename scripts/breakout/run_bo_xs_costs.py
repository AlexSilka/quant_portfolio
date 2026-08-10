"""What the cross-sectional breakout leg costs once it pays what the rest of the book pays.

The time-series leg runs through `src/backtest/engine.py`: commission split by venue, half-spread,
Almgren √-impact per name from ADV, and funding charged at every 8h settlement. The cross-sectional
leg does not — `run_bo_xs_tf.xs_daily` charges a **flat 6bps of turnover and nothing else**, so two
cost lines the rest of the repo treats as mandatory are simply absent from it:

  **√-impact.** `src/config.py` states the rule — total per-trade cost is always commission +
  half-spread + √-impact, never a flat constant, because a flat spread lets the illiquid tail of a
  panel trade at the majors' cost. A flat 6bps happens to equal the perp taker (5) plus half-spread
  (1), so the constant is not wrong, it is *incomplete*.

  **Funding.** `xsect.xs_backtest` says it outright: "Crypto perps pay funding, not borrow, so those
  callers leave borrow_bps_annual=0 and charge funding separately." This caller never charged it.
  Dollar-neutrality does not make it cancel: the leg is long the most broken-out names and short the
  least, and funding is highest exactly on what a breakout signal buys — so the two legs are expected
  to *both* cost, not to offset.

This script does not fix anything. It reproduces the shipped leg and adds one cost line at a time, so
the size of each is a measurement rather than an argument, and then prices the venue split the
time-series leg already uses (long on spot, short on perps).

    python scripts/breakout/run_bo_xs_costs.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import bo_common as bo  # noqa: E402
from src.backtest.costs import panel_impact_cost  # noqa: E402
from src.config import (BINANCE_FUT_TAKER_BPS, BINANCE_SPOT_TAKER_BPS, BREAKOUT_DIR,  # noqa: E402
                        CRYPTO_HALF_SPREAD_BPS, IMPACT_K, OOS_START, VOL_TARGET_ANNUAL)
from src.sleeves.cross_sectional import breakout_signal  # noqa: E402
from scripts.breakout.run_bo_deep_history import full_metrics, show  # noqa: E402
from scripts.breakout.run_bo_xs_big import NONCRYPTO, symbols_with_tf  # noqa: E402
from scripts.breakout.run_bo_xs_tf import BPD, PPY  # noqa: E402

N = 30
TFS = ["1d", "4h", "1h"]
FLAT_BPS = 6.0                      # what the shipped leg charges, all in


# --- panels -------------------------------------------------------------------------------------

def panels(tf: str, min_days: int = 150):
    """Close, quote-volume and per-bar funding panels for the crypto perps with enough history."""
    min_obs = min_days * BPD[tf]
    close, qv, fund = {}, {}, {}
    for s in symbols_with_tf(tf):
        if s[:-4] in NONCRYPTO:
            continue
        px = bo.load_crypto(s, tf)
        if px is None or "quote_volume" not in px or px["close"].notna().sum() < min_obs:
            continue
        close[s], qv[s] = px["close"], px["quote_volume"]
        fund[s] = bo.safe_funding(s)
    idx = sorted(set().union(*[c.index for c in close.values()]))
    C = pd.DataFrame(close).reindex(idx).sort_index().ffill(limit=5)
    Q = pd.DataFrame(qv).reindex(idx).sort_index()
    # funding accrues on an 8h grid; bin every settlement into the bar it falls in so none is dropped
    bar = C.index.to_series().diff().dropna().median()
    F = pd.DataFrame({s: (f.sort_index().resample(bar, origin=C.index[0]).sum() if len(f) else None)
                      for s, f in fund.items() if len(f)}).reindex(C.index).fillna(0.0)
    return C, Q, F.reindex(columns=C.columns).fillna(0.0)


def pit_members(qv: pd.DataFrame, n: int, win: int) -> pd.DataFrame:
    tv = qv.rolling(win, min_periods=win // 2).median().shift(1)
    return tv.rank(axis=1, ascending=False) <= n


def weights(sig: pd.DataFrame, rebal: int) -> pd.DataFrame:
    """The shipped construction: long top 30% / short bottom 30%, dollar-neutral, t+2 fill."""
    ranks = sig.rank(axis=1, pct=True)
    wl, ws = (ranks >= 0.7).astype(float), (ranks <= 0.3).astype(float)
    w = (wl.div(wl.sum(axis=1).replace(0, np.nan), axis=0)
         - ws.div(ws.sum(axis=1).replace(0, np.nan), axis=0))
    hold = pd.Series(False, index=w.index)
    hold.iloc[::rebal] = True
    return w.where(hold, np.nan).ffill().shift(2).fillna(0.0)


# --- the cost stack, one line at a time -----------------------------------------------------------

def run(C, Q, F, sig, tf, *, impact: bool, funding: bool, venue: str = "perp") -> pd.Series:
    """Daily net return of the XS leg under a chosen cost stack.

    `venue='split'` fills the long book on spot (no funding, 2x taker) and the short book on perps
    (funding, 1x taker) — the same asymmetry the time-series leg exploits.
    """
    rets = C.pct_change().clip(-0.5, 0.5)
    w = weights(sig, BPD[tf])
    dw = w.diff().abs()
    turn = dw.sum(axis=1)
    gross = (w * rets.fillna(0.0)).sum(axis=1)

    if venue == "split":
        lng, sht = w.clip(lower=0.0), w.clip(upper=0.0)
        fee = (lng.diff().abs().sum(axis=1) * (BINANCE_SPOT_TAKER_BPS + CRYPTO_HALF_SPREAD_BPS)
               + sht.diff().abs().sum(axis=1) * (BINANCE_FUT_TAKER_BPS + CRYPTO_HALF_SPREAD_BPS)) / 1e4
        # only the short book sits on perps, so only it touches funding; spot longs pay none
        carry = pd.Series(0.0, index=w.index) if not funding else -(sht * F).sum(axis=1)
    else:
        fee = turn * FLAT_BPS / 1e4
        carry = pd.Series(0.0, index=w.index) if not funding else -(w * F).sum(axis=1)

    adv = Q.rolling(20).median().shift(1)
    imp = (panel_impact_cost(dw, rets.rolling(20).std(), adv, bo.CAP, IMPACT_K)
           if impact else pd.Series(0.0, index=w.index))

    net_bar = gross - fee - imp + carry
    scale = (VOL_TARGET_ANNUAL / (net_bar.rolling(60).std() * np.sqrt(PPY[tf]))).clip(upper=3.0)
    net_bar = net_bar * scale.shift(1).fillna(0.0)
    return ((1 + net_bar).resample("D").prod() - 1).dropna()


def main():
    print("=== CROSS-SECTIONAL BREAKOUT LEG — what it costs once it pays what the book pays ===")
    print(f"PIT top-{N} by trailing dollar volume, 52w-high nearness, dollar-neutral, t+2\n")

    out, series = {}, {}
    for tf in TFS:
        C, Q, F = panels(tf)
        sig = breakout_signal(C, "nearness", 126 * BPD[tf]).where(pit_members(Q, N, 63 * BPD[tf]))
        fp = float((F != 0).any().sum())
        print(f"--- {tf}: {C.shape[1]} coins in pool, funding series on {fp:.0f} of them ---")
        rows = []
        for lab, kw in (("as shipped (flat 6bps, no funding)", dict(impact=False, funding=False)),
                        ("+ √-impact from ADV", dict(impact=True, funding=False)),
                        ("+ funding at every settlement", dict(impact=False, funding=True)),
                        ("both (the book's own cost model)", dict(impact=True, funding=True)),
                        ("both + venue split (long spot/short perp)",
                         dict(impact=True, funding=True, venue="split"))):
            r = run(C, Q, F, sig, tf, **kw)
            series[f"{tf} {lab}"] = r
            rows.append(full_metrics(r, lab))
            rows.append(full_metrics(r[r.index >= OOS_START], f"    ^ OOS {OOS_START.date()}+"))
        show(rows, f"{tf}")
        out[tf] = rows

    pd.DataFrame(series).to_parquet(BREAKOUT_DIR / "bo_xs_costs.parquet")
    (BREAKOUT_DIR / "bo_xs_costs.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nBO XS COSTS OK")


if __name__ == "__main__":
    main()
