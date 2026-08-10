"""The time-series breakout book, shipped assembly vs the corrected one, on one window.

The shipped book (`run_bo_final.py`) is: frozen core-10, all-perp execution, 1d raw chandelier plus
4h/1h gated by a meta-label model fitted with purged k-fold on gross labels. Three separate findings
from this review change it, and this script prices them one at a time so the reader can see which
part of the delta comes from where:

  venue   long leg on spot, short leg on perps. A perp long has paid ~23%/yr in funding *conditional
          on the book being long* (it is long exactly when funding is extreme); spot pays none. A
          perp short collects only ~2%/yr, but that still beats the ~2.9%/yr coin-borrow a spot
          short pays. Neither venue alone can express this.
  label   the meta-label is priced at execution, net of commission, spread and funding, instead of
          gross at the signal bar.
  gate    the meta-label model is fitted walk-forward on resolved trades only, instead of purged
          k-fold, which trains each fold on the future as well as the past.

    python scripts/breakout/run_bo_ts_book.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import bo_common as bo  # noqa: E402
from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.config import BREAKOUT_DIR, CRYPTO_PPY, OOS_START  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from scripts.breakout.run_bo_spot import CORE10, VENUE, _borrow, load, stats  # noqa: E402
from scripts.breakout.run_bo_ml_wf import (THR, proba_kfold, proba_walkforward,  # noqa: E402
                                           sleeve_data)

WINDOW = ("2020-01-01", "2026-07-31")     # both venues live — the only window the split can trade
TFS_ML = ["4h", "1h"]


def _net(venue: str, px: pd.DataFrame, posv: pd.Series, sym: str, tf: str) -> pd.Series:
    v = VENUE[venue]
    bt = backtest(px["close"], posv, capital=bo.CAP,
                  funding=bo.safe_funding(sym) if v["funding"] else None,
                  adv=px["quote_volume"].rolling(20).median().shift(1), **v["costs"])
    return bt["net_ret"] - _borrow(bt["position"], v["borrow_bps"], bo.CRYPTO_TF[tf])


def execute(venue: str, sym: str, tf: str, pos: pd.Series, px: pd.DataFrame) -> pd.Series:
    """Fill a target position on one venue, or split it: long leg on spot, short leg on perps."""
    posv = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    if venue != "split":
        net = _net(venue, px, posv, sym, tf)
    else:
        sp, pp = load("spot", sym, tf, *WINDOW), load("perp", sym, tf, *WINDOW)
        if sp is None or pp is None:
            return None
        idx = sp.index.intersection(pp.index).intersection(posv.index)
        net = (_net("spot", sp.loc[idx], posv.reindex(idx).clip(lower=0.0), sym, tf)
               + _net("perp", pp.loc[idx], posv.reindex(idx).clip(upper=0.0), sym, tf))
    return (1 + net).resample("D").prod() - 1


def leg_1d(venue: str) -> dict:
    """The 1d leg is never gated — ~30 Donchian-55 trades per sleeve is too few to meta-label."""
    src = "spot" if venue == "split" else venue
    out = {}
    for sym in CORE10:
        px = load(src, sym, "1d", *WINDOW)
        if px is None:
            continue
        c, h, l = px["close"], px["high"], px["low"]
        pos = bl.hold_atr_trailing(c, h, l, bl.donchian_side(c, h, l, 55), 3.0, 14)
        r = execute(venue, sym, "1d", pos, px)
        if r is not None:
            out[f"{sym}_1d"] = r
    return out


def leg_gated(venue: str, gate: str) -> dict:
    """4h+1h legs gated by the meta-label model. `gate` picks the CV and the label definition."""
    src = "spot" if venue == "split" else venue
    out = {}
    for tf in TFS_ML:
        for sym in CORE10:
            s = sleeve_data(src, sym, tf)
            if s is None:
                continue
            p = (proba_kfold(s, s["y_gross"]) if gate == "kfold/gross"
                 else proba_walkforward(s, s["y_net"]))
            if not len(p):
                continue
            keep = p.index[p.values >= THR]
            pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], keep)
            pos = pos.loc[p.index.min():]          # a walk-forward gate is silent before its model
            r = execute(venue, sym, tf, pos, s["px"].loc[pos.index])
            if r is not None:
                out[f"{sym}_{tf}"] = r
    return out


def assemble(venue: str, gate: str) -> pd.Series:
    """Equal-weight over live sleeves, clipped to the common window so rows compare like for like
    (the spot legs carry more history than the perp ones, which would otherwise flatter them)."""
    sl = {**leg_1d(venue), **leg_gated(venue, gate)}
    port = pd.DataFrame(sl).sort_index().mean(axis=1)
    return port.loc[WINDOW[0]:WINDOW[1]]


def main():
    print("=== TIME-SERIES BREAKOUT BOOK — shipped assembly vs the corrected one ===")
    print(f"frozen core-10, 1d raw chandelier + 4h/1h meta-labelled, {WINDOW[0]}..{WINDOW[1]}\n")

    steps = [("shipped: all-perp, k-fold gate, gross labels", "perp", "kfold/gross"),
             ("+ honest gate: walk-forward, net-of-cost labels", "perp", "walkfwd/net"),
             ("+ venue split: long spot, short perp", "split", "walkfwd/net"),
             ("(reference) all-spot, honest gate", "spot", "walkfwd/net")]

    rows, series, prev = [], {}, None
    print(f"{'assembly':<48}{'Sharpe':>8}{'MC-P5':>8}{'CAGR':>9}{'vol':>8}{'maxDD':>9}"
          f"{'mo+':>6}{'step':>7}   OOS")
    for label, venue, gate in steps:
        port = assemble(venue, gate)
        s, o = stats(port, label), stats(port[port.index >= OOS_START], label)
        step = "" if prev is None else f"{s['sharpe'] - prev:+.2f}"
        if not label.startswith("("):
            prev = s["sharpe"]
        mc = f"{s['mc_p5']:+.2f}" if np.isfinite(s.get("mc_p5", np.nan)) else "  — "
        print(f"{label:<48}{s['sharpe']:+8.2f}{mc:>8}{s['cagr']:+9.1%}{s['vol']:8.1%}"
              f"{s['max_dd']:+9.1%}{s['months_in_profit']:6.0%}{step:>7}   {o['sharpe']:+.2f}")
        rows.append({**s, "venue": venue, "gate": gate, "oos_sharpe": o["sharpe"],
                     "oos_cagr": o["cagr"]})
        series[label] = port

    df = pd.DataFrame(series)
    py = {k: {int(y): round(float(np.sqrt(CRYPTO_PPY) * g.mean() / g.std(ddof=1)), 2)
              for y, g in v.dropna().groupby(v.dropna().index.year)} for k, v in series.items()}
    print("\nper-year Sharpe:")
    print(pd.DataFrame(py).T.to_string(float_format=lambda v: f"{v:+.2f}"))

    df.to_parquet(BREAKOUT_DIR / "bo_ts_book.parquet")
    (BREAKOUT_DIR / "bo_ts_book.json").write_text(json.dumps({"steps": rows, "per_year": py},
                                                             indent=2, default=float))
    print("\nBO TS BOOK OK")


if __name__ == "__main__":
    main()
