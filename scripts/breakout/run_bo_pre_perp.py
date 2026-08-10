"""Can the breakout book start in 2017 instead of 2020? Spot history, honest about shorting.

Spot klines reach back to 2017-08 and the construction scores +1.34 on that block
(`run_bo_spot.py`), which invites simply starting the book there. The invitation is a trap, and the
reason is not statistical: **for most of that block there was no way to short.**

  - **before 2019-07-11** — Binance had no margin trading at all (it launched with "Binance 2.0" on
    that date), and Binance USD-M perps are not in the cache until 2020-01. A short leg backtested
    over 2017-08 → 2019-07 is an instrument that did not exist. BitMEX had a BTC perp from 2016, but
    that is one coin on another venue in a quanto contract — not this panel and not this cost model.
  - **2019-07-11 → the symbol's perp listing** — shortable on spot margin, paying the coin borrow.
  - **after the symbol's perp listing** — shortable on perps, which pay funding instead of borrow.

So this script builds the book three ways over 2017-08 → 2026-07 and lets the difference speak:

  naive       long-short throughout, shorts free from 2017 — what "just start in 2017" means
  tradeable   the short leg exists only when a venue for it existed, era by era, symbol by symbol
  long-only   no short leg at all, as the floor

Universe is point-in-time top-10 by trailing spot dollar volume (the core-10 is a 2026 list and half
of it has not listed in 2017), with the frozen core-10 alongside so the two are comparable. The
verdict is then carried to the master book: does an earlier-starting breakout leg move the five
scored targets, or is it a longer line on a chart that changes nothing?

    python scripts/breakout/run_bo_pre_perp.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import bo_common as bo  # noqa: E402
from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.config import (BREAKOUT_DIR, CRYPTO_PPY, CRYPTO_SPOT_BORROW_BPS_ANNUAL,  # noqa: E402
                        OOS_START, RAW_DIR)
from src.sleeves import breakout_lab as bl  # noqa: E402
from scripts.breakout.run_bo_spot import (CORE10, VENUE, _borrow, load, pit_membership,  # noqa: E402
                                          spot_dollar_volume_panel, stats)

# Binance margin trading went live with "Binance 2.0" — before this there is no short on this venue.
MARGIN_LIVE = pd.Timestamp("2019-07-11", tz="UTC")
WINDOW = ("2017-08-01", "2026-07-31")
BLOCKS = {"2017-08..2019-07 (no short venue)": ("2017-08-01", "2019-07-10"),
          "2019-07..2019-12 (spot margin only)": ("2019-07-11", "2019-12-31"),
          "2020+ (perps live)": ("2020-01-01", "2026-07-31")}
TFS = ["1d", "4h"]


def perp_start(sym: str, tf: str) -> pd.Timestamp | None:
    """First bar this symbol's perp actually has — the date its short leg can move to futures."""
    d = RAW_DIR / "futures/um/klines" / sym / tf
    files = sorted(d.glob("[0-9]*.parquet")) if d.exists() else []
    if not files:
        return None
    return pd.read_parquet(files[0]).index.min()


def sleeve(sym: str, tf: str, mode: str, mask: pd.Series | None) -> pd.Series | None:
    """One sleeve's daily net return. Signal and longs on spot; the short leg depends on `mode`."""
    sp = load("spot", sym, tf, *WINDOW)
    if sp is None:
        return None
    c, h, l = sp["close"], sp["high"], sp["low"]
    pos = bl.hold_atr_trailing(c, h, l, bl.donchian_side(c, h, l, 55), 3.0, 14)
    valid = None
    if mask is not None:
        pos = pos.where(mask.reindex(pos.index.normalize()).fillna(False).to_numpy(), 0.0)
        # outside a point-in-time universe the slot does not exist — its return must be NaN, not a
        # zero that drags the book's equal weighting down. Widened by the execution lag so the
        # closing trade's cost still lands inside the window.
        days = pd.date_range(pos.index[0].normalize(), pos.index[-1].normalize(), freq="D", tz="UTC")
        valid = mask.reindex(days).fillna(False).rolling(3, min_periods=1).max().astype(bool)
    if pos.abs().sum() == 0:
        return None
    posv = vol_target(pos, c, bo.TVOL, bo.CRYPTO_TF[tf])
    lng, sht = posv.clip(lower=0.0), posv.clip(upper=0.0)
    sp_adv = sp["quote_volume"].rolling(20).median().shift(1)

    net = backtest(c, lng, capital=bo.CAP, funding=None, adv=sp_adv, **VENUE["spot"]["costs"])["net_ret"]
    if mode != "long_only":
        # `naive` differs from `tradeable` in exactly one thing: whether the short leg is allowed
        # before a venue for it existed. Both route shorts to perps once the symbol lists one, so
        # the comparison isolates the shorting-availability constraint and nothing else.
        pstart = perp_start(sym, tf)
        floor = posv.index[0] if mode == "naive" else MARGIN_LIVE
        era_perp = pd.Series(False if pstart is None else posv.index >= pstart, index=posv.index)
        era_margin = pd.Series((posv.index >= floor) & ~era_perp.to_numpy(), index=posv.index)
        s_margin = sht.where(era_margin.to_numpy(), 0.0)
        bt = backtest(c, s_margin, capital=bo.CAP, funding=None, adv=sp_adv, **VENUE["spot"]["costs"])
        net = net + bt["net_ret"] - _borrow(bt["position"], CRYPTO_SPOT_BORROW_BPS_ANNUAL,
                                            bo.CRYPTO_TF[tf])
        if era_perp.any():
            pp = load("perp", sym, tf, *WINDOW)
            if pp is not None:
                idx = c.index.intersection(pp.index)
                s_perp = sht.where(era_perp.to_numpy(), 0.0).reindex(idx).fillna(0.0)
                bp = backtest(pp["close"], s_perp, capital=bo.CAP, funding=bo.safe_funding(sym),
                              adv=pp["quote_volume"].rolling(20).median().shift(1), **VENUE["perp"]["costs"])
                net = net.add(bp["net_ret"], fill_value=0.0)
    daily = (1 + net).resample("D").prod() - 1
    return daily if valid is None else daily.where(valid.reindex(daily.index).fillna(False))


def book(syms: list[str], mode: str, memb: pd.DataFrame | None) -> pd.Series:
    cols = {}
    for tf in TFS:
        for s in syms:
            m = memb[s] if (memb is not None and s in memb.columns) else None
            r = sleeve(s, tf, mode, m)
            if r is not None:
                cols[f"{s}_{tf}"] = r
    return pd.DataFrame(cols).sort_index().mean(axis=1) if cols else pd.Series(dtype=float)


def show(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'variant':<44}{'Sharpe':>8}{'MC-P5':>8}{'CAGR':>9}{'vol':>8}{'maxDD':>9}{'mo+':>6}")
    for r in rows:
        if not np.isfinite(r.get("sharpe", np.nan)):
            print(f"{r['label']:<44}{'n/a':>8}"); continue
        mc = f"{r['mc_p5']:+.2f}" if np.isfinite(r.get("mc_p5", np.nan)) else "  — "
        print(f"{r['label']:<44}{r['sharpe']:+8.2f}{mc:>8}{r['cagr']:+9.1%}{r['vol']:8.1%}"
              f"{r['max_dd']:+9.1%}{r['months_in_profit']:6.0%}")


def main():
    print("=== CAN THE BREAKOUT BOOK START IN 2017? spot history vs shorting reality ===")
    print(f"signal + longs on spot from {WINDOW[0]}; shorts only where a venue existed "
          f"(margin from {MARGIN_LIVE.date()}, perps at each symbol's listing)\n")

    memb = pit_membership(spot_dollar_volume_panel(), top_n=10)
    pit_syms = [c for c in memb.columns if memb[c].any()]
    series, rows = {}, []
    for uname, syms, m in (("PIT-top10", pit_syms, memb), ("frozen-core10", CORE10, None)):
        for mode in ("naive", "tradeable", "long_only"):
            port = book(syms, mode, m)
            if port.dropna().empty:
                continue
            key = f"{uname} {mode}"
            series[key] = port
            rows.append(stats(port, key))
    show(rows, f"FULL WINDOW {WINDOW[0]}..{WINDOW[1]}")

    for bname, (a, b) in BLOCKS.items():
        show([stats(p.loc[a:b], k) for k, p in series.items() if len(p.loc[a:b].dropna()) > 60], bname)

    show([stats(p[p.index >= OOS_START], k) for k, p in series.items()], f"OOS {OOS_START.date()}+")

    py = pd.DataFrame({k: {int(y): (round(float(np.sqrt(CRYPTO_PPY) * g.mean() / g.std(ddof=1)), 2)
                                    if g.std(ddof=1) > 0 else 0.0)
                           for y, g in v.dropna().groupby(v.dropna().index.year)}
                       for k, v in series.items()}).T.sort_index(axis=1)
    print("\n=== per-year Sharpe ===")
    print(py.to_string(float_format=lambda v: f"{v:+.2f}" if np.isfinite(v) else "   —"))

    pd.DataFrame(series).to_parquet(BREAKOUT_DIR / "bo_pre_perp_series.parquet")
    py.to_csv(BREAKOUT_DIR / "bo_pre_perp_per_year.csv")
    (BREAKOUT_DIR / "bo_pre_perp.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\nBO PRE-PERP OK")


if __name__ == "__main__":
    main()
