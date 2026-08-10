"""How far back does the breakout book actually go, and what does it earn there — full metrics.

Binance spot starts 2017-08 because Binance opened in July 2017; that is a venue limit, not the
limit of crypto price history. Coin Metrics' free community reference price reaches **2010-07 for
BTC**, 2013 for LTC, 2014 for DOGE/DASH/XMR/XRP and 2015 for ETH — sixteen years instead of six.

Two things have to be paid for to use it, and both are measured here rather than waved through:

  1. **The reference price is close-only.** There is no high, no low and no volume, so Donchian
     channels and Wilder ATR — which read high/low — are not computable as specified. The
     construction is degraded to a close-only twin (channel on closes, true range from
     close-to-close), and §1 calibrates that twin against the real one over 2017-2026 where Binance
     OHLC exists. Whatever the twin loses there is the error carried back into the deep window.
  2. **Shorting did not exist for most of it.** Binance margin went live 2019-07-11; USD-M perps in
     this cache start 2020-01. So the deep book is long-only for its first nine years, and that is
     not a modelling choice — there was nothing to sell.

Reported on the full metric set, not on Sharpe: CAGR, total return, volatility, max drawdown,
Calmar, worst and best month, months in profit, longest losing streak, Sortino, the Monte-Carlo 5th
percentile, and dollars on the brief's $500k. A cost ladder is included because 2011-2013 execution
was Mt.Gox-era and nothing like the 10bps the model charges.

    python scripts/breakout/run_bo_deep_history.py
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
from src.config import (BREAKOUT_DIR, CACHE_DIR, CAPITAL_USD, CRYPTO_PPY,  # noqa: E402
                        CRYPTO_SPOT_BORROW_BPS_ANNUAL, OOS_START)
from src.data.onchain import _cm_get  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.breakout.run_bo_pre_perp import MARGIN_LIVE  # noqa: E402
from scripts.breakout.run_bo_spot import VENUE, load  # noqa: E402

# CM asset -> the Binance symbol it corresponds to, for the calibration overlap
ASSETS = {"btc": "BTCUSDT", "ltc": "LTCUSDT", "doge": "DOGEUSDT", "dash": "DASHUSDT",
          "xmr": "XMRUSDT", "xrp": "XRPUSDT", "eth": "ETHUSDT", "xlm": "XLMUSDT",
          "etc": "ETCUSDT", "zec": "ZECUSDT", "bnb": "BNBUSDT", "bch": "BCHUSDT",
          "link": "LINKUSDT", "ada": "ADAUSDT"}
DEEP_START = "2010-01-01"
CAL = ("2017-08-01", "2026-07-31")          # where Binance OHLC and the CM close both exist
LOOKBACK, K_ATR = 55, 3.0


# --- data ------------------------------------------------------------------------------------

def cm_prices() -> pd.DataFrame:
    """Daily CM community reference close per asset, cached. One network pull, then offline."""
    cache = CACHE_DIR / "book_bo" / "cm_price_1d.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = {}
    for a, sym in ASSETS.items():
        df = _cm_get(a, ["PriceUSD"], DEEP_START, pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
        if df.empty or "PriceUSD" not in df:
            print(f"    ! no CM PriceUSD for {a}")
            continue
        s = pd.to_numeric(df["PriceUSD"], errors="coerce").dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).tz_localize("UTC")
        cols[sym] = s[~s.index.duplicated(keep="first")]
        print(f"    {a:<5} -> {sym:<10} {s.index.min().date()} .. {s.index.max().date()}  n={len(s)}")
    panel = pd.DataFrame(cols).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache)
    return panel


# --- the two constructions ---------------------------------------------------------------------

def position_ohlc(px: pd.DataFrame) -> pd.Series:
    """The shipped construction: Donchian on the high/low channel, chandelier on Wilder ATR."""
    c, h, l = px["close"], px["high"], px["low"]
    return bl.hold_atr_trailing(c, h, l, bl.donchian_side(c, h, l, LOOKBACK), K_ATR, 14)


def position_close_only(close: pd.Series) -> pd.Series:
    """The close-only twin, for a series that has no high/low.

    The channel reads trailing closes instead of trailing highs and lows, and true range collapses
    to |close - previous close|. Both are strictly less informative — a close-only channel breaks
    later than a high/low one and a close-only ATR understates range, so the twin trades a little
    less and stops a little tighter. §1 measures what that costs.
    """
    hi = close.rolling(LOOKBACK).max().shift(1)
    lo = close.rolling(LOOKBACK).min().shift(1)
    side = pd.Series(0.0, index=close.index)
    side[close > hi] = 1.0
    side[close < lo] = -1.0
    return bl.hold_atr_trailing(close, close, close, side, K_ATR, 14)


def run_leg(close: pd.Series, pos: pd.Series, shortable_from: pd.Timestamp | None,
            cost_mult: float = 1.0) -> pd.Series:
    """Daily net return for one sleeve: longs on spot, shorts only once a venue existed."""
    if shortable_from is not None:
        pos = pos.where((pos > 0) | (pos.index >= shortable_from), 0.0)
    else:
        pos = pos.clip(lower=0.0)
    if pos.abs().sum() == 0:
        return None
    posv = vol_target(pos, close, bo.TVOL, CRYPTO_PPY)
    costs = dict(VENUE["spot"]["costs"])
    costs["commission_bps"] *= cost_mult
    costs["half_spread_bps"] *= cost_mult
    bt = backtest(close, posv, capital=bo.CAP, funding=None, adv=None, **costs)
    borrow = bt["position"].clip(upper=0.0).abs() * (CRYPTO_SPOT_BORROW_BPS_ANNUAL / 1e4) / CRYPTO_PPY
    return bt["net_ret"] - borrow


# --- metrics ------------------------------------------------------------------------------------

def full_metrics(r: pd.Series, label: str) -> dict:
    """Everything the brief scores plus what Sharpe hides — return, path and tail, side by side."""
    r = r.dropna()
    if len(r) < 60:
        return {"label": label, "n": len(r)}
    s = summarise(r, CRYPTO_PPY)
    yrs = len(r) / CRYPTO_PPY
    eq = (1 + r).cumprod()
    mo = (1 + r).resample("ME").prod() - 1
    neg, streak, mx = (mo <= 0).astype(int).to_numpy(), 0, 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else float("nan")
    mc = bootstrap_sharpe(r, CRYPTO_PPY, 1000, bo.SEED) if s["sharpe_ann"] > 0.2 else {}
    return {"label": label, "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "years": round(yrs, 1), "sharpe": s["sharpe_ann"], "sortino": s["sortino_ann"],
            "mc_p5": mc.get("sharpe_p5", np.nan), "cagr": cagr,
            "total_return": s["total_return"], "vol": float(r.std(ddof=1) * np.sqrt(CRYPTO_PPY)),
            "max_dd": s["max_dd"], "calmar": abs(cagr / s["max_dd"]) if s["max_dd"] else np.nan,
            "worst_month": float(mo.min()), "best_month": float(mo.max()),
            "months_in_profit": s["months_in_profit"], "streak": int(mx),
            "pnl_usd": float(r.sum() * CAPITAL_USD), "pnl_usd_yr": float(r.sum() * CAPITAL_USD / yrs)}


def show(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    hdr = (f"{'variant':<34}{'yrs':>5}{'CAGR':>8}{'total':>10}{'vol':>7}{'maxDD':>8}{'Calmar':>8}"
           f"{'worst mo':>9}{'mo+':>6}{'strk':>5}{'Sharpe':>8}{'MC-P5':>7}{'$/yr':>10}")
    print(hdr)
    for r in rows:
        if "sharpe" not in r:
            print(f"{r['label']:<34}  (too short, n={r.get('n', 0)})"); continue
        mc = f"{r['mc_p5']:+.2f}" if np.isfinite(r.get("mc_p5", np.nan)) else "  — "
        print(f"{r['label']:<34}{r['years']:>5.1f}{r['cagr']:>8.1%}{r['total_return']:>10.0%}"
              f"{r['vol']:>7.1%}{r['max_dd']:>8.1%}{r['calmar']:>8.2f}{r['worst_month']:>9.1%}"
              f"{r['months_in_profit']:>6.0%}{r['streak']:>5d}{r['sharpe']:>+8.2f}{mc:>7}"
              f"{r['pnl_usd_yr']:>10,.0f}")


def book(rets: dict[str, pd.Series]) -> pd.Series:
    return pd.DataFrame(rets).sort_index().mean(axis=1) if rets else pd.Series(dtype=float)


def main():
    print("=== HOW FAR BACK DOES THE BREAKOUT BOOK GO — full metrics, not just Sharpe ===")
    px = cm_prices()
    print(f"CM reference-price panel: {px.shape[1]} assets, "
          f"{px.index.min().date()} .. {px.index.max().date()}")
    live = px.notna().sum(axis=1)
    for y in [2011, 2014, 2016, 2018, 2020, 2024]:
        n = int(live.loc[str(y)].mean()) if str(y) in live.index.strftime("%Y") else 0
        print(f"    assets live in {y}: {n}")

    # --- 1. calibration: what does the close-only twin cost, where both are computable? -----
    ohlc, twin = {}, {}
    for sym in ASSETS.values():
        b = load("spot", sym, "1d", *CAL)
        if b is None:
            continue
        a = run_leg(b["close"], position_ohlc(b), MARGIN_LIVE)
        t = run_leg(b["close"], position_close_only(b["close"]), MARGIN_LIVE)
        if a is not None and t is not None:
            ohlc[sym], twin[sym] = (1 + a).resample("D").prod() - 1, (1 + t).resample("D").prod() - 1
    cal = [full_metrics(book(ohlc), "full OHLC construction (shipped)"),
           full_metrics(book(twin), "close-only twin (what CM allows)")]
    show(cal, f"1. CALIBRATION on Binance spot {CAL[0]}..{CAL[1]} — {len(ohlc)} assets, same rules")
    d = cal[1]["sharpe"] - cal[0]["sharpe"]
    print(f"    -> the close-only degradation is worth {d:+.2f} Sharpe and "
          f"{cal[1]['cagr'] - cal[0]['cagr']:+.1%} CAGR on the overlap; carry that as the error bar.")

    # --- 2. the deep run ------------------------------------------------------------------
    rows, series = [], {}
    for mode, short_from in (("long+short where possible", MARGIN_LIVE), ("long-only throughout", None)):
        legs = {}
        for sym in px.columns:
            c = px[sym].dropna()
            if len(c) < 400:
                continue
            r = run_leg(c, position_close_only(c), short_from)
            if r is not None:
                legs[sym] = r
        port = book(legs)
        series[mode] = port
        rows.append(full_metrics(port, mode))
    show(rows, f"2. DEEP RUN on CM reference prices, {px.index.min().date()}+ (close-only twin)")

    # --- 3. the same book cut by era, so the deep window is not one number -----------------
    eras = {"2010-2013 (BTC only, Mt.Gox era)": ("2010-01-01", "2013-12-31"),
            "2014-2016 (pre-Binance)": ("2014-01-01", "2016-12-31"),
            "2017-2019 (Binance, no short until 2019-07)": ("2017-01-01", "2019-12-31"),
            "2020-2026 (perps live)": ("2020-01-01", "2026-12-31")}
    for name, (a, b) in eras.items():
        show([full_metrics(p.loc[a:b], k) for k, p in series.items()], f"3. {name}")

    # --- 4. cost ladder: early-era execution was nothing like 10bps -----------------------
    ladder = []
    for m in (1.0, 3.0, 5.0, 10.0):
        legs = {}
        for sym in px.columns:
            c = px[sym].dropna()
            if len(c) < 400:
                continue
            r = run_leg(c, position_close_only(c), MARGIN_LIVE, cost_mult=m)
            if r is not None:
                legs[sym] = r
        ladder.append(full_metrics(book(legs), f"{m:.0f}x base cost ({10 * m:.0f}bps taker)"))
    show(ladder, "4. COST LADDER on the deep run — how much execution the edge survives")

    show([full_metrics(p[p.index >= OOS_START], k) for k, p in series.items()],
         f"5. FROZEN OOS BLOCK {OOS_START.date()}+")

    pd.DataFrame(series).to_parquet(BREAKOUT_DIR / "bo_deep_history.parquet")
    (BREAKOUT_DIR / "bo_deep_history.json").write_text(
        json.dumps({"calibration": cal, "deep": rows, "cost_ladder": ladder}, indent=2, default=float))
    print("\nBO DEEP HISTORY OK")


if __name__ == "__main__":
    main()
