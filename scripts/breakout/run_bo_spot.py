"""Breakout on SPOT vs PERP — venue A/B, the pre-2020 history, and a point-in-time universe.

The shipped breakout book trades USD-M perpetuals and therefore starts 2020-01, the month perps
list. Binance SPOT klines for the same names reach back to 2017-08, so three questions that the
perp-only book cannot answer are answerable here:

  1. **Venue** — does the construction survive spot economics? The two venues differ in three ways
     that pull in opposite directions: spot pays 2x the taker fee (10bps vs 5bps), spot pays no
     funding (a perp long has paid ~10%/yr on average since 2020), and a spot short is not free —
     the coin must be borrowed on margin (~2.9%/yr for the core-10, live rate). Matched window
     2020-01+ so the venue is the only thing that changes.
  2. **History** — 2017-08 -> 2019-12 is data the construction has never been exposed to, and it
     contains the 2018 bear market. It is also *less* survivorship-biased than the perp panel:
     the early spot cross-section is the 2017-bubble cohort (EOS, ICX, IOTA, HOT), names that
     later faded and that no current-perp universe contains.
  3. **Universe** — the shipped book freezes a core-10 chosen by 2026 market cap. Ranking the
     spot panel by *trailing* dollar volume rebuilds the universe point-in-time, so the hindsight
     premium in that frozen list can be measured instead of assumed.

The venue asymmetry implies a fourth variant that neither pure venue can express: run the LONG leg
on spot (no funding drag) and the SHORT leg on perps (no borrow, and it *collects* funding).

Every leg is reported with return and volatility beside the ratio: these are vol-targeted streams,
so Sharpe alone is blind to how much money a variant actually makes.

    python scripts/breakout/run_bo_spot.py
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
from src.config import (BINANCE_FUT_TAKER_BPS, BINANCE_SPOT_TAKER_BPS,  # noqa: E402
                        CACHE_DIR, CRYPTO_HALF_SPREAD_BPS, CRYPTO_PPY,
                        BREAKOUT_DIR, CRYPTO_SPOT_BORROW_BPS_ANNUAL, IMPACT_K,
                        OOS_START, RAW_DIR)
from src.metrics import summarise  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CORE10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

VENUE = {
    "perp": dict(root=RAW_DIR / "futures/um/klines",
                 costs=dict(commission_bps=BINANCE_FUT_TAKER_BPS,
                            half_spread_bps=CRYPTO_HALF_SPREAD_BPS,
                            impact_k=IMPACT_K, exec_lag=2),
                 funding=True, borrow_bps=0.0),
    "spot": dict(root=RAW_DIR / "spot/klines",
                 costs=dict(commission_bps=BINANCE_SPOT_TAKER_BPS,
                            half_spread_bps=CRYPTO_HALF_SPREAD_BPS,
                            impact_k=IMPACT_K, exec_lag=2),
                 funding=False, borrow_bps=CRYPTO_SPOT_BORROW_BPS_ANNUAL),
}
MATCHED = ("2020-01-01", "2026-07-31")     # both venues live -> venue is the only difference
EXTENDED = ("2017-08-01", "2026-07-31")    # everything spot has
PRE_PERP = ("2017-08-01", "2019-12-31")    # the block the construction has never seen
CACHE = CACHE_DIR / "book_bo"


# --- data -------------------------------------------------------------------------

def load(venue: str, sym: str, tf: str, lo: str, hi: str) -> pd.DataFrame | None:
    """Monthly parquets for one symbol/timeframe on one venue, clipped to [lo, hi]. Offline only."""
    d = VENUE[venue]["root"] / sym / tf
    files = sorted(d.glob("[0-9]*.parquet")) if d.exists() else []
    if not files:
        return None
    px = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    px = px[~px.index.duplicated(keep="first")]
    px = px[(px.index >= pd.Timestamp(lo, tz="UTC")) & (px.index <= pd.Timestamp(hi, tz="UTC"))]
    return px if len(px) >= 500 else None


def _clip_epoch(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop the handful of rows Binance stamps at the epoch instead of the real bar time."""
    return panel[panel.index >= pd.Timestamp("2017-01-01", tz="UTC")]


def spot_dollar_volume_panel() -> pd.DataFrame:
    """Daily quote-volume panel over every cached spot symbol — the PIT liquidity ranking input."""
    cache = CACHE / "spot_qv_1d.parquet"
    if cache.exists():
        return _clip_epoch(pd.read_parquet(cache))
    cols = {}
    for p in sorted(VENUE["spot"]["root"].iterdir()):
        files = sorted((p / "1d").glob("[0-9]*.parquet")) if (p / "1d").exists() else []
        if not files:
            continue
        px = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
        cols[p.name] = px[~px.index.duplicated(keep="first")]["quote_volume"]
    panel = _clip_epoch(pd.DataFrame(cols).sort_index())
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache)
    return panel


def pit_membership(panel: pd.DataFrame, top_n: int = 10, lookback: int = 63) -> pd.DataFrame:
    """Boolean daily membership: top-N by *trailing* median dollar volume, lagged one day.

    Ranking on trailing volume only — a coin joins once it has actually become liquid and leaves
    when it fades, so the universe knows nothing the market did not already know that day.
    """
    trail = panel.rolling(lookback, min_periods=lookback // 2).median().shift(1)
    rank = trail.rank(axis=1, ascending=False, method="first")
    return (rank <= top_n) & trail.notna()


# --- one sleeve --------------------------------------------------------------------

def _signal(px: pd.DataFrame, long_only: bool, lookback: int = 55, k_atr: float = 3.0) -> pd.Series:
    close, high, low = px["close"], px["high"], px["low"]
    pos = bl.hold_atr_trailing(close, high, low,
                               bl.donchian_side(close, high, low, lookback), k_atr, 14)
    return pos.clip(lower=0.0) if long_only else pos


def _borrow(pos: pd.Series, bps_annual: float, ppy_bar: float) -> pd.Series:
    """Per-bar coin-borrow charge on short gross — same convention as the equity short-borrow."""
    if not bps_annual:
        return pd.Series(0.0, index=pos.index)
    return pos.clip(upper=0.0).abs() * (bps_annual / 1e4) / ppy_bar


def _to_daily(net: pd.Series, cost: pd.Series, valid: pd.Series | None) -> pd.DataFrame:
    daily = pd.DataFrame({"ret": (1 + net).resample("D").prod() - 1,
                          "cost": cost.resample("D").sum()}).dropna(subset=["ret"])
    if valid is not None:                       # outside a PIT universe the slot does not exist
        v = valid.reindex(daily.index).fillna(False)
        daily = daily.where(v, np.nan)
    return daily


def sleeve(venue: str, sym: str, tf: str, lo: str, hi: str, *, long_only: bool = False,
           mask: pd.Series | None = None) -> pd.DataFrame | None:
    """Donchian-55 -> chandelier(3) on one symbol/venue/timeframe -> daily net return + cost.

    `mask` (daily boolean, optional) forces the position flat outside a point-in-time universe. It
    is applied to the *position* before the backtest so the membership churn pays its cost, and the
    daily return is blanked outside the window (widened by the execution lag so the closing trade's
    cost still lands inside) — an unheld slot must not dilute the book's equal weighting.
    """
    px = load(venue, sym, tf, lo, hi)
    if px is None:
        return None
    v = VENUE[venue]
    pos = _signal(px, long_only)
    valid = None
    if mask is not None:
        days = pos.index.normalize()
        pos = pos.where(mask.reindex(days).fillna(False).to_numpy(), 0.0)
        valid = mask.reindex(pd.date_range(pos.index[0].normalize(), pos.index[-1].normalize(),
                                           freq="D", tz="UTC")).fillna(False)
        valid = valid.rolling(3, min_periods=1).max().astype(bool)   # keep the exit bar's cost
    if pos.abs().sum() == 0:
        return None
    posv = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bt = backtest(px["close"], posv, capital=bo.CAP,
                  funding=bo.safe_funding(sym) if v["funding"] else None,
                  adv=px["quote_volume"].rolling(20).median().shift(1), **v["costs"])
    borrow = _borrow(bt["position"], v["borrow_bps"], bo.CRYPTO_TF[tf])
    return _to_daily(bt["net_ret"] - borrow, bt["cost"] + borrow, valid)


def sleeve_split(sym: str, tf: str, lo: str, hi: str) -> pd.DataFrame | None:
    """Venue-split execution: the LONG leg on spot, the SHORT leg on perps.

    Spot longs skip the ~10%/yr funding a perp long pays; perp shorts skip the coin borrow a spot
    short pays and collect that same funding instead. The side is taken from the spot series and
    each leg is filled on its own venue's prices and cost schedule; a flip therefore pays two
    trades (close the spot long, open the perp short), which is charged.
    """
    sp, pp = load("spot", sym, tf, lo, hi), load("perp", sym, tf, lo, hi)
    if sp is None or pp is None:
        return None
    idx = sp.index.intersection(pp.index)
    sp, pp = sp.loc[idx], pp.loc[idx]
    posv = vol_target(_signal(sp, long_only=False), sp["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bl_ = backtest(sp["close"], posv.clip(lower=0.0), capital=bo.CAP, funding=None,
                   adv=sp["quote_volume"].rolling(20).median().shift(1), **VENUE["spot"]["costs"])
    bs_ = backtest(pp["close"], posv.clip(upper=0.0), capital=bo.CAP, funding=bo.safe_funding(sym),
                   adv=pp["quote_volume"].rolling(20).median().shift(1), **VENUE["perp"]["costs"])
    return _to_daily(bl_["net_ret"] + bs_["net_ret"], bl_["cost"] + bs_["cost"], None)


# --- book aggregation ---------------------------------------------------------------

def stats(port: pd.Series, label: str = "") -> dict:
    """Sharpe plus the things Sharpe hides on a vol-targeted stream: CAGR, vol, exposure, DD."""
    r = port.dropna()
    if len(r) < 30:
        return {"label": label, "n": len(r), "sharpe": np.nan}
    s = summarise(r, CRYPTO_PPY)
    yrs, eq = len(r) / CRYPTO_PPY, (1 + r).prod()
    out = {"label": label, "sharpe": s["sharpe_ann"],
           "cagr": float(eq ** (1 / yrs) - 1) if yrs > 0 and eq > 0 else float("nan"),
           "vol": float(r.std(ddof=1) * np.sqrt(CRYPTO_PPY)),
           "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
           "total": s["total_return"], "n": int(len(r)),
           "start": str(r.index[0].date()), "end": str(r.index[-1].date())}
    mc = (bootstrap_sharpe(r, CRYPTO_PPY, n_reps=1000, seed=bo.SEED)
          if s["sharpe_ann"] > 0.3 else {})
    out["mc_p5"] = mc.get("sharpe_p5", np.nan)
    return out


def combine(sleeves: dict[str, pd.DataFrame], how: str = "active") -> pd.Series:
    """Equal-weight the sleeve daily returns.

    `active` divides by the sleeves actually live that day — the shipped book divides by the full
    slot count (`legacy`), which silently under-invests while names are still listing.
    """
    if not sleeves:
        return pd.Series(dtype=float)
    rets = pd.DataFrame({k: v["ret"] for k, v in sleeves.items()}).sort_index()
    return rets.mean(axis=1) if how == "active" else rets.fillna(0.0).mean(axis=1)


def build(venue: str, syms: list[str], tfs: list[str], lo: str, hi: str, *,
          long_only: bool = False, memb: pd.DataFrame | None = None) -> dict:
    out = {}
    for tf in tfs:
        for sym in syms:
            mask = memb[sym] if (memb is not None and sym in memb.columns) else None
            fn = (lambda: sleeve_split(sym, tf, lo, hi)) if venue == "split" else \
                 (lambda: sleeve(venue, sym, tf, lo, hi, long_only=long_only, mask=mask))
            d = fn()
            if d is not None:
                out[f"{sym}_{tf}"] = d
    return out


def show(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'variant':<38}{'Sharpe':>8}{'MC-P5':>8}{'CAGR':>9}{'vol':>8}{'maxDD':>9}{'mo+':>6}  window")
    for r in rows:
        if not np.isfinite(r.get("sharpe", np.nan)):
            print(f"{r['label']:<38}{'n/a':>8}   (n={r.get('n', 0)})")
            continue
        mc = f"{r['mc_p5']:+.2f}" if np.isfinite(r.get("mc_p5", np.nan)) else "  — "
        print(f"{r['label']:<38}{r['sharpe']:+8.2f}{mc:>8}{r['cagr']:+9.1%}{r['vol']:8.1%}"
              f"{r['max_dd']:+9.1%}{r['months_in_profit']:6.0%}  {r['start']}..{r['end']}")


def per_year(series: dict) -> pd.DataFrame:
    out = {}
    for key, port in series.items():
        p = port.dropna()
        out[key] = {int(y): (round(float(np.sqrt(CRYPTO_PPY) * g.mean() / g.std(ddof=1)), 2)
                             if g.std(ddof=1) > 0 else 0.0)
                    for y, g in p.groupby(p.index.year)}
    return pd.DataFrame(out).T.sort_index(axis=1)


# --- stages ---------------------------------------------------------------------------

def stage_venue_ab() -> tuple[list[dict], dict]:
    """Matched 2020+ window: perp vs spot vs venue-split, long-short vs long-only, per timeframe."""
    lo, hi = MATCHED
    rows, series = [], {}
    cases = [("perp", "LS", False), ("perp", "LO", True),
             ("spot", "LS", False), ("spot", "LO", True),
             ("split", "LS", False)]
    for venue, direction, lo_only in cases:
        for tfs, tag in ([["1d"], "1d"], [["4h"], "4h"], [["1h"], "1h"], [["1d", "4h"], "1d+4h"]):
            port = combine(build(venue, CORE10, tfs, lo, hi, long_only=lo_only))
            key = f"{venue} {direction} {tag}"
            series[key] = port
            rows.append(stats(port, key))
    return rows, series


def stage_history() -> tuple[list[dict], dict]:
    """Spot back to 2017-08: frozen core-10 vs point-in-time top-10."""
    lo, hi = EXTENDED
    memb = pit_membership(spot_dollar_volume_panel(), top_n=10)
    pit_syms = [c for c in memb.columns if memb[c].any()]
    print(f"    PIT universe: {len(pit_syms)} symbols ever in the top-10 by trailing dollar volume")
    rows, series = [], {}
    for direction, lo_only in (("LS", False), ("LO", True)):
        for tfs, tag in ([["1d"], "1d"], [["1d", "4h"], "1d+4h"]):
            for label, port in (
                    (f"spot {direction} {tag} frozen-core10",
                     combine(build("spot", CORE10, tfs, lo, hi, long_only=lo_only))),
                    (f"spot {direction} {tag} PIT-top10",
                     combine(build("spot", pit_syms, tfs, lo, hi, long_only=lo_only, memb=memb)))):
                series[label] = port
                rows.append(stats(port, label))
    return rows, series


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    print("=== BREAKOUT ON SPOT — venue A/B, pre-perp history, point-in-time universe ===")
    print("construction frozen: Donchian-55 entry -> chandelier ATR(3) exit, vol-target 15%, t+2 exec")
    print(f"perp {BINANCE_FUT_TAKER_BPS:.0f}bps + funding | spot {BINANCE_SPOT_TAKER_BPS:.0f}bps, no funding, "
          f"shorts pay {CRYPTO_SPOT_BORROW_BPS_ANNUAL/100:.2f}%/yr coin-borrow | split = long spot, short perp")

    ab_rows, ab_series = stage_venue_ab()
    show(ab_rows, "VENUE A/B — matched window 2020-01..2026-07, frozen core-10")

    h_rows, h_series = stage_history()
    show(h_rows, "SPOT EXTENDED HISTORY 2017-08..2026-07")

    blk = [stats(p.loc[PRE_PERP[0]:PRE_PERP[1]], f"{k} | 2017-19")
           for k, p in h_series.items() if len(p.loc[PRE_PERP[0]:PRE_PERP[1]].dropna()) > 200]
    show(blk, "NEVER-SEEN BLOCK — 2017-08..2019-12 (before perps existed)")

    allser = {**ab_series, **h_series}
    oos = [stats(p[p.index >= OOS_START], f"{k} | OOS") for k, p in allser.items()
           if len(p[p.index >= OOS_START].dropna()) > 100]
    show(oos, f"FROZEN OOS BLOCK {OOS_START.date()}+")

    py = per_year(allser)
    print("\n=== per-year Sharpe ===")
    print(py.to_string(float_format=lambda v: f"{v:+.2f}" if np.isfinite(v) else "   —"))

    pd.DataFrame(allser).to_parquet(BREAKOUT_DIR / "bo_spot_series.parquet")
    py.to_csv(BREAKOUT_DIR / "bo_spot_per_year.csv")
    (BREAKOUT_DIR / "bo_spot_summary.json").write_text(json.dumps(
        {"venue_ab": ab_rows, "history": h_rows, "pre_perp": blk, "oos": oos},
        indent=2, default=float))
    print("\nBO SPOT OK")


if __name__ == "__main__":
    main()
