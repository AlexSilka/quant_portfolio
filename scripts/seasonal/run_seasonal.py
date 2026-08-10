"""Calendar-seasonality deep-dive (HYPOTHESES.md H4) — pre-FOMC announcement drift + turn-of-month,
run through the same funnel as every other family (in/out-of-window decomposition, per-year + decay
split, shuffled-calendar placebo, cost sensitivity, block-bootstrap MC, deflated Sharpe, correlation
to the deliverable book + lift curve). Because a calendar window is *deterministic and known in
advance*, the honest execution model is hold-through-the-window (cost charged only at the edges, not
the daily round-trip that killed the overnight sleeve) — see src/sleeves/seasonal.py.

Returns are dividend-inclusive and intraday prices are re-stamped at the instant they are observed;
both matter more here than in a signal-driven family, because a calendar study reads a handful of
specific bars and the dividend calendar and the bar-labelling convention both land inside them
(src/sleeves/seasonal.py). The variant sweep over window anchor, side, assets and timeframe lives in
run_seasonal_variants.py — this driver holds the two a-priori windows.

Coverage (the request: crypto + stocks + FX, several timeframes, top-10/50/100, parameter variations):
  • pre-FOMC drift  — SPY/QQQ/IWM/DIA daily 2005→ (long history, incl. the classic-vs-decayed split),
                      the *precise* intraday 2pm→2pm window (SPY/QQQ, 5-min, 2020→), and the crypto
                      analogue (BTC/ETH daily + hourly, funding-charged).
  • turn-of-month   — SPY / BTC / FX-basket time-series, a (days_before × days_after) parameter
                      surface, and the cross-sectional top-N breadth cut (N ∈ {10,50,100,200}) on
                      crypto and stocks.
  • combined sleeve — pre-FOMC ∪ turn-of-month, the portfolio funnel (MC, deflated SR, corr-to-book).

Verdict → reports/seasonal_summary.json + docs/strategies/SEASONAL.md; artifacts → reports/figures/seasonal.png.

    python scripts/seasonal/run_seasonal.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, RAW_DIR, REPORTS_DIR, SEASONAL_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.fomc import announce_days, announce_timestamps_utc  # noqa: E402
from src.data.twelvedata import load_dividends  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import seasonal as sz  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP = REPORTS_DIR
FIG = REP / "figures"
TD = RAW_DIR / "twelvedata"
ETF_RAW = RAW_DIR / "equity_td"
CACHE = CACHE_DIR / "xs"
SEED, TVOL = SEED, VOL_TARGET_ANNUAL
rng = np.random.default_rng(SEED)

# a-priori config (declared before fit; surfaces reported, never peak-picked)
FOMC_OFFSETS = [-1]              # headline pre-FOMC window: the trading day *before* the announcement
TOM_BEFORE, TOM_AFTER = 1, 3     # classic Lakonishok-Smidt turn-of-month (−1,+3)
EQ_COST, CR_COST, FX_COST = 3.0, 6.0, 0.9   # per-side bps (equity / crypto taker / FX majors)
START_EQ = "2005-01-01"          # ETF history floor = FOMC calendar floor (src/data/fomc.py)
DECAY_SPLIT = "2018-01-01"       # classic (pre) vs recent (post) — the repo's frozen train_start


# ── loaders ─────────────────────────────────────────────────────────────────────────────────
def _etf_close(ticker: str, start=START_EQ) -> pd.Series:
    """Split-adjusted daily closes (the level series — momentum features need prices, not returns)."""
    s = pd.read_parquet(ETF_RAW / f"{ticker}_1d.parquet")["close"]
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s[s.index >= pd.Timestamp(start, tz="UTC")]


def _etf_returns(ticker: str, start=START_EQ) -> pd.Series:
    """Dividend-inclusive daily returns. `equity_td` closes are split-adjusted only, and an ETF's
    ex-dates are themselves a calendar — SPY goes ex two trading days after a quarter-end FOMC in
    ~a quarter of all meetings — so price returns book a fake loss inside the measured window."""
    s = _etf_close(ticker, start)
    try:
        div = load_dividends(ticker, start="2004-01-01")
    except Exception:
        div = None                                          # non-distributing or unavailable
    return sz.total_return(s, div)


def _intraday_price_at(ticker: str, tf: str) -> pd.Series:
    """Intraday closes (Twelve Data), UTC, RE-STAMPED at the instant each price is observed.

    Twelve Data labels a bar by its open, so the close stamped 13:55 is the 14:00 price; `asof(T)` on
    the raw index hands back a price one bar past T, which for the pre-announcement window means it
    ends after the statement instead of at it. sz.price_at_instant undoes the labelling."""
    cands = sorted(TD.glob(f"{ticker}_{tf}_*.parquet"))
    if not cands:
        return pd.Series(dtype=float)
    df = pd.concat([pd.read_parquet(p) for p in cands])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return sz.price_at_instant(df["close"], pd.Timedelta(tf))


def _panel(tag: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    C = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    apath = CACHE / f"{tag}_adv.parquet"                 # FX has no ADV panel (not a x-sectional universe)
    A = pd.read_parquet(apath).reindex_like(C) if apath.exists() else None
    if C.index.tz is None:                               # localise AFTER reindex_like (both were tz-naive) —
        C.index = C.index.tz_localize("UTC")             # localising C first would strand a tz-naive A on a
        if A is not None:                                # UTC index and reindex it to all-NaN
            A.index = A.index.tz_localize("UTC")
    return C, A


def _btc_funding_daily(index: pd.DatetimeIndex, name="BTCUSDT") -> pd.Series:
    """Daily funding drag a long pays (sum of the day's 8h settlements), aligned to `index`."""
    fr = pd.read_parquet(RAW_DIR / f"futures/um/fundingRate/{name}")["last_funding_rate"]
    daily = fr.resample("1D").sum()
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    return daily.reindex(index).fillna(0.0)


def _sh(net: pd.Series, ppy: float) -> float:
    r = net.dropna()
    sd = r.std(ddof=1)
    return float(np.sqrt(ppy) * r.mean() / sd) if sd > 0 and len(r) > 2 else 0.0


def _wf_oos(M: pd.DataFrame, ppy: float, train_bars: int, test_bars: int, embargo: int,
            top_k: int = 3) -> tuple[pd.Series, int]:
    """Purged/embargoed walk-forward: on each train block pick the best-Sharpe calendar-window
    configs, apply to the next block, stitch OOS. Same machinery as run_bab._wf_oos, applied to the
    (window-shape) grid — it shows whether *selecting* a calendar window in-sample survives OOS."""
    segs, n_ref = [], 0
    start = train_bars
    while start + test_bars <= len(M):
        train = M.iloc[max(0, start - train_bars):max(0, start - embargo)]
        test = M.iloc[start:start + test_bars]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        chosen = list(sr.nlargest(top_k).index)
        segs.append(test[chosen].mean(axis=1))
        n_ref += 1
        start += test_bars
    return (pd.concat(segs) if segs else pd.Series(dtype=float)), n_ref


# ── placebo: same-shaped window on random anchors (kills the calendar, keeps the marginals) ──
def _placebo_sharpe(ret: pd.Series, n_anchors: int, offsets: list[int], cost_bps: float,
                    ppy: float, n_iter: int = 200, funding: pd.Series | None = None) -> np.ndarray:
    """Distribution of net Sharpe from `n_anchors` *random* anchor bars (same window shape, same count).

    The real calendar must beat this: if a random set of days with the identical window shape earns the
    same Sharpe, the "effect" is just the base drift of being long, not the event. Anchors are drawn far
    enough from the edges that every offset lands in-sample.
    """
    idx = ret.index
    lo, hi = -min(offsets) if offsets else 0, max(offsets) if offsets else 0
    pool = np.arange(lo + 1, len(idx) - hi - 1)
    out = np.empty(n_iter)
    for i in range(n_iter):
        picks = idx[rng.choice(pool, size=min(n_anchors, len(pool)), replace=False)]
        pos = sz.window_position(idx, pd.DatetimeIndex(picks), offsets)
        net = sz.hold_backtest(ret, pos, cost_bps=cost_bps)["net"]
        if funding is not None:
            net = net - pos * funding
        out[i] = _sh(net, ppy)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — pre-FOMC drift, equity ETFs, daily, long history + decay split
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_fomc_equity() -> dict:
    print(f"\n{'='*82}\nSECTION 1 — PRE-FOMC DRIFT (US equity index ETFs, daily, {START_EQ[:4]}→)\n{'='*82}")
    ann = announce_days("UTC")
    out, books = {}, {}
    for t in ("SPY", "QQQ", "IWM", "DIA"):
        ret = _etf_returns(t)
        n_ev = int(((ann >= ret.index.min()) & (ann <= ret.index.max())).sum())

        # window comparison: day-before / announce-day / both / two-days-before
        wins = {"day_before[-1]": [-1], "announce[0]": [0], "both[-1,0]": [-1, 0], "run[-2,-1,0]": [-2, -1, 0]}
        wtab = {}
        for name, off in wins.items():
            pos = sz.window_position(ret.index, ann, off)
            dec = sz.in_vs_out(ret, pos)
            net = sz.hold_backtest(ret, pos, cost_bps=EQ_COST)["net"]
            wtab[name] = {**dec, "net_sharpe": round(_sh(net, 252), 3)}

        # per-offset drift map: mean bps on each single offset −3..+2 (where does the drift live?)
        offmap = {off: round(float(sz.offset_event_returns(ret, ann, off).mean() * 1e4), 2)
                  for off in range(-3, 3)}

        # headline book = day-before timing; decay split + fill-timing robustness
        pos = sz.window_position(ret.index, ann, FOMC_OFFSETS)
        bt = sz.hold_backtest(ret, pos, cost_bps=EQ_COST)
        net = bt["net"]
        shift1 = sz.hold_backtest(ret, pos, cost_bps=EQ_COST, exec_shift=1)["net"]   # enter a day later
        pre = net[net.index < pd.Timestamp(DECAY_SPLIT, tz="UTC")]
        post = net[net.index >= pd.Timestamp(DECAY_SPLIT, tz="UTC")]
        per_year = {int(y): round(_sh(g, 252), 2) for y, g in net.dropna().groupby(net.dropna().index.year)}

        # placebo + cost
        placebo = _placebo_sharpe(ret, n_ev, FOMC_OFFSETS, EQ_COST, 252)
        pctile = float((_sh(net, 252) > placebo).mean() * 100)
        cost_sens = {f"{m:.0f}x": round(_sh(sz.hold_backtest(ret, pos, cost_bps=EQ_COST * m)["net"], 252), 3)
                     for m in (0, 1, 2, 3)}

        out[t] = {"n_events": n_ev, "windows": wtab, "offset_mean_bps": offmap,
                  "headline_day_before": {"net_sharpe": round(_sh(net, 252), 3),
                                          "in_window_mean_bps": wtab["day_before[-1]"]["in_window_mean_bps"],
                                          "in_window_sharpe": wtab["day_before[-1]"]["in_window_sharpe"],
                                          "share_of_total_logret": wtab["day_before[-1]"]["in_window_share_of_total_logret"],
                                          "sharpe_pre_2018": round(_sh(pre, 252), 3),
                                          "sharpe_post_2018": round(_sh(post, 252), 3),
                                          "sharpe_exec_shift1": round(_sh(shift1, 252), 3)},
                  "per_year": per_year,
                  "placebo": {"real_pctile": round(pctile, 0), "placebo_mean": round(float(placebo.mean()), 3),
                              "placebo_p95": round(float(np.percentile(placebo, 95)), 3)},
                  "cost_sensitivity": cost_sens}
        books[f"fomc_{t.lower()}"] = net
        w = wtab["day_before[-1]"]
        print(f"  {t}: {n_ev} events | day-before mean {w['in_window_mean_bps']:+.1f}bps (out "
              f"{sz.in_vs_out(ret, sz.window_position(ret.index, ann, [-1]))['out_window_ann_ret_pct']/252*100:+.1f}) "
              f"in-win Sharpe {w['in_window_sharpe']:+.2f} | net {out[t]['headline_day_before']['net_sharpe']:+.2f} "
              f"(pre-18 {out[t]['headline_day_before']['sharpe_pre_2018']:+.2f} → post-18 "
              f"{out[t]['headline_day_before']['sharpe_post_2018']:+.2f}) | placebo p{pctile:.0f}")
    return {"by_etf": out, "_books": books}


# ════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — pre-FOMC drift, precise intraday 2pm→2pm window (SPY/QQQ, 5-min, 2020→)
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_fomc_intraday() -> dict:
    print(f"\n{'='*82}\nSECTION 2 — PRE-FOMC precise intraday 2pm→2pm window (5-min, 2020→)\n{'='*82}")
    ann_ts = announce_timestamps_utc(14, 0)      # 14:00 ET localized → UTC per date (DST-correct)
    out = {}
    for t in ("SPY", "QQQ", "IWM", "DIA"):
        s = _intraday_price_at(t, "5min")
        tf_used = "5min"
        if s.empty:
            s, tf_used = _intraday_price_at(t, "1h"), "1h"  # 5-min not archived for all ETFs → 1h fallback
        if s.empty:
            continue
        ev = sz.event_window_returns(s, ann_ts, 24.0)
        # daily proxy over the same events for comparison (day-before close-to-close)
        dret = _etf_returns(t, "2020-01-01")
        pos = sz.window_position(dret.index, announce_days("UTC"), [-1])
        proxy = (dret * pos).replace(0.0, np.nan).dropna()
        out[t] = {"n_events": int(len(ev)), "bar": tf_used,
                  "precise_24h_mean_bps": round(float(ev.mean() * 1e4), 2),
                  "precise_24h_hit_rate": round(float((ev > 0).mean()), 3),
                  "precise_24h_t_stat": round(float(ev.mean() / (ev.std(ddof=1) / np.sqrt(len(ev)))), 2),
                  "daily_proxy_mean_bps": round(float(proxy.mean() * 1e4), 2)}
        print(f"  {t}: precise 24h mean {out[t]['precise_24h_mean_bps']:+.1f}bps "
              f"(t={out[t]['precise_24h_t_stat']:+.1f}, hit {out[t]['precise_24h_hit_rate']:.0%}, "
              f"n={out[t]['n_events']})  vs daily-proxy {out[t]['daily_proxy_mean_bps']:+.1f}bps")
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — pre-FOMC crypto analogue (BTC/ETH, daily + hourly precise, funding-charged)
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_fomc_crypto() -> dict:
    print(f"\n{'='*82}\nSECTION 3 — PRE-FOMC crypto analogue (BTC/ETH perp, funding-charged, 2020→)\n{'='*82}")
    C, _ = _panel("crypto_1d")
    ann = announce_days("UTC")
    out, books = {}, {}
    for name in ("BTCUSDT", "ETHUSDT"):
        ret = C[name].pct_change(fill_method=None)
        fund = _btc_funding_daily(ret.index, name)
        pos = sz.window_position(ret.index, ann, FOMC_OFFSETS)
        net = sz.hold_backtest(ret, pos, cost_bps=CR_COST)["net"] - pos * fund   # long pays funding
        dec = sz.in_vs_out(ret, pos)
        placebo = _placebo_sharpe(ret, int(((ann >= ret.index.min()) & (ann <= ret.index.max())).sum()),
                                  FOMC_OFFSETS, CR_COST, 365, funding=fund)
        out[name] = {"in_window_mean_bps": dec["in_window_mean_bps"], "in_window_sharpe": dec["in_window_sharpe"],
                     "net_sharpe_after_funding": round(_sh(net, 365), 3),
                     "placebo_pctile": round(float((_sh(net, 365) > placebo).mean() * 100), 0),
                     "funding_drag_bps_per_held_day": round(float(fund[pos > 0].mean() * 1e4), 2)}
        books[f"fomc_{name[:3].lower()}"] = net
        print(f"  {name}: day-before mean {dec['in_window_mean_bps']:+.1f}bps in-win Sharpe {dec['in_window_sharpe']:+.2f}"
              f" | net after funding {out[name]['net_sharpe_after_funding']:+.2f} (funding drag "
              f"{out[name]['funding_drag_bps_per_held_day']:+.1f}bps/day) | placebo p{out[name]['placebo_pctile']:.0f}")
    # hourly precise 24h window for BTC
    h = pd.read_parquet(CACHE / "crypto_1h_close.parquet", columns=["BTCUSDT"])["BTCUSDT"]
    if h.index.tz is None:
        h.index = h.index.tz_localize("UTC")
    ev = sz.event_window_returns(sz.price_at_instant(h, pd.Timedelta("1h")),
                                 announce_timestamps_utc(14, 0), 24.0)
    out["BTC_hourly_precise"] = {"n_events": int(len(ev)), "mean_bps": round(float(ev.mean() * 1e4), 2),
                                 "hit_rate": round(float((ev > 0).mean()), 3),
                                 "t_stat": round(float(ev.mean() / (ev.std(ddof=1) / np.sqrt(len(ev)))), 2)}
    print(f"  BTC hourly precise 24h→2pmET: mean {out['BTC_hourly_precise']['mean_bps']:+.1f}bps "
          f"(t={out['BTC_hourly_precise']['t_stat']:+.1f}, hit {out['BTC_hourly_precise']['hit_rate']:.0%}, "
          f"n={out['BTC_hourly_precise']['n_events']})")
    return {"by_name": out, "_books": books}


# ════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — turn-of-month, time-series (SPY / BTC / FX), param surface + placebo
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_tom_ts() -> dict:
    print(f"\n{'='*82}\nSECTION 4 — TURN-OF-MONTH time-series (SPY / BTC / FX-basket)\n{'='*82}")
    out, books = {}, {}

    def run_one(label, ret, cost, ppy, funding=None):
        anch = sz.month_end_anchors(ret.index)
        off = sz.turn_of_month_offsets(TOM_BEFORE, TOM_AFTER)
        pos = sz.window_position(ret.index, anch, off)
        bt = sz.hold_backtest(ret, pos, cost_bps=cost)
        net = bt["net"] - (pos * funding if funding is not None else 0.0)
        dec = sz.in_vs_out(ret, pos)
        # param surface: (days_before × days_after)
        grid = []
        for b in (1, 2, 4):
            for a in (1, 3, 5):
                p = sz.window_position(ret.index, anch, sz.turn_of_month_offsets(b, a))
                n = sz.hold_backtest(ret, p, cost_bps=cost)["net"] - (p * funding if funding is not None else 0.0)
                grid.append({"asset": label, "days_before": b, "days_after": a, "net_sharpe": round(_sh(n, ppy), 3)})
        placebo = _placebo_sharpe(ret, len(anch), off, cost, ppy, funding=funding)
        per_year = {int(y): round(_sh(g, ppy), 2) for y, g in net.dropna().groupby(net.dropna().index.year)}
        out[label] = {"in_window_mean_bps": dec["in_window_mean_bps"], "in_window_sharpe": dec["in_window_sharpe"],
                      "out_window_sharpe": dec["out_window_sharpe"], "frac_days": dec["frac_days_in_window"],
                      "share_of_total_logret": dec["in_window_share_of_total_logret"],
                      "net_sharpe": round(_sh(net, ppy), 3),
                      "placebo_pctile": round(float((_sh(net, ppy) > placebo).mean() * 100), 0),
                      "placebo_mean": round(float(placebo.mean()), 3), "per_year": per_year}
        books[f"tom_{label.lower()}"] = net
        print(f"  {label:10s}: ToM mean {dec['in_window_mean_bps']:+.1f}bps (out-of-win Sharpe {dec['out_window_sharpe']:+.2f}) "
              f"in-win Sharpe {dec['in_window_sharpe']:+.2f} | net {out[label]['net_sharpe']:+.2f} | "
              f"placebo p{out[label]['placebo_pctile']:.0f} (mean {out[label]['placebo_mean']:+.2f}) | "
              f"captures {dec['in_window_share_of_total_logret']} of total in {dec['frac_days_in_window']*100:.0f}% of days")
        return grid

    grids = []
    grids += run_one("SPY", _etf_returns("SPY"), EQ_COST, 252)
    Cc, _ = _panel("crypto_1d")
    btc_ret = Cc["BTCUSDT"].pct_change(fill_method=None)
    grids += run_one("BTC", btc_ret, CR_COST, 365, funding=_btc_funding_daily(btc_ret.index, "BTCUSDT"))
    fx, _ = _panel("fx_1d")
    # FX "market" = equal-weight basket of USD-quoted majors expressed as USD-long (sign so USDxxx pairs align)
    usd_base = [c for c in fx.columns if c.startswith("USD")]
    fx_basket = fx[usd_base].pct_change(fill_method=None).mean(axis=1)
    grids += run_one("FXbasket", fx_basket, FX_COST, 252)
    return {"by_asset": out, "_books": books, "_grid": pd.DataFrame(grids)}


# ════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — turn-of-month, cross-sectional top-N breadth (the top-10/50/100 cut)
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_tom_xsection() -> dict:
    print(f"\n{'='*82}\nSECTION 5 — TURN-OF-MONTH cross-sectional top-N breadth (crypto & stocks)\n{'='*82}")
    off = sz.turn_of_month_offsets(TOM_BEFORE, TOM_AFTER)
    out = {}
    for tag, ppy, cost, Ns, flat_fund in [("crypto_1d", 365, CR_COST, (10, 50, 100, 200), 0.000325),
                                          ("stocks_broad_1d", 252, EQ_COST, (50, 100, 200, 500), 0.0)]:
        C, A = _panel(tag)
        anch = sz.month_end_anchors(C.index)
        pos = sz.window_position(C.index, anch, off)
        row = {}
        for N in Ns:
            bt = sz.xs_window_backtest(C, pos, top_n=N, cost_bps=cost, adv=A, impact_k=0.1)
            net = bt["net"] - pos * flat_fund       # crypto: flat avg-funding drag while long (documented)
            row[f"top{N}"] = round(_sh(net, ppy), 3)
        out[tag] = row
        print(f"  {tag:18s}: net Sharpe by basket size {row}"
              + ("   (crypto incl. ~11.9%/yr flat funding drag)" if flat_fund else ""))
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — combined calendar sleeve (pre-FOMC ∪ turn-of-month) + portfolio funnel
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_combined(fomc_books: dict, tom_books: dict) -> dict:
    print(f"\n{'='*82}\nSECTION 6 — COMBINED calendar sleeve + portfolio funnel\n{'='*82}")
    ann = announce_days("UTC")
    out, books = {}, {}
    # SPY combined: long inside (pre-FOMC day-before) ∪ (turn-of-month) — one book, cost amortised
    ret = _etf_returns("SPY")
    anch = sz.month_end_anchors(ret.index)
    pos = ((sz.window_position(ret.index, ann, FOMC_OFFSETS) +
            sz.window_position(ret.index, anch, sz.turn_of_month_offsets(TOM_BEFORE, TOM_AFTER))) > 0).astype(float)
    net = sz.hold_backtest(ret, pos, cost_bps=EQ_COST)["net"]
    dec = sz.in_vs_out(ret, pos)
    mc = bootstrap_sharpe(net.dropna(), 252, 1000, SEED)
    r = net.dropna()

    # purged/embargoed walk-forward over the calendar-window grid (ToM shapes + pre-FOMC offset variants):
    # does *selecting* a window in-sample survive OOS? — parity with every other family's WFO.
    grid = {}
    for b in (1, 2, 4):
        for a in (1, 3, 5):
            p = sz.window_position(ret.index, anch, sz.turn_of_month_offsets(b, a))
            grid[f"tom_{b}_{a}"] = sz.hold_backtest(ret, p, cost_bps=EQ_COST)["net"]
    for off, tag in [([-1], "fomc_m1"), ([0], "fomc_0"), ([-1, 0], "fomc_m1_0"), ([-2, -1, 0], "fomc_run")]:
        p = sz.window_position(ret.index, ann, off)
        grid[f"{tag}"] = sz.hold_backtest(ret, p, cost_bps=EQ_COST)["net"]
    M = pd.DataFrame(grid).dropna(how="all").fillna(0.0)
    n_trials = M.shape[1]
    wf, n_ref = _wf_oos(M, 252, int(2 * 252), int(0.5 * 252), embargo=5)
    var_tr = float((M.mean() / M.std(ddof=1)).clip(-3, 3).var())
    dsr = deflated_sharpe(r.mean() / r.std(ddof=1), len(r), r.skew(), r.kurt() + 3.0, n_trials=n_trials,
                          var_across_trials=max(var_tr, 1e-8))
    books["combined_spy"] = net
    out["combined_spy"] = {"net_sharpe": round(_sh(net, 252), 3), "frac_days": dec["frac_days_in_window"],
                           "in_window_sharpe": dec["in_window_sharpe"], "maxdd": round(summarise(r, 252)["max_dd"], 3),
                           "mc_p5": mc.get("sharpe_p5"), "mc_p50": mc.get("sharpe_p50"),
                           "wf_oos_sharpe": round(_sh(wf, 252), 3), "wf_n_refits": n_ref, "n_grid_trials": n_trials,
                           "deflated_sharpe": round(dsr, 3), "vs_buyhold_sharpe": round(_sh(ret, 252), 3)}
    print(f"  SPY combined (pre-FOMC ∪ ToM): net Sharpe {out['combined_spy']['net_sharpe']:+.2f} "
          f"[MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f}] WF-OOS {out['combined_spy']['wf_oos_sharpe']:+.2f} "
          f"deflated {out['combined_spy']['deflated_sharpe']:.2f} | "
          f"in market {dec['frac_days_in_window']*100:.0f}% of days | buy&hold Sharpe {out['combined_spy']['vs_buyhold_sharpe']:+.2f}")

    # correlation to the deliverable book + does adding the best calendar leg lift it?
    corr, lift = {}, {}
    cand = books["combined_spy"]
    bp_path, bs_path = REP / "master_book.parquet", REP / "master_book_legs.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path)["ret"]; bs = pd.read_parquet(bs_path)
        for f in (bp, bs):
            if f.index.tz is not None:
                f.index = f.index.tz_localize(None)
        h = cand.copy(); h.index = h.index.tz_localize(None)
        bk = bp.dropna()
        if len(bk) > 50:
            # the candidate is mapped onto the BOOK's calendar (flat on days it does not trade) and
            # annualised by the book's own obs/yr — intersecting instead would quietly re-annualise the
            # book on the equity calendar and print a baseline that is not the book's headline Sharpe
            corr["book"] = round(float(h.reindex(bk.index).fillna(0.0).corr(bk)), 3)
            for c in bs.columns:
                corr[c] = round(float(h.reindex(bk.index).fillna(0.0).corr(bs[c].reindex(bk.index))), 3)
            ppy_book = len(bk) / ((bk.index.max() - bk.index.min()).days / 365.25)
            hm = h.reindex(bk.index).fillna(0.0)
            hm = hm * (bk.std() / hm.std()) if hm.std() > 0 else hm
            lift = {f"{int(w*100)}%": round(_sh((1 - w) * bk + w * hm, ppy_book), 3)
                    for w in (0.0, 0.15, 0.3, 0.5)}
            # control: a naked long-SPY stub blended in the same way. The book is market-neutral, so any
            # long-equity exposure lifts it; a calendar sleeve is only interesting if it lifts it MORE.
            beta = _etf_returns("SPY").copy()
            beta.index = beta.index.tz_localize(None)
            beta = beta.reindex(bk.index).fillna(0.0)
            beta = beta * (bk.std() / beta.std())
            lift["buy_hold_spy_at_15%"] = round(_sh(0.85 * bk + 0.15 * beta, ppy_book), 3)
            print(f"  corr to book {corr.get('book')}  → book-lift by weight {lift}")
    out["corr_to_book"], out["book_lift_by_weight"] = corr, lift
    return {"summary": out, "_books": books}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    s1 = section_fomc_equity()
    s2 = section_fomc_intraday()
    s3 = section_fomc_crypto()
    s4 = section_tom_ts()
    s5 = section_tom_xsection()
    fomc_books = {**s1.pop("_books"), **s3.pop("_books")}
    tom_books = s4.pop("_books")
    s6 = section_combined(fomc_books, tom_books)

    # persist candidate return series + surfaces
    allbooks = {**fomc_books, **tom_books, **s6.pop("_books")}
    rets = pd.DataFrame({k: v for k, v in allbooks.items()})
    rets.to_parquet(SEASONAL_DIR / "seasonal_returns.parquet")
    s4.pop("_grid").to_csv(SEASONAL_DIR / "seasonal_tom_grid.csv", index=False)

    summ = {"config": {"fomc_offsets": FOMC_OFFSETS, "tom_before": TOM_BEFORE, "tom_after": TOM_AFTER,
                       "eq_cost_bps": EQ_COST, "cr_cost_bps": CR_COST, "fx_cost_bps": FX_COST,
                       "window_equity": [START_EQ, str(rets.index.max().date())], "decay_split": DECAY_SPLIT},
            "fomc_equity": s1, "fomc_intraday": s2, "fomc_crypto": s3,
            "tom_timeseries": s4, "tom_xsection": s5, "combined": s6["summary"]}
    (SEASONAL_DIR / "seasonal_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    _figure(summ, allbooks)

    print(f"\n{'='*82}\nRUN SEASONAL OK — reports/seasonal_summary.json + figures/seasonal.png\n{'='*82}")


def _figure(summ, books):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Calendar seasonality (H4) — pre-FOMC drift + turn-of-month (net of costs)",
                 fontsize=13, fontweight="bold")

    # (1) pre-FOMC per-offset drift map (SPY): where the drift lives, relative to the announcement
    a = ax[0, 0]
    off = summ["fomc_equity"]["by_etf"]["SPY"]["offset_mean_bps"]
    xs = [int(k) for k in off]; ys = [off[k] for k in off]
    a.bar([str(x) for x in xs], ys, color=["#a63" if x < 0 else "#68a" for x in xs])
    a.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(ys):
        a.text(i, v, f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    a.set_title("SPY mean return (bps) by trading-day offset from FOMC announce\n(−1 = day before; drift concentrates pre-announce)", fontsize=9)
    a.set_xlabel("offset from announce day")

    # (2) pre-FOMC decay: pre-2018 vs post-2018 net Sharpe by ETF
    a = ax[0, 1]
    ets = ["SPY", "QQQ", "IWM", "DIA"]; x = np.arange(len(ets)); wd = 0.38
    pre = [summ["fomc_equity"]["by_etf"][t]["headline_day_before"]["sharpe_pre_2018"] for t in ets]
    post = [summ["fomc_equity"]["by_etf"][t]["headline_day_before"]["sharpe_post_2018"] for t in ets]
    a.bar(x - wd/2, pre, wd, label="2011–2017", color="#2b6")
    a.bar(x + wd/2, post, wd, label="2018–2026", color="#c53")
    a.axhline(0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(ets)
    a.set_title("Pre-FOMC day-before net Sharpe by sub-period (both sub-bar)", fontsize=10); a.legend(fontsize=8)

    # (3) turn-of-month in-window vs out-of-window Sharpe by asset
    a = ax[1, 0]
    assets = list(summ["tom_timeseries"]["by_asset"].keys()); x = np.arange(len(assets)); wd = 0.38
    inw = [summ["tom_timeseries"]["by_asset"][k]["in_window_sharpe"] for k in assets]
    outw = [summ["tom_timeseries"]["by_asset"][k]["out_window_sharpe"] for k in assets]
    a.bar(x - wd/2, inw, wd, label="in-window (ToM)", color="#2b6")
    a.bar(x + wd/2, outw, wd, label="out-of-window", color="#889")
    a.axhline(0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(assets)
    a.set_title("Turn-of-month: in-window vs rest-of-month Sharpe", fontsize=10); a.legend(fontsize=8)

    # (4) equity curves of the candidate calendar books
    a = ax[1, 1]
    for c, lab in [("combined_spy", "SPY combined (FOMC∪ToM)"), ("tom_spy", "SPY turn-of-month"),
                   ("fomc_spy", "SPY pre-FOMC day-before")]:
        if c in books:
            r = books[c].dropna()
            a.plot((1 + r).cumprod().index, (1 + r).cumprod().values, label=lab, lw=1.3)
    a.axhline(1.0, color="k", lw=0.6, ls=":")
    a.set_title("Candidate calendar books (cumulative, un-levered)", fontsize=10); a.legend(fontsize=8); a.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "seasonal.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
