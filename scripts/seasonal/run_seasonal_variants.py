"""Calendar seasonality — the variant sweep the first pass did not run (HYPOTHESES.md H4, round 2).

The first pass declared one a-priori window per effect (pre-FOMC = the trading day *before* the
announcement; turn-of-month = the classic −1/+3), tested it long-only on four ETFs plus BTC/ETH, and
excluded the family. This driver re-opens it along the axes that were never swept — which bar the
window is anchored on, which side of the event is traded, which assets, how many of them, and which
timeframe — after fixing three defects in the first pass that between them moved the headline numbers:

  1. an intraday look-ahead: Binance and Twelve Data bars are labelled by the bar's OPEN, so `asof(T)`
     returned a price one bar past T and the "24h into the 2pm-ET statement" window actually ended an
     hour *after* the statement. That single hour is the announcement reaction (see §1).
  2. price-only returns: `equity_td` closes are split-adjusted only, and the dividend calendar is
     itself a calendar — SPY goes ex two trading days after a quarter-end FOMC in ~a quarter of all
     meetings, bond ETFs go ex on the first business day of every month (inside the turn-of-month
     window). The "post-FOMC fade" was part ex-dividend drop.
  3. a calendar that started in 2011 while the prices start in 2005, throwing away 48 events (+38%) —
     including the whole era the pre-FOMC-drift literature was written on.

Sweep (every arm is reported, and the deflated Sharpe is deflated over all of them):
  §2 offset map     — where the drift lives, per asset × sub-period, on total returns, 2005→
  §3 event books    — long/short/paired arms × single name / equal-weight / risk-parity baskets
  §4 FOMC cycle     — the even-week/odd-week structure (Cieslak-Morse-Vissing-Jorgensen 2019)
  §5 intraday       — corrected precise windows, 2h→48h, pre- and post-announcement, equity + crypto
  §6 turn-of-month  — beta-neutral calendar spread, bond ToM, dash-for-cash conditioning, baskets
  §7 funnel         — placebo, block-bootstrap MC, purged WF over the grid, deflated SR, book lift

    python scripts/seasonal/run_seasonal_variants.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, RAW_DIR, REPORTS_DIR, SEASONAL_DIR, SEED  # noqa: E402
from src.data.fomc import announce_days, announce_timestamps_utc  # noqa: E402
from src.data.twelvedata import load_dividends  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import seasonal as sz  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP, FIG = REPORTS_DIR, REPORTS_DIR / "figures"
ETF_RAW, CACHE = RAW_DIR / "equity_td", CACHE_DIR / "xs"
rng = np.random.default_rng(SEED)

START = "2005-01-01"                        # ETF history floor = FOMC calendar floor
EQ_COST, CR_COST = 3.0, 6.0                 # per-side bps (equity ETF / crypto taker)
SUBS = [("2005-01-01", "2010-12-31"), ("2011-01-01", "2015-12-31"),
        ("2016-01-01", "2020-12-31"), ("2021-01-01", "2026-12-31")]
SUB_LABELS = ["05-10", "11-15", "16-20", "21-26"]

EQ_US = ["SPY", "QQQ", "IWM", "DIA"]
EQ_SECTOR = ["XLK", "XLF", "XLE", "XLU", "XLP", "XLV", "XLY", "XLI", "XLB"]
EQ_INTL = ["EEM", "EFA"]
CREDIT, RATES, METALS = ["HYG", "LQD"], ["TLT", "IEF", "AGG", "SHY"], ["GLD", "SLV"]
ALL_ETF = EQ_US + EQ_SECTOR + EQ_INTL + CREDIT + RATES + METALS


# ── data layer ──────────────────────────────────────────────────────────────────────────────
def etf_total_returns(tickers: list[str]) -> pd.DataFrame:
    """Dividend-inclusive daily returns for the ETF universe (see sz.total_return for why)."""
    out = {}
    for t in tickers:
        px = pd.read_parquet(ETF_RAW / f"{t}_1d.parquet")["close"]
        if px.index.tz is None:
            px.index = px.index.tz_localize("UTC")
        px = px[px.index >= pd.Timestamp(START, tz="UTC")]
        try:
            div = load_dividends(t, start="2004-01-01")
        except Exception:
            div = None                                  # non-distributing (GLD/SLV) or unavailable
        out[t] = sz.total_return(px, div)
    return pd.DataFrame(out)


def crypto_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    C = pd.read_parquet(CACHE / "crypto_1d_close.parquet", columns=["BTCUSDT", "ETHUSDT"])
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC")
    R = C.pct_change(fill_method=None)
    F = {}
    for name in ("BTCUSDT", "ETHUSDT"):
        fr = pd.read_parquet(RAW_DIR / f"futures/um/fundingRate/{name}")["last_funding_rate"]
        d = fr.resample("1D").sum()
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        F[name] = d.reindex(R.index).fillna(0.0)
    return R, pd.DataFrame(F)


def inv_vol_basket(R: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Risk-parity basket return: weights ∝ 1/trailing vol, lagged, renormalised to one unit gross."""
    w = 1.0 / R.rolling(lookback).std()
    w = w.div(w.sum(axis=1), axis=0).shift(1)
    return (w * R).sum(axis=1).where(R.notna().any(axis=1))


# ── arm evaluation ──────────────────────────────────────────────────────────────────────────
def sharpe(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(np.sqrt(ppy) * r.mean() / sd) if sd > 0 and len(r) > 2 else 0.0


def tstat(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


class Arm:
    """One tradable calendar book: a return stream, a signed event window, and its cost model."""

    def __init__(self, name: str, ret: pd.Series, anchors: pd.DatetimeIndex, longs=(), shorts=(),
                 cost_bps: float = EQ_COST, ppy: float = 252, funding: pd.Series | None = None,
                 cost_mult: float = 1.0, group: str = ""):
        self.name, self.ret, self.anchors, self.group = name, ret.dropna(), anchors, group
        self.longs, self.shorts = tuple(longs), tuple(shorts)
        self.cost, self.ppy, self.funding, self.cost_mult = cost_bps, ppy, funding, cost_mult
        self.pos = sz.signed_position(self.ret.index, anchors, self.longs, self.shorts)
        self.net = self._book(self.pos)

    def _book(self, pos: pd.Series) -> pd.Series:
        bt = sz.hold_backtest(self.ret, pos, cost_bps=self.cost * self.cost_mult)
        net = bt["net"]
        if self.funding is not None:
            net = net - pos * self.funding.reindex(net.index).fillna(0.0)
        return net

    def placebo(self, n_iter: int = 200) -> np.ndarray:
        """Same window shape on random anchors — the only test that separates event from drift."""
        idx = self.ret.index
        offs = list(self.longs) + list(self.shorts)
        lo, hi = -min(offs), max(offs)
        pool = np.arange(max(lo + 1, 1), len(idx) - hi - 1)
        n = len(np.unique(idx.searchsorted(self.anchors)))
        out = np.empty(n_iter)
        for i in range(n_iter):
            picks = pd.DatetimeIndex(idx[rng.choice(pool, size=min(n, len(pool)), replace=False)])
            out[i] = sharpe(self._book(sz.signed_position(idx, picks, self.longs, self.shorts)), self.ppy)
        return out

    def row(self, with_placebo: bool = True) -> dict:
        r = self.net.dropna()
        sub = {}
        for (a, b), lab in zip(SUBS, SUB_LABELS):
            w = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index <= pd.Timestamp(b, tz="UTC"))]
            sub[lab] = round(sharpe(w, self.ppy), 2) if len(w) > 20 else None
        d = {"arm": self.name, "group": self.group,
             "long": ",".join(map(str, self.longs)) or "-", "short": ",".join(map(str, self.shorts)) or "-",
             "sharpe": round(sharpe(r, self.ppy), 3),
             "ann_ret_pct": round(float(r.mean() * self.ppy * 100), 2),
             "ann_vol_pct": round(float(r.std(ddof=1) * np.sqrt(self.ppy) * 100), 2),
             "days_in_market_pct": round(float((self.pos != 0).mean() * 100), 1),
             "max_dd": round(summarise(r, self.ppy)["max_dd"], 3),
             **{f"sh_{k}": v for k, v in sub.items()}}
        if with_placebo:
            p = self.placebo()
            d["placebo_pctile"] = round(float((d["sharpe"] > p).mean() * 100), 0)
            d["placebo_p95"] = round(float(np.percentile(p, 95)), 3)
        return d


def evaluate(arms: list[Arm], with_placebo: bool = True, show: int = 0) -> pd.DataFrame:
    rows = [a.row(with_placebo) for a in arms]
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    if show:
        cols = ["arm", "sharpe", "ann_ret_pct", "ann_vol_pct", "days_in_market_pct"] + \
               [f"sh_{k}" for k in SUB_LABELS] + (["placebo_pctile", "placebo_p95"] if with_placebo else [])
        print(df[cols].head(show).to_string(index=False))
    return df


# ════════════════════════════════════════════════════════════════════════════════════════════
# §1 — the three defects, quantified against the shipped numbers
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_defects(R: pd.DataFrame) -> dict:
    print(f"\n{'='*100}\n§1 — DEFECTS IN THE FIRST PASS (each reproduced, then corrected)\n{'='*100}")
    ann_ts = announce_timestamps_utc(14, 0)
    out = {}

    h = pd.read_parquet(CACHE / "crypto_1h_close.parquet", columns=["BTCUSDT"])["BTCUSDT"]
    if h.index.tz is None:
        h.index = h.index.tz_localize("UTC")
    shipped = sz.event_window_returns(h, ann_ts, 24.0)                       # asof on open-labelled bars
    fixed = sz.event_window_returns(sz.price_at_instant(h, pd.Timedelta("1h")), ann_ts, 24.0)
    leak = sz.event_window_returns(sz.price_at_instant(h, pd.Timedelta("1h")), ann_ts, 1.0, end_offset_hours=1.0)
    out["intraday_lookahead"] = {
        "asset": "BTC 24h→14:00ET", "n": int(len(fixed)),
        "shipped_mean_bps": round(float(shipped.mean() * 1e4), 1), "shipped_t": round(tstat(shipped), 2),
        "corrected_mean_bps": round(float(fixed.mean() * 1e4), 1), "corrected_t": round(tstat(fixed), 2),
        "leaked_announcement_hour_bps": round(float(leak.mean() * 1e4), 1)}
    print("  (1) open-labelled bars + asof → the window ended an hour AFTER the statement:")
    print(f"      BTC 24h window  shipped {out['intraday_lookahead']['shipped_mean_bps']:+.1f}bps "
          f"(t={out['intraday_lookahead']['shipped_t']:+.2f})  →  corrected "
          f"{out['intraday_lookahead']['corrected_mean_bps']:+.1f}bps (t={out['intraday_lookahead']['corrected_t']:+.2f})"
          f"   [the leaked hour alone: {out['intraday_lookahead']['leaked_announcement_hour_bps']:+.1f}bps]")

    ann = announce_days("UTC")
    div_rows = {}
    for t in ("SPY", "QQQ", "XLK", "DIA"):
        px = pd.read_parquet(ETF_RAW / f"{t}_1d.parquet")["close"]
        if px.index.tz is None:
            px.index = px.index.tz_localize("UTC")
        px = px[px.index >= pd.Timestamp(START, tz="UTC")]
        pr = px.pct_change(fill_method=None)
        tr = R[t].reindex(pr.index)
        div = (load_dividends(t, start="2004-01-01").reindex(px.index).fillna(0.0) > 0).astype(float)
        hits = sz.offset_event_returns(div, ann, 2)
        div_rows[t] = {"ex_div_on_offset+2_of_n": f"{int(hits.sum())}/{len(hits)}",
                       "price_only_bps": round(float(sz.offset_event_returns(pr, ann, 2).mean() * 1e4), 1),
                       "total_return_bps": round(float(sz.offset_event_returns(tr, ann, 2).mean() * 1e4), 1)}
    out["dividend_artifact"] = div_rows
    print("  (2) split-adjusted-only closes → a fake 'post-FOMC fade' on the day the ETF goes ex:")
    for t, d in div_rows.items():
        print(f"      {t}: offset +2 ex-div in {d['ex_div_on_offset+2_of_n']} events | "
              f"price-only {d['price_only_bps']:+.1f}bps → total-return {d['total_return_bps']:+.1f}bps")

    n_now = int(((ann >= R.index.min()) & (ann <= R.index.max())).sum())
    n_2011 = int((ann >= pd.Timestamp("2011-01-01", tz="UTC")).sum() -
                 (ann > R.index.max()).sum())
    out["calendar_extension"] = {"events_2011_start": n_2011, "events_2005_start": n_now,
                                 "gain_pct": round(100 * (n_now / n_2011 - 1), 1)}
    print(f"  (3) calendar started 2011 while prices start 2005: {n_2011} → {n_now} events "
          f"(+{out['calendar_extension']['gain_pct']:.0f}%)")
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# §2 — where the drift actually lives: offset × asset × sub-period
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_offset_map(R: pd.DataFrame, Rc: pd.DataFrame) -> dict:
    print(f"\n{'='*100}\n§2 — OFFSET MAP (total returns, 2005→): which bar carries the premium?\n{'='*100}")
    ann = announce_days("UTC")
    offs = list(range(-3, 4))
    out = {}

    def per_event(ret: pd.Series, off: int) -> pd.Series:
        return sz.offset_event_returns(ret, ann, off)

    print(f"  {'asset':6s} " + "".join(f"{o:>15d}" for o in offs))
    for t in list(R.columns) + list(Rc.columns):
        ret = R[t] if t in R.columns else Rc[t]
        row = {}
        for o in offs:
            v = per_event(ret.dropna(), o)
            row[o] = {"mean_bps": round(float(v.mean() * 1e4), 1), "t": round(tstat(v), 2), "n": int(len(v))}
        out[t] = row
        if t in ("SPY", "QQQ", "XLK", "EFA", "HYG", "TLT", "GLD", "BTCUSDT", "ETHUSDT"):
            print(f"  {t:6s} " + "".join(f"{row[o]['mean_bps']:+8.1f}(t{row[o]['t']:+.1f})" for o in offs))

    print("\n  Sub-period decay of the two live offsets (equal-weight equity basket, mean bps):")
    eq = R[EQ_US + EQ_SECTOR + EQ_INTL].mean(axis=1)
    decay = {}
    for o in (-1, 0, 1):
        v = per_event(eq, o)
        cells = {}
        for (a, b), lab in zip(SUBS, SUB_LABELS):
            w = v[(v.index >= pd.Timestamp(a, tz="UTC")) & (v.index <= pd.Timestamp(b, tz="UTC"))]
            cells[lab] = {"mean_bps": round(float(w.mean() * 1e4), 1), "t": round(tstat(w), 2), "n": int(len(w))}
        decay[o] = cells
        print(f"    offset {o:+d}: " + "  ".join(f"{lab} {c['mean_bps']:+7.1f}(t{c['t']:+.1f})"
                                                 for lab, c in cells.items()))
    return {"by_asset": out, "basket_decay": decay}


# ════════════════════════════════════════════════════════════════════════════════════════════
# §3 — event books: which side, which assets, how many
# ════════════════════════════════════════════════════════════════════════════════════════════
def event_streams(R: pd.DataFrame, Rc: pd.DataFrame, Fc: pd.DataFrame) -> dict:
    """Return streams to run the event windows on: single names, basket sizes, relative-value pairs.

    The basket ladder (1 → 4 → 13 → 15 → 21 names) is the "how many assets" axis: eight events a year
    is a thin sample, and if the premium is a shared macro response then averaging more names should
    cut the noise without cutting the signal. Baskets are fixed a-priori groups, never a top-K picked
    on realised event returns — that would select the winners with the answer in hand.
    """
    eq13, eq15 = EQ_US + EQ_SECTOR, EQ_US + EQ_SECTOR + EQ_INTL
    return {
        "SPY": (R["SPY"], EQ_COST, 252, None, 1.0),
        "QQQ": (R["QQQ"], EQ_COST, 252, None, 1.0),
        "TLT": (R["TLT"], EQ_COST, 252, None, 1.0),
        "GLD": (R["GLD"], EQ_COST, 252, None, 1.0),
        "HYG": (R["HYG"], EQ_COST, 252, None, 1.0),
        "eqUS4ew": (R[EQ_US].mean(axis=1), EQ_COST, 252, None, 1.0),
        "eq13ew": (R[eq13].mean(axis=1), EQ_COST, 252, None, 1.0),
        "eq15ew": (R[eq15].mean(axis=1), EQ_COST, 252, None, 1.0),
        "eq15rp": (inv_vol_basket(R[eq15]), EQ_COST, 252, None, 1.0),
        "all21rp": (inv_vol_basket(R[ALL_ETF]), EQ_COST, 252, None, 1.0),
        "xasset8rp": (inv_vol_basket(R[["SPY", "QQQ", "EFA", "EEM", "HYG", "TLT", "GLD", "LQD"]]),
                      EQ_COST, 252, None, 1.0),
        "bond3rp": (inv_vol_basket(R[["TLT", "IEF", "LQD"]]), EQ_COST, 252, None, 1.0),
        "BTC": (Rc["BTCUSDT"], CR_COST, 365, Fc["BTCUSDT"], 1.0),
        "crypto2": (Rc.mean(axis=1), CR_COST, 365, Fc.mean(axis=1), 1.0),
        # relative-value streams: both legs trade, so the cost model charges two sides
        "TLTvsSPY": (R["TLT"] - R["SPY"], EQ_COST, 252, None, 2.0),
        "bondVsEq": (inv_vol_basket(R[["TLT", "IEF"]]) - inv_vol_basket(R[EQ_US]), EQ_COST, 252, None, 2.0),
    }


# Every shape is run in BOTH directions. The offset map says equities rise into the statement and fall
# after it while bonds do the reverse, so a grid that only trades one side would be choosing the sign
# with the answer already on screen. Both signs are swept, both are reported, and the deflated Sharpe
# is charged for all of them.
EVENT_WINDOWS = [("pre1", (-1,), ()), ("ann", (0,), ()), ("pre_ann", (-1, 0), ()),
                 ("post1", (1,), ()), ("post12", (1, 2), ()),
                 ("ann_fade", (0,), (1,)), ("preann_fade", (-1, 0), (1, 2))]


def build_event_arms(streams: dict) -> list[Arm]:
    ann = announce_days("UTC")
    arms = []
    for sname, (ret, cost, ppy, fund, mult) in streams.items():
        for wname, lo, sh in EVENT_WINDOWS:
            for tag, a, b in ((wname, lo, sh), (f"{wname}_rev", sh, lo)):
                if not (a or b):
                    continue
                arms.append(Arm(f"{sname}:{tag}", ret, ann, a, b, cost_bps=cost, ppy=ppy,
                                funding=fund, cost_mult=mult, group="fomc_event"))
    return arms


def build_asset_attribution(R: pd.DataFrame, Rc: pd.DataFrame, Fc: pd.DataFrame) -> pd.DataFrame:
    """Same two headline windows on every asset separately — which names carry the premium."""
    print(f"\n{'='*100}\n§3b — PER-ASSET ATTRIBUTION of the two headline windows\n{'='*100}")
    ann = announce_days("UTC")
    rows = []
    for t in ALL_ETF + ["BTCUSDT", "ETHUSDT"]:
        crypto = t.endswith("USDT")
        ret = (Rc[t] if crypto else R[t]).dropna()
        kw = dict(cost_bps=CR_COST if crypto else EQ_COST, ppy=365 if crypto else 252,
                  funding=Fc[t] if crypto else None, group="attribution")
        row = {"asset": t, "n_years": round((ret.index.max() - ret.index.min()).days / 365.25, 1)}
        for wname, lo, sh in (("ann", (0,), ()), ("ann_fade", (0,), (1,)), ("preann_fade", (-1, 0), (1, 2))):
            a = Arm(f"{t}:{wname}", ret, ann, lo, sh, **kw)
            row[f"{wname}_sharpe"] = round(sharpe(a.net, a.ppy), 2)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("preann_fade_sharpe", ascending=False)
    print(df.to_string(index=False))
    return df


# ════════════════════════════════════════════════════════════════════════════════════════════
# §4 — FOMC-cycle even/odd weeks (Cieslak-Morse-Vissing-Jorgensen 2019)
# ════════════════════════════════════════════════════════════════════════════════════════════
def section_cycle(R: pd.DataFrame) -> dict:
    print(f"\n{'='*100}\n§4 — FOMC CYCLE: is the equity premium concentrated in even weeks?\n{'='*100}")
    ann = announce_days("UTC")
    eq = R[EQ_US + EQ_SECTOR + EQ_INTL].mean(axis=1).dropna()
    cd = sz.cycle_day(eq.index, ann)
    even = cd.notna() & (((cd // 5) % 2) == 0)
    odd = cd.notna() & (((cd // 5) % 2) == 1)
    cmvj = cd.isin([0, 2, 3, 4, 8, 9, 10, 16, 17, 18, 22, 23, 24])
    out = {}
    for lab, mask in (("even_week", even), ("odd_week", odd), ("cmvj_days", cmvj)):
        cells = {}
        for (a, b), sub in zip(SUBS, SUB_LABELS):
            w = eq[mask & (eq.index >= pd.Timestamp(a, tz="UTC")) & (eq.index <= pd.Timestamp(b, tz="UTC"))]
            cells[sub] = round(float(w.mean() * 1e4), 1)
        v = eq[mask]
        out[lab] = {"mean_bps": round(float(v.mean() * 1e4), 2), "t": round(tstat(v), 2),
                    "in_window_sharpe": round(sharpe(v, 252), 2), "days_pct": round(100 * len(v) / len(eq), 0),
                    "by_sub": cells}
        print(f"  {lab:10s} mean {out[lab]['mean_bps']:+5.2f}bps  t {out[lab]['t']:+5.2f}  "
              f"in-window Sharpe {out[lab]['in_window_sharpe']:+.2f}  ({out[lab]['days_pct']:.0f}% of days)  "
              f"by sub-period {cells}")
    out["buy_hold_sharpe"] = round(sharpe(eq, 252), 2)
    # the tradable version: long even weeks / short odd weeks, and long-only even weeks
    pos_ls = pd.Series(0.0, index=eq.index).mask(even, 1.0).mask(odd, -1.0)
    pos_lo = pd.Series(0.0, index=eq.index).mask(even, 1.0)
    for lab, pos in (("even_week_long_short", pos_ls), ("even_week_long_only", pos_lo)):
        net = sz.hold_backtest(eq, pos, cost_bps=EQ_COST)["net"]
        out[lab] = {"net_sharpe": round(sharpe(net, 252), 3)}
        print(f"  {lab:22s} net Sharpe {out[lab]['net_sharpe']:+.2f}   (buy&hold {out['buy_hold_sharpe']:+.2f})")
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# §5 — precise intraday windows, corrected (2020→)
# ════════════════════════════════════════════════════════════════════════════════════════════
def _intraday_close(ticker: str) -> tuple[pd.Series, pd.Timedelta]:
    for tf, bar in (("5min", pd.Timedelta("5min")), ("1h", pd.Timedelta("1h"))):
        files = sorted((RAW_DIR / "twelvedata").glob(f"{ticker}_{tf}_*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(p) for p in files])
        s = df[~df.index.duplicated(keep="last")].sort_index()["close"]
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s, bar
    return pd.Series(dtype=float), pd.Timedelta("1h")


class IntradayArm:
    """An event book on a precise intraday window: one round trip per announcement, flat between.

    The per-event return is stamped on the announcement's calendar day and the book is zero on every
    other day, so its Sharpe is directly comparable to the daily arms and to the master book — a day
    spent flat is a real zero, not a missing observation.
    """

    def __init__(self, name: str, price_at: pd.Series, ts: pd.DatetimeIndex, hours: float,
                 end_offset: float, calendar: pd.DatetimeIndex, cost_bps: float,
                 funding_daily: pd.Series | None = None, ppy: float = 252, group: str = "intraday"):
        self.name, self.group, self.ppy = name, group, ppy
        self.price_at, self.hours, self.end_offset = price_at, hours, end_offset
        self.cost, self.funding, self.calendar = cost_bps, funding_daily, calendar
        self.longs, self.shorts = (0,), ()
        self.events = sz.event_window_returns(price_at, ts, hours, end_offset_hours=end_offset)
        self.net = self._book(self.events)
        self.pos = (self.net != 0).astype(float)

    def _book(self, ev: pd.Series) -> pd.Series:
        if ev.empty:
            return pd.Series(0.0, index=self.calendar)
        r = ev - 2 * self.cost / 1e4                              # one entry, one exit per event
        if self.funding is not None:
            f = self.funding.reindex(pd.DatetimeIndex(ev.index.date).tz_localize("UTC")).fillna(0.0)
            r = r - self.hours / 24.0 * f.to_numpy()              # funding accrues pro-rata while held
        daily = pd.Series(r.to_numpy(), index=pd.DatetimeIndex(ev.index.date).tz_localize("UTC"))
        daily = daily.groupby(level=0).sum()
        return daily.reindex(self.calendar).fillna(0.0)

    def placebo(self, n_iter: int = 200) -> np.ndarray:
        """Same window, same time-of-day, random dates — the shuffled-calendar null for an event book."""
        pool = self.calendar[(self.calendar >= self.price_at.index.min() + pd.Timedelta(days=3)) &
                             (self.calendar <= self.price_at.index.max() - pd.Timedelta(days=1))]
        out = np.empty(n_iter)
        for i in range(n_iter):
            picks = pd.DatetimeIndex(rng.choice(pool, size=min(len(self.events), len(pool)), replace=False))
            fake = pd.DatetimeIndex([d.tz_convert("America/New_York").normalize() for d in picks]) \
                .tz_localize(None).tz_localize("America/New_York", ambiguous=True,
                                               nonexistent="shift_forward") + pd.Timedelta(hours=14)
            ev = sz.event_window_returns(self.price_at, fake.tz_convert("UTC"), self.hours,
                                         end_offset_hours=self.end_offset)
            out[i] = sharpe(self._book(ev), self.ppy)
        return out

    def row(self, with_placebo: bool = True) -> dict:
        r = self.net.dropna()
        sub = {}
        for (a, b), lab in zip(SUBS, SUB_LABELS):
            w = r[(r.index >= pd.Timestamp(a, tz="UTC")) & (r.index <= pd.Timestamp(b, tz="UTC"))]
            sub[lab] = round(sharpe(w, self.ppy), 2) if len(w) > 20 else None
        d = {"arm": self.name, "group": self.group, "long": f"{self.hours:.0f}h@{self.end_offset:+.0f}h",
             "short": "-", "sharpe": round(sharpe(r, self.ppy), 3),
             "ann_ret_pct": round(float(r.mean() * self.ppy * 100), 2),
             "ann_vol_pct": round(float(r.std(ddof=1) * np.sqrt(self.ppy) * 100), 2),
             "days_in_market_pct": round(float((r != 0).mean() * 100), 1),
             "max_dd": round(summarise(r, self.ppy)["max_dd"], 3),
             "mean_bps_per_event": round(float(self.events.mean() * 1e4), 1),
             "t_per_event": round(tstat(self.events), 2), "n_events": int(len(self.events)),
             **{f"sh_{k}": v for k, v in sub.items()}}
        if with_placebo:
            p = self.placebo()
            d["placebo_pctile"] = round(float((d["sharpe"] > p).mean() * 100), 0)
            d["placebo_p95"] = round(float(np.percentile(p, 95)), 3)
        return d


def build_intraday_arms(R: pd.DataFrame, Fc: pd.DataFrame) -> list[IntradayArm]:
    print(f"\n{'='*100}\n§5 — PRECISE INTRADAY WINDOWS around the 14:00-ET statement (corrected, 2020→)\n{'='*100}")
    ts = announce_timestamps_utc(14, 0)
    eq_cal = R.index[R.index >= pd.Timestamp("2020-01-01", tz="UTC")]
    cr_cal = pd.date_range(eq_cal.min(), eq_cal.max(), freq="D", tz="UTC")
    arms = []
    for t in ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLK", "HYG"):
        s, bar = _intraday_close(t)
        if s.empty:
            continue
        for hrs, end in ((2.0, 0.0), (6.0, 0.0), (24.0, 0.0), (48.0, 0.0), (24.0, 24.0)):
            arms.append(IntradayArm(f"{t}:w{hrs:.0f}h@{end:+.0f}", sz.price_at_instant(s, bar), ts,
                                    hrs, end, eq_cal, EQ_COST))
    h = pd.read_parquet(CACHE / "crypto_1h_close.parquet", columns=["BTCUSDT", "ETHUSDT"])
    if h.index.tz is None:
        h.index = h.index.tz_localize("UTC")
    pa = {c: sz.price_at_instant(h[c], pd.Timedelta("1h")) for c in h.columns}
    pa["crypto2"] = None
    for c in ("BTCUSDT", "ETHUSDT"):
        for hrs, end in ((2.0, 0.0), (6.0, 0.0), (12.0, 0.0), (24.0, 0.0), (48.0, 0.0), (24.0, 24.0)):
            arms.append(IntradayArm(f"{c[:3]}:w{hrs:.0f}h@{end:+.0f}", pa[c], ts, hrs, end, cr_cal,
                                    CR_COST, funding_daily=Fc[c], ppy=365))
    return arms


# ════════════════════════════════════════════════════════════════════════════════════════════
# §6 — turn-of-month variants
# ════════════════════════════════════════════════════════════════════════════════════════════
def build_tom_arms(R: pd.DataFrame, Rc: pd.DataFrame, Fc: pd.DataFrame) -> tuple[list[Arm], dict]:
    print(f"\n{'='*100}\n§6 — TURN-OF-MONTH variants (beta-neutral spread, bonds, conditioning)\n{'='*100}")
    eq = R[EQ_US + EQ_SECTOR + EQ_INTL].mean(axis=1)
    streams = {
        "SPY": (R["SPY"], EQ_COST, 252, None, 1.0),
        "eq15ew": (eq, EQ_COST, 252, None, 1.0),
        "TLT": (R["TLT"], EQ_COST, 252, None, 1.0),
        "bond3rp": (inv_vol_basket(R[["TLT", "IEF", "LQD"]]), EQ_COST, 252, None, 1.0),
        "xasset8rp": (inv_vol_basket(R[["SPY", "QQQ", "EFA", "EEM", "HYG", "TLT", "GLD", "LQD"]]),
                      EQ_COST, 252, None, 1.0),
        "crypto2": (Rc.mean(axis=1), CR_COST, 365, Fc.mean(axis=1), 1.0),
    }
    arms = []
    for sname, (ret, cost, ppy, fund, mult) in streams.items():
        ret = ret.dropna()
        anch = sz.month_end_anchors(ret.index)
        for wlab, b, a in (("m1p3", 1, 3), ("m1p1", 1, 1), ("m4p3", 4, 3)):
            arms.append(Arm(f"tom_{sname}:{wlab}", ret, anch, sz.turn_of_month_offsets(b, a), (),
                            cost_bps=cost, ppy=ppy, funding=fund, cost_mult=mult, group="tom"))

    # beta-neutral calendar spread: long the window, short the rest of the month at matched $-days
    extra = {}
    for sname, (ret, cost, ppy, fund, mult) in streams.items():
        ret = ret.dropna()
        anch = sz.month_end_anchors(ret.index)
        inw = sz.window_position(ret.index, anch, sz.turn_of_month_offsets(1, 3))
        frac = float(inw.mean())
        pos = inw - (1 - inw) * frac / max(1 - frac, 1e-9)      # equal long and short exposure-days
        net = sz.hold_backtest(ret, pos, cost_bps=cost * mult)["net"]
        if fund is not None:
            net = net - pos * fund.reindex(net.index).fillna(0.0)
        a, b = ret[inw > 0].dropna(), ret[inw == 0].dropna()
        welch = float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        extra[f"tom_spread_{sname}"] = {"sharpe": round(sharpe(net, ppy), 3),
                                        "ann_ret_pct": round(float(net.mean() * ppy * 100), 2),
                                        "mean_in_minus_out_bps": round(float((a.mean() - b.mean()) * 1e4), 2),
                                        "t_in_vs_out": round(welch, 2)}
        print(f"  spread {sname:10s} long-in/short-out net Sharpe {extra[f'tom_spread_{sname}']['sharpe']:+.2f}"
              f"   (in−out {extra[f'tom_spread_{sname}']['mean_in_minus_out_bps']:+.2f}bps, t {welch:+.2f})")

    # dash-for-cash conditioning: month-end liquidity demand should be strongest after a weak month
    for sname in ("SPY", "eq15ew"):
        ret = streams[sname][0].dropna()
        anch = sz.month_end_anchors(ret.index)
        off = sz.turn_of_month_offsets(1, 3)
        inw = sz.window_position(ret.index, anch, off)
        prior = ret.rolling(21).sum().reindex(ret.index)
        gate_dn = (prior.shift(1) < 0).astype(float)            # prior-month return known at entry
        for glab, gate in (("after_down_month", gate_dn), ("after_up_month", 1 - gate_dn)):
            pos = inw * gate
            net = sz.hold_backtest(ret, pos, cost_bps=EQ_COST)["net"]
            extra[f"tom_{sname}:{glab}"] = {"sharpe": round(sharpe(net, 252), 3),
                                            "days_pct": round(float((pos > 0).mean() * 100), 1),
                                            "in_window_mean_bps": round(float(ret[pos > 0].mean() * 1e4), 2)}
            print(f"  conditional {sname:8s} {glab:18s} net Sharpe "
                  f"{extra[f'tom_{sname}:{glab}']['sharpe']:+.2f} "
                  f"(in-window {extra[f'tom_{sname}:{glab}']['in_window_mean_bps']:+.1f}bps, "
                  f"{extra[f'tom_{sname}:{glab}']['days_pct']:.0f}% of days)")
    return arms, extra


# ════════════════════════════════════════════════════════════════════════════════════════════
# §7 — funnel for the best arms
# ════════════════════════════════════════════════════════════════════════════════════════════
def _wf_oos(M: pd.DataFrame, ppy: float, train: int, test: int, embargo: int, top_k: int = 3):
    segs, n_ref, start = [], 0, train
    while start + test <= len(M):
        tr = M.iloc[max(0, start - train):max(0, start - embargo)]
        sr = (tr.mean() / tr.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        segs.append(M.iloc[start:start + test][list(sr.nlargest(top_k).index)].mean(axis=1))
        n_ref += 1
        start += test
    return (pd.concat(segs) if segs else pd.Series(dtype=float)), n_ref


def ppy_of(s: pd.Series) -> float:
    """Observations per year of the series itself — the book blends 252-day and 365-day legs, so a
    flat 365 would overstate any sub-365 series (same convention as run_master_book)."""
    yrs = (s.index.max() - s.index.min()).days / 365.25
    return len(s) / yrs if yrs > 0 else 252.0


def section_beta_control(R: pd.DataFrame, arms: list) -> dict:
    """The control the first pass never ran: does a naked beta stub lift the book by MORE?

    A calendar book that helps the master book is only interesting if the help came from the calendar.
    The master book is market-neutral and very slightly short beta, so ANY long-equity stub raises its
    Sharpe. Buy-&-hold SPY and a random-days-long stub are blended in at the same weights, vol-matched
    the same way, and the OOS block is shown separately — a lift that a coin-flip calendar matches is
    exposure, not timing.
    """
    print(f"\n{'='*100}\n§8 — BETA CONTROL for the book-lift claim\n{'='*100}")
    book = pd.read_parquet(REP / "master_book.parquet")["ret"].dropna()
    if book.index.tz is not None:
        book.index = book.index.tz_localize(None)
    oos = pd.Timestamp(json.loads((REP / "master_book_summary.json").read_text())["oos_start"])
    spy = R["SPY"].copy()
    spy.index = spy.index.tz_localize(None)
    mask = pd.Series(rng.random(len(spy)) < 0.334, index=spy.index).astype(float)   # match the widest ToM arm
    stubs = {"buy_hold_SPY": spy,
             "random_33pct_days_SPY": spy * mask - mask.diff().abs().fillna(0.0) * EQ_COST / 1e4}
    cands = dict(stubs)
    for a in arms:
        s = a.net.copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        cands[a.name] = s
    ppy_b, out = ppy_of(book), {}
    for lab, c in cands.items():
        h = c.reindex(book.index).fillna(0.0)
        if h.std() == 0:
            continue
        hm = h * (book.std() / h.std())
        lift = {f"{int(w*100)}%": round(sharpe((1 - w) * book + w * hm, ppy_b), 3) for w in (0.0, 0.1, 0.2, 0.3)}
        b_o, h_o = book[book.index >= oos], hm[hm.index >= oos]
        ppy_o = ppy_of(b_o)              # the OOS block is fully live (~366 obs/yr), not the blend's ~323
        out[lab] = {"lift": lift, "oos_0pct": round(sharpe(b_o, ppy_o), 2),
                    "oos_20pct": round(sharpe(0.8 * b_o + 0.2 * h_o, ppy_o), 2),
                    "corr_to_book": round(float(h.corr(book)), 3)}
    ranked = sorted(out.items(), key=lambda kv: -kv[1]["lift"]["20%"])
    print(f"  {'candidate':26s} {'corr':>6s} " + " ".join(f"{w:>7s}" for w in ("0%", "10%", "20%", "30%")) +
          "   OOS 0%→20%")
    for lab, d in ranked[:10] + [(k, v) for k, v in out.items() if k in stubs and k not in dict(ranked[:10])]:
        print(f"  {lab:26s} {d['corr_to_book']:>6.2f} " + " ".join(f"{v:>7.3f}" for v in d["lift"].values()) +
              f"   {d['oos_0pct']:.2f}→{d['oos_20pct']:.2f}" +
              ("   <-- pure beta, no calendar" if lab in stubs else ""))
    best_cal = max((v["lift"]["20%"] for k, v in out.items() if k not in stubs), default=0.0)
    print(f"\n  best calendar arm at 20%: {best_cal:.3f}   vs   buy-&-hold SPY: "
          f"{out['buy_hold_SPY']['lift']['20%']:.3f}  →  "
          f"{'the calendar adds nothing beta would not' if best_cal <= out['buy_hold_SPY']['lift']['20%'] else 'the calendar beats plain beta'}")
    return out


def section_funnel(arms: list, table: pd.DataFrame, n_trials: int) -> dict:
    """Full funnel on the arms that survive the two selection criteria that mean different things.

    Ranking by raw Sharpe finds whatever is longest the market; ranking by the margin over the arm's
    OWN shuffled-calendar placebo finds what the *calendar* earned. Both lists are run — a variant
    that only makes the first is beta wearing an event's clothes.
    """
    print(f"\n{'='*100}\n§7 — FUNNEL ({n_trials} arms swept → deflated Sharpe charged for all of them)\n{'='*100}")
    by_name = {a.name: a for a in arms}
    tbl = table.copy()
    tbl["margin"] = tbl["sharpe"] - tbl["placebo_p95"]
    best = list(dict.fromkeys(list(tbl.sort_values("sharpe", ascending=False)["arm"].head(5)) +
                              list(tbl.sort_values("margin", ascending=False)["arm"].head(5))))
    best = [n for n in best if n in by_name]
    book = pd.read_parquet(REP / "master_book.parquet")["ret"] if (REP / "master_book.parquet").exists() else None
    legs = pd.read_parquet(REP / "master_book_legs.parquet") if (REP / "master_book_legs.parquet").exists() else None
    if book is not None and book.index.tz is not None:
        book.index = book.index.tz_localize(None)
    out = {}
    for name in best:
        a = by_name[name]
        r = a.net.dropna()
        mc = bootstrap_sharpe(r, a.ppy, 1000, SEED)
        M = pd.DataFrame({x.name: x.net for x in arms if x.ppy == a.ppy and x.group == a.group}).fillna(0.0)
        wf, n_ref = _wf_oos(M, a.ppy, int(3 * a.ppy), int(0.5 * a.ppy), embargo=5)
        var_tr = float((M.mean() / M.std(ddof=1)).clip(-3, 3).var())
        dsr = deflated_sharpe(r.mean() / r.std(ddof=1), len(r), r.skew(), r.kurt() + 3.0,
                              n_trials=n_trials, var_across_trials=max(var_tr, 1e-8))
        row = {"sharpe": round(sharpe(r, a.ppy), 3), "mc_p5": mc.get("sharpe_p5"),
               "mc_p50": mc.get("sharpe_p50"), "wf_oos_sharpe": round(sharpe(wf, a.ppy), 3),
               "wf_refits": n_ref, "n_arms_in_wf_pool": M.shape[1], "deflated_sharpe": round(float(dsr), 3)}
        if book is not None:
            h = r.copy()
            h.index = h.index.tz_localize(None)
            bk = book.dropna()
            # the candidate is mapped onto the BOOK's calendar and is flat (0.0) on days it does not
            # trade — intersecting instead would silently shorten the book to the candidate's calendar
            hm = h.reindex(bk.index).fillna(0.0)
            row["corr_to_book"] = round(float(hm.corr(bk)), 3)
            ppy_b = ppy_of(bk)
            hm = hm * (bk.std() / hm.std()) if hm.std() > 0 else hm
            row["book_lift"] = {f"{int(w*100)}%": round(sharpe((1 - w) * bk + w * hm, ppy_b), 3)
                                for w in (0.0, 0.1, 0.2, 0.3)}
            if legs is not None:
                lg = legs.copy()
                if lg.index.tz is not None:
                    lg.index = lg.index.tz_localize(None)
                row["corr_to_legs"] = {c: round(float(hm.corr(lg[c].reindex(bk.index))), 2) for c in lg.columns}
        out[name] = row
        print(f"  {name:22s} Sharpe {row['sharpe']:+.2f} | MC-P5 {row['mc_p5']:+.2f} | "
              f"WF-OOS {row['wf_oos_sharpe']:+.2f} | deflated {row['deflated_sharpe']:.2f} | "
              f"corr {row.get('corr_to_book')} | lift {row.get('book_lift')}")
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    SEASONAL_DIR.mkdir(parents=True, exist_ok=True)
    R = etf_total_returns(ALL_ETF)
    Rc, Fc = crypto_returns()

    defects = section_defects(R)
    offmap = section_offset_map(R, Rc)

    print(f"\n{'='*100}\n§3 — EVENT BOOKS: side × asset × basket size (net of cost, funding-charged)\n{'='*100}")
    ev_arms = build_event_arms(event_streams(R, Rc, Fc))
    ev_table = evaluate(ev_arms, show=14)
    attribution = build_asset_attribution(R, Rc, Fc)

    cycle = section_cycle(R)
    intra_arms = build_intraday_arms(R, Fc)
    intra_table = evaluate(intra_arms, show=12)
    tom_arms, tom_extra = build_tom_arms(R, Rc, Fc)
    print()
    tom_table = evaluate(tom_arms, show=8)

    all_arms = ev_arms + intra_arms + tom_arms
    table = pd.concat([ev_table, intra_table, tom_table], ignore_index=True)
    n_trials = len(all_arms) + len(tom_extra) + 5          # + the cycle arms swept in §4
    funnel = section_funnel(all_arms, table, n_trials)
    beta_control = section_beta_control(R, all_arms)

    table.to_csv(SEASONAL_DIR / "seasonal_variants_grid.csv", index=False)
    attribution.to_csv(SEASONAL_DIR / "seasonal_variants_attribution.csv", index=False)
    pd.DataFrame({a.name: a.net for a in all_arms}).to_parquet(SEASONAL_DIR / "seasonal_variants_returns.parquet")
    summary = {"config": {"start": START, "eq_cost_bps": EQ_COST, "cr_cost_bps": CR_COST,
                          "n_arms_swept": n_trials, "subperiods": SUB_LABELS},
               "defects": defects, "offset_map": offmap, "cycle": cycle,
               "attribution": attribution.to_dict("records"),
               "tom_extra": tom_extra, "funnel": funnel, "beta_control": beta_control,
               "best_arms": table.sort_values("sharpe", ascending=False).head(20).to_dict("records")}
    (SEASONAL_DIR / "seasonal_variants_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n{'='*100}\nRUN SEASONAL VARIANTS OK — reports/seasonal/seasonal_variants_*.{{json,csv,parquet}}\n{'='*100}")


if __name__ == "__main__":
    main()
