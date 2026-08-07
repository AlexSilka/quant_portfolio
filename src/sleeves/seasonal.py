"""Calendar-seasonality sleeve — event-timing books that are long *only inside a deterministic
calendar window* and flat otherwise. Two documented effects live here (docs/HYPOTHESES.md H4):

  • pre-FOMC announcement drift (Lucca-Moench 2015): abnormal equity returns accrue in the ~24h
    before a scheduled FOMC statement — historically ~80% of the annual equity premium in that
    handful of hours. Event anchor = the announcement day (src/data/fomc.py).
  • turn-of-month (Lakonishok-Smidt 1988): returns concentrate on the last trading day of a month
    and the first few of the next (month-end pension/index flows). Anchor = each month-end bar.

Why a separate engine from src/sleeves/xsect.py. Those are cross-sectional *ranking* books whose
signal is estimated from market data (hence the t+2 delay to avoid look-ahead). A calendar window
is **known years in advance** — there is no signal to estimate and no look-ahead in knowing that
tomorrow is the day before an FOMC meeting. So the honest execution model is different, and it is
the crux of whether these survive (H4 spec): you *hold through* the multi-day window, so you pay
commission+spread only at the window's **edges** (one entry, one exit), never the daily round-trip
that killed the overnight sleeve. Costs are otherwise the same liquidity-aware machinery as the rest
of the book, so the Sharpes are directly comparable.

Execution convention (stated so it can be audited): position[d] = 1 means "hold over daily bar d",
entered at the close of bar d-1 and exited at the close of bar d — i.e. the return earned is ret[d].
Because the window is deterministic this needs no estimation lag; a robustness that shifts the whole
window one bar later (enter a day after) is run in the driver to prove the effect is not a fill-timing
artifact. Everything is computable-at-bar by construction (the calendar is exogenous).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import panel_impact_cost

from src.sleeves.xsect import top_n_liquid


# ── event → position mask ───────────────────────────────────────────────────────────────────
def window_position(index: pd.DatetimeIndex, anchors: pd.DatetimeIndex,
                    offsets: list[int]) -> pd.Series:
    """1.0 on every bar whose *trading-day offset* from an anchor bar is in `offsets`, else 0.0.

    Offsets are in **trading days** (positions in `index`), not calendar days, so a window is
    measured in real bars and never lands on a holiday/weekend gap. An anchor that is not itself a
    bar (e.g. a Sunday FOMC date, or an announce day the exchange was shut) is snapped to the next
    available bar via searchsorted, so its window is still placed. Overlapping windows (offsets from
    two nearby anchors hitting the same bar) collapse to a single long position — you cannot be
    "twice long" — which is the correct book.

    offsets examples: pre-FOMC day-before = [-1]; announce-day bar = [0]; both = [-1, 0];
    turn-of-month (-1,+3) around a month-end anchor = [0, 1, 2, 3]; broad (-4,+3) = [-3..3].
    """
    n = len(index)
    pos = np.zeros(n, dtype=float)
    # locate each anchor's bar position (next bar if the anchor date itself is not a trading bar)
    locs = index.searchsorted(anchors)
    for a in np.unique(locs):
        if a >= n:
            continue
        for off in offsets:
            j = a + off
            if 0 <= j < n:
                pos[j] = 1.0
    return pd.Series(pos, index=index)


def window_instances(index: pd.DatetimeIndex, anchors: pd.DatetimeIndex,
                     offsets: list[int]) -> pd.Series:
    """Like window_position, but stamp each window bar with its *anchor's* integer id (NaN off-window).

    Groups the bars of one multi-day window (all offsets of the same anchor) under a single id, so a
    cross-asset book can aggregate a whole turn-of-month / pre-FOMC episode into one "instance" return
    and trade the cross-section once per event — the unit a relative-value seasonal book rebalances on.
    """
    n = len(index)
    ids = np.full(n, np.nan)
    locs = index.searchsorted(anchors)
    for a in np.unique(locs):
        if a >= n:
            continue
        for off in offsets:
            j = a + off
            if 0 <= j < n:
                ids[j] = float(a)
    return pd.Series(ids, index=index)


def month_end_anchors(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading bar of each calendar month present in `index` — the turn-of-month anchor.

    Built by taking the max bar within each (year, month) group; `.tolist()` keeps the elements as
    tz-aware Timestamps (a plain `.values`/numpy round-trip would silently drop the tz and break the
    tz-aware searchsorted in window_position).
    """
    idx = pd.DatetimeIndex(index)
    last = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max()
    return pd.DatetimeIndex(sorted(last.tolist()))


def turn_of_month_offsets(days_before: int = 1, days_after: int = 3) -> list[int]:
    """Trading-day offsets for a (−days_before, +days_after) turn-of-month window around month-end.

    Classic Lakonishok-Smidt is (−1,+3): the last trading day (offset 0) + the first three of the
    next month (offsets +1..+3). days_before counts the last N days of the month *inclusive* of the
    last (so days_before=1 → just offset 0; days_before=4 → offsets −3..0).
    """
    return list(range(-(days_before - 1), days_after + 1))


# ── time-series book: long the index only inside the window ─────────────────────────────────
def hold_backtest(ret: pd.Series, position: pd.Series, *, cost_bps: float = 3.0,
                  exec_shift: int = 0) -> dict:
    """Long-in-window time-series book on one return stream; charge cost only at window edges.

    position ∈ {0,1} (target exposure per bar). The return earned is position·ret; turnover is
    |Δposition| so a 0→1 entry and a 1→0 exit each cost one side of `cost_bps` (a round trip = 2×) —
    and, crucially, *nothing is charged on the days held inside the window*. exec_shift>0 slides the
    whole realised position that many bars later (the fill-timing robustness). Returns the same
    5-field shape as xsect.xs_backtest so the caller can vol-target the net series.
    """
    p = position.reindex(ret.index).fillna(0.0)
    if exec_shift:
        p = p.shift(exec_shift).fillna(0.0)
    gross = p * ret
    turn = p.diff().abs().fillna(p.abs())
    cost = turn * cost_bps / 1e4
    net = gross - cost
    return {"net": net, "gross": gross, "turnover": turn, "cost": cost, "position": p}


# ── cross-sectional book: long the top-N liquid names only inside the window ─────────────────
def xs_window_backtest(close: pd.DataFrame, position: pd.Series, *, top_n: int = 50,
                       cost_bps: float = 6.0, exec_shift: int = 0, adv: pd.DataFrame | None = None,
                       impact_k: float = 0.0, capital: float = 500_000.0, vol_lb: int = 20,
                       lookback_days: int = 30, bpd: int = 1) -> dict:
    """Equal-weight long-only book across the `top_n` most-liquid names, live only inside the window.

    This is the "top-10 / top-50 / top-100" breadth cut the request asks for: during a window bar the
    book holds an equal-weight basket of the top_n names (by trailing daily $-volume, survivorship-free
    via top_n_liquid), and is flat (cash) otherwise. It answers whether the calendar premium is a broad
    market-timing effect (bigger basket ≈ same Sharpe) or concentrated in a few names. Cost is charged
    on the basket's turnover — one build at entry, one unwind at exit, plus the small daily drift of
    membership — with the optional √-impact term (liquidity-aware), never a flat daily round-trip.
    """
    ret = close.pct_change(fill_method=None)
    # eligible = top_n most liquid each bar; equal weight among them, but only on window bars
    elig = top_n_liquid(pd.DataFrame(1.0, index=close.index, columns=close.columns), adv, top_n,
                        bpd_=bpd, lookback_days=lookback_days).notna()
    p = position.reindex(close.index).fillna(0.0)
    n_elig = elig.sum(axis=1).replace(0.0, np.nan)
    w = elig.div(n_elig, axis=0).mul(p, axis=0).fillna(0.0)   # rows sum to p (1 inside window, 0 out)
    if exec_shift:
        w = w.shift(exec_shift).fillna(0.0)
    gross = (w * ret).sum(axis=1)
    dw = w.diff().abs()
    turn = dw.sum(axis=1)
    cost = turn * cost_bps / 1e4
    if adv is not None and impact_k > 0.0:
        cost = cost + panel_impact_cost(dw, ret.rolling(vol_lb).std(), adv, capital, impact_k)
    net = gross - cost
    return {"net": net, "gross": gross, "turnover": turn, "cost": cost, "weights": w}


# ── window diagnostics (the honest "is it timing or beta?" split) ───────────────────────────
def in_vs_out(ret: pd.Series, position: pd.Series) -> dict:
    """Decompose a buy-&-hold stream into its in-window and out-of-window parts.

    The Lucca-Moench headline is that most of the *total* return accrues inside a tiny window. This
    reports, on the same asset: annualised mean and Sharpe of the in-window bars vs the out-of-window
    bars, the fraction of days spent in-window, and the share of total cumulative log-return captured
    inside it — the single most telling number for a calendar effect (concentration, not just sign).
    """
    p = position.reindex(ret.index).fillna(0.0).astype(bool)
    r = ret.reindex(p.index).fillna(0.0)
    lg = np.log1p(r.clip(-0.99, None))
    tot = float(lg.sum())
    n = len(r)
    n_in = int(p.sum())

    def _sh(x):
        x = x.dropna()
        sd = x.std(ddof=1)
        return float(np.sqrt(252) * x.mean() / sd) if sd > 0 and len(x) > 2 else 0.0

    return {
        "frac_days_in_window": round(n_in / n, 4) if n else 0.0,
        "in_window_share_of_total_logret": round(float(lg[p].sum() / tot), 3) if tot != 0 else None,
        "in_window_sharpe": round(_sh(r[p]), 3),
        "out_window_sharpe": round(_sh(r[~p]), 3),
        "in_window_ann_ret_pct": round(float(r[p].mean() * 252 * 100), 2),
        "out_window_ann_ret_pct": round(float(r[~p].mean() * 252 * 100), 2),
        "in_window_mean_bps": round(float(r[p].mean() * 1e4), 2),
        "n_in_window_bars": n_in,
    }
