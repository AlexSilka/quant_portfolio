"""Calendar / session sleeve — the overnight-vs-intraday return decomposition.

The brief's §4 lists "Calendar and session effects" as a required feature family; this is the
strategy sleeve that trades it. Each equity bar splits into two disjoint sessions:

    overnight[t] = open[t]  / close[t-1] - 1     (the close-to-open gap, held through the night)
    intraday[t]  = close[t] / open[t]   - 1      (the open-to-close move, held through the day)

The documented anomaly (Cliff-Cooper-Gulen; Lou-Polk-Skouras; Hendershott-Livdan-Rösch) is that
the equity premium accrues *overnight* while the intraday leg is flat/negative, and that the two
sessions are traded by different clienteles — so overnight returns behave differently from intraday.
This module builds the market-neutral cross-sectional version and, crucially, prices the two honest
execution models, because *where* the return is captured decides whether the effect is tradable:

    execution="overnight_only" : be flat intraday — enter at close(t-1), exit at open(t). This isolates
                                 the overnight leg but pays a FULL round-trip every single day.
    execution="hold_24h"       : hold the position through the whole day (earn close->close) and only
                                 tilt it by the overnight-derived signal — normal rebalance turnover,
                                 but you are no longer harvesting "overnight" per se, you are running a
                                 close-to-close book whose signal happens to come from the night leg.

Every signal is stamped at bar t from data <= t and the book is delayed exec_lag bars, so a signal
never fills at its own bar's close (the shared t+2-style convention). Costs are commission+half-spread
on traded notional plus an optional √-impact term (liquidity-aware, never flat) — identical machinery
to src/sleeves/xsect.py, so overnight numbers are directly comparable to the rest of the book.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.sleeves.xsect import held_turnover

from src.backtest.costs import panel_impact_cost

from src.config import RAW_DIR  # noqa: E402
from src.sleeves.xsect import top_n_liquid

_RAW = RAW_DIR / "equity_td"


# ── session-return panels ─────────────────────────────────────────────────────────────────
def session_returns(close_panel: pd.DataFrame, raw_dir: Path = _RAW, winsor: float = 0.5,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overnight and intraday return panels aligned to a close panel's index and columns.

    open/close come from the per-ticker raw daily parquets (Twelve Data, split-adjusted). The two
    panels inherit the close panel's NaN mask, so a name participates in a session exactly when it
    is a valid member that bar — the panel's survivorship policy is carried through unchanged, no
    fresh survivorship is introduced here.

    Data hygiene (essential, not cosmetic): open and close are split/dividend-adjusted on their own
    schedules, so on an adjustment day `open / close.shift(1)` can print a spurious ±100%+ (or ∞ when
    a prior close rounds to zero) that is a *data artifact*, not an overnight gap. Left in, those few
    hundred name-days manufacture a fake cross-sectional reversal "edge" (a name spikes, then
    "reverts"). ±inf is dropped and |session return| > `winsor` is NaN-ed out (the name simply is not
    ranked that bar). The driver reports the raw-vs-clean delta so the artifact is visible, not hidden.
    """
    idx, cols = close_panel.index, close_panel.columns
    on = pd.DataFrame(index=idx, columns=cols, dtype=float)
    intraday = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t in cols:
        p = raw_dir / f"{t}_1d.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).reindex(idx)
        o, c = df["open"], df["close"]
        on[t] = o / c.shift(1) - 1.0
        intraday[t] = c / o - 1.0
    mask = close_panel.notna()
    on, intraday = on.replace([np.inf, -np.inf], np.nan), intraday.replace([np.inf, -np.inf], np.nan)
    if np.isfinite(winsor):
        on = on.where(on.abs() <= winsor)
        intraday = intraday.where(intraday.abs() <= winsor)
    return on.where(mask), intraday.where(mask)


def dense_rows(close_panel: pd.DataFrame, min_valid: int = 100) -> pd.DataFrame:
    """Drop union-calendar junk rows (holidays / mis-aligned dates with almost no live names).

    A cross-sectional book must rank across a real trading day's names; a row with < min_valid
    valid closes is a calendar artifact that otherwise leaves the book flat and corrupts the
    vol-target's rolling-std denominator. Same guard build_book applies to the equity panel.
    """
    return close_panel[close_panel.notna().sum(axis=1) >= min_valid]


# ── signal ────────────────────────────────────────────────────────────────────────────────
def trailing_session(session_panel: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Trailing-mean session return over `lookback` bars — the ranking signal, computable at bar t.

    Ranking long the high tail and short the low tail is *continuation*; flip the sign (direction=-1
    in xs_book) for *reversal* (short recent overnight winners). Post-2016 on liquid US names the
    reversal sign is the one that is (weakly) positive — the classic overnight-momentum sign has
    decayed/inverted as the effect crowded, a finding the driver documents rather than assumes.
    """
    return session_panel.rolling(lookback).mean()


# ── book ──────────────────────────────────────────────────────────────────────────────────
def xs_book(close: pd.DataFrame, earn: pd.DataFrame, signal: pd.DataFrame, *,
            direction: float = -1.0, top_n: int = 100, top_frac: float = 0.2,
            execution: str = "overnight_only", exec_lag: int = 1, cost_bps: float = 3.0,
            adv: pd.DataFrame | None = None, impact_k: float = 0.0, capital: float = 500_000.0,
            min_names: int = 6, vol_lb: int = 20) -> dict:
    """Dollar-neutral long/short book on `signal`, earning the `earn` session panel.

    direction=-1 shorts the top of the signal and longs the bottom (reversal). Weights rank only the
    `top_n` most-liquid names each bar (survivorship-free, tradable), are delayed `exec_lag` bars, and
    are held flat between rebalances. The cost model is the crux:

      execution="overnight_only": you must be flat during the day, so every bar you trade the FULL
        position in (at the prior close) and out (at the open) — traded notional = 2·gross each bar,
        plus the rebalance from yesterday's target. This is what makes "harvest the overnight leg"
        expensive.
      execution="hold_24h": you keep the position over the day, so traded notional is only the
        rebalance (w.diff) — but `earn` should then be the close-to-close panel, not the overnight leg.

    Returns net/gross/turnover/cost/weights, in the same 5-field shape as xsect.xs_backtest so the
    caller can vol-target the net series (leverage multiplies gross-return and cost alike — exact).
    """
    sig = top_n_liquid(signal, adv, top_n) * direction
    ranks = sig.rank(axis=1, pct=True)
    n_valid = sig.notna().sum(axis=1)
    longs = (ranks >= 1.0 - top_frac).astype(float)
    shorts = (ranks <= top_frac).astype(float)
    wl = longs.div(longs.sum(axis=1).replace(0.0, np.nan), axis=0)
    ws = shorts.div(shorts.sum(axis=1).replace(0.0, np.nan), axis=0)
    w = (wl - ws).where(n_valid >= min_names, 0.0).fillna(0.0).shift(exec_lag).fillna(0.0)

    gross_ret = (w * earn).sum(axis=1)
    gross_expo = w.abs().sum(axis=1)
    # A book that is FLAT intraday is unwound and rebuilt every bar, so the round trip below already
    # pays for everything and there is no drift left to restore. A book HELD through the day does
    # drift, and putting it back on target is a trade `w.diff()` never sees (`xsect.held_turnover`).
    if execution == "overnight_only":
        dw = w.diff().abs()
        traded = 2.0 * gross_expo + dw.sum(axis=1)   # full round-trip every bar, plus the reweight
    elif execution == "hold_24h":
        dw = held_turnover(w, earn.reindex_like(w))
        traded = dw.sum(axis=1)                       # held — the target change AND the drift back
    else:
        raise ValueError(execution)
    lin_cost = traded * cost_bps / 1e4
    if adv is not None and impact_k > 0.0:
        sig_bar = close.pct_change(fill_method=None).rolling(vol_lb).std()
        lin_cost = lin_cost + panel_impact_cost(dw, sig_bar, adv, capital, impact_k)
    net = gross_ret - lin_cost
    return {"net": net, "gross": gross_ret, "turnover": traded, "cost": lin_cost,
            "weights": w, "exposure": gross_expo}
