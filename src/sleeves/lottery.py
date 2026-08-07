"""Cross-sectional skewness / lottery (MAX) sleeve — the retail-mispricing hypothesis.

Investors overpay for lottery-like assets — high idiosyncratic skew, high recent maximum daily
return — so those assets *underperform* (Bali-Cakici-Whitelaw 2011 "MAX"; Kumar 2009 "gambling").
The tradable expression is dollar-neutral **short the high-lottery tail / long the low-lottery
tail**, vol-targeted, run through the exact same engine (`src/sleeves/xsect.py::xs_backtest`) as the
momentum and carry books so its numbers and its cost model are directly comparable. Crypto is the
extreme case the spec flags — acute retail lottery demand for high-skew memecoins — so crypto is the
primary test; equity is secondary.

Two a-priori signals (declared before fitting, both reported — never peak-picked):

    skew  : trailing skewness of daily log-returns over `lookback` bars (20-60d).
    MAX   : mean of the top-`k` daily returns over the past ~month (Bali et al.'s MAX(5)).

Both are ranking signals fed to `xs_backtest`, which longs the top-ranked and shorts the
bottom-ranked names. To express the *lottery* bet (short high, long low) the driver passes the
**negated** signal (`-skew`, `-MAX`); the un-negated sign is the opposite bet (long the high tail),
reported alongside so the reader can see whether any positive number is a real lottery premium or
merely re-labelled momentum (high-MAX names are last month's winners).

`vol_signal` (trailing return-vol) is here only to build a low-volatility / BAB proxy book *through
the same engine*, so the driver can regress the skew book on it and show the lottery effect is (or is
not) independent of low-vol — the "is this just re-labelled low-beta?" orthogonality test the bar asks
for. Every signal is stamped at bar t from data <= t; execution delay + costs live in `xs_backtest`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ── signals (wide bars × names, value at t from data <= t) ─────────────────────────────────
def skew_signal(px: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """Trailing skewness of daily log-returns over `lookback` bars (the same measure as
    `features/engine.py::higher_moments`). High positive skew = lottery-like (rare big up-days);
    the lottery bet shorts it. Rolling, backward-only — computable at bar t."""
    return np.log(px).diff().rolling(lookback).skew()


def max_signal(px: pd.DataFrame, lookback: int = 21, k: int = 5) -> pd.DataFrame:
    """MAX(k): mean of the k largest *simple* daily returns over the trailing `lookback` window.

    Bali-Cakici-Whitelaw's lottery proxy — the recent maximum daily payoff a name has printed, the
    thing a lottery-seeking buyer chases. Vectorised per name via a sliding window with a NaN-aware
    top-k (missing bars sort to −∞). A window needs at least `max(k, lookback//2)` real returns to be
    ranked — enough to trust the MAX without demanding a fully-populated window, which would drop the
    gappy memecoins the hypothesis is precisely about. Simple returns (not log) so "max daily return"
    means the literal largest up-move, matching the source. The sliding window is materialised in
    column blocks sized to a ~500MB cap, so an intraday panel (tens of thousands of bars × a long
    window) stays bounded instead of allocating a T×N×lookback array."""
    r = px.pct_change(fill_method=None).to_numpy()
    T, N = r.shape
    out = np.full((T, N), np.nan)
    min_valid = max(k, lookback // 2)
    if T >= lookback:
        block = max(1, int(5e8 / (max(T, 1) * lookback * 8)))   # cap the materialised window at ~500MB
        for j0 in range(0, N, block):
            col = r[:, j0:j0 + block]
            win = sliding_window_view(col, lookback, axis=0)    # (T-lb+1, b, lb) view
            valid = np.isfinite(win).sum(axis=2)
            w = np.where(np.isfinite(win), win, -np.inf)
            topk = np.partition(w, -k, axis=2)[..., -k:].mean(axis=2)
            topk[valid < min_valid] = np.nan                    # too thin a window to rank
            out[lookback - 1:, j0:j0 + block] = topk
    return pd.DataFrame(out, index=px.index, columns=px.columns)


def vol_signal(px: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """Trailing daily-return volatility — the low-vol / BAB proxy used only for the orthogonality
    test (is the skew book just re-labelled low-vol?). Not a lottery signal itself."""
    return px.pct_change(fill_method=None).rolling(lookback).std()


# ── data integrity: are the extreme returns that drive skew/MAX real, or artifacts? ─────────
def return_diagnostics(px: pd.DataFrame, spike: float = 1.0, revert: float = 0.5) -> dict:
    """Quantify the data hazards specific to a moment / extreme-return signal on a crypto panel.

    skew and MAX are *by construction* dominated by a name's largest daily returns, so a single bad
    print (a spurious tick, a split-style mis-adjustment) would hijack the ranking — the exact trap
    that faked a +0.18 in the overnight sleeve. This reports, without hiding anything:

      * extreme_ge_{50,100,300}pct : count of |daily return| over each threshold.
      * spike_revert : |return| > `spike` immediately undone by ≥ `revert`× the next bar — the
        signature of a round-trip data glitch (a real pump/crash does *not* revert). If this is ~0
        the extreme moves are genuine and the signal is built on real economics, not errors.
      * ffill_flats : identical-close bars (build_panels ffills gaps up to 5 bars) — these inject
        return-0 rows that merely dilute a moment window, they do not manufacture an edge.
      * delisted_crashed : names whose last price is < 2% of their all-time max (LUNA-style) — the
        panel *keeps* dead names (good: no survivorship into survivors), but a position held into a
        delisting stops marking-to-market when the series goes NaN, so terminal crash losses are
        under-captured; the driver runs a delisting-trimmed variant to bound that bias.
    """
    r = px.pct_change(fill_method=None)
    a = r.abs().to_numpy()
    fin = np.isfinite(a)
    n = int(fin.sum())
    r1 = r.shift(-1)
    spike_revert = ((r.abs() > spike) & (np.sign(r) != np.sign(r1)) & (r1.abs() >= revert * r.abs()))
    last_vs_max = px.ffill().iloc[-1] / px.max()
    crashed = last_vs_max[last_vs_max < 0.02].dropna()
    return {
        "finite_name_days": n,
        "extreme_ge_50pct": int((a > 0.5).sum()),
        "extreme_ge_100pct": int((a > 1.0).sum()),
        "extreme_ge_300pct": int((a > 3.0).sum()),
        "max_daily_ret": round(float(np.nanmax(r.to_numpy())), 3),
        "min_daily_ret": round(float(np.nanmin(r.to_numpy())), 3),
        "spike_revert_glitches": int(spike_revert.to_numpy().sum()),
        "ffill_flat_bars": int((px.diff() == 0).to_numpy().sum()),
        "ffill_flat_pct": round(float((px.diff() == 0).to_numpy().sum()) / max(n, 1) * 100, 2),
        "delisted_crashed_names": int(len(crashed)),
    }


def predelist_mask(px: pd.DataFrame, tail: int = 5, crash_frac: float = 0.5) -> pd.DataFrame:
    """Boolean panel: True on a name's final `tail` valid bars *iff* it delists having crashed.

    A name is "delisted" if its series ends before the panel does; "crashed" if its last price is
    below `crash_frac`× its trailing peak. Ranking on / holding these bars is where the un-captured
    terminal-crash loss lives; masking them out gives the driver a delisting-trimmed book to prove
    the verdict does not hinge on that under-capture."""
    mask = pd.DataFrame(False, index=px.index, columns=px.columns)
    panel_end = px.notna()[::-1].idxmax().max()                 # last bar any name is live
    for c in px.columns:
        s = px[c].dropna()
        if len(s) < tail + 1 or s.index[-1] >= panel_end:
            continue                                            # still live at panel end → not delisted
        if s.iloc[-1] < crash_frac * s.cummax().iloc[-1]:
            mask.loc[s.index[-tail:], c] = True
    return mask
