"""Convexity / tail-overlay sleeve — explicit long-gamma via TRADABLE vol instruments.

The master book is six short-gamma risk-premium harvesters (short-vol / carry / momentum / breakout /
x-sect / BAB-adjacent). Its worst month (Oct-2018 -6.0%, right on the -6% gate) and its losing streaks
are correlated risk-off events. The crisis sleeve (managed-futures trend + defensive rotation) hedges
SUSTAINED crashes, but it needs a durable trend — it does NOT catch a one-day volatility SHOCK
(Feb-2018 volmageddon: VIX +115% in a day; the Aug-2024 yen-carry unwind: VIX->65 intraday). The only
instrument that pays in a one-day shock is explicit convexity: long volatility.

Naive always-on long-vol bleeds catastrophically through contango (VIXY has decayed ~32000x since 2011),
which would gut months-in-profit and return. So the overlay is TERM-STRUCTURE-TIMED: hold mid-term
long-vol (VIXM — decays only ~22x vs VIXY's ~32000x, the cleaner carry) ONLY while the VIX curve is
backwardated (spot VIX >= VIX3M) — the mechanical storm signal that (a) coincides with vol shocks and
(b) is the regime where the roll actually pays you to be long. Otherwise the sleeve is in cash (0 return).
Signal decided on the prior close (shift 1), no look-ahead. The book SELLS vol in calm (volprem family);
this sleeve BUYS the tail cheap only when the curve says a storm is here — self-financed convexity.

Honesty gates. VIXM is live from 2011-01 (before the 2016-08 reporting window) — no pre-inception vol is
synthesised. Prices are TwelveData daily, split-adjusted (the paid feed, cross-checked against Yahoo);
the ETP's roll cost is already inside its price. Triggers are mechanical (a fixed curve inequality), not
fitted to Feb-2018 / COVID dates.

Emits the raw sleeve return (0 on cash days) — the book-integration rescaling to 15% vol lives in the
candidate script, identical to every other family. Verdict on whether it fixes the book: run_convexity_book.py.

    python scripts/run_convexity.py   ->  reports/lab/convexity_sleeve.parquet  (+ crash-window diagnostics)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
from src.config import LAB_DIR, REPORTS_DIR  # noqa: E402
from src.data.twelvedata import load_bars  # noqa: E402
from src.data.cboe import load_cboe_vol  # noqa: E402
from src.metrics import summarise  # noqa: E402

PPY = 365
VETP = ROOT / "data/raw/vol_etp"
# crash windows where a one-day vol shock hit — the months a trend crisis-sleeve structurally misses
CRASH_WINDOWS = {
    "Feb-2018 volmageddon": ("2018-02-01", "2018-02-28"),
    "Q4-2018 selloff":      ("2018-10-01", "2018-12-31"),
    "Aug-2019 trade/curve":  ("2019-08-01", "2019-08-31"),
    "COVID-2020":            ("2020-02-15", "2020-04-30"),
    "2022 rate repricing":   ("2022-01-01", "2022-06-30"),
    "Aug-2024 yen unwind":   ("2024-08-01", "2024-08-16"),
    "Apr-2025 tariff shock": ("2025-04-01", "2025-04-30"),
}


def etp_ret(symbol: str) -> pd.Series:
    """Split-adjusted daily return for a vol ETP. Cache the clean {close,ret} under a stable name
    (TwelveData's own cache filename embeds today's date); fetch once from the paid feed if absent."""
    stable = VETP / f"{symbol}.parquet"
    if stable.exists():
        s = pd.read_parquet(stable)["ret"]
    else:
        df = load_bars(symbol, "1day", "2009-01-01")
        if df.empty:
            raise RuntimeError(f"convexity: no TwelveData bars for {symbol}")
        close = df["close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.sort_index()
        VETP.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"close": close, "ret": close.pct_change()}).to_parquet(stable)
        s = close.pct_change()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.rename(symbol)


def backwardation_signal(thr: float = 1.0) -> pd.Series:
    """1 when the VIX term structure is backwardated (spot VIX >= thr * VIX3M) — front richer than
    3-month = stress / positive long-vol roll. Both series are Cboe official closes."""
    vix = load_cboe_vol("VIX"); vix.index = pd.to_datetime(vix.index)
    vix3m = load_cboe_vol("VIX3M"); vix3m.index = pd.to_datetime(vix3m.index)
    ratio = vix / vix3m.reindex(vix.index).ffill()
    return (ratio >= thr).astype(float).rename("backwardation")


def build_sleeve(underlying: str = "VIXM", thr: float = 1.0) -> pd.Series:
    """Timed long-vol return: hold the underlying when yesterday's close was backwardated, else cash (0).
    0-on-cash is a real return (the strategy is flat), and keeps the series gap-free for downstream vol
    estimation — exactly how a go-to-cash family is represented."""
    ur = etp_ret(underlying)
    sig = backwardation_signal(thr).shift(1).reindex(ur.index).fillna(0.0)
    off = sig.eq(0.0)
    if off.all():
        print(f"WARNING convexity: signal never fired for {underlying} @thr={thr} — sleeve is all-cash")
    return (ur * sig).fillna(0.0).rename("ret")


def diagnostics(sleeve: pd.Series, thr: float) -> dict:
    s = sleeve[sleeve.index >= "2016-08-01"]
    active = float((s != 0).mean())
    ss = summarise(s, PPY)
    # per-crash payoff (compounded sleeve return across the window) — the reason the sleeve exists
    crash = {}
    for name, (a, b) in CRASH_WINDOWS.items():
        w = sleeve[(sleeve.index >= a) & (sleeve.index <= b)]
        crash[name] = float((1.0 + w).prod() - 1.0) if len(w) else float("nan")
    # calm-year bleed: full-calendar years, compounded — the carry cost the timing is meant to contain
    yearly = {int(y): float((1.0 + g).prod() - 1.0) for y, g in s.groupby(s.index.year)}
    return {"active_fraction": active, "standalone_sharpe": ss["sharpe_ann"],
            "months_in_profit": ss["months_in_profit"], "total_return": ss["total_return"],
            "crash_payoffs": crash, "yearly": yearly}


def main() -> None:
    thr = 1.0
    underlying = "VIXM"
    sleeve = build_sleeve(underlying, thr)
    d = diagnostics(sleeve, thr)

    print(f"=== CONVEXITY SLEEVE: backwardation-timed long {underlying} (VIX >= {thr:.2f}*VIX3M) ===")
    print(f"coverage {sleeve.index.min().date()}..{sleeve.index.max().date()}  ({len(sleeve)} days)")
    print(f"active (in-market) fraction: {d['active_fraction']:.1%}   standalone Sharpe {d['standalone_sharpe']:+.2f}   "
          f"months+ {d['months_in_profit']:.0%}   total {d['total_return']:+.0%}")
    print("\ncrash-window payoffs (the shocks a trend crisis-sleeve misses):")
    for name, v in d["crash_payoffs"].items():
        print(f"  {name:24s} {v:+7.1%}")
    print("\nper-year sleeve return (calm-year bleed vs crash-year payoff):")
    for y, v in sorted(d["yearly"].items()):
        print(f"  {y}: {v:+6.1%}")

    REPORTS_DIR.mkdir(exist_ok=True)
    sleeve.to_frame().to_parquet(LAB_DIR / "convexity_sleeve.parquet")
    _figure(sleeve, d)
    print("\nartifact -> reports/lab/convexity_sleeve.parquet   figure -> reports/figures/convexity_sleeve.png")
    print("CONVEXITY SLEEVE OK")


def _figure(sleeve: pd.Series, d: dict) -> None:
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    s = sleeve[sleeve.index >= "2016-08-01"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    (1 + s).cumprod().plot(ax=ax[0], color="#8c1d40", lw=1.4)
    ax[0].set_yscale("log"); ax[0].set_title("Convexity sleeve equity (backwardation-timed long VIXM) — flat in calm, spikes in shocks")
    yrs = sorted(d["yearly"]); ax[1].bar(range(len(yrs)), [d["yearly"][y] for y in yrs],
        color=["#2ca02c" if d["yearly"][y] > 0 else "#d62728" for y in yrs])
    ax[1].set_xticks(range(len(yrs))); ax[1].set_xticklabels([str(y)[2:] for y in yrs]); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("Per-year sleeve return: crash-year payoff (2018/2020) vs calm-year timed bleed")
    fig.tight_layout(); (REPORTS_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / "figures" / "convexity_sleeve.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
