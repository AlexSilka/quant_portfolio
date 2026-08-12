"""How long a window should the vol-target look back over? Sweep it — and defuse the three traps first.

`run_master_book._scale` sizes every leg off a trailing 60-bar vol estimate. That 60 was a convention,
never swept, so this is the sweep. Run naively it says "shorter is better" monotonically, and all three
reasons that reading is wrong are corrected here:

  BURN-IN. `_scale` fills its warm-up with zero, so a longer window holds each leg FLAT for longer at the
  start of its life. A long window then looks safer for free — BAB's worst day (2020-04-18) reads −0.0%
  at a 120-bar window purely because BAB lists in 2020 and the window had not filled. Every arm here
  starts each leg after the same 250-bar burn-in, so the samples are identical.

  RISK. A shorter window is a noisier estimate that misses more of the tail, so it sizes UP on average:
  the book runs at 24.2% vol at 10 bars against 22.5% at 60. Comparing those two directly compares two
  different amounts of risk. Every arm is rescaled by ONE constant to the shipped window's volatility
  before anything is read off it.

  THE UNCHARGED COST. Each sleeve's cost model runs INSIDE the sleeve. The vol-target multiplier is
  applied afterwards, to the sleeve's finished returns, so the rebalancing it implies is charged nowhere.
  A shorter window re-sizes far more often and in a naive backtest that trading is free. This reports the
  multiplier's own turnover and the break-even cost at which each arm stops beating the shipped one.

Verdict recorded by the run: 60 stays. At matched risk the whole 10–120 range is a plateau worth ~2%
relative, the gain is not robust across blocks, and on the master book a shorter window pushes
months-in-profit below its target. The 2020-04-18 blow-up it was meant to fix is identical at every
window from 20 to 90 — that failure is the ceiling, not the lookback.

    python scripts/run_volwindow_lab.py    ->  reports/lab/volwindow.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_live_book as lb  # noqa: E402
import scripts.run_master_book as mb  # noqa: E402
from src.config import LAB_DIR, VOL_SCALE_CAP, VOL_TARGET_ANNUAL  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

WINDOWS = [10, 20, 30, 40, 60, 90, 120, 250]
SHIPPED = 60
BURN = 250          # bars dropped from every leg in every arm, so no arm gets a longer flat warm-up
OOS = pd.Timestamp(str(mb.OOS_START)[:10])


def scale_at(net: pd.Series, window: int) -> pd.Series:
    """`_scale` with the lookback exposed. The 3x cap, the one-bar lag and the zero warm-up fill are held
    identical, so the sweep moves exactly one thing."""
    return vol_target_scale(net, VOL_TARGET_ANNUAL, mb.PPY, lookback=window, cap=VOL_SCALE_CAP)


def arm(labels: list[str], window: int, start: str, leverage: float):
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES if lab in labels}
    raw = {k: v for k, v in raw.items() if v is not None}
    df = mb.hold_started(pd.DataFrame({k: (v * scale_at(v, window)).iloc[BURN:]
                                       for k, v in raw.items()}).sort_index())
    df = df[df.index >= pd.Timestamp(start)].dropna(how="all")
    sc = pd.DataFrame({k: scale_at(v, window) for k, v in raw.items()}).reindex(df.index)
    b = (mb.book_stack(df) * leverage).dropna()
    turnover = float(sc.diff().abs().sum(axis=1).sum() / ((df.index[-1] - df.index[0]).days / 365.25))
    return b, df, sc, turnover


def stats(s: pd.Series) -> dict:
    """Un-reinvested, like the live page: P&L as a fraction of the fixed sizing capital."""
    s = s.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    pnl = s.cumsum()
    mo = s.resample("ME").sum()
    return {"ret": float(s.sum()) / yrs, "sharpe": mb.scorecard(s)["sharpe"],
            "max_dd": float((pnl - pnl.cummax()).min()), "worst_month": float(mo.min()),
            "worst_day": float(s.min()), "vol": float(s.std() * np.sqrt(mb.ppy_of(s))),
            "months_in_profit": float((mo > 0).mean())}


def blocks_of(s: pd.Series, n: int = 5) -> list:
    return [round(float(s.loc[ix].sum()) / ((ix[-1] - ix[0]).days / 365.25), 3)
            for ix in np.array_split(s.index, n)]


def targets_hit(s: pd.Series) -> int:
    sc, t = mb.scorecard(s), mb.TARGETS
    return sum([t["sharpe"][0] <= sc["sharpe"] <= t["sharpe"][1], sc["max_dd"] >= t["max_dd"],
                sc["months_in_profit"] >= t["months_in_profit"], sc["worst_month"] >= t["worst_month"],
                sc["longest_losing_streak_mo"] <= t["longest_losing_streak_mo"]])


def sweep(labels: list[str], start: str, leverage: float, tag: str, score: bool) -> dict:
    b0, _, _, t0 = arm(labels, SHIPPED, start, leverage)
    v0, s0, blk0 = float(b0.std() * np.sqrt(mb.ppy_of(b0))), stats(b0), blocks_of(b0)
    out = {}
    print(f"\n=== {tag} — every arm rescaled to the shipped window's {100 * v0:.1f}% vol ===")
    head = (f"{'window':>7s} {'own vol':>8s} {'return':>8s} {'Sharpe':>7s} {'max DD':>8s} "
            f"{'worst mo':>9s} {'months up':>10s} {'resize/yr':>10s} {'blocks':>7s} {'break-even':>11s}")
    print(head + ("  targets" if score else ""))
    for w in WINDOWS:
        b, df, sc, turn = arm(labels, w, start, leverage)
        own = float(b.std() * np.sqrt(mb.ppy_of(b)))
        bm = b * (v0 / own)                       # ONE constant over the whole history, not a daily re-fit
        st, blk = stats(bm), blocks_of(bm)
        wins = sum(x > y for x, y in zip(blk, blk0))
        dt = turn - t0
        be = 1e4 * (st["ret"] - s0["ret"]) / dt if dt > 0 else None
        row = st | {"own_vol": own, "resize_turnover_yr": turn, "blocks": blk,
                    "blocks_beating_shipped": wins, "break_even_bps": be,
                    "pct_days_at_cap": float((sc >= VOL_SCALE_CAP - 1e-3).to_numpy().mean()),
                    }
        if score:
            row["targets_full"], row["targets_oos"] = targets_hit(b), targets_hit(b[b.index >= OOS])
        out[str(w)] = row
        print(f"{w:>7d} {100 * own:7.1f}% {100 * st['ret']:7.1f}% {st['sharpe']:7.2f} "
              f"{100 * st['max_dd']:7.1f}% {100 * st['worst_month']:8.1f}% "
              f"{100 * st['months_in_profit']:9.1f}% {turn:9.1f}x {wins:5d}/5 "
              f"{(f'<{be:5.1f}bps' if be else '   cheaper'):>11s}"
              + (f"  {row['targets_full']}/5 {row['targets_oos']}/5" if score else "")
              + ("   <- shipped" if w == SHIPPED else ""))
    return out


def main() -> None:
    live = sweep(lb.LEGS, lb.START, lb.DEFAULT_LEVERAGE, "LIVE BOOK (4 legs, 2x)", score=False)
    master = sweep([lab for lab, _, _ in mb.FAMILIES], "2011-01-03", mb.BOOK_LEVERAGE,
                   "MASTER BOOK (1.15x) — the same _scale serves both", score=True)

    # The day the whole question came from. Read on the UNTRIMMED leg, and as the multiplier rather than
    # the return: with a 250-bar burn-in the day is not in the sample at all, and a long window that has
    # simply not filled yet reports a flat position that looks like risk control and is not.
    bab = mb.load("bab", *[(f, c) for lab, f, c in mb.FAMILIES if lab == "bab"][0])
    day, first = pd.Timestamp("2020-04-18"), bab.dropna().index.min()
    print(f"\nBAB on {day.date()} — the day the cap bound, and what each window did with it")
    print(f"(the leg's own history starts {first.date()}, so a window is only meaningful once it has filled):")
    diag = {}
    for w in WINDOWS:
        sc = scale_at(bab, w).get(day, float("nan"))
        filled = (day - first).days >= w
        diag[str(w)] = {"multiplier": round(float(sc), 3), "window_filled": bool(filled),
                        "leg_return": round(float(bab.get(day, float("nan")) * sc), 4)}
        print(f"  window {w:>3d}: multiplier {sc:4.2f}x -> leg takes {100 * bab.get(day, 0) * sc:6.1f}%"
              + ("" if filled else "   (window NOT filled — the leg is flat because it has no history)"))

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    (LAB_DIR / "volwindow.json").write_text(json.dumps(
        {"windows": WINDOWS, "shipped": SHIPPED, "burn_in_bars": BURN,
         "note": "every arm rescaled by one constant to the shipped window's vol before reading",
         "live": live, "master": master, "bab_2020_04_18": diag}, indent=2))
    print(f"\nwrote {LAB_DIR / 'volwindow.json'}")


if __name__ == "__main__":
    main()
