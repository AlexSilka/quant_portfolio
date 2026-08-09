"""Does a SHORT trend leg buy back the months the book loses? — the one candidate with full history.

The book's two binding misses (months-in-profit on both windows, and the full window's 3-month streak)
come from three short-gamma legs falling together through the Dec-2021→Feb-2022 crypto unwind. §6c/§6d-bis
established what a fix has to look like — a source that *earns* in those months — and closed every
candidate: long-gamma sleeves at parity break other targets, and the one that works (long crypto variance)
only lists from 2021-03, so its whole effect on a fifteen-year scorecard sits inside the window containing
the miss.

A short trend leg does not have that defect. It is the same machinery already validated in TREND.md with
the sign flipped, it runs on the same point-in-time universe, and it exists over the **whole** window —
so it can be judged on fifteen years rather than on the five that contain the problem.

TREND.md finding 1 is not a contradiction of this. It says long-only beats long-short *as the trend leg's
own construction*, because the short leg drags a structurally-upward book. That is a statement about
replacing the leg. This asks a different question: whether the short leg, held **separately** at its own
risk parity, pays for that drag with what it does in the months the rest of the book cannot cover.

Three variants, each on the shipped PIT universe and otherwise the a-priori config:
  short_only   the pure bear leg — replaces the long-only trend family entirely
  asym 70/30   full long + 30% short, the deep-dive's own compromise
  ls           symmetric long-short
  long+short   the family split into two equal-risk sleeves inside its own 1/8 slot

Scored on the selection window with the frozen block as a read-out (§10), on all five targets.

    python scripts/trend/run_trend_short_leg.py  ->  reports/trend/trend_short_leg.json
"""
from __future__ import annotations

import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
import scripts.trend.run_trend_pit_universe as P  # noqa: E402
from src.config import OOS_START, TREND_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402

OOS = pd.Timestamp(OOS_START).tz_localize(None)
SELECT_END = pd.Timestamp("2024-06-30")
STREAK = ("2021-12-01", "2022-02-28")
MODES = {"long_only": "SHIPPED — long only", "short_only": "REPLACED — short only",
         "ls": "REPLACED — long and short"}
TARGETS = {"sharpe": (2.5, 4.0), "months_in_profit": (0.80, None), "max_dd": (-0.15, None),
           "longest_losing_streak_mo": (None, 2), "worst_month": (-0.06, None)}


def n_targets(c: dict) -> int:
    return sum((lo is None or c[k] >= lo) and (hi is None or c[k] <= hi)
               for k, (lo, hi) in TARGETS.items())


def leg(mode: str) -> pd.Series:
    """The trend block rebuilt with a different direction mode, same PIT universe and config.

    Cached to parquet per mode. Rebuilding a direction variant walks ~1,640 equities and ~330 perps and
    costs minutes; re-scoring one against a different book assembly costs milliseconds. The first run of
    this script threw the series away and kept only the summary numbers, which made a change of question
    ("add a ninth leg" -> "replace the family") a full recompute for no reason."""
    cache = TREND_DIR / f"trend_dir_{mode}.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)["ret"]
        s.index = pd.DatetimeIndex(s.index)
        return s.dropna()
    spec = {"entry": "ema", "direction": mode, "exit": "reversal"}
    cols = {}
    for tf in P.BLOCK_TFS:
        rets, vol = P.pool(tf, spec=spec)
        mem = P.pit_members(vol, P.TOP_N, P.LOOKBACK_D)
        for sym in rets.columns:
            r = rets[sym].where(mem[sym].reindex(rets.index).fillna(False))
            if r.notna().sum() > 60 and r.std(ddof=1) > 0:
                cols[f"{sym}_{tf}"] = r
    for sym, r in P.equity_legs(pit=True, spec=spec).items():
        cols[sym] = r
    out = pd.DataFrame(cols).mean(axis=1).dropna().rename("ret")
    out.to_frame().to_parquet(cache)
    return out


def book_with(trend_leg: pd.Series | None, end=None) -> pd.Series:
    """The canonical book with the TREND FAMILY rebuilt — eight families throughout.

    A short trend sleeve is not a ninth family: trend is one of the eight, and its direction mode is a
    property of that family's construction. Adding a short leg alongside the shipped long-only one would
    dilute every family to 1/9 *and* double-count the long side, which measures nothing."""
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    if trend_leg is not None:
        raw["trend_momentum"] = trend_leg.rename("trend_momentum")
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
    if end is not None:
        df = df[df.index <= end]
    df = df[df.notna().sum(axis=1) >= 2]
    return mb.risk_overlay(df.mean(axis=1, skipna=True).dropna(), leverage=mb.BOOK_LEVERAGE)[0]


def main():
    print("=== the trend family REBUILT with a short side — eight families throughout ===")
    print("(selection window 2011-01..2024-06; the frozen block is a read-out only)\n")
    base_sel, base_full = book_with(None, SELECT_END), book_with(None)
    bs, bo_ = mb.scorecard(base_sel), mb.scorecard(base_full.loc[OOS:])
    print(f"  {'BASELINE (8 families)':24s} sel {n_targets(bs)}/5  Sh {bs['sharpe']:.2f} "
          f"months {100 * bs['months_in_profit']:.1f}% streak {bs['longest_losing_streak_mo']}   "
          f"OOS {n_targets(bo_)}/5  months {100 * bo_['months_in_profit']:.1f}%\n")

    out = {"baseline": {"selection": bs, "oos": bo_}}
    for mode, what in MODES.items():
        s = leg(mode)
        seg = s.loc[STREAK[0]:STREAK[1]]
        m3 = (1 + seg).resample("ME").prod() - 1
        st = summarise(s.dropna(), 365)
        sel = mb.scorecard(book_with(s, SELECT_END))
        full = book_with(s)
        oos = mb.scorecard(full.loc[OOS:])
        out[mode] = {"standalone_sharpe": round(st["sharpe_ann"], 2), "streak_window": round(float((1 + seg).prod() - 1), 4),
                     "selection": sel, "oos": oos}
        print(f"  {mode + ' (' + what + ')':44s}")
        print(f"      leg standalone Sharpe {st['sharpe_ann']:+.2f}  |  over the 3 months "
              f"{100 * ((1 + seg).prod() - 1):+.1f}%  (" + " ".join(f"{d.strftime('%b')} {100 * v:+.1f}%" for d, v in m3.items()) + ")")
        print(f"      book sel {n_targets(sel)}/5  Sh {sel['sharpe']:.2f} months "
              f"{100 * sel['months_in_profit']:.1f}% streak {sel['longest_losing_streak_mo']} "
              f"worst {100 * sel['worst_month']:.2f}%   |   OOS {n_targets(oos)}/5 Sh {oos['sharpe']:.2f} "
              f"months {100 * oos['months_in_profit']:.1f}%")

    (TREND_DIR / "trend_short_leg.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {TREND_DIR / 'trend_short_leg.json'}")


if __name__ == "__main__":
    main()
