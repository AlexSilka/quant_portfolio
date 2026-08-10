"""Search for a NINTH family: a second long-gamma source, judged against the CURRENT book.

Why this exists. The book's binding limits are months-in-profit and the ≤2-month streak, and §5d proved
they cannot be bought by any tactic that *reduces* exposure — de-grossing turns a losing month into a
flat one, and a flat month is still not a profitable month. The only mechanism that converts a bad month
into a good one is owning something that PAYS while the short-gamma legs bleed. Two such legs are already
in (crisis-alpha, global-macro); both are trend, so both need a crash to last long enough to trend into.
This searches for a third that is convex in a different way.

Candidates, each either an existing lab sleeve re-judged or built here from data already in the repo:

  A convexity (existing)   — term-structure-timed long VIX exposure (lab/convexity_sleeve).
  B defensive (existing)   — vol-timed haven basket: gold, duration, JPY/CHF (lab/defensive_sleeve).
  C long crypto variance   — the MIRROR of a finding the volprem deep-dive already established: under the
                             honest OHLC realised leg crypto short-vol is NEGATIVE (BTC −0.41, ETH −0.86)
                             because the intraday path is unhedgeable for a short. If that is real, the
                             long side of the same swap is a paid long-gamma leg, not a bleed.
  D long correlation       — long index variance, short a single-name variance basket. Correlations spike
                             in a crash, so long-correlation is convex; it is also the one vol structure
                             the 18-leg short-vol book does NOT already contain.
  E curve-timed long vol   — long variance only while the VIX curve is inverted (the exact mirror of the
                             shipped gate, reusing a signal already validated on this book).

Both existing sleeves were last judged against a stale baseline (the 6-family book at Sharpe 3.05, months
73.6%, streak 3) — the book has since gained the two-segment gate, an 8th family and 1.15× sizing, so the
verdict is re-taken here, not inherited.

Each candidate is scored standalone AND as a 9th equal-risk family through the canonical assembler, on all
five §11 targets plus return, on the selection window (pre-OOS) with the frozen block as a read-out only.

    python scripts/run_longgamma_search.py  ->  reports/lab/longgamma_search.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
from src.config import LAB_DIR, OOS_START  # noqa: E402
from src.data.cboe import load_cboe_vol  # noqa: E402
from src.data.deribit import load_dvol  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402
from scripts.volprem.run_vol_premium_book import COST_BY_CLASS, naive_df, naive_dt, vt  # noqa: E402

PPY = 365
OOS = pd.Timestamp(OOS_START).tz_localize(None)
SELECT_END = pd.Timestamp("2024-06-30")     # §10: the final block is a read-out, never a selection input
SINGLES = [("VXAPL", "AAPL"), ("VXAZN", "AMZN"), ("VXGOG", "GOOGL"), ("VXGS", "GS"), ("VXIBM", "IBM")]


def _naive(s):
    ix = pd.DatetimeIndex(s.index)
    s = pd.Series(np.asarray(s), index=ix.tz_convert("UTC").tz_localize(None) if ix.tz else ix)
    return s.groupby(level=0).last().sort_index()


def long_variance(iv_sym: str, und: str, cls: str, source: str = "cboe", ppy: int = 252) -> pd.Series:
    """The LONG side of the same discrete variance swap the short-vol book sells: +1 side, so it pays
    when realised variance exceeds the strike. Same OHLC realised leg, same charged vega spread."""
    if source == "deribit":
        iv = naive_dt(load_dvol(iv_sym, "2021-01", "2026-08")["close"])
        bars = naive_df(load_klines(und, "1d", "2021-01", "2026-08", market="um")[["open", "high", "low", "close"]])
    else:
        iv = naive_dt(load_cboe_vol(iv_sym))
        bars = naive_df(load_equity_daily(und, start="2005-01-01")[["open", "high", "low", "close"]])
    f = vp.short_vol_book(bars["close"], iv, ppy=ppy, timed=False, var_cap=1e9, bars=bars,
                          vega_cost_volpts=COST_BY_CLASS.get(cls, 1.5))
    # short_vol_book returns the SHORT's P&L net of cost; the long is its mirror, and pays the same
    # spread rather than earning it — so flip the gross and charge the cost again, never credit it.
    return (-(f["gross"]) - f["cost"]).rename("ret")


def candidates() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for tag, f in (("A convexity (existing)", "convexity_sleeve"), ("B defensive (existing)", "defensive_sleeve")):
        p = LAB_DIR / f"{f}.parquet"
        if p.exists():
            out[tag] = _naive(pd.read_parquet(p).iloc[:, 0].dropna())

    # C — long crypto variance, the mirror of the volprem deep-dive's negative crypto short-vol
    legs = {}
    for sym in ("BTC", "ETH"):
        try:
            legs[sym] = vt(long_variance(sym, f"{sym}USDT", "crypto", source="deribit", ppy=PPY), PPY)
        except Exception as e:                                  # a missing cache must be visible
            print(f"  SKIP long-var {sym}: {str(e)[:70]}")
    if legs:
        out["C long crypto variance"] = _naive(pd.DataFrame(legs).mean(axis=1, skipna=True).dropna())

    # D — long correlation: long index variance, short the single-name variance basket
    try:
        idx = vt(long_variance("VIX", "SPY", "eq_index"), 252)
        sing = {s: vt(long_variance(s, u, "single"), 252) for s, u in SINGLES}
        basket = pd.DataFrame(sing).mean(axis=1, skipna=True)
        out["D long correlation"] = _naive(vt((idx - basket.reindex(idx.index)).dropna(), 252))
    except Exception as e:
        print(f"  SKIP long correlation: {str(e)[:70]}")

    # E — long variance only while the curve is inverted (the mirror of the shipped short-vol gate)
    if "D long correlation" in out or True:
        try:
            lv = long_variance("VIX", "SPY", "eq_index")
            vix, v3 = naive_dt(load_cboe_vol("VIX")), naive_dt(load_cboe_vol("VIX3M"))
            inv = ((v3.reindex(lv.index).ffill() / vix.reindex(lv.index).ffill()) < 1.0).astype(float).shift(1).fillna(0.0)
            out["E curve-timed long vol"] = _naive(vt((lv * inv).dropna(), 252))
        except Exception as e:
            print(f"  SKIP curve-timed long vol: {str(e)[:70]}")
    return out


def card(s: pd.Series, ppy: float | None = None) -> dict:
    s = s.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ppy = len(s) / yrs if ppy is None else ppy
    sc = summarise(s, ppy)
    m = (1 + s).resample("ME").prod() - 1
    neg, streak, mx = (m <= 0).astype(int).to_numpy(), 0, 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"sharpe": round(sc["sharpe_ann"], 2),
            "cagr": round(float((1 + s).prod() ** (1 / yrs) - 1), 3) if yrs > 0 else 0.0,
            "max_dd": round(sc["max_dd"], 3), "worst_month": round(float(m.min()), 3),
            "months_in_profit": round(float((m > 0).mean()), 3), "streak": int(mx),
            "skew": round(float(s.skew()), 1)}


def n_targets(c: dict) -> int:
    return sum([2.5 <= c["sharpe"] <= 4.0, c["months_in_profit"] >= 0.80, c["max_dd"] >= -0.15,
                c["worst_month"] >= -0.06, c["streak"] <= 2])


def book_with(extra: pd.Series | None, end: pd.Timestamp | None, w: float = 1.0) -> pd.Series:
    """The canonical assembly with `extra` added as one more equal-risk family."""
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
    if end is not None:
        df = df[df.index <= end]
    df = df[df.notna().sum(axis=1) >= 2]
    stack = df.mean(axis=1, skipna=True).dropna()
    if extra is not None:
        # `w` is the leg's share of ONE equal-risk slot: 1.0 is a full ninth family, 0.25 a quarter of
        # one. A tail hedge is not a return source, so parity is the wrong default size for it — parity
        # hands it the same risk budget as an earner and makes it pay for that budget every calm month.
        e = mb.rescale(extra.rename("longgamma")).reindex(stack.index).fillna(0.0)
        stack = ((stack * len(raw) + e * w) / (len(raw) + w)).dropna()
    return mb.risk_overlay(stack, leverage=mb.BOOK_LEVERAGE)[0]


def crisis_table(s: pd.Series) -> dict:
    """What a long-gamma leg is actually bought for: its return through the book's worst windows."""
    wins = {"2011 Aug selloff": ("2011-08-01", "2011-10-03"), "2015 China": ("2015-08-17", "2015-09-30"),
            "2018 volmageddon": ("2018-02-01", "2018-02-28"), "2018 Q4": ("2018-10-01", "2018-12-31"),
            "COVID crash": ("2020-02-19", "2020-03-23"), "2022 bear": ("2022-01-01", "2022-12-31"),
            "yen unwind 2024": ("2024-07-25", "2024-08-09")}
    return {k: round(float((1 + s.loc[a:b]).prod() - 1), 3) for k, (a, b) in wins.items() if len(s.loc[a:b])}


def main():
    print("=== searching for a 9th family: a second long-gamma source ===\n")
    cands = candidates()
    base_sel, base_full = book_with(None, SELECT_END), book_with(None, None)
    bs, bf = card(base_sel), card(base_full)
    print(f"BASELINE book (8 families, 1.15x)  selection window: Sh {bs['sharpe']:+.2f} CAGR {bs['cagr']:+.0%} "
          f"DD {bs['max_dd']:+.1%} worst {bs['worst_month']:+.1%} mo {bs['months_in_profit']:.0%} "
          f"strk {bs['streak']} [{n_targets(bs)}/5]\n")

    out = {"baseline": {"selection": bs, "full": bf, "oos": card(base_full[base_full.index >= OOS])}}
    print("standalone (own span, own risk):")
    for tag, s in cands.items():
        c = card(s)
        out[tag] = {"standalone": c, "crisis": crisis_table(s),
                    "span": [str(s.index[0].date()), str(s.index[-1].date())]}
        print(f"  {tag:26s} {s.index[0].date()}..{s.index[-1].date()}  Sh {c['sharpe']:+.2f} "
              f"CAGR {c['cagr']:+.1%} DD {c['max_dd']:+.1%} skew {c['skew']:+.1f} mo+ {c['months_in_profit']:.0%}")
        print(f"  {'':26s} crisis: " + "  ".join(f"{k} {v:+.1%}" for k, v in out[tag]["crisis"].items()))

    print("\nas a 9th equal-risk family, canonical assembly at 1.15x — SELECTION WINDOW (pre-OOS):")
    for tag, s in cands.items():
        sel = card(book_with(s, SELECT_END))
        full_s = book_with(s, None)
        full, oos = card(full_s), card(full_s[full_s.index >= OOS])
        out[tag] |= {"book_selection": sel, "book_full": full, "book_oos": oos}
        d = lambda a, b, k: a[k] - b[k]  # noqa: E731
        print(f"  {tag:26s} Sh {sel['sharpe']:+.2f} ({d(sel, bs, 'sharpe'):+.2f})  CAGR {sel['cagr']:+.1%} "
              f"({d(sel, bs, 'cagr'):+.1%})  DD {sel['max_dd']:+.1%}  worst {sel['worst_month']:+.1%} "
              f"({d(sel, bs, 'worst_month'):+.1%})  mo {sel['months_in_profit']:.0%} "
              f"({d(sel, bs, 'months_in_profit'):+.1%})  strk {sel['streak']} [{n_targets(sel)}/5]")
        print(f"  {'':26s} read-out OOS: Sh {oos['sharpe']:+.2f}  CAGR {oos['cagr']:+.0%}  "
              f"mo {oos['months_in_profit']:.0%}  strk {oos['streak']}")

    # --- the size sweep: parity is the wrong default for a hedge, so ask what fraction of a slot works
    print("\nfractional sizing — share of ONE equal-risk slot (selection window):")
    # Both windows per rung, because the report argues the sizing on the selection window and reads the
    # frozen block beside it — quoting one from the artifact and typing the other by hand is how that
    # table went stale. 0.15/0.40 are in the grid because they are the rungs the argument turns on.
    sweep = {}
    for tag in [t for t in cands if not t.startswith("D")]:
        row = {}
        for w in (0.10, 0.15, 0.25, 0.40, 0.50, 1.00):
            c = card(book_with(cands[tag], SELECT_END, w=w))
            c["oos"] = card(book_with(cands[tag], None, w=w).loc[OOS:])
            row[f"{w:.2f}"] = c
            flag = "  <-- 5/5" if n_targets(c) == 5 else ""
            print(f"  {tag:26s} w={w:.2f}  Sh {c['sharpe']:+.2f}  CAGR {c['cagr']:+.1%}  DD {c['max_dd']:+.1%}  "
                  f"worst {c['worst_month']:+.1%}  mo {c['months_in_profit']:.0%}  strk {c['streak']} "
                  f"[{n_targets(c)}/5]{flag}")
        sweep[tag] = row
    base = card(book_with(None, SELECT_END))
    base["oos"] = card(book_with(None, None).loc[OOS:])
    sweep["baseline (no extra leg)"] = {"0.00": base}
    out["size_sweep"] = sweep

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    (LAB_DIR / "longgamma_search.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {LAB_DIR / 'longgamma_search.json'}")


if __name__ == "__main__":
    main()
