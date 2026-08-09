"""Would a narrower crypto universe remove the full window's 3-month losing streak? (No — measured.)

The one target the book misses is the full-window losing streak: Dec-2021 −2.5%, Jan-2022 −0.8%,
Feb-2022 −1.7%. Decomposed by family, that stretch is carry (−16.4%), x-sect (−13.0%) and trend
(−11.2%); BAB (top-25 crypto) and breakout (top-30) were *positive* through it. So the obvious question
is whether the losing legs are losing because their crypto universe is too broad — BAB and breakout run
narrow universes and survived, x-sect runs top-100 and did not.

**This test is deliberately built to be able to say no.** Choosing a universe size because it removes a
missed target on a known window is target-fitting, and it is the exact failure this project spends its
validation budget on — the x-sect deep-dive's own headline is that narrower/curated universes flatter
results (curated-50 +1.06 vs honest PIT +0.70). So the sweep is scored the way a suspicious result has
to be:

  * the whole surface is reported, not the best cell — a lone N that fixes the streak while its
    neighbours do not is noise, not a finding;
  * scoring stops at SELECT_END (the frozen OOS block is a read-out only, §10);
  * every target is scored, not just the streak — a narrower universe that fixes the streak by breaking
    months-in-profit has fixed nothing;
  * the a-priori top-100 stays the shipped choice unless the surface says something structural.

    python scripts/xs/run_xs_universe_sweep.py  ->  reports/xs/xs_universe_sweep.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
from scripts.residmom.run_residmom import ASSETS as RM_ASSETS, _book as rm_book, _load as rm_load  # noqa: E402
from scripts.xs.build_xs_book import equity_xsect  # noqa: E402
from src import config as cfg  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import idio_mom  # noqa: E402

SELECT_END = pd.Timestamp("2024-06-30")
OOS = pd.Timestamp(cfg.OOS_START).tz_localize(None)
GRID = [10, 25, 50, 75, 100, 150, 200]        # shipped is 100
STREAK_WIN = ("2021-12-01", "2022-02-28")


def _naive(s):
    ix = pd.DatetimeIndex(s.index)
    return pd.Series(np.asarray(s, dtype="float64"),
                     index=ix.tz_convert("UTC").tz_localize(None) if ix.tz else ix).groupby(level=0).last()


def crypto_leg(topn: int) -> pd.Series:
    """The crypto x-sect leg at a given point-in-time universe size — everything else a-priori."""
    c = dict(RM_ASSETS["crypto"], topn=topn)
    Craw, adv = rm_load(c["tag"])
    C = bab.winsorize_panel(Craw, c["winsor"])
    sig = idio_mom(C, c["base"]["lb"], c["beta_lb"], c["base"]["sk"], market=None)
    net, _ = rm_book(C, sig, adv, c)
    return _naive(net.dropna())


def xs_block(topn: int, eq: pd.Series) -> pd.Series:
    """The published x-sect block: crypto + equity legs combined at inverse-vol risk parity."""
    R = pd.concat([crypto_leg(topn).rename("crypto"), eq.rename("equity")], axis=1)
    w = (1.0 / R.std()).replace([np.inf, -np.inf], np.nan)
    return (R * w).sum(axis=1, min_count=1).div(w.sum()).where(R.notna().any(axis=1)).dropna()


def card(s: pd.Series) -> dict:
    s = s.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    sc = summarise(s, len(s) / yrs)
    m = (1 + s).resample("ME").prod() - 1
    neg, run, mx = (m <= 0).astype(int).to_numpy(), 0, 0
    for v in neg:
        run = run + 1 if v else 0
        mx = max(mx, run)
    return {"sharpe": round(sc["sharpe_ann"], 2), "max_dd": round(sc["max_dd"], 4),
            "worst_month": round(float(m.min()), 4),
            "months": round(float((m > 0).mean()), 4), "streak": int(mx),
            "cagr": round(float((1 + s).prod() ** (1 / yrs) - 1), 3)}


def n_targets(c: dict) -> int:
    return sum([2.5 <= c["sharpe"] <= 4.0, c["months"] >= 0.80, c["max_dd"] >= -0.15,
                c["worst_month"] >= -0.06, c["streak"] <= 2])


def book_with(leg: pd.Series, end=None) -> pd.Series:
    raw = {lab: mb.load(lab, f, col) for lab, f, col in mb.FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    raw["xs_momentum"] = leg.rename("xs_momentum")
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
    if end is not None:
        df = df[df.index <= end]
    df = df[df.notna().sum(axis=1) >= 2]
    return mb.risk_overlay(df.mean(axis=1, skipna=True).dropna(), leverage=mb.BOOK_LEVERAGE)[0]


def main():
    print("=== does a narrower crypto x-sect universe remove the 3-month streak? ===")
    print("(scored to 2024-06-30; the frozen block is a read-out. Shipped universe is top-100.)\n")
    eq = equity_xsect()
    rows = []
    for n in GRID:
        leg = xs_block(n, eq)
        sel = card(book_with(leg, SELECT_END))
        full = book_with(leg, None)
        oos = card(full[full.index >= OOS])
        seg = leg.loc[STREAK_WIN[0]:STREAK_WIN[1]]
        rows.append({"topn": n, **{f"leg_{k}": v for k, v in card(leg).items()},
                     "leg_streak_window": round(float((1 + seg).prod() - 1), 4),
                     **{f"book_{k}": v for k, v in sel.items()}, "book_targets": n_targets(sel),
                     "oos_sharpe": oos["sharpe"], "oos_targets": n_targets(oos)})
        r = rows[-1]
        star = "  <-- streak fixed" if sel["streak"] <= 2 else ""
        print(f"  top-{n:<4d} leg Sharpe {r['leg_sharpe']:+.2f}  leg over the 3 months "
              f"{100 * r['leg_streak_window']:+6.1f}%   book Sh {sel['sharpe']:+.2f} "
              f"months {sel['months']:.1%} worst {sel['worst_month']:+.2%} "
              f"streak {sel['streak']} [{n_targets(sel)}/5]{star}")

    d = pd.DataFrame(rows)
    d.to_csv(cfg.XS_DIR / "xs_universe_sweep.csv", index=False)
    fixed = d[d.book_streak <= 2]
    print(f"\n  universes that reach streak <= 2: {len(fixed)} of {len(d)}"
          + (f" ({', '.join('top-' + str(int(x)) for x in fixed.topn)})" if len(fixed) else ""))
    print(f"  wrote {cfg.XS_DIR / 'xs_universe_sweep.csv'}")


if __name__ == "__main__":
    main()
