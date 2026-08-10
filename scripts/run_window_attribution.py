"""Decompose any window of the book into per-family contribution — the tool for "the book stalled,
which leg stopped paying?"

The book is an equal-weight risk-parity stack, so a family's *standalone* return is not what it did to
the book: a leg running at half the risk of its neighbours moves the book half as much. This reports the
weighted contribution (leg return x its risk-parity weight, summed over the window), which adds up to the
book's own return and is therefore the only decomposition that answers "where did the year go".

Alongside it, two things that separate a leg failing from a leg being *hedged into a loss on purpose*:

  * realised beta of each family to the crypto market, per window. A market-neutral or net-SHORT leg
    losing money in a crypto bear is an ALPHA failure, not a directional one — and the prescription for
    the two is opposite ("add a short" is wrong when beta is already negative).
  * the alpha left after that beta is removed, with its t-stat, so a small sample is visible as one.

    python scripts/run_window_attribution.py                 # every calendar year
    python scripts/run_window_attribution.py 2026            # one year
    python scripts/run_window_attribution.py 2026-01 2026-08 # an explicit window
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, LAB_DIR, REPORTS_DIR  # noqa: E402

LEGS = REPORTS_DIR / "master_book_legs.parquet"
WEIGHTS = REPORTS_DIR / "master_book_weights.parquet"
BOOK = REPORTS_DIR / "master_book.parquet"
CRYPTO_PANEL = CACHE_DIR / "xs" / "crypto_spotwide_1d_close.parquet"


def _crypto_market() -> pd.Series:
    """Equal-weight return of the survivorship-free crypto panel — the beta benchmark. EW, not BTC:
    three of the six families trade the cross-section, so cap-weighting would hide their exposure."""
    px = pd.read_parquet(CRYPTO_PANEL)
    idx = pd.DatetimeIndex(px.index)
    px.index = (idx.tz_localize(None) if idx.tz is None else idx.tz_convert("UTC").tz_localize(None)).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px[px.index >= "2017-01-01"]                          # panel starts sparse; pre-2017 is a handful of names
    return px.pct_change(fill_method=None).mean(axis=1).rename("mkt")


def _ols(y: pd.Series, x: pd.Series) -> tuple[float, float, float, float]:
    """(beta, t_beta, alpha_bps_per_day, t_alpha) — plain OLS, no HAC: the point here is to size the
    sample, and a t of 0.2 is a t of 0.2 under any covariance estimator."""
    d = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(d) < 30:
        return (np.nan,) * 4
    X = np.column_stack([np.ones(len(d)), d["mkt"].to_numpy()])
    yv = d["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    dof = len(d) - 2
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * (resid @ resid) / dof))
    return float(coef[1]), float(coef[1] / se[1]), float(coef[0] * 1e4), float(coef[0] / se[0])


def windows(argv: list[str], index: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if not argv:
        return [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))
                for y in sorted(index.year.unique())]
    if len(argv) == 1:
        y = argv[0]
        return [(y, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))]
    a, b = argv[0], argv[1]
    return [(f"{a}..{b}", pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0))]


def main() -> None:
    legs = pd.read_parquet(LEGS)
    w = pd.read_parquet(WEIGHTS)
    book = pd.read_parquet(BOOK)["ret"]
    mkt = _crypto_market()
    cols = [c for c in legs.columns if c in w.columns]
    contrib = (legs[cols] * w[cols])                            # adds up to the equal-weight book return

    rows = []
    for label, lo, hi in windows(sys.argv[1:], legs.index):
        m = (legs.index >= lo) & (legs.index <= hi)
        if not m.any():
            continue
        c, b = contrib[m], book[(book.index >= lo) & (book.index <= hi)]
        print(f"\n=== {label} — book {float((1 + b).prod() - 1) * 100:+.2f}%  ({int(m.sum())} obs) ===")
        print(f"  {'family':13s} {'contrib pp':>11s} {'standalone %':>13s} {'beta_crypto':>12s} "
              f"{'(t)':>7s} {'alpha bps/d':>12s} {'(t)':>7s}")
        for col in cols:
            leg = legs[col][m].dropna()
            if leg.empty:
                continue
            pp = float(c[col].sum()) * 100
            solo = float((1 + leg).prod() - 1) * 100
            beta, tb, alpha, ta = _ols(leg, mkt)
            print(f"  {col:13s} {pp:+11.2f} {solo:+13.1f} {beta:+12.3f} {tb:+7.2f} {alpha:+12.1f} {ta:+7.2f}")
            rows.append(dict(window=label, family=col, contrib_pp=round(pp, 3), standalone_pct=round(solo, 2),
                             beta_crypto=round(beta, 4), t_beta=round(tb, 2),
                             alpha_bps_day=round(alpha, 2), t_alpha=round(ta, 2)))
        print(f"  {'TOTAL':13s} {float(c.sum().sum()) * 100:+11.2f}")

    if rows:
        LAB_DIR.mkdir(parents=True, exist_ok=True)
        p = LAB_DIR / "window_attribution.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"\nwrote {p}")
        print("read it as: a NEGATIVE beta with a negative alpha is a leg whose direction paid and whose "
              "signal did not — adding more short exposure there makes it worse, not better.")


if __name__ == "__main__":
    main()
