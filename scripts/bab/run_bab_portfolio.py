"""Does BAB earn a seat in the master book? — add-as-6th vs swap-for-xsect, measured.

The master book is five risk-parity families (trend, carry, volprem, xs-momentum, breakout). BAB
clears the robust bar and is ~decorrelated, so the portfolio question is concrete: (A) baseline five,
(B) add BAB as a sixth family, (C) swap the weak xs-momentum leg for BAB. All on the common window
where BAB exists, same risk-parity recipe as run_master_book (rescale each leg to 15% trailing-lagged
vol, equal-weight). Reported with and without the tail-heavy volprem leg (whose +4.6 Sharpe dominates
the headline and understates BAB's marginal value on the rest of the book).

    python scripts/bab/run_bab_portfolio.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import BAB_DIR, CACHE_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import top_n_liquid, vol_target  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

REP, CACHE = REPORTS_DIR, CACHE_DIR / "xs"
SEED, PPY = SEED, 365
COST_BPS = 6.0        # per-side crypto taker+spread on the BAB legs — the shipped charge


def _bab_net(top_n, cost_bps=COST_BPS):
    """`cost_bps` is a parameter only so the same book can be re-run costless — that pair is what §9's
    "cost as a share of gross P&L" is measured from; the shipped book always uses the default."""
    C = bab.winsorize_panel(pd.read_parquet(CACHE / "crypto_1d_close.parquet"), 1.0)
    A = pd.read_parquet(CACHE / "crypto_1d_adv.parquet").reindex_like(C)
    beta = top_n_liquid(bab.panel_beta(C, 90), A, top_n)
    w = bab.bab_weights(beta, top_frac=0.2, neutral="beta", rebal=21)
    net = vol_target(bab.bab_backtest(C, w, exec_lag=2, cost_bps=cost_bps, adv=A, impact_k=0.1)["net"], PPY, VOL_TARGET_ANNUAL)
    net.index = net.index.tz_localize(None)
    return net.dropna()


def _rp(df):
    """Risk-parity: each leg rescaled to 15% trailing-lagged vol, equal-weighted."""
    scaled = df.apply(lambda c: vol_target(c.dropna(), PPY, VOL_TARGET_ANNUAL).reindex(df.index))
    return scaled.mean(axis=1, skipna=True)


def _metrics(book, label):
    b = book.dropna()
    s = summarise(b, PPY)
    mc = bootstrap_sharpe(b, PPY, 1000, SEED)
    py = {int(y): round(np.sqrt(PPY) * g.mean() / g.std(ddof=1), 2)
          for y, g in b.groupby(b.index.year) if g.std(ddof=1) > 0}
    return {"config": label, "sharpe": round(s["sharpe_ann"], 3), "max_dd": round(s["max_dd"], 3),
            "months_in_profit": round(s["months_in_profit"], 3), "mc_p5": round(mc.get("sharpe_p5", float("nan")), 3),
            "mc_p50": round(mc.get("sharpe_p50", float("nan")), 3), "per_year": py, "n_obs": int(len(b))}


def main():
    L = pd.read_parquet(REP / "master_book_legs.parquet")
    if L.index.tz is not None:
        L.index = L.index.tz_localize(None)
    bab25, bab100 = _bab_net(25), _bab_net(100)

    # publish the canonical BAB family series for the master-book assembly. top-100 is the a-priori,
    # sibling-consistent (x-sect uses top-100 liquid) choice; the concentrated top-25 is kept alongside
    # as the studied stronger-but-thinner (5 names/leg) variant.
    bab100.rename("ret").to_frame().to_parquet(BAB_DIR / "bab_book.parquet")
    bab25.rename("ret").to_frame().to_parquet(BAB_DIR / "bab_book_c25.parquet")

    # common window: every non-BAB leg the book actually trades, plus BAB. This was a typed list naming
    # trend and carry, dropped from the book, so it asked master_book_legs.parquet for columns that are
    # not there and died on a KeyError — the same stale-list failure the assembler's consumers had.
    legs5 = [c for c in L.columns if c != "bab"]
    idx = L[legs5].dropna().index.intersection(bab25.index)
    L, b25, b100 = L.reindex(idx), bab25.reindex(idx), bab100.reindex(idx)
    print(f"common window: {idx.min().date()}..{idx.max().date()}  ({len(idx)} bars)")

    # BAB correlation to each existing leg (are they independent? esp. vs xs-momentum)
    corr = {c: round(float(b25.corr(L[c])), 3) for c in legs5}
    print(f"\ncorr(BAB top-25, leg): {corr}")

    # ── configurations, full 5-leg book (volprem included) and ex-volprem ───────────────────────
    def book(cols, extra=None):
        df = L[cols].copy()
        if extra is not None:
            df = pd.concat([df, extra.rename("bab")], axis=1)
        return _rp(df)

    configs = {
        "A. baseline 5": book(legs5),
        "B. +BAB top-25 (6 fam)": book(legs5, b25),
        "B2.+BAB top-100 (6 fam)": book(legs5, b100),
        "C. swap xs→BAB25 (5 fam)": book([c for c in legs5 if c != "xs_momentum"], b25),
    }
    exvp = [c for c in legs5 if c != "volprem"]
    configs_exvp = {
        "A. baseline 4 (ex-volprem)": book(exvp),
        "B. +BAB top-25 (5, ex-vp)": book(exvp, b25),
        "C. swap xs→BAB25 (4, ex-vp)": book([c for c in exvp if c != "xs_momentum"], b25),
    }

    out = {"window": [str(idx.min().date()), str(idx.max().date())], "n_bars": len(idx),
           "bab_corr_to_legs": corr, "with_volprem": [], "ex_volprem": []}
    for title, group, key in [("WITH volprem (headline; volprem +4.6 dominates)", configs, "with_volprem"),
                              ("EX-volprem (where BAB's marginal value shows)", configs_exvp, "ex_volprem")]:
        print(f"\n=== {title} ===")
        print(f"{'config':<28} {'Sharpe':>7} {'MC-P5':>7} {'maxDD':>7} {'months+':>8}")
        for label, bk in group.items():
            m = _metrics(bk, label)
            out[key].append(m)
            print(f"{label:<28} {m['sharpe']:>+7.2f} {m['mc_p5']:>+7.2f} {m['max_dd']:>+7.1%} {m['months_in_profit']:>7.0%}")

    (BAB_DIR / "bab_portfolio_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nRUN BAB PORTFOLIO OK  -> reports/bab_portfolio_summary.json")


if __name__ == "__main__":
    main()
