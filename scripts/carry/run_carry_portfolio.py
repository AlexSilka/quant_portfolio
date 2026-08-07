"""Carry's diversification value + the carry headline series (reports/carry_headline.parquet, consumed
by the vol-premium deep-dives). Diagnostics: correlation to the master book (with carry held out), a
risk-budget blend sweep (marginal contribution), cost sensitivity + break-even, per-year Sharpe and
crisis-window behaviour.

Cross-sectional carry is ~0.03-correlated to price momentum — a structurally distinct return source —
so it enters the master book as its own decorrelated family (see scripts/run_master_book.py).

    python scripts/carry/run_carry_portfolio.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.carry.run_carry import load_panel  # noqa: E402

PPY, TVOL, SEED, CB = 365, VOL_TARGET_ANNUAL, SEED, 6.0


def vt(net, target=TVOL):
    scale = (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def per_year(net):
    out = {}
    for y, g in net.groupby(net.index.year):
        g = g.dropna()
        out[int(y)] = round(float(np.sqrt(PPY) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    return out


def main():
    C, fd = load_panel()
    # headline carry book (level-7, top-20) with full cost/attribution components
    bk = carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, 7), direction=-1.0, top_frac=0.2, cost_bps=CB)
    carry = vt(bk["ret"]).rename("carry")
    s = summarise(carry, PPY)
    mc = bootstrap_sharpe(carry, PPY, 1000, SEED)
    print("=== HEADLINE carry (XScarry level-7 top-20, vol-targeted 15%) ===")
    print(f"  Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"MC[P5 {mc['sharpe_p5']:+.2f} P50 {mc['sharpe_p50']:+.2f}]")
    print(f"  per-year Sharpe: {per_year(carry)}")

    # ---- cost sensitivity + break-even (charge the cost multiple on the pre-vt book, then re-vt) ----
    print("\n=== cost sensitivity (carry) ===")
    base_cost = bk["cost"]
    def at_mult(m):
        return vt(bk["price"] + bk["funding"] - m * base_cost)
    for m, lab in [(1, "1x"), (2, "2x"), (3, "3x"), (5, "5x")]:
        sm = summarise(at_mult(m), PPY)
        print(f"  {lab:3s} base cost: Sharpe {sm['sharpe_ann']:+.2f}  maxDD {sm['max_dd']:+.1%}")
    be = next((m for m in np.linspace(1, 40, 391) if (1 + at_mult(m)).prod() - 1 <= 0), None)
    print(f"  break-even cost multiple: {be:.0f}x base" if be else "  break-even: >40x base")

    # ---- correlation to the master book (carry held out) + risk-budget blend sweep ----
    book = pd.read_parquet(REPORTS_DIR / "master_book_legs.parquet").drop(columns=["carry"]).mean(axis=1)
    if carry.index.tz is not None:
        book.index = book.index.tz_localize(carry.index.tz)
    idx = carry.index.intersection(book.index)
    cc, bb = carry.reindex(idx).fillna(0.0), vt(book.reindex(idx)).reindex(idx).fillna(0.0)
    corr = float(pd.concat([cc, bb], axis=1).corr().iloc[0, 1])
    print(f"\n=== portfolio integration (overlap {idx.min().date()}..{idx.max().date()}) ===")
    print(f"  corr(carry, book ex-carry) = {corr:+.2f}   (structurally decorrelated)")
    print("  risk-budget blend  a*carry + (1-a)*book  (both vol-matched 15%):")
    best = None
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        blend = (a * cc + (1 - a) * bb)
        sb = summarise(blend, PPY)
        p5 = bootstrap_sharpe(blend, PPY, 500, SEED).get("sharpe_p5", np.nan)
        tag = ""
        if best is None or sb["sharpe_ann"] > best[1]:
            best = (a, sb["sharpe_ann"]); tag = ""
        print(f"    a={a:.1f}  Sharpe {sb['sharpe_ann']:+.2f}  maxDD {sb['max_dd']:+.1%}  "
              f"months+ {sb['months_in_profit']:.0%}  MC-P5 {p5:+.2f}")
    # book-only reference on the same overlap
    sbb = summarise(bb, PPY)
    print(f"  book-only (this overlap): Sharpe {sbb['sharpe_ann']:+.2f}  maxDD {sbb['max_dd']:+.1%}  months+ {sbb['months_in_profit']:.0%}")
    print(f"  -> best blend a={best[0]:.1f} lifts book Sharpe {sbb['sharpe_ann']:+.2f} -> {best[1]:+.2f}")

    # ---- crisis / regime windows ----
    print("\n=== crisis windows (carry Sharpe) ===")
    wins = [("2020 COVID", "2020-01", "2020-06"), ("2021 bull", "2021-01", "2021-12"),
            ("2022 bear", "2022-01", "2022-12"), ("FTX Nov22", "2022-11", "2022-12"),
            ("2023-24 recov", "2023-01", "2024-12"), ("2025-26", "2025-01", "2026-07")]
    for name, a, b in wins:
        g = carry.loc[a:b].dropna()
        sh = float(np.sqrt(PPY) * g.mean() / g.std(ddof=1)) if len(g) > 20 and g.std(ddof=1) > 0 else float("nan")
        dd = summarise(g, PPY)["max_dd"] if len(g) > 20 else float("nan")
        print(f"  {name:14s} Sharpe {sh:+.2f}  maxDD {dd:+.0%}  (n={len(g)})")

    carry.to_frame().to_parquet(CARRY_DIR / "carry_headline.parquet")
    pd.DataFrame({"metric": ["sharpe", "max_dd", "months_in_profit", "mc_p5", "corr_to_book", "best_blend_a"],
                  "value": [s["sharpe_ann"], s["max_dd"], s["months_in_profit"], mc["sharpe_p5"], corr, best[0]]}
                 ).to_csv(CARRY_DIR / "carry_headline.csv", index=False)
    print("\nCARRY-PORTFOLIO OK")


if __name__ == "__main__":
    main()
