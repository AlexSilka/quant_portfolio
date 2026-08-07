"""Cross-asset carry in one picture: crypto vs FX vs equity. Left — equity curves (only crypto
compounds). Right — the carry-accrual vs price-leg decomposition per asset that explains why: the
carry premium is real everywhere, but only in crypto does the price leg help instead of offsetting it.

    python scripts/carry/make_carry_xasset_fig.py
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import CARRY_DIR, FIGURES_DIR  # noqa: E402
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

# per asset: (carry_accrual %/yr, price-leg %/yr, price-leg Sharpe, net Sharpe, placebo pct).
# %/yr magnitudes scale with each asset's own vol (crypto alts ~70% vs FX ~10%), so only the SIGN
# compares across assets; the price-leg Sharpe and net Sharpe are the vol-normalised comparable metrics.
DECOMP = {
    "Crypto\n(funding)": (28, 40, 0.97, 1.21, 99),
    "FX\n(rates)": (5, -1, -0.20, 0.39, 17),
    "Equity\n(dividends)": (5, -21, -0.95, -0.69, 20),
}


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # (1) equity curves
    a = ax[0]
    series = [("Crypto carry (Sh +1.21)", CARRY_DIR / "carry_headline.parquet", "carry", "#1f77b4"),
              ("FX carry (Sh +0.39)", CARRY_DIR / "carry_fx_headline.parquet", "ret", "#ff7f0e"),
              ("Equity div carry (Sh −0.69)", CARRY_DIR / "carry_equity_headline.parquet", "ret", "#2ca02c")]
    for label, path, col, c in series:
        try:
            s = pd.read_parquet(path)[col].dropna()
            (1 + s).cumprod().plot(ax=a, label=label, color=c, lw=1.8)
        except Exception:
            pass
    a.axhline(1.0, color="k", lw=0.5)
    a.set_title("Cross-asset carry — only crypto compounds"); a.legend(); a.set_yscale("log"); a.set_ylabel("equity (log, vol-targeted 15%)")

    # (2) accrual vs price-leg decomposition (up-bar = helps, regardless of which leg)
    a = ax[1]
    labels = list(DECOMP); x = np.arange(len(labels)); w = 0.38
    carry = [DECOMP[k][0] for k in labels]; price = [DECOMP[k][1] for k in labels]
    a.bar(x - w / 2, carry, w, label="carry accrual (%/yr)", color="#2ca02c")
    a.bar(x + w / 2, price, w, label="price / spot leg (%/yr)", color="#4c78c8")
    a.axhline(0, color="k", lw=0.6)
    a.set_ylim(-30, 52)
    for i, k in enumerate(labels):
        a.annotate(f"price-leg Sh {DECOMP[k][2]:+.2f}\nnet Sh {DECOMP[k][3]:+.2f} ({DECOMP[k][4]}th pct)",
                   (i, 44), ha="center", fontsize=8, fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel("annualised % (magnitude scales with asset vol)")
    a.set_title("Why: carry accrual is real everywhere;\nonly in crypto does the price leg help (up), not offset (down)", pad=12)
    a.legend(loc="lower left")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "carry_xasset.png", bbox_inches="tight")
    print("wrote reports/figures/carry_xasset.png")


if __name__ == "__main__":
    main()
