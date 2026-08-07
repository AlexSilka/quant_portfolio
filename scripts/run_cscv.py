"""Probability of backtest overfitting (CSCV) on the full 1,279-sleeve trial set (§6, mandatory).

Reads the discovery zoo's per-sleeve return matrix (reports/book/all_returns.parquet, T×1279) and runs
CSCV (src/validation/cscv.py) to estimate PBO — how often the in-sample-best sleeve falls below the OOS
median. This is the multiple-testing overfit probability the brief names alongside the deflated Sharpe
(already in src/metrics.py) and the placebo FDR (run_book.py). No external dependency.

    python scripts/run_cscv.py
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.validation.cscv import pbo_cscv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main():
    mat = pd.read_parquet(R / "book" / "all_returns.parquet")
    # dense common window: perps + equities all live from 2021, so coverage is high and the trial set
    # is not dominated by one asset class's history.
    res = pbo_cscv(mat, n_blocks=16, min_coverage=0.95, window=("2021-01-01", "2026-12-31"))
    lam = res.pop("_lambdas")
    rel = res.pop("_oos_rel_rank")
    print(f"CSCV PBO on {res['n_strategies']} sleeves ({res['window'][0]}..{res['window'][1]}, "
          f"{res['n_splits']} splits):")
    print(f"  PBO (P[IS-best below OOS median]) = {res['pbo']:.3f}")
    print(f"  P(selected loses OOS)             = {res['prob_oos_loss']:.3f}")
    print(f"  IS Sharpe(sel) {res['is_sharpe_mean']:+.3f}/bar -> OOS {res['oos_sharpe_mean']:+.3f}/bar "
          f"(degradation)")
    (R / "book" / "cscv_pbo.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(lam, bins=40, color="#4C78A8", edgecolor="white")
    ax[0].axvline(0, color="#d62728", lw=1.5, label=f"PBO={res['pbo']:.2f}")
    ax[0].set_title("CSCV logits λ (overfit if λ<0)"); ax[0].set_xlabel("λ = logit(OOS rel-rank)")
    ax[0].legend()
    ax[1].hist(rel, bins=40, color="#72B7B2", edgecolor="white")
    ax[1].axvline(0.5, color="#d62728", lw=1.5)
    ax[1].set_title("OOS relative rank of IS-best"); ax[1].set_xlabel("ω (0.5 = median)")
    fig.tight_layout()
    (R / "figures").mkdir(exist_ok=True)
    fig.savefig(R / "figures" / "cscv.png", bbox_inches="tight", dpi=120)
    print("artifacts -> reports/book/cscv_pbo.json · reports/figures/cscv.png")


if __name__ == "__main__":
    main()
