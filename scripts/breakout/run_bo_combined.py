"""The final combined breakout squeeze: the trend-following-with-ML leg (time-series, 1d/4h/1h) and the
point-in-time cross-sectional leg (top-30-liquid, 52w-high nearness) blended at RISK PARITY.

Both legs are re-scaled to a common ~15% annualised vol using a trailing (lagged) vol estimate — PIT,
no look-ahead — then equal-weighted, so each contributes equal risk (inverse-vol weighting). Because
they correlate only +0.13, the blend's Sharpe exceeds either leg. This is the single sleeve to hand to
the master portfolio alongside momentum/carry.

    python scripts/breakout/run_bo_combined.py
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from src.config import OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

PPY = 365


def rescale(net, target=0.15):
    """Re-scale a return series to `target` annualised vol using trailing (lagged) 60-day vol."""
    scale = vol_target_scale(net, target, PPY)
    return (net * scale).dropna()


def stats(s, label):
    ss = summarise(s, PPY)
    mc = bootstrap_sharpe(s, PPY, 2000, bo.SEED)
    oos = s[s.index >= OOS_START]
    py = {int(y): round(float(np.sqrt(PPY) * g.dropna().mean() / g.dropna().std(ddof=1)), 2)
          for y, g in s.groupby(s.index.year) if g.dropna().std(ddof=1) > 0}
    return {"leg": label, "sharpe": ss["sharpe_ann"], "max_dd": ss["max_dd"],
            "months_in_profit": ss["months_in_profit"], "mc_p5": mc.get("sharpe_p5", np.nan),
            "mc_p50": mc.get("sharpe_p50", np.nan), "oos_sharpe": summarise(oos, PPY)["sharpe_ann"],
            "per_year": py}


def main():
    ts = pd.read_parquet(bo.BREAKOUT / "bo_final_portfolio.parquet")["ret"].dropna()
    pit = pd.read_parquet(bo.BREAKOUT / "bo_xs_pit_returns.parquet")
    xs = pit[["1d_PIT_top30", "4h_PIT_top30", "1h_PIT_top30"]].mean(axis=1).dropna()  # XS sub-book

    ts_s, xs_s = rescale(ts), rescale(xs)
    idx = ts_s.index.intersection(xs_s.index)
    ts_s, xs_s = ts_s.reindex(idx).fillna(0.0), xs_s.reindex(idx).fillna(0.0)
    combined = 0.5 * ts_s + 0.5 * xs_s          # equal risk (both already re-scaled to 15% vol)
    corr = float(np.corrcoef(ts_s, xs_s)[0, 1])

    legs = [stats(ts_s, "time_series (trend+ML)"), stats(xs_s, "cross_sectional (PIT top30)"),
            stats(combined, "COMBINED risk-parity")]
    print("=== Final combined breakout squeeze (risk-parity, trailing-vol scaled, no look-ahead) ===")
    print(f"leg correlation (TS vs XS): {corr:+.2f}\n")
    for l in legs:
        print(f"  {l['leg']:28s} Sharpe {l['sharpe']:+.2f}  maxDD {l['max_dd']:+.1%}  "
              f"months+ {l['months_in_profit']:.0%}  MC[P5 {l['mc_p5']:+.2f} P50 {l['mc_p50']:+.2f}]  "
              f"OOS {l['oos_sharpe']:+.2f}")
    print(f"\ncombined per-year Sharpe: {legs[-1]['per_year']}")

    # marginal contribution: does the second leg pay? (§7)
    solo = summarise(ts_s, PPY)["sharpe_ann"]
    print(f"marginal: TS alone {solo:+.2f} -> +XS {legs[-1]['sharpe']:+.2f} "
          f"({legs[-1]['sharpe'] - solo:+.2f} from a +{corr:.2f}-correlated leg)")

    combined.rename("ret").to_frame().to_parquet(bo.BREAKOUT / "bo_combined_portfolio.parquet")
    pd.DataFrame({"time_series": ts_s, "cross_sectional": xs_s, "combined": combined}).to_parquet(
        bo.BREAKOUT / "bo_combined_legs.parquet")
    (bo.BREAKOUT / "bo_combined_summary.json").write_text(
        json.dumps({"legs": legs, "corr_ts_xs": corr}, indent=2, default=float))
    print("\nBO COMBINED OK")


if __name__ == "__main__":
    main()
