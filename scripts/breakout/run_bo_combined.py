"""The final combined breakout squeeze: the trend-following-with-ML leg (time-series, 1d/4h/1h) and the
point-in-time cross-sectional leg (top-30-liquid, 52w-high nearness) blended at RISK PARITY.

Risk parity means the legs are weighted by their trailing (lagged) volatility, not held at equal
notional. Equal notional was what this did, on two legs whose realised volatilities differ by an order of
magnitude — the time-series book is the average of many lowly-correlated sleeves and is diversified down
to a fraction of the cross-sectional book's volatility — so the "equal risk" blend was almost entirely
one leg while saying it was half of each. The blended book is then vol-targeted once, and both the re-weighting and
the re-sizing are charged: this layer moves whole sleeves, so it pays the book rebalance rate.

The achieved risk shares are published in the summary rather than asserted here, so the claim cannot
drift away from the construction again.

    python scripts/breakout/run_bo_combined.py
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from src.config import BOOK_REBALANCE_BPS, OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import (equal_risk_combine, held_weight_turnover,  # noqa: E402
                             resize_cost, vol_target_scale)

PPY = 365


def rescale(net, target=0.15, charge=True):
    """Re-scale a return series to `target` annualised vol using trailing (lagged) 60-day vol.

    Re-sizing is a trade. This layer moves a whole finished sleeve rather than named instruments, so it
    pays the same blended rate the master book's assembly pays for the same act."""
    scale = vol_target_scale(net, target, PPY)
    out = net * scale
    if charge:
        out = out - resize_cost(scale, BOOK_REBALANCE_BPS)
    return out.dropna()


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

    idx = ts.index.intersection(xs.index)
    L = pd.DataFrame({"time_series": ts.reindex(idx), "cross_sectional": xs.reindex(idx)})
    # equal RISK: trailing inverse vol over the two legs, not equal notional over legs whose volatility
    # differs by an order of magnitude. The weights are what the blend trades, so they come back and
    # are charged; the blended book is vol-targeted once on top, and that is charged too.
    blend, w = equal_risk_combine(L)
    blend = blend - held_weight_turnover(w, L) * (BOOK_REBALANCE_BPS / 1e4)
    combined = rescale(blend.dropna())
    corr = float(L.corr().iloc[0, 1])
    # what the blend ACTUALLY holds, measured rather than claimed — the number the old docstring got
    # wrong by asserting it. Marginal risk contribution: wᵢ·(Σw)ᵢ / wᵀΣw.
    cov = L.cov()
    wm = w.reindex(L.index).mean().reindex(L.columns).fillna(0.0).to_numpy()
    denom = float(wm @ cov.to_numpy() @ wm)
    risk_share = {c: round(float(wm[i] * (cov.to_numpy() @ wm)[i] / denom), 3) if denom > 0 else None
                  for i, c in enumerate(L.columns)}
    ts_s, xs_s = rescale(L["time_series"].dropna()), rescale(L["cross_sectional"].dropna())

    legs = [stats(ts_s, "time_series (trend+ML)"), stats(xs_s, "cross_sectional (PIT top30)"),
            stats(combined, "COMBINED risk-parity")]
    print("=== Final combined breakout squeeze (risk-parity, trailing-vol scaled, no look-ahead) ===")
    print(f"leg correlation (TS vs XS): {corr:+.2f}")
    print("raw leg vol: " + "  ".join(f"{c} {L[c].std() * np.sqrt(PPY):.2%}" for c in L.columns))
    print(f"mean weight: { {c: round(float(w[c].mean()), 3) for c in L.columns} }  ->  "
          f"risk share {risk_share}\n")
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
        json.dumps({"legs": legs, "corr_ts_xs": corr, "risk_share": risk_share,
                    "mean_weight": {c: float(w[c].mean()) for c in L.columns},
                    "raw_leg_vol": {c: float(L[c].std() * np.sqrt(PPY)) for c in L.columns}},
                   indent=2, default=float))
    print("\nBO COMBINED OK")


if __name__ == "__main__":
    main()
