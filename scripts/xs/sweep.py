"""Map the cross-sectional-momentum construction surface on one price panel.

Not a peak-picker: the grid exists to answer "is the edge a broad plateau or a lucky spike?"
(APPROACH.md §5b / §6). It runs every (signal × lookback × skip × breadth × weighting ×
rebalance) config through the same vol-targeted, cost-charged backtest, plus a random-signal
placebo, and writes one row per config. Walk-forward selection and the deflated-Sharpe penalty
at the true trial count live in sibling drivers; this one produces the map.

    python scripts/xs/sweep.py crypto_1d
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, SEED, XS_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.xsect import (mom, risk_adj_mom, blend_rank, xs_backtest,  # noqa: E402
                               vol_target, liquidity_mask)
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

# tradable-universe floor on trailing median daily $-volume: rank only fillable names each bar.
# On a broad survivorship-free crypto universe (hundreds of perps) this is the honest, tradable
# cross-section — not the hand-picked "major coins" list (selection bias) nor micro-cap noise.
LIQ_FLOOR = {"crypto": 20e6, "stocks": 10e6, "fx": 0.0}

CACHE = CACHE_DIR / "xs"
OUT = XS_DIR
OUT.mkdir(parents=True, exist_ok=True)

# bars per calendar day and per year, and per-trade cost (commission + half-spread, bps)
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96, "5m": 288}
PPY = {"crypto": {"1d": 365, "4h": 6 * 365, "1h": 24 * 365, "15m": 96 * 365, "5m": 288 * 365},
       "stocks": {"1d": 252, "4h": 2 * 252, "1h": 7 * 252},     # ~6.5h US session
       "fx": {"1d": 252, "4h": 6 * 252, "1h": 24 * 252}}         # 24h weekday
COST_BPS = {"crypto": 6.0, "stocks": 3.0, "fx": 1.0}

# a-priori grid (days for lookback/skip; bars for rebalance is derived from cadence-days)
LOOKBACK_D = [10, 20, 30, 45, 60, 90, 120, 180, 252]
SKIP_D = [0, 2, 7]
TOP_FRAC = [0.1, 0.2, 0.3]
WEIGHTING = ["equal", "rank", "volinv"]
REBAL_D = [1, 5, 10, 21]          # daily / weekly / biweekly / monthly cadence
SIGNALS = ["raw", "riskadj", "blend"]


def load_panel(tag: str):
    close = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    advp = CACHE / f"{tag}_adv.parquet"
    adv = pd.read_parquet(advp) if advp.exists() else None
    return close, adv


def make_signal(kind: str, px: pd.DataFrame, lb_bars: int, sk_bars: int) -> pd.DataFrame:
    if kind == "raw":
        return mom(px, lb_bars, sk_bars)
    if kind == "riskadj":
        return risk_adj_mom(px, lb_bars, sk_bars)
    # blend: three horizons around lb, risk-adjusted, rank-averaged
    return blend_rank([risk_adj_mom(px, max(2, int(lb_bars * f)), sk_bars) for f in (0.5, 1.0, 2.0)])


def run(tag: str) -> None:
    kind, tf = tag.split("_")           # e.g. "crypto", "1d"
    px, adv = load_panel(tag)
    bpd, ppy, cost = BARS_PER_DAY[tf], PPY[kind][tf], COST_BPS[kind]
    # intraday panels are large (57k–230k bars) and daily-cadence rebalancing is turnover
    # suicide there — so on any sub-daily timeframe use a focused, low-cadence grid
    if tf == "1d":
        lookbacks, skips, weights, rebals = LOOKBACK_D, SKIP_D, WEIGHTING, REBAL_D
    else:
        lookbacks = [5, 10, 20, 30, 45, 60, 90]
        skips, weights, rebals = [0, 2], ["equal", "volinv"], [5, 10, 21]
    rows = []
    for sig_kind in SIGNALS:
        for lb_d in lookbacks:
            for sk_d in skips:
                lb, sk = lb_d * bpd, sk_d * bpd
                if lb < 2:
                    continue
                sig = liquidity_mask(make_signal(sig_kind, px, lb, sk), adv,
                                     LIQ_FLOOR[kind], bpd)
                for tf_frac in TOP_FRAC:
                    for wt in weights:
                        for reb_d in rebals:
                            reb = max(1, reb_d * bpd)
                            bt = xs_backtest(px, sig, top_frac=tf_frac, weighting=wt, rebal=reb,
                                             cost_bps=cost, adv=adv, impact_k=0.1 if adv is not None else 0.0)
                            netv = vol_target(bt["net"], ppy).dropna()
                            if len(netv) < 100:
                                continue
                            s = summarise(netv, ppy)
                            p5 = (bootstrap_sharpe(netv, ppy, 400, SEED).get("sharpe_p5", np.nan)
                                  if s["sharpe_ann"] > 0.6 else np.nan)
                            rows.append({
                                "signal": sig_kind, "lookback_d": lb_d, "skip_d": sk_d,
                                "top_frac": tf_frac, "weighting": wt, "rebal_d": reb_d,
                                "sharpe": s["sharpe_ann"], "sortino": s["sortino_ann"],
                                "mc_p5": p5, "max_dd": s["max_dd"],
                                "months_in_profit": s["months_in_profit"],
                                "ann_turnover": float(bt["turnover"].sum() / (len(px) / ppy)),
                                "cost_drag": float(bt["cost"].sum()), "n_obs": s["n_obs"]})
    # placebo arm: 24 fresh random signals through a representative config — the distribution
    # of Sharpes pure noise earns here IS the pipeline's false-discovery rate (APPROACH.md §6)
    for i in range(24):
        plc = pd.DataFrame(np.random.default_rng(SEED + 100 + i).standard_normal(px.shape),
                           index=px.index, columns=px.columns)
        plc = liquidity_mask(plc, adv, LIQ_FLOOR[kind], bpd)
        bt = xs_backtest(px, plc, top_frac=0.3, weighting="equal",
                         rebal=max(1, 5 * bpd), cost_bps=cost)
        netv = vol_target(bt["net"], ppy).dropna()
        if len(netv) < 100:
            continue
        s = summarise(netv, ppy)
        rows.append({"signal": "PLACEBO", "lookback_d": 0, "skip_d": 0, "top_frac": 0.3,
                     "weighting": "equal", "rebal_d": 5, "sharpe": s["sharpe_ann"],
                     "sortino": s["sortino_ann"], "mc_p5": np.nan, "max_dd": s["max_dd"],
                     "months_in_profit": s["months_in_profit"],
                     "ann_turnover": float(bt["turnover"].sum() / (len(px) / ppy)),
                     "cost_drag": float(bt["cost"].sum()), "n_obs": s["n_obs"]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sweep_{tag}.csv", index=False)
    real = df[df.signal != "PLACEBO"]
    plac = df[df.signal == "PLACEBO"]
    print(f"\n=== {tag}: {len(real)} configs, panel {px.shape[0]}×{px.shape[1]} ===")
    print(f"  Sharpe  min {real.sharpe.min():+.2f} / median {real.sharpe.median():+.2f} / "
          f"max {real.sharpe.max():+.2f}   ({(real.sharpe > 0).mean():.0%} of grid positive)")
    print(f"  placebo Sharpe median {plac.sharpe.median():+.2f} / max {plac.sharpe.max():+.2f}")
    top = real.sort_values("sharpe", ascending=False).head(8)
    for r in top.itertuples():
        print(f"  {r.signal:8s} lb{r.lookback_d:>3d} sk{r.skip_d:>2d} tf{r.top_frac} {r.weighting:6s} "
              f"reb{r.rebal_d:>2d}  Sh {r.sharpe:+.2f}  P5 {r.mc_p5:+.2f}  DD {r.max_dd:+.0%}  "
              f"turn {r.ann_turnover:.0f}x")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["crypto_1d"]):
        run(t)
