"""Short-vol / variance-risk-premium deep-dive: is there a harvestable VRP in crypto, and does it
diversify the trend book's *source* of return (short gamma vs long gamma)?

Runs, all vol-targeted to 15% so they compare on equal risk, net of vega-spread costs, exec at t+2:
  1. always-short baseline vs implied-rich-timed short-vol, on BTC and ETH;
  2. a shuffled-DVOL placebo (destroys the implied-vs-realised information);
  3. parameter sensitivity (k_rich x rv_lookback x var_cap x restrike) + an exec-lag leakage ladder;
  4. cost sensitivity (base / 3x / break-even);
  5. portfolio value-add: correlation of the VRP book to the momentum and carry books, and the
     Sharpe / drawdown of momentum+carry with and without VRP added.

    python scripts/volprem/run_vol_premium.py

Implied vol: Deribit DVOL (src/data/deribit.py, free, cached). Realised vol and the paid leg:
Binance perp bars. Artifacts: reports/volprem_*.{csv,parquet}.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, REPORTS_DIR, SEED, TREND_DIR, VOLPREM_DIR, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.deribit import load_dvol  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402
from src.sleeves.vol_premium import realized_vol  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

SEED, TVOL, PPY = SEED, VOL_TARGET_ANNUAL, 365
START, END = "2021-01", "2026-08"
ASSETS = ["BTCUSDT", "ETHUSDT"]
CUR = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
UNCAPPED = 1e9        # honest baseline: no free tail truncation (a real cap costs wing premium)
rng = np.random.default_rng(SEED)


def vt(net: pd.Series) -> pd.Series:
    """Vol-target a daily P&L series to 15% annualised (lagged 1 bar, look-ahead-free).

    Clip at -0.999: a short-vol book can lose more than its capital in one crash day; clipping
    models liquidation at a total loss for that day rather than letting equity go negative. Days
    that hit the clip are ruin events and are surfaced by `profile`.
    """
    scale = vol_target_scale(net, TVOL, PPY)
    return (net * scale).clip(lower=-0.999).dropna()


def daily(s: pd.Series) -> pd.Series:
    """Normalise any daily series to a tz-naive midnight index so books align across sources."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.Series(s.to_numpy(), index=idx.normalize()).groupby(level=0).last()


def load_inputs():
    close, dvol = {}, {}
    for a in ASSETS:
        close[a] = load_klines(a, "1d", START, END, market="um")["close"]
        dvol[a] = load_dvol(CUR[a], START, END)["close"]
    return close, dvol


def book(close, dvol, *, timed=True, **kw):
    """VRP book across ASSETS, each vol-targeted to 15% then equal-risk averaged. Uncapped by default."""
    kw.setdefault("var_cap", UNCAPPED)
    legs = {}
    for a in ASSETS:
        bk = vp.short_vol_book(close[a], dvol[a], timed=timed, **kw)
        legs[a] = vt(bk["net"])
    port = pd.concat(legs, axis=1).mean(axis=1).dropna()
    return port, legs


def profile(name, net, mc=True):
    s = summarise(net, PPY)
    p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan) if mc else np.nan
    ruin = int((net <= -0.99).sum())          # days the sleeve was liquidated (short-vol crash)
    return {"book": name, "sharpe": s["sharpe_ann"], "mc_p5": p5, "max_dd": s["max_dd"],
            "worst_day": float(net.min()), "ruin_days": ruin,
            "months_in_profit": s["months_in_profit"], "skew": float(net.skew()),
            "psr_gt0": s["psr_gt0"], "n_obs": s["n_obs"]}


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    close, dvol = load_inputs()
    span = f"{min(v.index.min() for v in dvol.values()).date()}..{max(v.index.max() for v in dvol.values()).date()}"
    print(f"DVOL span {span} | assets {ASSETS}\n")

    rows, RET = [], {}

    # ---- 1) baseline (always-short) vs timed (short-when-rich) ----
    print("=== standalone short-vol books (vol-targeted 15%, net of vega costs, t+2) ===")
    base_port, _ = book(close, dvol, timed=False)
    timed_port, timed_legs = book(close, dvol, timed=True)
    RET["VRP_baseline_alwaysshort"] = base_port
    RET["VRP_timed_rich"] = timed_port
    rows.append(profile("VRP_baseline_alwaysshort", base_port))
    rows.append(profile("VRP_timed_rich", timed_port))
    for a, leg in timed_legs.items():
        RET[f"VRP_{a}"] = leg
        rows.append(profile(f"VRP_{a}_timed", leg))
    print("  per-year Sharpe (always-short baseline; short-vol should bleed in crash years):")
    py = base_port.groupby(base_port.index.year).apply(lambda x: summarise(x, PPY)["sharpe_ann"])
    print("   " + "  ".join(f"{y}:{s:+.2f}" for y, s in py.items()))

    # ---- 1b) same book but capped at 2.5x (risk control; wing cost NOT modelled -> Sharpe overstated) ----
    capped_port, _ = book(close, dvol, timed=True, var_cap=2.5)
    RET["VRP_timed_capped2.5"] = capped_port
    rows.append(profile("VRP_timed_capped2.5_wingfree", capped_port))

    # ---- 2) placebo: fair strike = trailing realised vol (NO implied premium) ----
    # If the edge is the real variance premium (implied > realised), a swap struck at realised
    # vol should earn ~nothing. Feeding realised*100 as the "implied" makes short_vol_book strike
    # at realised, isolating the premium.
    print("=== placebo: fair-strike (struck at realised vol, no premium) ===")
    fair = {a: (realized_vol(close[a]) * 100.0).reindex(dvol[a].index).ffill() for a in ASSETS}
    plac_port, _ = book(close, fair, timed=False)
    RET["PLACEBO_fair_strike"] = plac_port
    rows.append(profile("PLACEBO_fair_strike", plac_port, mc=False))

    # ---- 3) parameter sensitivity + exec-lag leakage ladder (on the timed book) ----
    print("=== parameter sensitivity (Sharpe of timed book) ===")
    grid = []
    for k_rich in (0.8, 0.9, 1.0, 1.1, 1.2):
        for rv_lb in (14, 20, 30, 45, 60):
            for rs in (5, 7, 10, 14):
                p, _ = book(close, dvol, timed=True, k_rich=k_rich, rv_lookback=rv_lb, restrike_days=rs)
                grid.append({"k_rich": k_rich, "rv_lookback": rv_lb, "restrike": rs,
                             "sharpe": summarise(p, PPY)["sharpe_ann"]})
    gdf = pd.DataFrame(grid)
    pos = float((gdf.sharpe > 0).mean())
    print(f"  {len(gdf)} configs | positive {pos:.0%} | median {gdf.sharpe.median():+.2f} | "
          f"min {gdf.sharpe.min():+.2f} | max {gdf.sharpe.max():+.2f}")
    gdf.to_csv(VOLPREM_DIR / "volprem_sensitivity.csv", index=False)

    print("=== exec-lag ladder (edge must decay gracefully, not collapse -> no leak) ===")
    lag_row = {}
    for lag in (1, 2, 3, 5, 8):
        p, _ = book(close, dvol, timed=True, exec_lag=lag)
        lag_row[lag] = summarise(p, PPY)["sharpe_ann"]
        print(f"  exec_lag {lag}: Sharpe {lag_row[lag]:+.2f}")

    # ---- 4) cost sensitivity (base / 3x / break-even) ----
    print("=== cost sensitivity (vega half-spread, vol points per roll) ===")
    cost_row = {}
    for mult, tag in [(1.0, "base_0.75"), (3.0, "3x_2.25")]:
        p, _ = book(close, dvol, timed=True, vega_cost_volpts=0.75 * mult)
        cost_row[tag] = summarise(p, PPY)["sharpe_ann"]
        print(f"  {tag:10s}: Sharpe {cost_row[tag]:+.2f}")
    lo, hi = 0.75, 0.75
    while summarise(book(close, dvol, timed=True, vega_cost_volpts=hi)[0], PPY)["sharpe_ann"] > 0 and hi < 30:
        lo, hi = hi, hi * 1.6
    be = round((lo + hi) / 2, 2)
    cost_row["break_even_volpts"] = be
    print(f"  break-even ~{be} vol pts/roll ({be/0.75:.1f}x base)")

    # ---- 5) portfolio value-add vs momentum + carry books ----
    print("=== portfolio value-add (corr + Sharpe/DD with VRP added) ===")
    vrp = daily(timed_port)
    ext = {"VRP": vrp}
    mom_path = TREND_DIR / "trend_block_returns.parquet"
    if mom_path.exists():
        m = pd.read_parquet(mom_path)
        ext["momentum"] = daily(m["ret"] if "ret" in m else m.iloc[:, 0])
    for cp in (CARRY_DIR / "carry_refined.parquet", CARRY_DIR / "carry_headline.parquet"):
        if Path(cp).exists():
            c = pd.read_parquet(cp)
            col = "ret" if "ret" in c else c.select_dtypes("number").columns[0]
            ext["carry"] = daily(c[col])
            break
    E = pd.DataFrame(ext).dropna()
    # re-vol-target each book to 15% on the common window so weights are equal-risk
    E = E.apply(vt).dropna()
    corr = E.corr()
    print(f"  common window {E.index.min().date()}..{E.index.max().date()} ({len(E)} days)")
    print("  correlation:\n" + corr.round(2).to_string())

    have = [c for c in ("momentum", "carry") if c in E]
    baseline = E[have].mean(axis=1)
    sb = summarise(baseline, PPY)
    print(f"  baseline {'+'.join(have):16s} Sharpe {sb['sharpe_ann']:+.2f}  maxDD {sb['max_dd']:+.1%}")
    print("  marginal contribution of VRP by weight (mix = (1-w)*baseline + w*VRP):")
    mc_rows = []
    for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        mix = (1 - w) * baseline + w * E["VRP"]
        sm = summarise(mix, PPY)
        mc_rows.append({"w_vrp": w, "sharpe": sm["sharpe_ann"], "max_dd": sm["max_dd"],
                        "months_in_profit": sm["months_in_profit"]})
        print(f"    w={w:.2f}  Sharpe {sm['sharpe_ann']:+.2f}  maxDD {sm['max_dd']:+.1%}  months+ {sm['months_in_profit']:.0%}")
    mcdf = pd.DataFrame(mc_rows)
    best = mcdf.loc[mcdf.sharpe.idxmax()]
    print(f"  -> best Sharpe {best.sharpe:+.2f} at w_vrp={best.w_vrp:.2f} (vs baseline {sb['sharpe_ann']:+.2f})")
    mcdf.to_csv(VOLPREM_DIR / "volprem_marginal.csv", index=False)

    # ---- persist ----
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(VOLPREM_DIR / "volprem_results.csv", index=False)
    pd.DataFrame(RET).to_parquet(VOLPREM_DIR / "volprem_returns.parquet")
    corr.to_csv(VOLPREM_DIR / "volprem_portfolio_corr.csv")
    pd.DataFrame({"exec_lag": lag_row}).to_csv(VOLPREM_DIR / "volprem_execlag.csv")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== RESULTS (sorted by Sharpe) ===")
    print(df.round(2).to_string(index=False))
    print("\nVOLPREM OK")


if __name__ == "__main__":
    main()
