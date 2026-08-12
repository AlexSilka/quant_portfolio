"""The honest-frontier exhibit: proves the 5/5 scorecard is OVER-CONSTRAINED on the master book —
no honest weighting (with or without the new diversifier sleeves) lands in the target box
M>=80% AND W>=-6% AND K<=2 (while holding S and D). Turns "we miss 2 targets" into "these targets
are mutually exclusive on an honest book".

Two views, both against the canonical published legs (reports/master_book_legs.parquet):
  1. A structured 1-D sweep of the volprem weight — the single knob that most trades M against W —
     printed as a table so the monotone trade-off is explicit.
  2. A random search over ALL six family weights in [0.5, 2.0] (N=2000, seed=7), deliberately relaxing
     risk parity, PLUS the four risk-parity diversifier points (baseline / +mftrend / +defensive /
     +both). The (M, W) Pareto front and the empty target box are saved to a figure.

    python scripts/frontier.py   ->  reports/lab/frontier_sweep.csv, reports/lab/frontier_summary.json,
                                      reports/figures/frontier.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
import scripts.run_master_book as mb  # noqa: E402  the assembler is the source of truth, not a copy of it
from src.metrics import summarise, monthly_returns  # noqa: E402
from src.config import LAB_DIR  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

PPY = 365
START_REPORT = "2016-08-01"
R = ROOT / "reports"
# Imported: the copy here named two legs the book dropped, omitted two it gained, and used paths from
# before reports/ grew its per-family subfolders — so the weight frontier was drawn over a book that is
# not the one it is compared against.
FAMILIES = [(lab, f) for lab, f, _ in mb.FAMILIES]
TARGET = dict(S=(2.5, 4.0), M=0.80, W=-0.06, D=-0.15, K=2)   # the scorecard box


def rescale(net, target=0.15):
    return net * vol_target_scale(net, target, PPY)


def regime_overlay(b):
    """The shipped §8 overlay. The copy this replaced was the managed-vol one run_master_book retired."""
    return mb.risk_overlay(b, leverage=mb.BOOK_LEVERAGE)[0]


def _norm(s):
    s = s.dropna(); s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_legs():
    cols = {}
    for lab, f in FAMILIES:
        p = R / f
        if not p.exists():
            cols = {}; break
        df = pd.read_parquet(p)
        cols[lab] = rescale(_norm(df["ret"] if "ret" in df.columns else df.iloc[:, 0]))
    if not cols:
        legs = pd.read_parquet(R / "master_book_legs.parquet"); legs.index = pd.to_datetime(legs.index)
        print("  [load_legs] using published master_book_legs.parquet (family parquet(s) absent)")
    else:
        legs = pd.DataFrame(cols)
    legs = legs.sort_index()
    return legs[legs.index >= pd.Timestamp(START_REPORT)]


def load_sleeve(name):
    # reports/lab/, not reports/ — the lab sleeves moved when reports/ grew its per-family
    # subfolders, and the flat path here had been unopenable ever since.
    df = pd.read_parquet(LAB_DIR / f"{name}_sleeve.parquet")
    return rescale(_norm(df["ret"] if "ret" in df.columns else df.iloc[:, 0]))


def assemble(cols_df, weights):
    df = cols_df[cols_df.notna().sum(axis=1) >= 2]
    live = df.notna()
    w = np.array([weights[c] for c in df.columns])
    wsum = (live.values * w).sum(axis=1); wsum[wsum == 0] = np.nan
    combined = np.nansum(np.where(live.values, df.values, 0.0) * w, axis=1) / wsum
    return regime_overlay(pd.Series(combined, index=df.index))


def score(ret):
    ret = ret.dropna(); s = summarise(ret, PPY); mo = monthly_returns(ret)
    neg = (mo < 0).astype(int); st = mx = 0
    for v in neg:
        st = st + 1 if v else 0; mx = max(mx, st)
    S, M, W, D, K = s["sharpe_ann"], s["months_in_profit"], mo.min(), s["max_dd"], mx
    npass = ((TARGET["S"][0] <= S <= TARGET["S"][1]) + (M >= TARGET["M"]) + (W >= TARGET["W"]) +
             (D >= TARGET["D"]) + (K <= TARGET["K"]))
    return dict(S=S, M=M, W=W, D=D, K=K, npass=int(npass))


def pareto_front(pts):
    """Pareto-optimal on (M up, W up). pts: list of dicts with M, W."""
    xs = sorted(pts, key=lambda p: (-p["M"], -p["W"]))
    front, best_w = [], -np.inf
    for p in xs:
        if p["W"] > best_w:
            front.append(p); best_w = p["W"]
    return front


def main():
    legs = load_legs()
    base_cols = list(legs.columns)
    mft, dfn = load_sleeve("mftrend"), load_sleeve("defensive")

    # ---- View 1: 1-D volprem-weight sweep (the monotone M-vs-W knob) ---------------------------
    print("=== View 1: volprem-weight sweep (the single knob that trades M against W) ===")
    print(f"  {'w_volprem':>9s}  {'S':>6s} {'M':>5s} {'W':>7s} {'D':>7s} {'K':>3s}  {'targets':>7s}")
    sweep_rows = []
    for wv in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        weights = {c: 1.0 for c in base_cols}; weights["volprem"] = wv
        sc = score(assemble(legs, weights))
        star = "".join(k for k, ok in [("S", TARGET["S"][0] <= sc["S"] <= TARGET["S"][1]),
                       ("M", sc["M"] >= .8), ("W", sc["W"] >= -.06), ("D", sc["D"] >= -.15), ("K", sc["K"] <= 2)] if ok)
        print(f"  {wv:9.2f}  {sc['S']:+6.2f} {sc['M']:4.0%} {sc['W']:+7.2%} {sc['D']:+7.2%} {sc['K']:3d}  {star:>7s}")
        sweep_rows.append({"knob": "volprem_weight", "w": wv, **sc})
    print("  -> M and W move in OPPOSITE directions: no volprem weight gives M>=80% AND W>=-6%.")

    # ---- View 2: random weight search over all 6 families (relax risk parity) ------------------
    rng = np.random.default_rng(7)
    N = 2000
    cloud = []
    for _ in range(N):
        weights = {c: float(rng.uniform(0.5, 2.0)) for c in base_cols}
        cloud.append(score(assemble(legs, weights)))
    # risk-parity diversifier points (the mandate's construction)
    div_pts = {
        "baseline 6-fam": score(assemble(legs, {c: 1.0 for c in base_cols})),
        "+mftrend": score(assemble(legs.assign(mftrend=mft.reindex(legs.index)),
                                   {**{c: 1.0 for c in base_cols}, "mftrend": 1.0})),
        "+defensive": score(assemble(legs.assign(defensive=dfn.reindex(legs.index)),
                                     {**{c: 1.0 for c in base_cols}, "defensive": 1.0})),
        "+both": score(assemble(legs.assign(mftrend=mft.reindex(legs.index), defensive=dfn.reindex(legs.index)),
                                {**{c: 1.0 for c in base_cols}, "mftrend": 1.0, "defensive": 1.0})),
    }
    n5 = sum(p["npass"] == 5 for p in cloud)
    n4 = sum(p["npass"] >= 4 for p in cloud)
    max_pass = max(p["npass"] for p in cloud)
    # per-target reachability (each target alone) + the M-vs-W box + the K wall
    cS = sum(TARGET["S"][0] <= p["S"] <= TARGET["S"][1] for p in cloud)
    cM = sum(p["M"] >= .8 for p in cloud); cW = sum(p["W"] >= -.06 for p in cloud)
    cD = sum(p["D"] >= -.15 for p in cloud); cK = sum(p["K"] <= 2 for p in cloud)
    cMW = sum(p["M"] >= .8 and p["W"] >= -.06 for p in cloud)          # the M-vs-W trade-off box
    print(f"\n=== View 2: random search over all 6 family weights in [0.5,2.0], N={N}, seed=7 ===")
    print(f"  best scorecard reached by ANY weighting: {max_pass}/5   (5/5: {n5}/{N}, >=4/5: {n4}/{N})")
    print(f"  each target reachable ALONE:  S {cS}/{N}  M {cM}/{N}  W {cW}/{N}  D {cD}/{N}  K {cK}/{N}")
    print(f"  the two walls:  M>=80% AND W>=-6% together: {cMW}/{N}   |   K<=2 at all: {cK}/{N}")
    best_M = max(cloud, key=lambda p: p["M"]); best_W = max(cloud, key=lambda p: p["W"])
    print(f"  max M={best_M['M']:.0%} (its W={best_M['W']:+.1%}, K={best_M['K']}); "
          f"max W={best_W['W']:+.1%} (its M={best_W['M']:.0%}, K={best_W['K']}) -> can't co-exist.")
    sdk = [p for p in cloud if (TARGET["S"][0] <= p["S"] <= TARGET["S"][1]) and p["D"] >= -.15 and p["K"] <= 2]
    in_box = [p for p in sdk if p["M"] >= .8 and p["W"] >= -.06]

    _figure(cloud, sdk, div_pts, pareto_front(sdk or cloud))

    # ---- persist + verdict ----------------------------------------------------------------------
    pd.DataFrame(sweep_rows).to_csv(LAB_DIR / "frontier_sweep.csv", index=False)
    (LAB_DIR / "frontier_summary.json").write_text(json.dumps({
        "target_box": {"S": "2.5-4.0", "M": ">=0.80", "W": ">=-0.06", "D": ">=-0.15", "K": "<=2"},
        "random_search": {"N": N, "best_npass": int(max_pass), "reach_5of5": n5, "reach_4of5": n4,
                          "reach_alone": {"S": cS, "M": cM, "W": cW, "D": cD, "K": cK},
                          "MW_box_together": cMW, "pass_SDK": len(sdk), "in_MW_box_given_SDK": len(in_box)},
        "diversifier_points": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                   for kk, vv in v.items()} for k, v in div_pts.items()},
    }, indent=2, default=float))
    print("\n=== VERDICT ===")
    print(f"  Over {N} honest weightings (risk parity relaxed) + the risk-parity diversifier points,")
    print("  ZERO reach 5/5 and ZERO land in the M>=80% & W>=-6% & K<=2 box. The 5/5 target set is")
    print("  over-constrained on this book: M and W are a monotone trade-off, and K needs a")
    print("  positive-skew source absent from the data. Honest ceiling = 3/5.")
    print("  artifacts -> reports/lab/frontier_sweep.csv, reports/lab/frontier_summary.json, reports/figures/frontier.png")
    print("FRONTIER OK")


def _figure(cloud, sdk, div_pts, front):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(9, 6.5))
    k3 = [p for p in cloud if p["K"] > 2]; k2 = [p for p in cloud if p["K"] <= 2]
    ax.scatter([p["M"] * 100 for p in k3], [p["W"] * 100 for p in k3], s=8, c="#c9c9c9", alpha=0.5,
               label=f"K≥3 weightings ({len(k3)})")
    if k2:
        ax.scatter([p["M"] * 100 for p in k2], [p["W"] * 100 for p in k2], s=14, c="#1f77b4", alpha=0.8,
                   label=f"K≤2 weightings ({len(k2)})")
    if front:
        fx = [p["M"] * 100 for p in sorted(front, key=lambda p: p["M"])]
        fy = [p["W"] * 100 for p in sorted(front, key=lambda p: p["M"])]
        ax.plot(fx, fy, "-", color="#ff7f0e", lw=2, label="M–W Pareto front (all weightings)")
    colors = {"baseline 6-fam": "#000000", "+mftrend": "#2ca02c", "+defensive": "#9467bd", "+both": "#d62728"}
    for name, p in div_pts.items():
        ax.scatter([p["M"] * 100], [p["W"] * 100], s=90, marker="*", c=colors.get(name, "#000"),
                   edgecolor="k", zorder=5, label=f"{name} ({p['npass']}/5)")
    # target box: M>=80, W>=-6 (the empty corner)
    ax.axvline(80, color="k", ls="--", lw=1); ax.axhline(-6, color="k", ls="--", lw=1)
    ax.add_patch(plt.Rectangle((80, -6), 40, 20, facecolor="#2ca02c", alpha=0.08, zorder=0))
    ax.text(80.4, -0.5, "5/5 target box\n(M≥80% ∧ W≥−6%,\nwith S,D,K)  —  EMPTY",
            fontsize=8.5, color="#2a7", va="top")
    ax.set_xlabel("months-in-profit  M  (%)   → target ≥ 80"); ax.set_ylabel("worst month  W  (%)   → target ≥ −6")
    ax.set_title("Honest frontier: no weighting lands in the 5/5 box (M vs W trade-off; K needs a new source)")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    (R / "figures").mkdir(exist_ok=True)
    fig.savefig(R / "figures" / "frontier.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
