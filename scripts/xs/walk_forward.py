"""Walk-forward parameter selection for the cross-sectional sleeve — the honest number.

Peak-picking on the full sample is overfit by construction. The honest question is: if you had
to *choose* the construction out-of-sample and pay for that choice, what Sharpe survives? So:

  1. Precompute the vol-targeted net-return series of every grid config once.
  2. For several (train, test) window schemes — rolling and anchored, short and long — roll
     through time: on each train block pick the best-Sharpe config, apply it to the next block,
     stitch the OOS returns. That stitched series pays the cost of choosing parameters.
  3. Report in-sample-best (overfit ceiling) vs each scheme's walk-forward OOS (honest), plus a
     fixed a-priori default that does no selection at all.

A small peak↔walk-forward gap on a broad positive surface is the signature of a real edge; a
large gap is a fitted spike. Run after sweep.py.

    python scripts/xs/walk_forward.py crypto_1d stocks_1d
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CACHE_DIR, XS_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.xsect import (mom, risk_adj_mom, blend_rank, xs_backtest,  # noqa: E402
                               vol_target, liquidity_mask)
LIQ_FLOOR = {"crypto": 20e6, "stocks": 10e6, "fx": 0.0}   # tradable daily $-vol floor

CACHE, OUT = CACHE_DIR / "xs", XS_DIR
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
PPY = {"crypto": {"1d": 365, "4h": 6 * 365, "1h": 24 * 365, "15m": 96 * 365},
       "stocks": {"1d": 252}, "fx": {"1d": 252}}
COST_BPS = {"crypto": 6.0, "stocks": 3.0, "fx": 1.0}

# a common grid spanning both the fast-crypto and slow-equity plateaus (~432 configs)
GRID = [dict(signal=s, lb=lb, sk=sk, tf=tfr, wt=wt, rb=rb)
        for s in ("raw", "riskadj", "blend")
        for lb in (20, 30, 45, 90, 180, 252)
        for sk in (0, 7)
        for tfr in (0.1, 0.2, 0.3)
        for wt in ("equal", "volinv")
        for rb in (5, 21)]

# fixed a-priori defaults — textbook, chosen from the literature NOT from the sweep argmax:
# crypto momentum is fast (~30d risk-adjusted, Liu-Tsyvinski), equity is classic 12-1 (skip a
# month, decile). These are declared-before-fit defaults, the cleanest no-selection baseline.
APRIORI = {"crypto": dict(signal="riskadj", lb=30, sk=0, tf=0.3, wt="equal", rb=21),
           "stocks": dict(signal="riskadj", lb=252, sk=7, tf=0.1, wt="equal", rb=21),
           "fx": dict(signal="riskadj", lb=252, sk=7, tf=0.1, wt="equal", rb=21)}

SCHEMES = [("roll 2y/6m", 2.0, 0.5, False), ("roll 3y/1y", 3.0, 1.0, False),
           ("anch 2y/6m", 2.0, 0.5, True), ("anch 3y/1y", 3.0, 1.0, True)]
TOP_K = 10   # ensemble the top-K train configs per block — robust to plateau near-ties


def _signal(cfg, px, bpd):
    lb, sk = cfg["lb"] * bpd, cfg["sk"] * bpd
    if cfg["signal"] == "raw":
        return mom(px, lb, sk)
    if cfg["signal"] == "riskadj":
        return risk_adj_mom(px, lb, sk)
    return blend_rank([risk_adj_mom(px, max(2, int(lb * f)), sk) for f in (0.5, 1.0, 2.0)])


def net_series(cfg, px, adv, bpd, ppy, cost, kind="crypto"):
    sig = liquidity_mask(_signal(cfg, px, bpd), adv, LIQ_FLOOR[kind], bpd)
    bt = xs_backtest(px, sig, top_frac=cfg["tf"], weighting=cfg["wt"], rebal=max(1, cfg["rb"] * bpd),
                     cost_bps=cost, adv=adv, impact_k=0.1 if adv is not None else 0.0)
    return vol_target(bt["net"], ppy)


def walk_forward(M: pd.DataFrame, ppy: float, train_y: float, test_y: float, anchored: bool,
                 top_k: int = 1):
    """M = time × config net returns. Stitch OOS blocks; per block select by train Sharpe.

    top_k=1 picks the single best train config (classic, but noisy on a plateau of near-ties);
    top_k>1 equal-weights the top-K train configs' OOS returns — an ensemble robust to the fact
    that plateau configs are statistically indistinguishable in-sample.
    """
    idx = M.index
    tr_b, te_b = int(train_y * ppy), int(test_y * ppy)
    segs, picks = [], []
    start = tr_b
    while start + te_b <= len(idx):
        tr0 = 0 if anchored else max(0, start - tr_b)
        train = M.iloc[tr0:start]
        test = M.iloc[start:start + te_b]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        chosen = list(sr.nlargest(top_k).index)
        segs.append(test[chosen].mean(axis=1))
        picks.extend(chosen)
        start += te_b
    oos = pd.concat(segs) if segs else pd.Series(dtype=float)
    return oos, picks


def run(tag: str):
    kind, tf = tag.split("_")
    px = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    advp = CACHE / f"{tag}_adv.parquet"
    adv = pd.read_parquet(advp) if advp.exists() else None
    bpd, ppy, cost = BARS_PER_DAY[tf], PPY[kind][tf], COST_BPS[kind]

    # precompute all configs' vol-targeted net series once
    cols = {}
    for i, cfg in enumerate(GRID):
        cols[i] = net_series(cfg, px, adv, bpd, ppy, cost, kind)
    M = pd.DataFrame(cols).dropna(how="all")
    full_sr = (M.mean() / M.std(ddof=1) * np.sqrt(ppy))
    is_best = int(full_sr.idxmax())
    plateau = float((full_sr > 1.0).mean()) if kind == "crypto" else float((full_sr > 0.5).mean())
    bar = 1.0 if kind == "crypto" else 0.5
    print(f"\n=== {tag}  ({px.shape[0]}×{px.shape[1]}, {len(GRID)} configs) ===")
    print(f"  surface: {(full_sr > 0).mean():.0%} positive, {plateau:.0%} above Sharpe {bar:.1f} "
          f"(plateau breadth)")
    print(f"  in-sample BEST (overfit): Sharpe {full_sr.max():+.2f}  -> {GRID[is_best]}")

    rows = [{"scheme": "in-sample best (overfit)", "sharpe": float(full_sr.max()),
             "config": str(GRID[is_best])}]
    for name, tr_y, te_y, anch in SCHEMES:
        oos1, _ = walk_forward(M, ppy, tr_y, te_y, anch, top_k=1)
        oosk, picks = walk_forward(M, ppy, tr_y, te_y, anch, top_k=TOP_K)
        s1, sk = summarise(oos1, ppy), summarise(oosk, ppy)
        print(f"  WF {name:11s}: single-best {s1['sharpe_ann']:+.2f}  |  top{TOP_K}-ensemble "
              f"{sk['sharpe_ann']:+.2f}  DD {sk['max_dd']:+.0%}  months+ {sk['months_in_profit']:.0%} "
              f"({len(picks)//TOP_K} refits)")
        rows.append({"scheme": f"WF {name}", "sharpe_single": s1["sharpe_ann"],
                     "sharpe": sk["sharpe_ann"], "max_dd": sk["max_dd"],
                     "months_in_profit": sk["months_in_profit"], "n_refits": len(picks) // TOP_K,
                     "config": f"top{TOP_K} ensemble"})

    # fixed a-priori default (no selection at all — the cleanest honest baseline)
    ap = net_series(APRIORI[kind], px, adv, bpd, ppy, cost, kind).dropna()
    sa = summarise(ap, ppy)
    print(f"  FIXED a-priori    : Sharpe {sa['sharpe_ann']:+.2f}  DD {sa['max_dd']:+.0%}  "
          f"months+ {sa['months_in_profit']:.0%}  -> {APRIORI[kind]}")
    rows.append({"scheme": "fixed a-priori (no selection)", "sharpe": sa["sharpe_ann"],
                 "max_dd": sa["max_dd"], "months_in_profit": sa["months_in_profit"],
                 "config": str(APRIORI[kind])})
    pd.DataFrame(rows).to_csv(OUT / f"wf_{tag}.csv", index=False)
    ap.rename("ret").to_frame().to_parquet(OUT / f"apriori_{tag}.parquet")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["crypto_1d", "stocks_1d"]):
        run(t)
    print("\nWF OK")
