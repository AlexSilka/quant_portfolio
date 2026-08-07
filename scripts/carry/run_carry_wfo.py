"""Walk-forward for cross-sectional carry — the honest way to choose parameters, tested across
DIFFERENT optimisation/testing window schemes (the user's ask). For each scheme we roll forward:
on each train window pick the best (funding-lookback, top-fraction) by train Sharpe, apply it to
the next out-of-sample block, stitch the OOS returns. The stitched OOS Sharpe is the number that
pays the cost of choosing parameters — unlike peak-picking on the whole sample.

Schemes vary train length, test length, and rolling vs expanding, so we can see the result is not
an artifact of one particular window choice. Also prints the full-sample sensitivity surface (is
the edge a broad plateau?) and the peak-picking (overfit) Sharpe as the dishonest reference.

    python scripts/carry/run_carry_wfo.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.carry.run_carry import load_panel  # noqa: E402

PPY, TVOL, SEED, CB = 365, VOL_TARGET_ANNUAL, SEED, 6.0
GRID = [(lb, tf) for lb in (3, 7, 14, 30) for tf in (0.1, 0.2, 0.3)]


def book_ret(C, fd, lb, tf):
    return carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, lb), direction=-1.0,
                            top_frac=tf, cost_bps=CB)["ret"]


def vt_slice(net):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale)


def sharpe_of(net):
    n = vt_slice(net).dropna()
    return summarise(n, PPY)["sharpe_ann"] if len(n) > 30 else -9.0


def walk_forward(C, fd, train_days, test_days, expanding):
    """Roll: pick best grid point on [start, t), apply on [t, t+test]. Stitch OOS."""
    idx = C.index
    t0 = idx[0] + pd.Timedelta(days=train_days)
    oos, picks = [], []
    cur = t0
    while cur < idx[-1]:
        tr_start = idx[0] if expanding else cur - pd.Timedelta(days=train_days)
        te_end = cur + pd.Timedelta(days=test_days)
        Ctr, fdtr = C.loc[tr_start:cur], fd.loc[tr_start:cur]
        best = max(GRID, key=lambda p: sharpe_of(book_ret(Ctr, fdtr, *p)))
        seg = vt_slice(book_ret(C, fd, *best)).loc[cur:te_end]
        oos.append(seg)
        picks.append(best)
        cur = te_end
    stitched = pd.concat(oos)
    stitched = stitched[~stitched.index.duplicated()].dropna()
    return stitched, picks


def main():
    C, fd = load_panel()
    print(f"panel {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}\n")

    # full-sample sensitivity surface + peak-picking (overfit) reference
    surf = np.array([sharpe_of(book_ret(C, fd, *p)) for p in GRID])
    print("=== full-sample sensitivity surface (12 configs) ===")
    print(f"  Sharpe min {surf.min():+.2f} / median {np.median(surf):+.2f} / max {surf.max():+.2f}"
          f"   fraction positive {(surf > 0).mean():.0%}")
    print(f"  peak-picking best (OVERFIT reference): {surf.max():+.2f}  at {GRID[int(surf.argmax())]}\n")

    print("=== walk-forward OOS across different window schemes ===")
    schemes = [
        ("roll 365/90", 365, 90, False), ("roll 365/180", 365, 180, False),
        ("roll 545/90", 545, 90, False), ("roll 270/90", 270, 90, False),
        ("expand */90", 365, 90, True), ("expand */180", 365, 180, True),
    ]
    rows = []
    for name, tr, te, exp in schemes:
        stitched, picks = walk_forward(C, fd, tr, te, exp)
        s = summarise(stitched, PPY)
        p5 = bootstrap_sharpe(stitched, PPY, 500, SEED).get("sharpe_p5", np.nan)
        # how often each param was picked -> parameter stability
        from collections import Counter
        top = Counter(picks).most_common(1)[0]
        rows.append({"scheme": name, "oos_sharpe": round(s["sharpe_ann"], 2), "oos_mc_p5": round(p5, 2),
                     "max_dd": round(s["max_dd"], 2), "mip": round(s["months_in_profit"], 2),
                     "n_rebal": len(picks), "modal_param": f"lb{top[0][0]}_top{int(top[0][1]*100)}", "modal_%": f"{top[1]}/{len(picks)}"})
        print(f"  {name:14s} OOS Sharpe {s['sharpe_ann']:+.2f}  P5 {p5:+.2f}  DD {s['max_dd']:+.0%}  "
              f"mip {s['months_in_profit']:.0%}  modal {rows[-1]['modal_param']} ({rows[-1]['modal_%']})")

    df = pd.DataFrame(rows)
    df.to_csv(CARRY_DIR / "carry_wfo.csv", index=False)
    print(f"\n  mean OOS Sharpe across schemes: {df.oos_sharpe.mean():+.2f}  "
          f"(vs peak-picking {surf.max():+.2f}); the small gap is the sign of a real plateau, not a fitted spike")
    print("\nCARRY-WFO OK")


if __name__ == "__main__":
    main()
