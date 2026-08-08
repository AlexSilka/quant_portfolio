"""§6 — what a search budget buys: the winner's in-sample Sharpe vs. what it is worth out of sample.

REPORT §5 states the conclusion (best-sleeve deflated Sharpe ~0 at N=1,279; CSCV PBO 32%; the naive
in-sample selection walk-forwards to +0.20). This measures the *dose-response* behind it.

Two curves, both from the committed candidate matrix — no re-mining, so neither can drift from the
published zoo (`reports/book/all_returns.parquet`: every zoo sleeve's daily net return, survivors and
failures alike):

  A. SEARCH BUDGET. Sweep N from 1 to the pool; at each N draw many random candidate subsets, pick the
     in-sample winner, score it out of sample, and compare against picking a candidate at RANDOM. The
     in-sample Sharpe rises monotonically with N — that is an order statistic, no edge required. If the
     out-of-sample line does not rise with it, the search bought nothing.

  B. DECLARED TRIALS. Take the single best sleeve and deflate it at a range of *declared* trial counts.
     Same track record, same Sharpe — only the honesty about how much was searched changes. This is the
     penalty a reader cannot see unless the trial count is published.

Pool and window are the ones the committed CSCV run already uses (2021+, candidates with dense coverage),
so the two overfit diagnostics are computed on the same evidence.

    make selection-bias      # seconds; writes reports/book/selection_bias.json
"""
import json

import numpy as np
import pandas as pd

from src import bo_common as bo
from src.config import SEED
from src.metrics import deflated_sharpe

PPY = 365                      # zoo sleeves are carried as daily series on the crypto calendar
WINDOW_START = "2021-01-01"    # CSCV's window: the point where a large pool is simultaneously live
MIN_COVERAGE = 0.95            # a candidate must be this dense on the window to be rank-comparable
N_DRAWS = 2000                 # random candidate subsets per search-budget level
BUDGETS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 385]
DECLARED = [1, 2, 5, 10, 50, 100, 385, 1279]   # trial counts to deflate the same winner at
SPLITS = np.linspace(0.35, 0.65, 7)            # split points; one chronological cut is a single draw


def sharpe_bar(mat):
    """Per-bar Sharpe of every column at once (annualised on output; the ranking is scale-free)."""
    sd = mat.std(axis=0, ddof=1)
    return np.where(sd > 0, mat.mean(axis=0) / np.maximum(sd, 1e-12), 0.0)


def main():
    df = pd.read_parquet(bo.REPORTS / "book" / "all_returns.parquet")
    win = df.loc[WINDOW_START:]
    # the ≤5% of bars a dense candidate still misses are days it was not positioned -> flat, not dropped,
    # so every candidate is scored on the identical calendar
    panel = win[win.columns[win.notna().mean() >= MIN_COVERAGE]].fillna(0.0)
    pool = panel.shape[1]
    ann = np.sqrt(PPY)
    cuts = [int(len(panel) * f) for f in SPLITS]
    print(f"pool {pool} candidates on {panel.index[0].date()}..{panel.index[-1].date()} "
          f"({len(panel)} bars)  averaged over {len(cuts)} split points "
          f"{panel.index[cuts[0]].date()}..{panel.index[cuts[-1]].date()}")

    rng = np.random.default_rng(SEED)
    # accumulate every split's answer, then report the median across splits — a single chronological
    # cut is one draw from a noisy distribution and would read as precision the design does not have
    acc = {n: {"is": [], "oos": [], "lose": [], "vs_rand": []} for n in BUDGETS if n <= pool}
    randoms, var_list = [], []

    for cut in cuts:
        is_mat, oos_mat = panel.values[:cut], panel.values[cut:]
        sr_is, sr_oos = sharpe_bar(is_mat), sharpe_bar(oos_mat)
        var_list.append(float(np.var(sr_is, ddof=1)))
        random_oos = float(np.median(sr_oos) * ann)      # what an UNselected candidate returns
        randoms.append(random_oos)
        for n in acc:
            if n == pool:
                picks = np.array([int(np.argmax(sr_is))])   # whole pool: one deterministic winner
            else:
                subsets = rng.integers(0, pool, size=(N_DRAWS, n))
                picks = subsets[np.arange(N_DRAWS), np.argmax(sr_is[subsets], axis=1)]
            w_is, w_oos = sr_is[picks] * ann, sr_oos[picks] * ann
            acc[n]["is"].append(float(np.median(w_is)))
            acc[n]["oos"].append(float(np.median(w_oos)))
            acc[n]["lose"].append(float((w_oos <= 0).mean()))
            acc[n]["vs_rand"].append(float(np.median(w_oos) - random_oos))

    random_oos = float(np.median(randoms))
    var_trials = float(np.median(var_list))            # dispersion of trial Sharpes — the DSR null width
    print(f"baseline — a candidate picked at RANDOM: out-of-sample Sharpe {random_oos:+.2f}\n")

    rows = []
    for n, a in acc.items():
        rows.append({
            "n_candidates": n,
            "winner_sharpe_is": round(float(np.median(a["is"])), 3),
            "winner_sharpe_oos": round(float(np.median(a["oos"])), 3),
            "winner_sharpe_oos_min_split": round(float(np.min(a["oos"])), 3),
            "winner_sharpe_oos_max_split": round(float(np.max(a["oos"])), 3),
            # pooled over splits, not median-of-splits: at n == pool each split contributes a single
            # deterministic winner, so a median would degenerate to 0/1 and read as false precision
            "p_winner_loses_oos": round(float(np.mean(a["lose"])), 3),
            "edge_over_random_oos": round(float(np.median(a["vs_rand"])), 3),
            "inflation_is_over_oos": round(float(np.median(a["is"]) - np.median(a["oos"])), 3),
        })
        r = rows[-1]
        print(f"  N={n:>4}  winner IS {r['winner_sharpe_is']:+.2f}  ->  OOS {r['winner_sharpe_oos']:+.2f}"
              f"  (across splits {r['winner_sharpe_oos_min_split']:+.2f}..{r['winner_sharpe_oos_max_split']:+.2f})"
              f"  inflation {r['inflation_is_over_oos']:+.2f}"
              f"  P(loses OOS) {r['p_winner_loses_oos']:.0%}  vs random {r['edge_over_random_oos']:+.2f}")

    # B — the same winner, deflated at a range of declared trial counts (mid split, for one clean track)
    mid = cuts[len(cuts) // 2]
    is_mat, oos_mat = panel.values[:mid], panel.values[mid:]
    sr_is, sr_oos = sharpe_bar(is_mat), sharpe_bar(oos_mat)
    best = int(np.argmax(sr_is))
    col = pd.Series(is_mat[:, best])
    sk, ku = float(col.skew()), float(col.kurt() + 3.0)
    declared = [{"declared_trials": n,
                 "deflated_sharpe": round(float(deflated_sharpe(float(sr_is[best]), len(col), sk, ku,
                                                                max(n, 2), var_trials)), 3)}
                for n in DECLARED]
    print(f"\nbest sleeve ({panel.columns[best]}): in-sample Sharpe {sr_is[best] * ann:+.2f}, "
          f"out-of-sample {sr_oos[best] * ann:+.2f}")
    print("  same track record, deflated at a declared trial count of:")
    for d in declared:
        print(f"    N={d['declared_trials']:>5}  ->  deflated Sharpe {d['deflated_sharpe']:.2f}")

    out = {"window": [str(panel.index[0].date()), str(panel.index[-1].date())],
           "split_dates": [str(panel.index[c].date()) for c in cuts],
           "pool": pool, "n_bars": len(panel),
           "n_draws": N_DRAWS, "var_across_trials": round(var_trials, 8),
           "random_pick_sharpe_oos": round(random_oos, 3),
           "best_sleeve": {"name": str(panel.columns[best]),
                           "sharpe_is": round(float(sr_is[best] * ann), 3),
                           "sharpe_oos": round(float(sr_oos[best] * ann), 3)},
           "by_budget": rows, "by_declared_trials": declared}
    path = bo.REPORTS / "book" / "selection_bias.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    print("VERDICT: search budget buys in-sample Sharpe, not out-of-sample return — which is why the "
          "traded book selects nothing and combines structurally distinct premia instead.")


if __name__ == "__main__":
    main()
