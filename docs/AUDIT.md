# What running every script found

`run_all.py` executes fourteen steps. The repository has about a hundred and fifty scripts. The other
hundred and thirty-odd had never been run in this tree, so nothing — not ruff, not the report's
`--check` gates, not a passing pipeline — could see what had rotted in them. This is the record of
running all of them once.

The method is the finding as much as the results are: **execution, not inspection**. Every defect
below is syntactically valid Python that greps clean and imports fine. Half of them are a constant
that was written for a book with a different set of families; the rest are a rename or a signature
change that reached the caller inside the pipeline and no further.

## Defects found and fixed

| what | where | shape of the failure |
|---|---|---|
| `mb.regime_overlay` — the managed-vol overlay the §8 ladder replaced | `run_adversary`, `run_grind_timer`, `run_convexity_book`, and pasted copies in `run_dispersion_book`, `run_adaptive_book`, `candidate_book_mftrend`, `frontier` | a pasted copy outlives its original; four candidate books were scoring themselves against a portfolio that no longer exists |
| `FAMILIES` naming trend and carry, missing global-macro and BAB | the same four lab books, `run_bo_contribution`, `run_bab_portfolio` | `run_dispersion_book` documented its copy as "identical to run_master_book.FAMILIES" while differing by two families in each direction |
| `daily_ret_cost(sym, px, pos, tf, fund)` — signature moved when the breakout long leg went to spot | `run_ml_book_contribution` | died on a TypeError; the fix was not to re-thread the signature but to put both A/B arms back on one venue |
| paths from before `reports/` grew per-family subfolders | `run_adaptive_book`, `candidate_book_mftrend`, `frontier`, `run_bo_contribution` | `frontier` and `candidate_book_mftrend` died on a FileNotFoundError for a lab sleeve that has existed the whole time |
| a local `load()` without the assembler's timezone normalisation | `run_bo_contribution` | surfaced only after the paths were fixed and it finally read another family's leg — a copy diverges by what it forgets |
| relative imports (`from .run_gate_coverage import`) | `run_gate_coverage`, `run_gate_ablation` | the only two files in the repo that cannot run as `python scripts/...`; their own siblings use the package-qualified form |
| `drop(columns=["carry"])` on a book that dropped carry | `carry/make_carry_figures` | the figure had not been generated since §6d-ter |
| `PER_FAMILY_CAP = 1.0 / 8 * 1.5` | `run_master_book` | written for an eight-family book, read as 1.13x equal weight against the 1.5x it claimed |
| the headline checker enforcing a scorecard shape the README stopped using | `check_headline` | the guard against stale claims was itself stale, and failing on five counts that were all its own |
| `OTHERS` — "the same as run_master_book.FAMILIES, minus trend" | `trend/run_trend_in_portfolio` | four ways from it: named carry, missed crisis/global-macro/BAB, pre-reorganisation paths, and volprem's UNGATED column — so the trend counterfactual was scored against a book that never existed |
| the same `drop(columns=["carry"])`, one directory over | `carry/run_carry_portfolio` | second instance of one defect in one family, which is what a copied line looks like after the thing it copied from moves |

## Two defects about the numbers, not the running

**A study whose universe was set by network luck.** `run_carry_equity` builds its universe by looping
over symbols and skipping the ones that fail to load. The feed refused two of them mid-audit, the loop
absorbed both, and the study published a headline computed on 50 names instead of 52 — moving **3,639
of 3,736 days by up to 3.1e-02**. The committed artifact's disagreement with a fresh run had exactly
the same magnitude as two fresh runs' disagreement with each other, which is how it was found:
**run a script twice before concluding its artifact is merely old.** Fixed by distinguishing "the feed
refused" (`RateLimited`, re-raised) from "this symbol has no data" (still skipped).

**A cache that could not record an absence.** `load_dividends` treated a short or empty cache as
unfetched, so seven names that pay no dividends at all and ten whose payments begin later re-walked
their whole history on every run — **17 of 69 symbols, ~51 empty requests per run**. Under a parallel
sweep that tripped the feed's per-minute limit and killed unrelated studies. The data was fully
downloaded the whole time. Now the walk records how far back it has been.

## What reproduces, and what does not

The point of re-running is not only that a script exits 0. Against the committed artifacts:

- **The vol-premium leg reproduces byte-identically** — 0.00e+00 over 5,399 days on both `ret` and
  `ret_gated`. That is the book's dominant leg, 72% of its P&L, and it is deterministic.
- **The discovery grid reproduces byte-identically** — 2,129 candidates, 46 survivors, in-sample
  +2.00 against walk-forward +0.13.
- **The trend family does not.** Its committed artifacts predate the funding finding: the commit that
  established funding as the dominant, *conditional* cost (23.4%/yr on the long leg) and moved the leg
  to spot touched **no file under `reports/trend/`**. Re-running collapses the trend headline's total
  return from **227% to 69%**. Trend is not in the book, but §6d-ter's case for dropping it quotes its
  standalone Sharpe.
- **One committed trend artifact contains an impossible bar** — HBARUSDT_4h, 2021-03-17, a return of
  **−117.8%**. That timestamp is exactly where HBAR's perp history begins, so it is a listing-boundary
  artifact; the raw bars there are clean and a fresh run gives zero. The fix is already in the code and
  the artifact simply never regenerated. A −118% bar distorts every statistic computed from that panel.
- **The breakout deep-dive's artifacts do not reproduce either**, for a different and benign reason:
  its equity/FX universe is a glob over the data directory (`bo_common.py`), now 1,645 tickers against
  ~400 when those files were written. A glob universe means the study is not reproducible across time
  or machines; the shipped leg is crypto and unaffected.

## The finding that outranks the rest: an input nobody could diff

Six of the book's seven legs reproduce byte-identically. The seventh is the cross-sectional leg, and
chasing why took the audit somewhere it did not expect to go.

`xs` is built from panels in `data/cache/`, which is git-ignored, and the panel's universe is a **ranked
cut**: the 300 most liquid of whatever USDT perps happen to be on disk, ranked on **full-sample** median
volume. Today that binds — 300 selected out of 367 eligible, 67 names displaced. Download more symbols,
rebuild, and the cut changes *retroactively*, rewriting the leg's whole history.

Rebuilding the panels during this audit moved the leg on **4,612 of 4,869 days** and took the book's
scored block from **5/5 to 3/5** — Sharpe 3.96 → 4.02, just past the ceiling, and months-in-profit
80.8% → 76.9%, two months on a 26-month block. No code changed. Nothing was diffable, because the input
that changed was not tracked anywhere.

Two things follow, and they are different sizes.

The small one is fixed: `build_panels` now writes `reports/xs/<tag>_universe.json` beside each panel —
selected names, eligible count, whether the cap bound, the thresholds, the per-name liquidity the rank
used, and an explicit note that the rank is not point-in-time. Tracked, so the next rebuild that moves a
number shows which names moved with it.

The large one is not, and should not be decided by an audit: a full-sample rank over a growing disk is a
hindsight universe. The honest fix is for the panel to stop pre-filtering at all and let the strategy's
own **trailing** liquidity rank — which it already computes — decide. That is a change to the leg.

Also worth stating plainly: the block's 5/5 was resting on this. Two targets crossed on a rebuild that
involved no decision at all, which says the margin was thinner than the scorecard looked.

## The pattern

Every defect here is the same one seen from a different angle: **a correction was made, committed, and
never propagated to anything outside `run_all.py`**. The repo already has the rule — changing a number
means regenerating its whole chain — and it holds inside the pipeline. Outside it, the chain has no
runner, so it silently does not exist.

Two cheap habits close most of it:

1. **Import the assembler; never paste it.** Every stale `FAMILIES`, every retired overlay, every
   pre-reorganisation path came from a copy taken for convenience. A copy cannot be swept.
2. **Run the scripts.** `for f in scripts/*/*.py; do python $f; done` found what no grep did. It costs
   an hour, most of it parallelisable by family, and it is the only thing that sees this class at all.

And one warning about doing it: check what touches a metered feed first. Running everything at once is
what turned a dormant caching bug into an exhausted rate limit.
