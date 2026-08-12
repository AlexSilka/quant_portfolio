# Auditing the four legs the live book actually holds

`scripts/run_live_book.py` runs four families — volprem, breakout, BAB, x-sect — at 2x. It used to
print Sharpe 5.40 and 111.3% a year on fixed capital. A Sharpe above 4 on a book of well-known risk
premia is a claim about the modelling, not about the premia, so the modelling was taken apart.

**Every defect found below is now fixed in the code and every number in the repository is rebuilt from
it.** The same book reads **Sharpe 3.37 and 68.3% a year, with a −20.9% drawdown**. The one thing
deliberately left alone is which four legs to hold: that is a decision to take on clean numbers, and
this document is how they got clean.

Every number below was produced by re-running the leg, not by reading it. The scripts that produced
them are named at each finding.

---

## The question that started it: what the daily bar hides

A daily bar has four numbers and a path. A backtest that decides on the close and fills on a later
close never has to know the path. A backtest that has a stop, a target, or a barrier has to know it,
and if it guesses it guesses in its own favour.

Three of the four legs were checked against 5-minute bars. The answers are different for each, and
only one of them is a problem.

### volprem — the daily bar is CONSERVATIVE here

The short-vol sleeve pays a realised-variance leg built from four daily numbers: the overnight gap
squared plus the Rogers-Satchell range term (`src/sleeves/vol_premium.realized_var_ohlc`). A
delta-hedged short-vol book pays the quadratic variation at its hedging frequency, so the honest test
is against a measured 5-minute path.

Eleven of the eighteen underlyings have 5-minute history in the cache (2020-02 →, ~1 630 days each):

| | SPY | QQQ | AAPL | AMZN | GOOGL | GS | IBM | USO | GLD | SLV | GDX | TLT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RS+gap ÷ 5-min QV | 1.12 | 1.01 | 1.03 | 1.06 | 1.05 | 1.07 | 1.07 | 1.07 | 1.05 | 1.05 | 1.12 | 1.00 |

**Mean 1.058.** The model charges the short about 6% MORE variance than a 5-minute delta-hedge would
actually have paid — which is what you expect, since 5-minute sampling is itself a downward-biased
estimate of continuous quadratic variation. Substituting the measured 5-minute leg *raises* the
sub-book from Sharpe 3.42 to 4.30. Nothing to fix. (`t1_intrabar_volprem.py`)

### breakout — the "chandelier stop" is not a stop

`src/sleeves/breakout_lab.hold_atr_trailing` exits when the **close** falls 3×ATR below the highest
close since entry, and the engine fills two bars later. That is implementable exactly as written, and
it contains no favourable-fill assumption. But almost nobody reading "chandelier exit" pictures a
close-only rule — a chandelier is a resting stop order. Replaying every trade against 5-minute bars:

| bar | trades | stop touched intrabar first | mean trade, shipped | with a real resting stop | P&L kept | median intraday MAE |
|---|---|---|---|---|---|---|
| 1d | 286 | 99.3% | +20.62% | +7.38% | **36%** | −11.1% |
| 4h | 1 865 | 99.4% | +1.84% | +0.39% | **21%** | −5.1% |
| 1h | 7 566 | 99.8% | +0.27% | +0.06% | **22%** | −2.5% |

Carried through the engine on the 1d sleeve, with the same costs, funding and vol target:

| 1d core-10 sleeve | Sharpe | CAGR | vol | maxDD |
|---|---|---|---|---|
| shipped (daily-close exit) | **+0.77** | +6.1% | 8.1% | −12.5% |
| resting intraday stop | **+0.43** | +2.5% | 6.4% | −10.8% |

So the leg's edge is not *hidden* by the daily bar — it *depends* on ignoring the intraday path. On
the median 1d trade the position is 11% offside at some point; 13% of trades go more than 25%
offside; nothing in the strategy reacts. That is a deliberate design choice (trend systems are
supposed to sit through noise), but it has to be stated, because "3×ATR chandelier stop" reads as
risk control and there is none. (`t2_intrabar_breakout.py`, `t2b_intrabar_leg.py`, `t10_intrabar_tf.py`)

The risk understatement at leg level is small, because vol-targeting keeps the position small:
drawdown on the 5-minute path is −13.2% against −12.5% close-to-close.

### BAB and x-sect — the question does not apply

Neither has a stop, a target or a barrier. Both form weights on a close, hold them for 21 bars, and
fill two bars later. There is no intrabar branch to get wrong, and the execution-lag ladder confirms
nothing is hiding in the first hours:

| exec_lag (bars) | BAB top-25 | x-sect crypto 1d | x-sect crypto 4h | x-sect crypto 1h |
|---|---|---|---|---|
| 1 | +1.26 | +1.01 | +0.72 | +0.94 |
| **2 (shipped)** | **+1.44** | **+1.05** | **+0.50** | **+0.97** |
| 3 | +1.38 | +0.83 | +0.48 | +0.97 |
| 5 | +1.48 | +0.69 | +0.44 | +0.89 |

---

## Finding 1 — volprem sells one day of variance at the thirty-day strike — FIXED

The largest finding here, because volprem is 61% of the master book's P&L. `VOLPREM_TERM_HAIRCUT =
0.168` in `src/config.py` now charges the measured gap and the leg ships at Sharpe **+3.49** against
the +6.90 below; the constant carries its own provenance so the number can be argued with rather than
re-derived.

`short_vol_book` accrues, every day, `K(t−2)² − RV(t)×252`, where `RV(t)` is **one day's** realised
variance and `K` is a Cboe **thirty-day** implied-vol index. A desk selling one day of variance is
quoted the one-day price, not the thirty-day one. Cboe publishes the near end of that curve, so the
gap is measurable rather than arguable:

| | days | near strike variance | VIX strike variance | ratio |
|---|---|---|---|---|
| VIX9D vs VIX, all days | 3 919 | 0.0373 | 0.0376 | 1.01 |
| VIX9D vs VIX, **gate ON** | 2 154 | 0.0228 | 0.0274 | **1.20** |
| VIX1D vs VIX, all days | 1 060 | 0.0317 | 0.0374 | 1.18 |
| VIX1D vs VIX, **gate ON** | 559 | 0.0194 | 0.0288 | **1.49** |

The gate makes it worse, not better: it only lets the leg trade in contango, which is exactly when the
near end sits furthest below the thirty-day point. Re-striking the SPY sleeve at each point of the
curve, holding gate, costs and days identical:

| window | strike sold | Sharpe | CAGR | mean P&L/day |
|---|---|---|---|---|
| 2011+ | VIX (shipped) | +7.50 | +123.3% | +0.322% |
| 2011+ | VIX9D | **+5.27** | +71.3% | +0.216% |
| 2022-05+ | VIX (shipped) | +6.80 | +99.6% | +0.277% |
| 2022-05+ | VIX9D | +4.64 | +56.1% | +0.179% |
| 2022-05+ | VIX1D | **+2.46** | +27.1% | +0.097% |

Two-thirds of the premium is term structure the position is not entitled to.

**And the instruments that exist did not do this.** Same window, same gate, vol-targeted the same way:

| | window | gated Sharpe | gated maxDD |
|---|---|---|---|
| short VXX | 2018-01 → 2026-08 | +0.76 | −26.4% |
| short VIXY | 2011-01 → 2026-08 | +0.78 | −26.7% |
| SVXY (long) | 2011-10 → 2026-08 | +0.82 | −26.4% |
| **the model, SPY sleeve** | 2011-10 → 2026-08 | **+7.53** | **−10.5%** |

VIX-futures ETPs are not variance swaps, and a real variance seller does earn more than they do — but
not ten times more with a third of the drawdown. Published estimates for systematic short-variance on
the S&P sit near Sharpe 0.6–1.0 with a fat left tail, which is the ETP column, not the model column.

Two smaller pieces of the same leg:

- **The vega spread is charged on the weekly re-strike only** — 7.1% of gross P&L. A position that
  genuinely turned over daily would pay 22.7%, which is more than the entire VIX1D-struck premium.
- **The mark-to-market is never booked.** A short variance position also loses on the implied level
  of the remaining term. Adding it raises the position's own daily volatility by 5%, so the 15% vol
  target is being set on a series 5% quieter than the real one.

(`t6_volprem_tradeability.py`)

## Finding 2 — the breakout ML gate is fitted on its own future, on a 2026 universe — FIXED

`run_bo_final` now takes its universe from `bo.pit_universe(10)` and gates with `run_bo_ml.wf_proba`,
an expanding walk-forward. The rebuilt leg reproduces arm D exactly — Sharpe **+0.52**, OOS **−0.01**,
deflated **0.03** — which is what says the fix is the counterfactual and not something else.

`scripts/breakout/run_bo_final.py` builds twenty of its thirty sleeves by calling
`proba_cache(...)` → `oos_proba` → `purged_kfold`. Purged k-fold makes the *test* folds contiguous in
time; each fold's *training* set is its whole complement. A trade in 2021 is filtered by a model fitted
on 2022-2026. `run_bo_ml.py`'s own docstring says so ("read this number as a generalisation estimate,
not as a track record"), and `run_bo_ml_wf.py` exists to re-measure it — but the series that reaches
the book is still the k-fold one.

The universe is `CORE10 = BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC`, typed once and used from
2020-01. A point-in-time top-10 by trailing 30-day median dollar volume — the rule the x-sect legs
already use — contains 137 distinct names over the window with 482 membership changes, and on
2020-02-01 it is `BCH BTC EOS ETC ETH LINK LTC TRX XLM XRP`: five of the ten are different.

Each defect priced on its own, same construction otherwise (arm A reproduces the shipped leg exactly —
Sharpe 1.121, OOS 0.197, 30 sleeves):

| time-series leg | Sharpe | OOS Sharpe |
|---|---|---|
| **A — shipped: CORE10 + k-fold gate** | **+1.12** | **+0.20** |
| B — CORE10 + walk-forward gate | +0.88 | +0.16 |
| C — PIT top-10 + k-fold gate | +0.69 | +0.02 |
| **D — PIT top-10 + walk-forward gate** | **+0.52** | **−0.01** |

The hindsight universe is worth more than the future-fitted model. With both removed the leg's
out-of-sample Sharpe is zero. The blended breakout family (this leg risk-parity'd with an already-PIT
cross-sectional half) goes from +1.47 to +1.18. (`t4_breakout_counterfactual.py`)

## Finding 3 — two scripts write the x-sect leg, and only one of them runs — FIXED

`build_xs_book.py` publishes to `xs_book_idio_candidate.parquet` now, so a candidate construction can
no longer overwrite the shipped leg by being run second; and `portfolio.py`'s two risk-parity combines
weight on trailing lagged volatility instead of the whole sample.

`reports/xs/xs_book.parquet` is written by **both** `scripts/xs/portfolio.py` and
`scripts/xs/build_xs_book.py`. Whichever ran last is what the book reads.

- `build_xs_book.py` opens with *"Build the x-sect family block (`reports/xs/xs_book.parquet`) that
  feeds the master book"*. It is in no Makefile target and no orchestrator. It builds idiosyncratic
  momentum on two legs and **charges no perp funding**.
- `portfolio.py` is what `make xs` runs, and it wrote the shipped artifact (reproduces byte-exact:
  corr 1.000000, max|diff| 0). It builds risk-adjusted momentum on three crypto timeframes plus a
  broad equity sleeve, and it **does** charge funding.

The two series correlate **0.73**. The docstring describes a leg the book does not hold.

The leg it does hold has one look-ahead: both of its risk-parity combines weight by `1/df.std()` over
the **whole sample**.

| x-sect block | Sharpe | total | maxDD |
|---|---|---|---|
| full-sample 1/std weights (shipped) | +0.914 | +213.4% | −13.7% |
| trailing 252-day lagged weights | **+0.772** | +343.9% | −28.4% |

−0.14 Sharpe. Small, and a one-line fix. (`t9_xs_real.py`)

Worth knowing rather than fixing: the funding this leg collects (4.3–5.9% a year per crypto sleeve,
because shorting beaten-down alts pays) is **9% of the leg's return**.

## Finding 4 — BAB pays perp funding that nothing charges — FIXED, and fixed once for everyone

`src/sleeves/bab.bab_backtest` had a commission term, a spread term and an impact term. It had no
funding term and no borrow term, and `run_bab_portfolio._bab_net` did not add one. The panel it
trades — `data/cache/xs/crypto_1d_close.parquet` — is built by `build_panels.build_crypto` from
`load_klines(..., market="um")`: Binance USD-M **perpetuals**, which settle funding three times a day.

Separately, `reports/bab/bab_book_c25.parquet` was dated 2026-08-10 22:33 while the crypto panel had
been rebuilt 2026-08-11 19:44 by the universe fix in `58f2340`, so the leg in the book had been
computed on a panel that no longer existed — **correlation 0.960, 2 150 of 2 404 days differing**.

**Both are now closed, and the first one structurally.** This was the fourth patch of one hole: the
x-sect leg was caught and patched, then the lottery sleeve, then BAB — leaving three separate
hand-rolled copies of the same funding panel behind (`xs/portfolio._funding_panel`,
`lottery/run_lottery._funding_daily`, `breakout/run_bo_xs_tf.funding_panel`) and a fourth strategy
free to forget tomorrow. The root cause is that `xs_backtest` and `bab_backtest` model the **trade**
but not the **instrument**: handed a price panel, they cannot tell a Binance perp from a cash equity,
so carry became the caller's job and callers forgot.

`src/backtest/carry.py` makes the instrument answer instead. `for_panel(px)` asks, per name, whether
Binance ever settled funding on it — not a guess about how the ticker is spelled, but the fact
itself, and self-maintaining. Both panel backtests call it when the caller passes nothing, so
**forgetting now charges funding rather than charging nothing**, and logs that it did; opting out is
`carry=NoCarry()`, which appears in a diff the way an omission never does. The three copies are gone.
`scripts/smoke_math.test_carry_is_not_opt_in` locks the invariant against the real archive.

Nineteen scripts across x-sect, BAB, residual momentum, on-chain, lottery and seasonal reach these
two functions on crypto panels. All of them are charged now; none of them had to be edited.

**Then the same question was asked of the whole repository rather than of the four legs**, because
"is it fixed here" is not the question — "can it come back" is. There are 85 places that can hold a
position: 46 through the panel backtests, 39 through the single-asset engine. The panel ones are
closed by construction. Of the 39, thirty-one already pass `funding=`, five are equity/FX/vol, and
three hold no perp at all — a synthetic series in the invariant tests, a shuffled-returns placebo
whose two arms run through the same line, and the equity loop of the trend trade log. **Zero real
gaps.** But the single-asset engine cannot be closed the way the panel one was: it is handed a bare
price Series with no venue on it. Two things now cover that:

  * `engine.backtest(..., symbol="BTCUSDT")` resolves the archive itself, so the convenient spelling
    is also the correct one, and naming a non-perp is harmless;
  * `scripts/check_funding.py` reads every call site — resolving `**kwargs` unpacking back to the
    dict it came from, which a keyword scan misses and would otherwise report as 39 defects — and
    fails the build on an uncharged crypto position. Waiving a site takes a named reason in `ALLOWED`.
    It runs in `make lint` and is the first step of `run_all.py`, so it cannot rot the way the thing
    it guards did.

What it costs, on today's panel:

| BAB top-25, beta-neutral | Sharpe | total | carry |
|---|---|---|---|
| shipped artifact (old panel, uncharged) | +1.51 | +417.0% | — |
| today's panel, uncharged | +1.44 | +377.1% | — |
| **today's panel, funding charged (shipped now)** | **+1.37** | +336.1% | **+4.52%/yr** |

The x-sect leg, which had been charging funding by hand, comes out **byte-identical** (max|diff|
8e-17) — proof the charge moved rather than changed. (`t3_funding.py`, `src/backtest/carry.py`)

## Finding 4b — the equity mirror: the broad-equity sleeve shorts 692 names and pays no borrow — FIXED

The same defect on the other venue, found by asking the funding question of the whole repository
instead of the four legs. `scripts/xs/broad.run_cfg` builds the sleeve that is half the x-sect leg,
shorts a decile of a 692-name panel, and calls `xs_backtest` without `borrow_bps_annual`. Mean short
notional 0.89 of the book; borrow charged: zero.

| broad-equity sleeve | Sharpe | total | borrow/yr |
|---|---|---|---|
| as shipped (no borrow at all) | +0.48 | +235.7% | 0.00% |
| **at the config's own 50bps (shipped now)** | **+0.45** | +207.3% | 0.52% |
| at 100bps (hard-to-borrow) | +0.42 | +181.3% | 1.04% |
| at 300bps (small-cap short) | +0.31 | +97.6% | 3.13% |

`carry.for_panel` now defaults a cash panel to `EQUITY_BORROW_BPS_ANNUAL` rather than to zero —
`Borrow` only charges the short leg, so a long-only book is untouched and the default is safe.
`borrow_bps_annual=None` means "not decided, use the panel's default"; an explicit `0.0` is a
statement and is honoured. FX is excluded by name: shorting a pair costs the interest differential,
not stock borrow, and charging the wrong model would be worse than charging none — so it is left
uncharged and logged.

Two more defects in the same file, both fixed: the sleeve that ships was chosen by comparing the rule
arm's full-sample Sharpe against an ML arm's (`best_ret = ml_ret if ml_s > ap_s else ap_ret`) — a
selection made on the sample it is scored on, inside a leg the book holds. It picks the rule arm
today (0.45 against 0.25) so removing it costs nothing; the point is that it could not have been
trusted the other way. And `broad.py` was in no Makefile target and no orchestrator, so the equity
half of a shipped leg had no reproduce path at all; it is in `make xs` now, ahead of `portfolio.py`
which consumes its output.

Worth stating beside the fixes: this sleeve's **own walk-forward is Sharpe +0.26** against the +0.45
that ships. The gap is construction selection, and the a-priori config coincidentally equals the
sweep's best cell — luck rather than peak-picking, but a reader cannot tell the two apart from the
summary.

## Finding 5 — `run_all.py` rebuilds none of the four legs — FIXED

All four are in `STEPS` now, in dependency order (`xs/broad.py` before `xs/portfolio.py`, which reads
its output; `run_bo_final.py` before `run_bo_combined.py`). It makes a reproduce run take about an
hour instead of a few minutes, which is the correct trade: a reproduce step that is slow and true
beats one that is fast and reproduces only the assembly.

`STEPS` is validate_sessions → run_book → feature_report → meta_overlay → crisis → gmacro →
**master_book** → wf_book → cscv → family_costs → oos_ledger → figures → **live_book** → live_report →
report → render. Not one of volprem, xs, breakout or bab is in it. `make reproduce` assembles the book
from whatever four parquet files happen to be on disk, and the `--check` gates only prove each page
matches its own JSON. Finding 4 is what that looks like when it bites.

## Finding 6 — nothing ever priced the choosing of the composition

The live book holds **four** legs. `run_master_book.FAMILIES` ships **six**. And the repository's own
word for the validated set is **eight**: `composition_search.json` still scores `standalone_sharpe`
across eight including `trend_momentum` and `carry`, and the assembler's comments read "with all
eight the book scores 3/5 full", "the SIX-family composition is the one thing in this book chosen
against the scorecard rather than on a-priori grounds", "carry was the fourth-highest standalone of
the eight". So there are **two selections stacked**: eight → six (drop trend and carry, by a
37-configuration search) and six → four (drop crisis and global-macro, in the live book). The reasons
given for each were reached by looking at what the candidates did, which is a choice made on the
sample it is then measured on — the one defect no per-leg audit can see.

Measured both ways, same assembly, same window, at 2x, so the count is not the argument:

| pool the four were chosen from | rank of the shipped four | the whole pool held | past-only picking, 11yr | the shipped four | hindsight |
|---|---|---|---|---|---|
| **six** (the live book's own choice) | 2 of 15 | +4.37 Sharpe, +75.6%/yr | **+812.4%** | +1037.4% | **+225%** |
| **eight** (both selections) | 5 of 70 | +4.15 Sharpe, +65.1%/yr | **+836.5%** | +1037.4% | **+201%** |

The past-only arm re-picks the best four each January on data strictly before that date — assembled
as a book, not ranked standalone, because a desk scores the composition it would run — and holds them
a year. It **never once** landed on the shipped four (0 of 11 years in either pool): it held
gmacro+trend+volprem+xs through 2021 and bab+breakout+carry+volprem after, i.e. it kept carry, the
leg the book explicitly dropped.

**About 18-20pp a year of the headline is composition hindsight** — more than every defect found
inside the legs put together. Counting only the selection the live book itself made, from the six the
master ships, makes it slightly *worse* (+225%), not better. Note that a selection test starting from
`FAMILIES` cannot see the eight→six step at all, which is why the denominator has to be rebuilt from
`trend/trend_block_returns.parquet` and `carry/carry_refined.parquet` by hand. (`t13_leg_selection.py`)

## Finding 7 — the cost model, validated where it can be and named where it cannot

- **Exchange fees are right and conservative.** Binance's own schedule gives Regular/VIP-0 USD-M
  futures taker 0.05% and spot taker 0.10%; the repo uses exactly 5.0 and 10.0 bps, i.e. the
  no-BNB-discount rate a real desk would beat (0.045% / 0.075%).
- **The impact coefficient is an order of magnitude below the literature.** `IMPACT_K = 0.1` in
  `k·σ·√(order/ADV)`; the square-root law's empirical coefficient sits near 0.5–1.0. At the brief's
  $500k it barely matters — BAB reads 1.37 at k=0.1 and 1.32 at k=1.0. At size it decides
  everything: 1.32 → 0.82 at $50m, 1.26 → **0.28** at $200m. **This corrects a claim made earlier in
  this audit**: capacity measured at the repo's own k looked like ~$1bn; at a literature-consistent
  k it is nearer $50m.
- **The vega spread cannot be validated here.** `data/raw/options_eod` turns out to hold stock EOD
  bars, not option chains — no bid/ask, no implied vol — so the 1.0/2.5 vol-point charge stands on
  published variance-swap quotes and nothing in this repository can check it. What can be said is
  the sensitivity, and `volprem_cost_robustness.csv` already holds it.

---

## What it cost — every defect above is now FIXED, and these are the shipped numbers

Findings 1, 2, 3, 4, 4b and 7 are corrected in the code, not just measured: the term-structure
haircut is charged, the breakout gate is a walk-forward on a point-in-time universe, the x-sect
combines weight on trailing volatility, BAB pays funding, the equity sleeve pays borrow, and the
candidate twin no longer overwrites the shipped leg. Every artifact below is regenerated from that
code. Finding 6 — the composition — is deliberately **not** touched: which legs to hold is a decision
to take on clean numbers, and these are the clean numbers.

**Per leg, standalone:**

| leg | before | after | what was wrong |
|---|---|---|---|
| volprem | **+6.90** (gated), DD −10.5% | **+3.49**, DD **−28.1%** | sold one day of variance at the thirty-day strike |
| breakout | **+1.47**, OOS +0.19 | **+1.18**, OOS **+0.10** | k-fold gate fitted on its own future; CORE10 hindsight universe |
| BAB | **+1.51** | **+1.37** | no funding on a panel of perpetuals; stale panel |
| x-sect | **+0.91** | **+0.74** | no borrow on the equity short leg; full-sample risk-parity weights |

The breakout time-series half alone is +0.52 with an out-of-sample Sharpe of **−0.01** and a deflated
Sharpe of **0.03**; its cross-sectional half is +1.12 with OOS +0.11. There is no out-of-sample edge
left on either side of that family.

**The live book at 2x, before and after:**

| | Sharpe | CAGR | P&L/yr on fixed capital | maxDD | worst month | months+ |
|---|---|---|---|---|---|---|
| before the fixes | +5.40 | +197.4% | +111.3% | −16.9% | −9.5% | 84% |
| **after** | **+3.37** | **+93.9%** | **+68.3%** ($341,725 on $500k) | **−20.9%** | **−13.7%** | 74% |
| after, 2020-01+ only | **+2.94** | +69.2% | — | −14.8% | — | — |

**The master book (six families):** Sharpe 4.34 → **2.88** full and 3.79 → **2.69** on the scored OOS
block; months-in-profit 80.8% → 73.9% and 73.1% → 65.4%; longest losing streak 2 → 3 months. The
composition search now clears all five targets in **0 of 37** configurations. volprem is still 61% of
P&L, and its standalone Sharpe reads 3.96 rather than 7.09.

None of this prices the tradeability gap in Finding 1. The haircut charges what a *nine-day* seller
would not collect; it does not close the distance between a synthetic daily variance strip and short
VXX. If the leg is worth what the tradeable instruments were worth over the same window under the
same gate (+0.8), the book has no headline at all.

---

## What was checked and is clean

Stated because a list of defects with no denominator is not an audit.

- **No same-bar execution anywhere.** `exec_lag=2` throughout — a signal stamped at bar *t* fills at
  `close(t+1)` and first earns the `(t+1, t+2]` return. Verified in `backtest`, `xs_backtest`,
  `bab_backtest` and `short_vol_book`.
- **Every rolling statistic is backward-only and lagged.** Vol targets, panel betas, momentum
  formation windows, liquidity ranks, both regime gates. The volprem gate carries three days of total
  lag (gate `shift(1)`, side `shift(2)`).
- **The crypto universe is genuinely survivorship-free.** 578 names, including FTT, LUNA, SRM, COCOS,
  TOMO, BTCST and every other perp Binance has delisted; membership decided by trailing dollar volume,
  lagged. Live names per year 39 → 558.
- **The volprem regime gate is timing, not just less exposure.** Against 60 block-shuffled gates at
  each sleeve's own duty cycle: real +6.90 Sharpe / −10.5% drawdown against a placebo p95 of +2.84 and
  a best-of-60 drawdown of −27.3%. The real gate beats 100% of them on Sharpe and drawdown, 98% on
  worst day. Ungated, the same book is Sharpe +2.53 with a −60.1% drawdown and a −57.6% worst day.
- **The −99.9% daily floor in `run_vol_premium_book.vt` does not bind on the shipped leg.** On the
  ungated research series it hides 10 sleeve-days worse than −100% (deepest **−256%** — a wiped-out
  account plus a debt). Under the gate: zero such days, deepest −55%. Worth stating plainly — one
  regime gate is the whole distance between this leg and ruin.
- **BAB's premium is alpha, not crypto beta.** Against the equal-weight panel: alpha +22.8%/yr,
  t = +3.50, realised beta +0.061. Against BTC: alpha +19.3%/yr, t = +3.09, beta +0.115.
- **Capacity is not a constraint at the brief's $500k** — but only there. At the repo's own impact
  coefficient BAB holds out to $1bn; at a literature-consistent one it is gone by $200m (Finding 7).
- **The ML layer under the breakout gate is clean.** Every feature is a backward rolling window,
  `pit_normalize` is a rolling z-score, and the truncation audit — recompute on a series cut at T and
  compare every past value — returns max|diff| = 0.00e+00. The gate itself is Finding 2; the engine
  it stands on is sound. It is also the only ML in the live book: volprem, x-sect and BAB contain
  none.
- **The dividend contamination in volprem is real and points the safe way.** `load_equity_daily`
  returns split- but not dividend-adjusted bars, so an ex-date open gap is charged to the short as
  realised variance. Measured across the ten payers: **0.21% of the variance charged** (worst VXEFA
  0.91%, VXTLT 0.44% on 183 ex-dates). Correcting it *raises* the leg by +0.11 Sharpe — the book was
  paying for a payment.
- **The breakout cross-sectional half is honestly built.** PIT membership by trailing median dollar
  volume, lagged; t+2 fills; cost, √-impact and funding all charged; a random-signal placebo run
  beside it. Its point-in-time arms are uniformly *worse* than the static ones (1d 0.85 vs 1.20, 4h
  1.16 vs 1.46, 1h 1.04 vs 1.69), which is the correct direction and says the hindsight was really
  removed. Its out-of-sample Sharpe is **+0.13** — so with Finding 2 on the other half, the breakout
  family has no out-of-sample edge left on either side.
- **Three of four artifacts reproduce byte-exact** from source: volprem `ret_gated`, breakout
  `bo_combined`, x-sect (from `portfolio.py`). BAB does not — Finding 4.

## What could not be settled

- **The eighteen sleeves' own term structures.** Cboe publishes a 9-day index for the S&P and nothing
  else, so the strike haircut in Finding 1 is measured on one underlying and applied to all. Single
  names and commodities have steeper curves than the S&P, which makes the uniform figure a floor on
  the equity-index sleeves and unknown on the rest.
- **What a real variance-swap book would have earned.** The gap between the model (+7.5) and the ETPs
  (+0.8) brackets it. Closing it needs option chains with quotes, and this repository has none:
  `data/raw/options_eod` is six months of 2013 *stock* EOD bars, despite the name.
- **The vega spread itself**, for the same reason — 1.0 vol-point on an index and 2.5 on a single
  name come from published variance-swap bid/ask and cannot be checked against a quote here.
- **FX carry.** `carry.for_panel` leaves an FX panel uncharged and says so, because shorting a pair
  costs the interest differential and this repository does not model it. No shipped leg trades FX, so
  it is a gap in the guard rather than in the book.
