# Auditing the four legs the live book actually holds

`scripts/run_live_book.py` runs four families — volprem, breakout, BAB, x-sect — at 2x, and prints
Sharpe 5.40, 111.3% a year on fixed capital, a −16.9% drawdown. A Sharpe above 4 on a book of
well-known risk premia is a claim about the modelling, not about the premia. This is what happens to
each of the four when the modelling is taken apart.

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

## Finding 1 — volprem sells one day of variance at the thirty-day strike

This is the largest finding in the book, because volprem is 57% of its P&L.

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

## Finding 2 — the breakout ML gate is fitted on its own future, and its universe is the 2026 top ten

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

## Finding 3 — two scripts write the x-sect leg, and only one of them runs

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

## Finding 5 — `run_all.py` rebuilds none of the four legs

`STEPS` is validate_sessions → run_book → feature_report → meta_overlay → crisis → gmacro →
**master_book** → wf_book → cscv → family_costs → oos_ledger → figures → **live_book** → live_report →
report → render. Not one of volprem, xs, breakout or bab is in it. `make reproduce` assembles the book
from whatever four parquet files happen to be on disk, and the `--check` gates only prove each page
matches its own JSON. Finding 4 is what that looks like when it bites.

---

## What the book costs once the measurable defects are corrected

Each correction applied alone and then together, assembled exactly the way `run_live_book.py` does at
HEAD (legs held through the days their own market is shut, assembly re-sizing charged). The volprem
correction is a **stress, not a measurement** — Cboe publishes a 9-day index for the S&P only, so the
measured 30d/9d ratio is applied uniformly to all eighteen sleeves — and is therefore shown as a
ladder.

**Full window, 2011-01-03 → 2026-08-05, at 2x:**

| book | Sharpe | CAGR | P&L/yr on fixed capital | maxDD | worst month |
|---|---|---|---|---|---|
| **as shipped** | **+5.40** | +197.4% | **+111.3%** | −16.9% | −9.5% |
| only x-sect fixed (causal weights) | +5.41 | +204.3% | +113.7% | −16.9% | −9.5% |
| only BAB fixed (funding) | +5.38 | +195.1% | +110.5% | −16.9% | −9.5% |
| only breakout fixed (PIT + walk-forward) | +5.34 | +193.9% | +110.1% | −16.9% | −11.0% |
| volprem strike −5% | +4.90 | +165.7% | +100.0% | −18.1% | −10.8% |
| volprem strike −10% | +4.34 | +135.5% | +87.8% | −19.3% | −12.1% |
| only volprem fixed (measured −16.8%) | +3.50 | +97.2% | +70.0% | −20.9% | −13.8% |
| **all four fixed** | **+3.39** | +94.3% | **+68.5%** | −20.9% | −13.8% |

**Where all four legs are live, 2020-01+:**

| book | Sharpe | CAGR | P&L/yr | maxDD | worst month |
|---|---|---|---|---|---|
| as shipped | +4.45 | +127.3% | +84.0% | −12.0% | −8.3% |
| all four fixed | **+2.95** | +69.4% | +54.4% | −14.8% | −11.6% |

A 5% haircut on one leg's strike — a fifth of the gap that was actually measured — costs 11 points of
annual P&L. That sensitivity is the finding: the headline is not robust to the price of the instrument
it sells. (`t8_corrected_book.py`)

None of this prices the tradeability gap in Finding 1. If the volprem leg is worth what short VXX was
worth over the same window with the same gate, the book has no headline at all.

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
- **Capacity is not a constraint at the brief's $500k.** BAB holds Sharpe 1.44 → 1.20 out to $1bn;
  x-sect crypto 1d holds 0.90 → 0.51 at $200m and breaks at $1bn.
- **Three of four artifacts reproduce byte-exact** from source: volprem `ret_gated`, breakout
  `bo_combined`, x-sect (from `portfolio.py`). BAB does not — Finding 4.

## What could not be settled

- **The eighteen sleeves' own term structures.** Cboe publishes a 9-day index for the S&P and nothing
  else, so the strike haircut in Finding 1 is measured on one underlying and applied to all. Single
  names and commodities have steeper curves than the S&P, which makes the uniform figure a floor on
  the equity-index sleeves and unknown on the rest.
- **What a real variance-swap book would have earned.** The gap between the model (+7.5) and the ETPs
  (+0.8) brackets it; closing it needs option chains, and `data/raw/options_eod` starts in 2013.
