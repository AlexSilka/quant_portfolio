# Calendar seasonality (H4) — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.76** full / **3.61** OOS). Any master-book Sharpe quoted below is the book *snapshot at the time this family was evaluated*, not the current headline.

**Scope.** HYPOTHESES.md **H4**: two documented, calendar-deterministic effects — the **pre-FOMC
announcement drift** (Lucca-Moench 2015) and the **turn-of-month** effect (Lakonishok-Smidt 1988) —
run through the same funnel as every other family (hold-through-window cost model, in/out-of-window
decomposition, per-year + decay split, shuffled-**calendar** placebo, cost sensitivity, block-bootstrap
MC, purged walk-forward OOS, deflated Sharpe, correlation to the deliverable book + lift curve). All
numbers net of costs. Windows: pre-FOMC day-before, turn-of-month (−1,+3). Equity ETFs daily 2011→2026
(FOMC-date coverage floor), crypto 2020→, FX 2012→. Figure:
[reports/figures/seasonal.png](../../reports/figures/seasonal.png). Reproduce: `make seasonal`.

---

## 0. TL;DR

- **Both effects are real in the data, but neither is a tradable decorrelated sleeve. H4 joins the
  overnight/session family in the "real-as-beta, not net-of-cost alpha" pile** — a rigorous edge-map
  entry, not a hidden gap. The correct portfolio decision is to **exclude** it.
- **The pre-FOMC drift has a clean economic signature** — SPY drifts **+8.7bps** the day before a
  scheduled announcement and **+7.5bps** on announce day, then **fades −16.9 / −15.8bps** the two days
  after (run-up-into-the-meeting, sell-the-news). In-window Sharpe is high (**+1.25** vs +0.75 the rest
  of the time). But as a standalone *timing* book it nets only **+0.05 to +0.13** across SPY/QQQ/IWM/DIA:
  a one-day hold pays a full round-trip for ~8 events/year, and — decisively — it **does not beat a
  shuffled-calendar placebo** (real at the 63rd–74th percentile of random-day sets; noise's 95th
  percentile +0.36 > the real +0.07). In a bull market being long *any* small set of days pays.
- **The one genuinely significant precise result is in crypto.** BTC's exact 24h window into the 2pm-ET
  announcement returns **+102bps on average (t = +2.4, 61% hit, n=56)** — a real, statistically
  significant pre-FOMC risk-on drift. It is still not a deployable levered sleeve (8 events/yr, funding
  drag, doesn't clear the placebo as a book), but it is the strongest single finding and the honest
  edge-map entry for crypto macro-event risk.
- **Turn-of-month is beta, not a turn-of-month edge — and the parameter surface proves it.** The net
  Sharpe **rises monotonically as the window widens** (SPY (−1,+1) 0.08 → (−4,+5) 0.77), converging on
  buy-&-hold SPY (0.76). The *classic* tight (−1,+3) window nets **+0.29 and underperforms buy-&-hold**;
  it sits at the **57th** placebo percentile (random 4-day windows average +0.25). Widening the window
  just walks you toward being permanently long — i.e. harvesting market drift, not a calendar anomaly.
  Crypto turn-of-month is **dead** (BTC net −0.01, 32nd placebo percentile); FX is weak (+0.13).
- **Cross-sectional breadth confirms "market-timing, not name-selection":** the stock turn-of-month book
  is **flat at ~0.29 whether you hold the top 50, 100, 200 or 500 names** — basket size does not matter,
  so there is no cross-sectional signal, only broad exposure on certain days.
- **Decorrelated (+0.18 to the book) but sub-bar, so it drags** (3.47 → 3.39 → 3.16 at 0/15/30% weight).
  Deflated Sharpe 0.31, MC-P5 −0.11. Correctly excluded from the portfolio.
- **Trading it market-neutral *between* assets does not rescue it, and ML does not either** (§5-6). A
  dollar-neutral long/short across names, live only in-window, is negative or sub-bar (crypto pre-FOMC
  −0.47, stocks turn-of-month −0.41 below-random, crypto turn-of-month +0.36 at the 96th placebo pctile
  — one marginal near-miss, still sub-0.5). Three ML variants — a conditional pre-FOMC gate (VIX / 10y-2y
  slope / trailing drift, purged CV) and a cross-sectional LGBM ranker — all **make it worse** (SPY gate
  0.24→0.07 with negative OOS IC; the in-window ML book −1.44 vs −0.94 all-days). Removing the beta leaves
  ~nothing, confirming the effect is uniform beta, not a cross-sectional or conditionally-timeable edge.

---

## 1. Construction — why the execution model is different (and honest)

Two effects, one engine (`src/sleeves/seasonal.py`); the event calendar is in `src/data/fomc.py`
(scheduled FOMC announcement dates 2011-2026, sourced from the Fed's own calendars).

- **pre-FOMC drift.** Anchor = the announcement day; the tradable window is the trading day(s) leading
  into it. The 24h Lucca-Moench window (2pm ET the day before → 2pm ET announce day) is measured two
  ways: a **daily** proxy (the close-to-close bar of the day before, deep 2011→ history) and the
  **precise** intraday window (5-min/1h, `asof` prices around the DST-correct 14:00-ET timestamp, 2020→).
- **turn-of-month.** Anchor = each month-end bar; window = the last `days_before` + first `days_after`
  trading days (classic (−1,+3)).

**Why not the cross-sectional engine.** `xsect.py` books estimate a signal from market data, so they
carry a t+2 delay to avoid look-ahead. A calendar window is **known years in advance** — there is no
signal to estimate. So (a) there is no estimation look-ahead, and (b) the honest execution model is
*hold through the multi-day window and pay commission+spread only at the edges* (one entry, one exit),
**not** the daily round-trip that killed the overnight sleeve. This is the whole reason H4 was worth a
separate test from overnight: the cost structure is genuinely more forgiving. A fill-timing robustness
(shift the whole window one bar later) is run to prove the effect is not an artifact of *when* the bar
is priced. Crypto books are on perps and **charged realised funding** while long (BTC/ETH ~+12%/yr).

## 2. Pre-FOMC drift — a real shape, killed by cost and capacity

**Where the drift lives (SPY, mean return by trading-day offset from the announcement):**

| offset | −3 | −2 | **−1 (day before)** | **0 (announce)** | +1 | +2 |
|---|---|---|---|---|---|---|
| mean bps | −2.0 | −2.8 | **+8.7** | **+7.5** | −16.9 | −15.8 |

The run-up is sharply concentrated in the 48h into the meeting and reverses hard afterward — exactly the
documented pattern. In-window (day-before) Sharpe is **+1.25** vs +0.75 out-of-window; QQQ **+1.42**, DIA
**+1.54**. The effect is real.

**But it does not survive as a timing sleeve** (day-before book, net of 3bps/side):

| ETF | net Sharpe | in-window mean | placebo pctile | 2011-17 | 2018-26 | fill-shift +1 |
|---|---|---|---|---|---|---|
| SPY | **+0.07** | +8.7bps | 63rd | −0.16 | +0.27 | +0.03 |
| QQQ | +0.13 | +12.4bps | 66th | −0.13 | +0.29 | +0.31 |
| IWM | +0.05 | +8.7bps | 65th | +0.04 | +0.07 | — |
| DIA | +0.11 | +10.1bps | 74th | −0.13 | +0.30 | — |

- **Placebo is the decisive test.** A random set of 125 days with the identical one-day window shape
  averages ~0.0 net Sharpe with a **95th percentile of +0.36** — above the real +0.07. The pre-FOMC day
  is not statistically distinguishable from being long a random handful of days. The binding problem: a
  **1-day hold pays a full round-trip** (~6bps) for a ~9bps gross edge, ~8 times a year → almost nothing
  survives, and what does is not separable from base drift.
- **Not the documented decay, but not classic strength either.** Over 2011-2026 the day-before book is
  ~0 overall, negative 2011-17 and modestly positive 2018-26. The classic *strong* era (the +49bp/24h,
  ~80%-of-annual-return figure) is Lucca-Moench's pre-2011 sample, before this data begins; by 2011 it
  had already faded (consistent with "The disappearing pre-FOMC drift", 2020). Fill-timing sensitivity
  (SPY day-before +0.07 but shift-one-bar +0.03; QQQ +0.13 vs +0.31) confirms the daily edge is thin
  enough that *which* bar you price matters more than the signal.

**Precise intraday window (2020→, the exact 24h→2pm-ET):** SPY +8.6bps (t=+0.5), QQQ +23.8bps (t=+1.2),
IWM +25.9bps (t=+1.2), DIA −2.2bps (t=−0.1) — all **insignificant** on the 55-event post-2020 sample.

**Crypto analogue (2020→, funding-charged) — the strongest result:** BTC day-before **+37.4bps**,
in-window Sharpe **+2.06**; the exact **hourly 24h→2pm-ET window returns +102.5bps, t = +2.4, 61% hit
(n=56)** — a genuinely significant pre-FOMC risk-on drift, stronger than in equities (crypto is the most
rate-sensitive, 24/7 risk asset). ETH echoes it (+31.2bps, in-window +1.37). Net of 6bps/side + funding
the *books* are +0.22 (BTC) / +0.13 (ETH) and still sit at the 74th/54th placebo percentile — real
event, not a deployable levered sleeve.

## 3. Turn-of-month — the parameter surface exposes it as beta

**Net Sharpe by window shape (the (days_before × days_after) grid):**

| | +1 | +3 | +5 | | | +1 | +3 | +5 |
|---|---|---|---|---|---|---|---|---|
| **SPY −1** | 0.08 | 0.29 | 0.42 | | **BTC −1** | 0.15 | −0.01 | 0.06 |
| **SPY −2** | 0.39 | 0.51 | 0.60 | | **BTC −2** | 0.57 | 0.34 | 0.34 |
| **SPY −4** | 0.63 | 0.70 | **0.77** | | **BTC −4** | 0.74 | 0.53 | 0.50 |

The SPY surface **increases monotonically in window width**, converging on buy-&-hold SPY (**0.76**). If
turn-of-month were a concentrated anomaly the *tight* window would dominate and widening would dilute it;
instead widening improves it, because a wider window = more days long = more of the market's general
uptrend. The classic (−1,+3) window (**0.29**) actually **underperforms buy-&-hold**. Time-series summary:

| asset | in-window Sharpe | out-of-window | net (−1,+3) | placebo pctile | placebo mean |
|---|---|---|---|---|---|
| SPY | +0.88 | +0.74 | +0.29 | 57th | +0.25 |
| BTC | +0.35 | +0.76 | −0.01 | 32nd | +0.13 |
| FX basket | +0.49 | +0.35 | +0.13 | 60th | +0.09 |

SPY's in-window Sharpe (+0.88) is barely above out-of-window (+0.74) and its net book (+0.29) barely
beats random 4-day windows (+0.25 mean, 57th pctile). **BTC turn-of-month is dead** (in-window *worse*
than out-of-window; below-random placebo). The captured share of total return is only 21.8% in 19% of
days — a ~1.1× over-representation, not the concentration a real anomaly shows.

**Cross-sectional breadth (top-N, the requested 10/50/100 cut):**

| universe | top-10 | top-50 | top-100 | top-200 | top-500 |
|---|---|---|---|---|---|
| **US stocks** | — | 0.28 | 0.29 | 0.29 | 0.29 |
| **crypto** (incl. funding) | 0.06 | 0.11 | 0.22 | 0.29 | — |

The stock book is **flat across basket size** — hold 50 names or 500, the same ~0.29 — so there is no
cross-sectional signal, only broad market exposure on turn-of-month days. Crypto *rises* with N (smaller
baskets carry more idiosyncratic noise and funding drag), never clearing 0.3.

## 4. Portfolio value — decorrelated but sub-bar, so it drags

The combined SPY calendar sleeve (pre-FOMC ∪ turn-of-month, in market 22% of days) nets **+0.32**
[MC P5 −0.11, P50 +0.31], maxDD −18.9%, **deflated Sharpe 0.31** over the 13 window-shape trials. A
purged walk-forward over that grid returns **+0.53** — but this is the same beta artifact: the WFO simply
selects the *widest* (most-long) windows in a bull market, which is why it exceeds the fixed book and why
the placebo and deflated-SR (which hold exposure constant / correct for trials) both say sub-bar. It is
**decorrelated** (corr to book +0.18; only trend +0.31 is non-trivial, as both are long-biased), but at a
Sharpe far below every book leg (0.9–3.8) it **drags** on inclusion:

| calendar weight | 0% | 15% | 30% | 50% |
|---|---|---|---|---|
| book Sharpe | 3.47 | 3.39 | 3.16 | 2.54 |

Buy-&-hold SPY (0.76) beats the calendar book (0.32) over the same window — the cleanest statement that
this is a low-capacity *slice of beta*, not timing alpha. Excluded, as the canonical book does.

## 5. Cross-asset relative-value — trading it *between* assets (does removing the beta rescue it?)

Since the headline problem is that the premium is *beta*, the direct fix is a **dollar-neutral long/short
across names, live only inside the window**: long the names that respond MORE to the event, short those
that respond LESS, ranked on each name's own trailing in-window history (a "seasonal-momentum" signal
over the past 8 episodes). This removes the market factor by construction — if a cross-sectional spread
survives, it is a real relative-value edge; if not, the effect is uniform beta. Full round-trip charged
each episode (the book is flat between windows). Placebo = shuffle the seasonal signal across names.

| book | universe | net Sharpe | placebo pctile (p95) | corr-to-book |
|---|---|---|---|---|
| crypto **pre-FOMC** L/S | 162 names, 53 events | **−0.47** | 60th (p95 +0.00) | −0.12 |
| crypto **turn-of-month** L/S | 219 names, 79 months | **+0.36** | **96th** (p95 +0.35) | +0.02 |
| stocks **turn-of-month** L/S | 692 names, 200 months | **−0.41** | 18th (below random) | −0.00 |

Removing the beta **removes the return**: two of three are negative and the stock book is *below* its
random-shuffle placebo. The one exception — crypto turn-of-month L/S at **+0.36**, just past the 95th
placebo percentile — is a genuine whisper of cross-sectional structure, but it is **sub-bar (< 0.5)** and
carries the heavy per-episode round-trip of a long/short book run 12×/yr. So "combine several assets and
trade between them" does not convert the calendar effect into a market-neutral edge here: there is almost
no cross-sectional spread to harvest once the shared market move (the whole effect) is netted out.

## 6. ML variations — does conditioning or learning-to-rank rescue it?

Three leakage-controlled ML tests (an ML-forecast layer applied to the calendar family):

- **Conditional pre-FOMC gate.** Lucca-Moench show the drift is larger when the yield curve is flat /
  implied vol is high / recent drift was high. Per event, features known 2 days before the announce (VIX
  level + 5d change, 10y-2y slope, trailing-3-event drift, 20/60d momentum, 20d realised vol) → predict
  the outcome (Ridge / Logistic / LGBM) under a **purged event K-fold**, and trade only predicted-positive
  events. Result: **gating makes it worse, not better** — SPY unconditional +0.24 → best conditional
  **+0.07** (ridge OOS IC **−0.45**); BTC +0.28 → **−0.19**. The conditioners carry no out-of-sample
  predictive power on this sample; the in-sample relationships do not survive purging.
- **Cross-sectional LGBM learning-to-rank** (the sleeve's `xsect_ml` stack: 20-feature name panel,
  expanding walk-forward OOS, cross-sectionally-demeaned forward-return target) traded **all days vs
  in-window only**. The window is **not** a better time to run a cross-sectional book: **−0.94 all-days →
  −1.44 in-window**. (The negative *level* is the known daily-rebalance turnover-cost drag on a crypto
  cross-sectional ranker, matching the repo's cross-sectional-reversal-dead result; the point here is the
  *relative* comparison — conditioning on the calendar does not help.)

ML does not rescue the family. It confirms the rule-based read: there is no conditional or cross-sectional
structure that survives out-of-sample once look-ahead is controlled.

## 7. Honest verdict & ceiling

- **Reachable here:** nothing market-neutral or deployable — as a time-series timing book, as a
  cross-asset long/short, or with ML. Both effects are real (the pre-FOMC run-up / post-FOMC fade is
  textbook; crypto's 24h pre-FOMC drift is significant at t=2.4), but net of cost they are beta-timing,
  not alpha: they do not beat a shuffled-calendar placebo, they improve as you widen toward buy-&-hold,
  removing the beta removes the return, ML gating hurts, and they are low-capacity (~8 FOMC events/yr).
- **Binding constraints:** (1) capacity — a handful of event-days a year caps the effect's contribution;
  (2) beta — being long on a subset of up-drifting days is not separable from the drift itself; the
  placebo and the monotonic width-surface both prove it; (3) cost on the short single-day pre-FOMC hold.
- **What did not work (kept, not hidden):** both effects, four ETFs, the precise intraday window, the
  crypto analogue, the (before×after) grid, four cross-sectional basket sizes, the combined sleeve, the
  cross-asset relative-value long/short (§5) and three ML variants (§6). The value delivered is the
  **map** (H4 covered, and the genuine crypto pre-FOMC risk-on drift located for future event-study work)
  and the **methodology** (a placebo, a parameter-surface shape, a beta-removing cross-sectional test, and
  a purged ML gate that all cleanly separate a real *effect* from a tradable *edge*), not a sleeve.

## 8. Reproduce

```bash
make seasonal   # run_seasonal.py            -> reports/seasonal/seasonal_{summary.json,returns.parquet,tom_grid.csv}
                #                               + reports/figures/seasonal.png
                # run_seasonal_xasset_ml.py  -> reports/seasonal/seasonal_xasset_ml_summary.json
                #                               + reports/figures/seasonal_ml.png
```

Fixed seed (7). FOMC dates: `src/data/fomc.py` (Fed FOMC calendars, scheduled meetings only). Sources:
Lucca & Moench, "The Pre-FOMC Announcement Drift" (JF 2015); Lakonishok & Smidt, "Are Seasonal Anomalies
Real?" (RFS 1988); Kim & Suh / Vähämaa, "The disappearing pre-FOMC announcement drift" (Fin. Res. Lett.
2020).
