# Calendar seasonality (H4) — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the equal-weight master book assembled by `scripts/run_master_book.py`; its composition, scorecard, leverage and target verdict live in [REPORT.md](../../REPORT.md), which is RENDERED from the artifacts and so cannot disagree with the run. Restated here they would go stale the next time the book is re-run, which is exactly what happened to the numbers this line used to carry — so this page quotes none of them. Any master-book figure below is a snapshot from when this family was evaluated, and is labelled as one.

**Scope.** HYPOTHESES.md **H4**: two documented, calendar-deterministic effects — the **pre-FOMC
announcement drift** (Lucca-Moench 2015) and the **turn-of-month** effect (Lakonishok-Smidt 1988) —
run through the same funnel as every other family (hold-through-window cost model, in/out-of-window
decomposition, per-year + sub-period splits, shuffled-**calendar** placebo, cost sensitivity,
block-bootstrap MC, purged walk-forward OOS, deflated Sharpe, correlation to the deliverable book +
lift curve), plus a **309-arm variant sweep** over the four axes the family is actually free in: which
bar the window is anchored on, which side of the event is traded, which assets and how many, and which
timeframe. All numbers net of costs, dividend-inclusive, 2005→2026 (equity ETFs), 2020→ (crypto).
Figures: [seasonal.png](../../reports/figures/seasonal.png). Reproduce: `make seasonal`.

---

## 0. TL;DR

- **Both effects are real in the data; neither is a tradable decorrelated sleeve, and the sweep does not
  change that.** The correct portfolio decision is still to **exclude** the family — but three of the
  numbers this conclusion used to rest on were wrong, and the corrected map is different in ways that
  matter for anyone building on it.
- **The premium sits on the announce-day bar, not the day before.** On dividend-inclusive returns over
  the full 2005→ calendar, SPY earns **+22.6bps (t +2.3) on the announcement bar** (close before →
  close after, i.e. holding through the 14:00-ET statement) against **+11.8bps (t +1.1)** the day
  before; QQQ **+34.7 (t +3.2) vs +15.3 (t +1.3)**. The day-before window declared a-priori is the
  weaker half of the event, and the fill-timing robustness was already saying so — shifting that book
  one bar later (onto the announcement) *raises* it, SPY +0.12 → **+0.34**, QQQ +0.16 → **+0.53**.
- **It is an announcement-risk premium, and it decayed.** The announce-day bar on an equal-weight
  equity basket: **+59.5bps (t +2.9) in 2005-10, +28.0 (t +1.3) in 2011-15, −9.7 in 2016-20, +2.5 in
  2021-26.** The mirror-image fade the day after (−34.7bps in 2011-15) decays with it, to **+0.1bps**
  in 2021-26. What is left in the recent era is not tradable.
- **Removing the beta removes the return — now shown directly.** The turn-of-month long/short calendar
  spread (long the window, short the rest of the month at matched exposure-days) nets **−0.02 on SPY**,
  −0.04 on a 15-name basket, −0.44 on TLT. The in-window minus out-of-window mean is **+1.45bps** on
  SPY. The parameter surface tells the same story from the other side: net Sharpe still rises
  monotonically with window width (SPY (−1,+1) 0.11 → (−4,+3) 0.68) toward buy-&-hold.
- **The FOMC-cycle even-week structure (Cieslak-Morse-Vissing-Jorgensen 2019) does not hold here.** Even
  weeks beat odd weeks 2005-2015 (+8.7/+7.4 vs −2.6/+1.4bps) and the ordering **inverts** afterwards
  (+4.4/+4.8 vs +8.2/+6.8). Long-only even weeks nets **+0.50 against buy-&-hold +0.66**; the
  long-even/short-odd book nets **+0.09**.
- **The portfolio test, scored on all five brief targets rather than the Sharpe alone, still says no —
  for a better reason.** A 20% blend of *anything* at matched vol costs the book ~11 points of CAGR
  (48.9% → a median 37.4% across the 294 arms), because it swaps a piece of a 3.9-Sharpe book for a
  0.5-Sharpe one. Some arms do print 5/5 on the full window by fixing the losing streak — but so does the
  **same arm with its dates scrambled**: 8 of 294 arms hold 5/5 in both windows (2.7%) against **5.8%**
  of 1470 time-shifted copies of those same arms, i.e. the flip is not the calendar. And out-of-sample
  most arms *cost* a target (months-in-profit 80.8% → 76.9%).
  Meanwhile plain buy-&-hold SPY lifts the Sharpe the most (3.91 → **4.07**) and takes the scorecard
  **down** (4/5 → 3/5). **Nothing here is worth 11 points of CAGR.**
- **What the sweep did buy:** the corrected engine, a calendar 38% deeper, an asset map (the premium is
  a *risk-asset* premium — EFA/BTC/QQQ/SLV/EEM/XLK at the top, the whole Treasury complex negative and
  monotonically so in maturity), and the finding that the strongest surviving *effect* is crypto's **6h**
  pre-announcement window (BTC +0.69, ETH +0.64 net, 100th/99th placebo percentile) rather than the 24h
  window the literature uses.

---

## 1. Three ways a calendar study breaks silently

A calendar family is unusually fragile to plumbing, because it reads a handful of *specific bars* — so
any error that is itself pinned to a calendar lands squarely inside the measurement instead of
averaging out. Three did.

**(1) Bar-labelling look-ahead.** Binance klines and Twelve Data bars are indexed by the bar's **open**,
so the close stamped at 19:00 is the price at 20:00. `series.asof(T)` therefore returns a price one bar
*past* T, and the "24h into the 14:00-ET statement" window silently ended an hour *after* the statement
— folding in the announcement reaction, the one hour the window exists to exclude.

| BTC, 24h window ending at the statement | mean | t | n |
|---|---|---|---|
| as shipped (`asof` on open-labelled bars) | +102.5bps | +2.42 | 56 |
| the same window, events whose data actually exists | +108.3bps | +2.43 | 53 |
| **corrected (price re-stamped at the instant observed)** | **+83.5bps** | **+1.88** | 53 |
| the swapped hour alone, [T, T+1h] | +35.4bps | — | 53 |

The family's single significant headline was significant because of the hour after the announcement.
Corrected, it is t = +1.9. (Three of the original 56 events were the 2026 meetings that have not
happened yet: `asof` at a timestamp past the data returns the last price, so they were "measured".)

**(2) Ex-dividend drops read as losses.** `equity_td` closes are split-adjusted only. Dividend calendars
are calendars: SPY's quarterly ex-date is the third Friday of the quarter-end month, which is **two
trading days after the March/June/September/December FOMC in 39 of 172 meetings**.

| offset +2 from the announcement | ex-div hits | price-only | total-return |
|---|---|---|---|
| SPY | 39/172 | −15.7bps | **−5.5bps** |
| DIA | 39/173 | −11.8bps | **−6.7bps** |
| XLK | 25/172 | −14.6bps | **−3.2bps** |
| QQQ | 17/172 | −6.7bps | **−4.2bps** |

Two-thirds of the "post-FOMC fade" on day +2 was the dividend. (Day +1 is genuine: −15.3bps on SPY
before and after the correction.) The same trap is waiting one step over in turn-of-month: every
monthly-paying bond ETF goes ex on the **first business day of the month**, i.e. on day +1 of the ToM
window, worth 19-44bps — a bond ToM study on price returns is biased down by a fifth of its own window.

**(3) A calendar shallower than the prices.** The event calendar started in 2011; the ETF history starts
in 2005. Eight events a year is a thin sample to begin with, and the missing block was **+48 events
(+38%)** — and the era the pre-FOMC-drift literature was written on, which is what makes the decay
legible rather than ambiguous. Fixed in `src/data/fomc.py` (scheduled meetings only, sourced from the
Fed's per-year calendars; 2007/2008/2010 conference calls and the inter-meeting emergency cuts excluded).

A fourth, smaller one is in the same family: an anchor predating the price history used to `searchsorted`
to position 0, so a deep calendar piled all its early events onto the first bar and marked it as a
window. Anchors before the data are now dropped.

## 2. Construction — why the execution model is different (and honest)

Two effects, one engine (`src/sleeves/seasonal.py`); the event calendar is `src/data/fomc.py`.

- **pre-FOMC drift.** Anchor = the announcement day; the tradable windows are the bars around it. The
  24h Lucca-Moench window is measured two ways: a **daily** proxy (a whole bar, deep 2005→ history) and
  the **precise** intraday window (5-min/1h, instant-stamped prices around the DST-correct 14:00-ET
  timestamp, 2020→).
- **turn-of-month.** Anchor = each month-end bar; window = the last `days_before` + first `days_after`
  trading days (classic (−1,+3)).

**Why not the cross-sectional engine.** `xsect.py` books estimate a signal from market data, so they
carry a t+2 delay to avoid look-ahead. A calendar window is **known years in advance** — there is no
signal to estimate. So (a) there is no estimation look-ahead, and (b) the honest execution model is
*hold through the window and pay commission+spread only at the edges* (one entry, one exit), **not**
the daily round-trip that killed the overnight sleeve. That is the whole reason H4 was worth a separate
test: the cost structure is genuinely more forgiving. Crypto books are on perps and **charged realised
funding** while long, credited while short. Signed (long/short) event books charge two sides on the flip.

## 3. Where the premium lives — the offset map and its decay

**Mean return by trading-day offset from the announcement, total returns, 2005→ (173 events):**

| offset | −3 | −2 | **−1 (day before)** | **0 (announce bar)** | **+1 (day after)** | +2 |
|---|---|---|---|---|---|---|
| SPY | −1.2 | −3.2 | **+11.8** (t+1.1) | **+22.6** (t+2.3) | **−16.3** (t−1.6) | −5.5 |
| QQQ | −5.5 | −1.6 | **+15.3** (t+1.3) | **+34.7** (t+3.2) | **−16.1** (t−1.4) | −4.2 |

Run-up into the meeting, a large announcement-day return, a fade after — the documented shape, with the
mass on the announcement bar rather than the day before it. In-window Sharpe on the day-before bar is
**+1.34** (SPY) against **+0.62** out of window; the announce bar is stronger still.

**But the shape is a 2005-2015 fact.** Equal-weight 15-name equity basket, mean bps by sub-period:

| offset | 2005-10 | 2011-15 | 2016-20 | 2021-26 |
|---|---|---|---|---|
| −1 (day before) | +19.2 (t+0.7) | −4.5 (t−0.2) | +33.9 (t+1.9) | −3.1 (t−0.3) |
| **0 (announce)** | **+59.5 (t+2.9)** | **+28.0 (t+1.3)** | −9.7 (t−0.5) | +2.5 (t+0.1) |
| **+1 (day after)** | −8.6 (t−0.4) | **−34.7 (t−1.6)** | −24.3 (t−1.3) | +0.1 (t+0.0) |

The premium and its fade both go quiet after 2015 — consistent with "The disappearing pre-FOMC
announcement drift" (2020), and consistent with an announcement-risk premium being competed away once
it is published. The shipped day-before book nets **+0.09 to +0.17** across SPY/QQQ/IWM/DIA at the 67th
to 84th placebo percentile — still inside the noise of holding a random handful of days.

## 4. The sweep — 309 arms over side, assets, count and timeframe

Every window shape is run in **both directions** (the offset map says equities rise into the statement
and bonds do the opposite, so picking a side after looking at that map would be choosing the answer),
on 16 return streams, and the deflated Sharpe is charged for all 309.

**Side.** The two-sided books are what the map suggests, and they are the ones that clear their own
placebo by the widest margin — because a book that is long as many event-days as it is short has no
drift for the placebo to reproduce:

| arm | net Sharpe | 05-10 | 11-15 | 16-20 | 21-26 | placebo pctile (p95) |
|---|---|---|---|---|---|---|
| `xasset8rp` long (−1,0) / short (+1,+2) | **+0.48** | 0.54 | 0.49 | 0.61 | 0.31 | 100th (+0.10) |
| `xasset8rp` long (0) / short (+1) | +0.37 | 0.41 | 0.74 | 0.23 | 0.16 | 100th (−0.07) |
| `QQQ` long (−1,0) / short (+1,+2) | +0.52 | 0.78 | 0.27 | 1.07 | −0.03 | 100th (+0.25) |
| `HYG` long (0) / short (+1) | +0.45 | 0.67 | 0.78 | 0.20 | 0.20 | 100th (−0.02) |
| `SPY` long (0) only | +0.34 | 1.00 | 0.50 | −0.35 | −0.08 | 94th (+0.34) |

A market-neutral event book at Sharpe ~0.5 that clears its placebo at the 100th percentile is the best
thing in the family — and it still fails everything downstream (§7).

**Assets.** Same two windows, every asset separately (Sharpe of the long(−1,0)/short(+1,+2) book):

| top eight | | bottom six | |
|---|---|---|---|
| EFA +0.59 · BTC +0.54 · QQQ +0.52 · SLV +0.52 | EEM +0.52 · XLK +0.51 · SPY +0.46 · XLF +0.45 | LQD −0.02 · TLT −0.11 · XLU −0.18 | IEF −0.19 · AGG −0.32 · SHY −1.31 |

The ordering is the economics: it is a **risk-asset** premium (international and tech equity, silver,
credit, crypto at the top) and it runs backwards in duration (the whole Treasury complex negative,
monotonically in maturity: SHY worst). That is a coherent risk-on/risk-off signature, not a fluke of one
series — and it is also why the relative-value construction fails: long bonds / short equity around the
event is simply the wrong sign, and its reverse is just the equity book with extra cost.

**How many.** More names help only while they share the sign; the all-21 basket dilutes to nothing:

| basket (long (0)/short (+1)) | SPY alone | 4 US index | 13 +sectors | 15 +intl | 21 all incl. bonds | 8-asset risk-parity |
|---|---|---|---|---|---|---|
| net Sharpe | +0.38 | +0.36 | +0.31 | +0.36 | **+0.05** | +0.37 |

**Timeframe.** Precise intraday windows around the statement (2020→, instant-stamped, 5-min for the
ETFs where archived, 1h for crypto). Shortening the window from 24h to **6h** raises the Sharpe on both
crypto legs — the edge is concentrated in the last hours before the statement, and a shorter hold
carries less of the noise:

| window ending at 14:00 ET | BTC | ETH | QQQ | XLK | SPY |
|---|---|---|---|---|---|
| 2h | +20.4bps (t+1.3) | +25.2 (t+1.6) | −0.4 (t−0.0) | −7.2 (t−0.8) | −5.8 (t−0.7) |
| **6h** | **+56.7 (t+2.3)** | **+51.8 (t+2.2)** | +25.9 (t+1.2) | +28.6 (t+1.2) | +7.8 (t+0.4) |
| 24h | +83.5 (t+1.9) | +78.4 (t+1.6) | +27.9 (t+1.4) | +32.9 (t+1.5) | +11.1 (t+0.6) |
| 48h | +91.3 (t+1.5) | +77.0 (t+1.1) | +50.3 (t+1.8) | +59.8 (t+2.0) | +31.1 (t+1.4) |
| the 24h *after* the statement | +39.9 (t+0.6) | +73.0 (t+0.9) | +10.2 (t+0.3) | +8.8 (t+0.3) | −5.2 (t−0.2) |

As books: **BTC 6h nets +0.69** (100th placebo pctile, p95 +0.46) and **ETH 6h +0.64** (99th) — the
strongest arms in the sweep on a per-unit-risk basis, on 53 events each.

## 5. FOMC cycle — the even-week structure does not hold here

Cieslak-Morse-Vissing-Jorgensen (2019) report that the entire US equity premium since 1994 accrued in
*even* weeks of the FOMC cycle (days 0-4, 10-14, 20-24 …) and nothing in odd weeks. Measured in bars
since the announcement on the 15-name equity basket:

| | mean | t | in-window Sharpe | days | 05-10 | 11-15 | 16-20 | 21-26 |
|---|---|---|---|---|---|---|---|---|
| even weeks | +6.37bps | +2.80 | +0.83 | 53% | +8.7 | +7.4 | +4.4 | +4.8 |
| odd weeks | +3.27bps | +1.47 | +0.46 | 47% | −2.6 | +1.4 | **+8.2** | **+6.8** |
| CMVJ day-set | +6.85bps | +2.76 | +0.93 | 41% | +7.6 | +6.8 | +5.2 | +7.6 |

The alternation exists in the 2005-2015 half and **inverts** in the 2016-2026 half. As a book it never
pays: long-only even weeks nets **+0.50 against buy-&-hold +0.66** (you give up a third of the premium
to sit out half the days), and the long-even/short-odd book — the version that would be market-neutral —
nets **+0.09**.

## 6. Turn-of-month — the beta-neutral spread settles it

The first pass inferred "ToM is beta" from the shape of the parameter surface. The direct test is the
**calendar spread**: long inside the window, short outside it, exposure-days matched so the book carries
no average market exposure. If turn-of-month is a real concentration of return, the spread is positive.

| spread (long in-window / short out-of-window) | net Sharpe | in − out | t (Welch) |
|---|---|---|---|
| SPY | **−0.02** | +1.45bps | +0.37 |
| 15-name equity basket | −0.04 | +1.11bps | +0.29 |
| 8-asset risk-parity | −0.03 | +1.63bps | +0.89 |
| TLT | −0.44 | −4.95bps | −1.50 |
| BTC+ETH | +0.06 | +7.20bps | +0.35 |

Nothing survives. The in-window premium on SPY is **+1.45bps a day at t = +0.4** — right sign, no
significance, and worth less than the half-spread it costs to isolate. The 40-year-old effect is not
absent so much as too small to trade after 2005. The width surface agrees: net Sharpe rises as the window
widens toward permanently-long — SPY (−1,+1) **0.11** → (−1,+3) **0.28** → (−4,+3) **0.68**, against
buy-&-hold **0.64** — which is the signature of harvesting drift, not of a concentrated anomaly. The
classic (−1,+3) window sits at the **67th** placebo percentile (random 4-day windows average +0.20).

**Conditioning does not rescue it either.** The month-end liquidity story (Etula-Rinne-Suominen-Vaittinen)
predicts the effect should be *strongest after a weak month*, when the rebalancing flow into equities is
largest. It is the opposite: SPY turn-of-month after a down month nets **+0.07**, after an up month
**+0.31** (basket: +0.10 vs +0.25) — the conditioner has the wrong sign and neither branch clears the bar.

**Cross-sectional breadth** (unchanged, and the same conclusion): the US-stock ToM book is flat at ~0.2-0.3
whether you hold the top 50, 100, 200 or 500 names, so there is no name-selection signal, only broad
exposure on certain days; crypto rises with N (0.06 → 0.29 from top-10 to top-200) and never clears 0.3.

## 7. Portfolio value — and the control that decides it

The combined SPY calendar sleeve (pre-FOMC ∪ turn-of-month, in market 22% of days) nets **+0.31**
[MC-P5 −0.04], maxDD −18.9%, deflated Sharpe 0.51 over its 13 window-shape trials; the purged
walk-forward over that grid returns +0.60, which is the same beta artifact — the WFO selects the widest,
most-long windows in a rising market. It is **decorrelated** (corr to book −0.06, no leg above |0.09|)
and at 15% weight it moves the book 3.909 → 3.946, against 4.045 for a naked SPY stub at the same weight.

Across the whole 309-arm sweep the deflated Sharpe — charged for all 309 trials — clears nothing:
the best is BTC's 6h window at **0.65**, then the widest crypto ToM at 0.53; every market-neutral event
arm lands at **0.00**. Purged walk-forward selection over the arm grid returns **+0.27** for the daily
event books, **+0.50** for the turn-of-month grid, **+0.39** for the crypto intraday books and **−0.60**
for the crypto daily one — i.e. picking the window in-sample does not reliably survive either.

**The control that decides it — on all five targets, not the ratio.** A Sharpe alone cannot answer this
question: the book's Sharpe has room (3.91 in a 2.5-4.0 band) while the target it actually misses is the
**losing streak**, and a blend trades those against each other. Every arm is blended at matched vol and
scored on the brief's full card, against two stubs that contain no calendar at all:

**Full window** (book alone: Sharpe 3.91, CAGR 48.9%, maxDD −8.3%, months 81.4%, worst −5.8%, streak 3 → **4/5**):

| +20% of | Sharpe | CAGR | maxDD | months>0 | worst mo | streak | targets |
|---|---|---|---|---|---|---|---|
| **buy-&-hold SPY** (no calendar) | **4.06** | 40.1% | −6.8% | **79.8%** | −3.1% | 2 | **3/5** |
| **random 33%-of-days SPY** (no calendar) | 3.91 | 38.6% | −6.8% | **79.8%** | −3.5% | 3 | **3/5** |
| ToM (−4,+3), SPY | 4.00 | 39.7% | −6.6% | 81.4% | −4.8% | 3 | 4/5 |
| ToM (−4,+3), 8-asset risk-parity | 3.99 | 39.9% | −6.5% | 82.5% | −5.0% | 2 | **5/5** |
| event L/S (0)/(+1), 8-asset risk-parity | 3.91 | 38.6% | −7.9% | 81.9% | −3.5% | 2 | **5/5** |
| event L/S (0)/(+1), HYG | 3.92 | 38.7% | −7.2% | 81.9% | −3.5% | 2 | **5/5** |
| event L/S (−1,0)/(+1,+2), 8-asset | 3.95 | 38.9% | −7.6% | 80.8% | −3.5% | 3 | 4/5 |
| BTC 6h pre-announcement | 3.87 | 38.9% | −6.7% | 80.8% | −4.3% | 3 | 4/5 |

**OOS block** (book alone: Sharpe 3.54, CAGR 39.8%, maxDD −5.7%, months 80.8%, worst −1.7%, streak 2 → **5/5**):

| +20% of | Sharpe | CAGR | months>0 | targets |
|---|---|---|---|---|
| **buy-&-hold SPY** | **3.76** | 33.9% | 80.8% | **5/5** |
| **random 33%-of-days SPY** | 3.69 | 33.2% | 80.8% | **5/5** |
| ToM (−4,+3), 8-asset risk-parity | 3.67 | 33.6% | 76.9% | 4/5 |
| event L/S, 8-asset risk-parity | 3.36 | 30.0% | 80.8% | 5/5 |
| BTC 6h pre-announcement | 3.34 | 29.8% | 76.9% | 4/5 |

Three things the Sharpe column alone would have hidden, and they point the same way:

1. **Every 20% blend costs ~11 points of CAGR** — 48.9% → a median of 37.4% across all 294 arms — because
   at matched vol you are swapping a piece of a 3.9-Sharpe book for a 0.5-Sharpe one. The ratio can go up
   while the money goes down; on this book it does.
2. **Adding beta is not free either.** Buy-&-hold lifts the Sharpe most and takes the scorecard *down*
   (4/5 → 3/5): it fixes the losing streak and breaks months-in-profit (81.4% → 79.8%). So the earlier
   one-line summary — "beta lifts it more" — is true of the ratio and false of the deliverable.
3. **The 5/5 prints have nothing to do with the calendar.** Of the 294 arms, 24 take the full window to
   5/5 and **8 hold 5/5 in both windows (2.7%)**. Run each arm against its own null — the same series
   **circularly shifted in time**, which keeps its return distribution, sparsity, vol and cost drag and
   destroys only the alignment between its P&L and the book's bad months — and the scrambled versions
   print 5/5 **more** often: 8.6% full and **5.8% in both windows** over 1470 draws. The missing target
   is a *discrete streak count* one month can flip, so any decorrelated stream flips it now and then;
   picking the arm that happened to flip it is window-shopping with extra steps.

Standalone, the event books also show why they can never carry a months-in-profit target on their own:
they are flat by construction in the four months a year with no meeting, so months-in-profit runs 36-43%
and the longest losing streak 7-9 months even when the arm is profitable overall.

## 8. Honest verdict & ceiling

- **Reachable here:** nothing deployable. Not as a long-only timing book, not as an event long/short,
  not on a shorter timeframe, not on a wider or narrower asset set, not conditioned, and not with ML
  (§9). The effects are real — the announcement-day premium is textbook and was worth +59.5bps a
  meeting in 2005-10; crypto still shows a +57bps 6h run-up into the statement at t = +2.3 — but the
  equity premium is gone since 2015, the turn-of-month spread is +1.45bps a day, and the only book-level
  lift on offer is beta.
- **Binding constraints, in the order they bind:** (1) **decay** — the premium and its fade both went
  quiet after 2015, so the 21-26 column is ~0 for every equity arm; (2) **capacity** — eight events a
  year means even a real per-event edge is a thin annual Sharpe, and 309 arms on ~170 events is a
  multiple-testing problem the deflated Sharpe prices at ~0; (3) **beta** — the arms that survive the
  first two are the ones that are simply long, and they are dominated by holding the index.
- **What did not work (kept, not hidden):** both effects; 7 window shapes × 2 directions × 16 streams;
  25 assets individually; five basket constructions from 1 to 21 names; five intraday windows from 2h
  to 48h plus the post-announcement window; the FOMC-cycle even/odd-week structure; the turn-of-month
  beta-neutral spread, its (before × after) surface, its dash-for-cash conditioning and four
  cross-sectional basket sizes; the cross-asset relative-value long/short (§9) and three ML variants (§9).
- **The one extension not run:** the same announcement-day book on the *other* scheduled macro releases
  (CPI, employment) — the Savor-Wilson announcement premium, which would quadruple the event count and
  is the only fix for constraint (2). It needs a historical release-date calendar, and neither source
  reachable from here serves one (bls.gov refuses automated requests; the FRED release-date endpoint
  requires an API key that is not provisioned). Rule-of-thumb dates would be wrong often enough to bias
  the test toward zero, which is worse than not running it.
- **Value delivered:** the **map** (H4 covered along every axis it is free in, with the decay dated), the
  **corrected plumbing** (instant-stamped intraday prices, dividend-inclusive returns, a calendar as deep
  as the prices — all three now in `src/`, where the next calendar study inherits them), and the
  **methodology**: a two-sided event book whose placebo has no drift to hide behind, and a beta control
  on the book-lift claim that a decorrelated sub-bar sleeve cannot pass.

## 9. Cross-asset relative-value and ML (unchanged conclusions)

A dollar-neutral long/short *across names*, live only in-window, ranked on each name's own trailing
in-window history: crypto pre-FOMC **−0.47** (60th placebo pctile), stocks turn-of-month **−0.49**
(59th), crypto turn-of-month **+0.36** at the 96th percentile — one marginal near-miss, still sub-bar and
carrying a per-episode round trip 12×/yr. Three leakage-controlled ML variants: a conditional pre-FOMC
gate (VIX, 10y-2y slope, trailing drift; purged event K-fold) leaves SPY **+0.24 → +0.23** — the ridge
has real out-of-sample signal on the deeper 171-event sample (IC **+0.29**) and still adds nothing,
because gating away 41% of the events gives back exactly what the ranking earns — while BTC goes
**+0.28 → −0.19** (IC −0.32) and the tree/logit variants are worse on both; a cross-sectional LGBM
ranker is *worse* run in-window than run every day (−0.94 → −1.44). Removing the shared market move
removes the return, and conditioning on the calendar does not add — the same answer §6's spread gives
from the time-series side.

## 10. Reproduce

```bash
make seasonal   # run_seasonal.py             -> reports/seasonal/seasonal_{summary.json,returns.parquet,tom_grid.csv}
                #                                + reports/figures/seasonal.png
                # run_seasonal_variants.py    -> reports/seasonal/seasonal_variants_{summary.json,grid.csv,
                #                                attribution.csv,returns.parquet}
                # run_seasonal_xasset_ml.py   -> reports/seasonal/seasonal_xasset_ml_summary.json
                #                                + reports/figures/seasonal_ml.png
```

Fixed seed (7). FOMC dates: `src/data/fomc.py` (Fed FOMC calendars, scheduled meetings only, 2005→).
Sources: Lucca & Moench, "The Pre-FOMC Announcement Drift" (JF 2015); Lakonishok & Smidt, "Are Seasonal
Anomalies Real?" (RFS 1988); Savor & Wilson, "How Much Do Investors Care About Macroeconomic Risk?"
(JFQA 2013); Cieslak, Morse & Vissing-Jorgensen, "Stock Returns over the FOMC Cycle" (JF 2019);
Etula, Rinne, Suominen & Vaittinen, "Dash for Cash: Month-End Liquidity Needs" (RFS 2020); Kim & Suh /
Vähämaa, "The disappearing pre-FOMC announcement drift" (Fin. Res. Lett. 2020).
