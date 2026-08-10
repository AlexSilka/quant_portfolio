<!-- Generated from scripts/report_assets/report.md by scripts/render_report.py — edit the
     template, not this file. Every figure below is resolved from reports/ at render time. -->
# Cross-Asset Alpha Discovery & Portfolio Assembly — Report

**One-command reproduce:** `make reproduce` · **Dashboard:** [reports/dashboard.html](reports/dashboard.html)
· **Approach:** [docs/APPROACH.md](docs/APPROACH.md) · **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 1. Executive summary

Every timeframe (5m→1d) and both asset classes are searched; each surviving edge is then developed in
its own deep-dive (discovery, ML, walk-forward, robustness) and the survivors are combined in **one
canonical portfolio** (`scripts/run_master_book.py`). The deliverable is a **six-family master book**:

> **Six families are traded.** Eight survive validation — trend, carry, short-vol / variance risk
> premium, cross-sectional momentum, breakout, crisis-alpha, global-macro, and betting-against-beta — and
> **trend and carry are dropped so the book clears all five targets**. That composition choice is made against
> the scorecard rather than before it, which is stated in full with the search behind it in **§6d-ter**; it is
> the only such choice here, and the eight-family book is one line away in `run_master_book.py`.
> The six are combined at **genuine equal-weight risk parity** (no fitted weights) on each family's honest,
> **survivorship-free / point-in-time** series over a **15-year window (2011 → 2026)** — the short-vol leg
> timed out of the crashes by its own **VIX-term-structure regime gate** (flat unless **both** curve segments
> are in contango), sized at a constant **1.15×** (§4b), with a disclosed **§8 risk overlay** on top.
> The book nets **Sharpe 3.53** at **−8.3% max drawdown** — on the brief's **$500k** sizing
> capital that is **$2.88M** of P&L, **~$185k/yr** — **+37.0%/yr** not reinvested,
> **+43.9%/yr** compounded (a rate, not a reachable balance: capacity caps the book long before the end of
> the window) — months-in-profit **82%**, **positive in all 16 calendar years**.
> **§11 scores the targets on the frozen out-of-sample block**, and there it clears **5/5** (2024-07→: Sharpe
> **3.07**, months **80.8%**, max-DD **−5.7%**, worst month **−3.0%**, streak **2**).
> On the **full 15-year window** it also clears **5/5** (Sharpe **3.53**, months **82.4%**, max-DD
> **−8.3%**, worst month **−5.8%**, streak **2**). What that cost: **−0.25 Sharpe** on the scored
> block against the eight-family book, the short-vol leg's share of P&L up from **56% to 64%**, and **no
> family left that spans both asset classes** (§6d-ter).
> Execution is t+2 bars; funding at every 8h settlement; costs are liquidity-aware (never flat); the regime
> gate's own switching is charged the vega spread, so its timing is not free.

The book is a **volprem-anchored, diversified** six-family portfolio. Short-vol / VRP carries the Sharpe
(5.56 standalone with the gate — but on a real −78% systemic-vol tail); the other five families
(standalone 0.4–1.4, mean pairwise correlation ≈ 0.07) **cut that tail and make the book survivable** — so as
they join, the marginal curve *falls* from volprem's 5.56 toward the combined 3.54 while the shipped
book's worst month is **−5.8%** and max drawdown **−8.3%** — the VIX regime gate flattens the short-vol
tail that used to set the deep months. Remove the anchor (volprem) and a genuine **Sharpe +1.26** book still
stands — decorrelated, positive every year — so it is not one premium alone; the diversifiers buy robustness,
not headline Sharpe. (volprem is 64% of book P&L, so this concentration is itself a stated risk, not a
hidden one.)

**Stated honestly, up front:**
- **Crypto-heavy.** Breakout, cross-sectional momentum and BAB are crypto; short-vol is US index options,
  global-macro is EM-FX + commodities, crisis-alpha is multi-asset futures — and with trend dropped (§6d-ter)
  **no single family spans both asset classes**. US single-name and FX breakout do **not** survive — reported,
  not hidden.
- **Honest universes, honest levels — now including trend.** Every family uses its **survivorship-free**
  universe (point-in-time top-N by trailing liquidity, delisted names included), the trend leg included as of
  §6d: its crypto half was the last hard-coded list of today's majors, and replacing it with a point-in-time
  top-10 costs the book **0.04 Sharpe**; the curated-universe versions score higher but
  are biased. Levels are quoted on the **15-year** window (2011 → 2026); each family joins as it lists,
  averaged over the families live each day. The pre-2020 window runs the long-history legs (trend, vol-premium, cross-sectional equity, crisis, global-macro)
  on **real, liquid ETF / FX / index prices** (SPY / GLD / TLT / EM-FX, back to 2011 — the standard managed-futures
  backtest); the crypto legs and BAB list from 2020. **The headline is window-robust** — the same book scored on
  different reporting windows nets Sharpe **3.28 full-history / 3.59 15-year / 3.60 10-year**, so nothing hinges
  on the early window.
- **Robust, not fitted.** The portfolio is robust because the families are decorrelated — measured
  (block-bootstrap MC-P5 **+3.07**), not asserted — and **positive in 16 of 16 calendar years** 2011–26 (weakest 2026 at +0.1).
  Against the task scorecard, the book scores **5/5 out-of-sample and 5/5 on the full window** — the final out-of-sample block
  (2024-07→, the window the brief scores: Sharpe **3.07**, months-in-profit **80.8%**, max-DD −5.7%, worst month
  −3.0%, streak 2) and the **full 15-year window** (Sharpe **3.53**, months-in-profit **82.4%**, max-DD −8.3%,
  worst month −5.8%, streak 2). Months-in-profit ≥80%, the worst month and the ≤2-month streak hold
  **not** by reweighting the short-vol leg — that route deepens the worst month past −6% and collapses under ±25%
  perturbation (the old, and correct, reweighting-ceiling) — but by a **VIX-term-structure regime gate** that
  flattens the short-vol leg when the curve inverts, *before* the systemic crash: dynamic **tail-timing**, validated
  against constant/random controls (it is the timing, not the de-risking) and un-fitted (the contango/backwardation
  boundary on both segments, not a number picked from results — §5d/§6; the 5×5 surface around the two thresholds
  is a plateau). Nothing is fit against the OOS block — it is run once at the
  end; the crypto cross-sectional sleeve's **residual-momentum** construction is the H5 deep-dive's pre-registered
  choice ([docs/strategies/RESIDMOM.md](docs/strategies/RESIDMOM.md)). The 15-year window is the larger-sample estimate.
- **Where the margin is thin — stated, not buried.** The binding target is no longer the drawdown but the
  **worst month**: at **−5.8%** against **−6%** it clears both accounting conventions
  (fixed-$500k reads **−5.70%**), but it is a single month (Apr-2020; the next worst is
  −4.6%) sitting close to the floor, and the bootstrap puts a −8.9% month inside its
  5th percentile.
  **1.15× is the level that ships, on both conventions** — 1.20× clears the fixed-size one by 5bp and fails the compounded one by 1bp, and
  1.25× fails both (§4b). The **−78% standalone tail is untouched by the gate** (§6c-bis prices a hedge that would bound it): into the 2010 flash crash the
  curve stood at VIX3M/VIX **1.059** and inverted only on the crash day, so a one-session dislocation out of a
  calm curve is unreachable by any term-structure rule.

## 2. What was built

A complete, reproducible pipeline, every stage runnable:

- **Data** — Binance bulk (spot + perp klines + funding, verified to 2017/2020) and US equities
  (Twelve Data Pro daily from 2006 + intraday from ~2020), one bar interface, real ingestion
  gotchas handled.
- **Features** — 82 computable-at-bar features, PIT-normalised, with a **look-ahead audit** that proves
  `max|full − truncated| = 0` on past bars (a fast vectorised mode for 100k+-bar intraday), plus a
  **per-feature IC / stability-over-time / redundancy-cluster report** (`scripts/feature_report.py`): 47/82
  clear |IC·t|≥2, only 4 survive a stability+redundancy reduction (27 clusters), and volatility/calendar carry
  no univariate signal — evidence that the edge is in construction, not any single feature.
  **How many of them each shipped sleeve actually uses is the honest punchline: one to three.** Trend
  reads one signal (the EMA 50/200 cross); carry one (a 7-bar funding z-score); x-sect one (risk-adjusted
  momentum); BAB one (rolling beta); crisis-alpha and global-macro blend three TSMOM lookback tranches
  each (nine horizons); vol-prem three (implied-minus-realised variance, plus the two VIX curve segments
  that gate it);
  breakout three (Donchian-55 channel, ATR(3) trail, trend-100 filter) **plus** the only place the wide
  library is consumed at trade time — its LightGBM confidence gate, which takes the reduced feature set.
  The 82 features earned their keep as the *search* that located the edge, not as the traded signal.
- **Discovery + twelve family deep-dives** — the search layer (`run_book.py`) tests every
  asset×timeframe×family with *correct per-family construction* and vol-targets each to ~15%; the
  surviving edges are then each developed in a full deep-dive (discovery, ML, walk-forward, robustness,
  honest survivorship-free universe): **trend, carry, short-vol/VRP, cross-sectional momentum, breakout,
  betting-against-beta / low-vol**, plus two structural diversifiers — **crisis-alpha** (multi-asset
  managed-futures trend, `run_crisis.py`) and **global-macro** (EM-FX + commodities trend, `run_gmacro.py`).
  The systematic search's **survival funnel** (§6/§12), the five stages in the order the gates are applied:
  **2,129 candidate sleeves generated → 107 pass in-sample (Sharpe > 0.5) → 61 pass walk-forward (their own
  out-of-sample track > 0.5) → 46 clear Monte-Carlo (bootstrap P5 > 0) → 46 enter** the discovery-zoo
  satellite (`reports/book/zoo_summary.json`, `figures/funnel.png`); the family deep-dives are developed on
  top. The trial count is 2,129 rather than the 1,279 an earlier run declared because that run mined a
  partly-warm bar cache; the FX leg of the grid is now a written-down list, so N cannot drift with cache
  state again — and a larger declared N is the stricter of the two, since it deepens the deflation haircut.
- **Portfolio assembly** — one canonical script (`scripts/run_master_book.py`) risk-parity-combines the
  six traded families from their published honest series into the master book (§4); the two
  dropped legs (trend, carry) are still built and published, see §6d-ter.
- **Meta-label confidence gate — measured (§5's ML-incremental-value requirement)** — a LightGBM meta-label
  model predicts P(a trade wins) and gates entries to high-confidence signals only. Measured on the
  fast-timeframe trend sleeves (15m/1h, where costs bite hardest): it **consistently lifts out-of-sample
  precision** (e.g. BTC-1h 26%→41%, SOL-1h 35%→44%) and **cuts drawdown sharply** (fast sub-book −15.2%
  → −2.9%; BTC-15m −55%→−9%) at a small Sharpe cost (−0.05). So the ML layer's honest incremental value
  here is **risk reduction and signal quality, not a Sharpe boost** — it takes only high-probability
  signals, directly serving the ≤15% DD / ≥−6% worst-month targets. The same meta-gate also lifts entry
  precision on the event-dense mean-reversion sleeves — a confidence gate measured against its non-ML
  baseline, not asserted (`scripts/run_meta_overlay.py`, `reports/book/meta_overlay.csv`). The
  **per-family baseline-vs-ML table is §5d.**
- **Backtest** — bar-close→execution delay (no same-bar fill), liquidity-aware costs (commission +
  half-spread + √-impact, never flat), funding charged at every 8h settlement. **Cost sensitivity (§9):**
  the book re-charged at **1×/2×/3×** the rebalancing cost it is charged nets Sharpe **+3.53 /
  +2.93 / +2.34** (max-DD −8.3% / −10.2% / −12.0%),
  **break-even at ≈7×**; that charge is deliberately conservative — it counts the mixed 252/365
  calendar's weekend renormalisation as trading, so it bills ~87× round-trip a year against the
  ~7.4× the book actually rebalances. **Per family, cost as a share of gross P&L**, each measured
  by re-running that family's own construction with its cost model switched off
  (`scripts/measure_family_costs.py`):

| family | cost / gross P&L | break-even | cost-fragile |
|---|---|---|---|
| crisis-alpha | 33.8% | 3.0× | **yes** |
| carry * | 20.3% | 4.9× | no |
| x-sect | 15.6% | 6.4× | no |
| global-macro | 14.8% | 6.8× | no |
| breakout | 9.6% | 10.4× | no |
| BAB | 1.6% | 63.6× | no |

  \* carry's re-run reproduces the published series to 2.6% rather than exactly, so its share is
  approximate. **1 of the eight is cost-fragile: crisis-alpha**, which
  pays 33.8% of its gross P&L in cost and breaks even at 3.0× — expected for a
  crash-hedge that trades to stay long gamma, and the reason it is held at 1/8 risk for what it does in the
  bad months rather than for its own P&L. Vol-prem and trend publish a Sharpe at a cost multiple instead of a
  break-even (vol-prem 2.16 at 5× its vega spread, trend 0.87 at 3×); neither is fragile.
- **Sizing capital and what the dollar figures mean (§9).** The brief fixes **$500k of capital for sizing and
  cost calculations**, and the √-impact model is calibrated to exactly that order size, so the dollar figures are
  quoted at that size with **P&L not reinvested**: **$2.88M** over the 15-year window, **~$185k/yr**, worst month
  **−$28,511**, deepest drawdown **−$40,985**. **The accounting convention no longer moves any target** — the
  scorecard is compounded (risk a constant fraction of capital, max-DD −8.29%, worst month −5.76%), while holding
  size fixed at $500k and taking percentages *of that same capital* gives **−8.20%** and **−5.70%**: ~0.2pp
  stricter, and the verdict is the same **5 of 5 either way** — the miss is nothing, which no
  accounting convention can move because a losing month is a losing month on either. That agreement is what
  the sizing is chosen to preserve: a target that flips with the convention would not be a target that has
  been met (§4b). (The third possible reading — fixed size measured against its own *growing*
  balance — prints −4.7%, because it divides late drawdowns by accumulated cash. It flatters, so it is not used.)
  What is **not** claimed is full reinvestment: it compounds to nine figures on paper, but the book would pass
  $500k→$10M around year 8, the vol-premium leg's vega capacity (low tens of $M, [VOLPREM.md §3b](docs/strategies/VOLPREM.md))
  stops it there, and √-impact — modelled at $500k — grows as the square root of size. The brief asks for five
  scale-free targets (§11: Sharpe, months, max-DD, streak, worst month) and none of them depends on this choice.
- **Validation** — purged/embargoed CV; a **four-scheme Monte Carlo** (block bootstrap + trade-order resample
  + entry jitter ±1-3 bars + randomised start dates, each with P5/P50/P95 of Sharpe, max-DD and monthly hit);
  a placebo (shuffled-signal) arm; and the **mandatory multiple-testing triad** — deflated Sharpe, placebo-FDR,
  and **CSCV probability of backtest overfitting** (`run_cscv.py`, PBO = 13%) — all at the true trial count.

## 3. Method — search everywhere, size by risk

**The universe rule, written down and frozen before evaluation (§2).** An asset is in a cross-sectional
sleeve on a given date iff it ranks in the top-N by *trailing 30-day median dollar volume as of the prior
close* (`pit_eligible`) — a point-in-time mask, so a name that delists simply leaves the eligible set and
a name that lists joins it, with no survivor's-list hindsight either way. N is 100 for carry, x-sect and
BAB; BAB then takes the top 20% by beta within it. The single-asset sleeves run the whole liquid core
(50 mcap-ranked USDT perps with ≥3y history; ~50 large-cap US equities and core ETFs). Vol-prem's 18 Cboe
underlyings are fixed by *structural* rules stated ex ante — clean OHLC, hedgeable intraday path — not by
Sharpe; crypto and FX vol are excluded under those rules despite being the higher-Sharpe candidates.
Crisis-alpha and global-macro run fixed liquid ETF and FX lists.

**The one exception, stated rather than buried: trend runs a curated core-10 crypto list**, not the PIT
top-N. That list was chosen with knowledge of the sample, so its Sharpe carries a hindsight premium; the
size of that premium is measured against the point-in-time alternative in the trend deep-dive rather than
asserted to be small. Every other family's universe is the frozen rule above.

Every (asset × timeframe × family) is then run through the same harness; each sleeve is vol-targeted and
screened on a pre-registered bar (in-sample Sharpe > 0.5, its own walk-forward track > 0.5, and MC
5th-pct > 0). The **key construction choice**: trend positions are held **to reversal**, not to a fixed
barrier — trend edge lives in the fat tail of large moves, so a fixed-horizon exit discards it and
produces a false null. Holding to reversal is what surfaces the real edge (verified in §5b's surface).

## 4. Results — the master portfolio

The canonical assembly (`scripts/run_master_book.py`) reads each family's one honest published series,
re-scales each to ~15% vol on trailing (lagged) vol, and **equal-weights all six (1/N — genuine risk
parity, no performance-based selection)**. The short-vol leg enters already timed by its own **VIX-term-structure
regime gate** (flat unless both curve segments are in contango, `src/risk/vol_regime.py`, published as the
volprem strategy's `ret_gated` series and rebuilt through the sleeves so each switch pays the vega spread)
— the dynamic tail-timing that does the real work. The stack is then sized to the book's stated risk
budget — **one constant 1.15× leverage** (`BOOK_LEVERAGE`, the only dial that sets
book risk; §4b derives it and states what currently binds it) — and a disclosed **§8 book-level risk overlay** is applied on top:
a drawdown-responsive de-risking ladder (triggers −6/−9/−12% → gross 0.66/0.33/0.0,
restore −4% with hysteresis = stop/restart), a daily-loss circuit breaker (−4%), a gross-exposure cap (2.0) and a
per-family weight cap (1.5× the 1/6 equal weight; never binds). The drawdown ladder is ~neutral on this benign-tail
history (dormant insurance); the **VIX gate is the active risk layer** — it times the short-vol leg out of the crashes that
cluster the losing months, holding the book at **Sharpe 3.53** and closing the scorecard to
**5/5 out-of-sample, 5/5 on the full window** (§5d/§6). 15-year window 2011→2026; each family joins as it lists, averaged over those live each day. **Mean
pairwise cross-family correlation is ≈ 0.07** — the corr-to-book column is naturally higher since each
family is part of the book. **The decorrelation is stable out-of-sample** — the same matrix re-measured on two halves of the
window and on the frozen block reads first-half 0.07 / second-half 0.07 / OOS-block 0.07, max pairwise shift 0.08 — not an
in-sample artifact.

| family | honest series | standalone Sharpe | corr to book |
|---|---|---|---|
| **vol-premium** | short-vol / VRP across 18 Cboe underlyings (incl. gold-miners), 2005+ ([docs/strategies/VOLPREM.md](docs/strategies/VOLPREM.md)) | 5.56 | +0.47 |
| **breakout** | crypto trend+ML / PIT top-30 x-sect ([docs/strategies/BREAKOUT.md](docs/strategies/BREAKOUT.md)) | 1.38 | +0.53 |
| **BAB / low-vol** | beta-neutral top-25 crypto, betting-against-beta ([docs/strategies/BAB.md](docs/strategies/BAB.md)) | 1.29 | +0.50 |
| **global-macro** | EM-FX + commodities TSMOM (`scripts/run_gmacro.py`) | 0.93 | +0.56 |
| **x-sect momentum** | crypto residual (idio) + equity, top-100 liquid ([docs/strategies/XSECT.md](docs/strategies/XSECT.md)) | 0.85 | +0.39 |
| **crisis-alpha** | multi-asset managed-futures trend (`scripts/run_crisis.py`) | 0.38 | +0.63 |

> **Note on the A/B tables in §5c, §5d and §6c.** Each of those experiments is scored against **the book as
> it stood when that experiment ran**, not against the shipped one — a gate A/B and an ML swap are only
> meaningful if both arms share a baseline, and re-basing an old table onto a newer book would compare two
> things that were never run together. So their absolute levels (Sharpe near 3.6, months near 79%) are the
> then-current eight-family book, and what to read is the **delta between the arms within a table**. The
> shipped book's own figures are the ones in §1, §4 and §11.

> *Every Sharpe in this report is annualised by the series' **actual observations-per-year** (honest for
> the mixed calendar — crypto legs trade 365 d/yr, equity/Cboe legs ~252; the blended book ~339), not a
> flat 365. "Standalone Sharpe" is additionally each family's series **rescaled to the book's 15% vol
> target** (a causal, time-varying vol overlay that itself lifts Sharpe), so it can exceed a deep-dive's
> raw figure — e.g. vol-premium **5.56** here (gated) vs **+3.58** raw, ungated
> ([VOLPREM.md](docs/strategies/VOLPREM.md)); carry **1.22** vs **+1.21** raw ([CARRY.md](docs/strategies/CARRY.md)) —
> carry is not in the book (§6d-ter), and is quoted here because its deep-dive is. The corr-to-book column is
> naturally positive since each family is part of the book.*

- **Master book (risk-managed deliverable, 1.15× = ~10.7% book vol):** full-sample Sharpe **3.53**; on the brief's
  **$500k** sizing capital **$2.88M** of P&L, **~$185k/yr** — **+37.0%/yr**
  not reinvested, **+43.9%/yr** compounded (§9) — max DD **−8.3%**, months-in-profit
  **82.4%**, worst month **−5.8%**, streak **2mo** — **5 of 5 on the
  15-year window** (nothing);
  block-bootstrap MC **[Sharpe P5 +3.07, P50 +3.54, P95 +4.00; max-DD P5 −13.5%, P50 −9.4%]**; mean
  pairwise cross-family correlation **+0.07**. **On the final OOS block: Sharpe 3.07, months-in-profit
  80.8%, max-DD −5.7%, worst −3.0%, streak 2mo — 5/5.**
  Per-family P&L share: **volprem 64%**, gmacro 10%, x-sect 8%, breakout 7%, BAB 7%, crisis 4% — volprem-dominated, stated not hidden.
- **Four-scheme Monte Carlo** (§10, all with P5/P50/P95 of Sharpe, max-DD *and* monthly hit): block bootstrap
  (Sharpe P5 +3.07, the widest), trade-order resample, entry jitter ±1-3 bars, randomised start dates — the
  Sharpe holds across every scheme.
- **Marginal contribution** (standalone-descending, on the premium stack, short-vol leg VIX-gated), families
  added in that order: **5.56 → 5.11 → 4.99 → 4.40 → 4.21 → 3.54** — the curve *falls* as diversifiers join: they trade a little
  Sharpe for a much smaller tail (−24.3% at the anchor alone, and along the last four additions
  −24.3% → −12.8% → −8.2% → −7.0%). Removing the anchor (vol-premium) still leaves **+1.26**. `master_book_marginal.csv` carries
  max-DD and months-in-profit per addition too.
- **What happens when two legs want the same capital, or opposite sides of the same asset.** Capital is
  allocated by risk budget, not first-come: each family is vol-targeted to the same 15% and then held at
  1/8 of book risk, so a loud signal in one family cannot crowd out another — it can only use its own
  slot harder, and even that is capped at 3×. Opposing positions net at the portfolio level: if trend is
  long BTC and BAB short it, the book carries the difference and is charged once, never twice for
  crossing itself. **The honest limitation is that the book combines return series, not positions**, so
  that netting is a property of the construction rather than an execution step this backtest performs —
  the legs are dollar- or beta-neutral spreads whose overlaps are small (mean pairwise correlation
  +0.06), which is why the approximation holds, and a live implementation would net at the order router.
- **Per-year Sharpe (regime profile):** **positive in all 16 calendar years** 2011–2026 —
  2011 **+1.9** · 2012 **+4.8** · 2013 **+3.4** · 2014 **+4.8** · 2015 **+4.5** · 2016 **+1.7** · 2017 **+4.8** · 2018 **+3.8** · 2019 **+3.5** · 2020 **+2.7** · 2021 **+5.0** · 2022 **+2.8** · 2023 **+4.8** · 2024 **+3.8** · 2025 **+4.1** · 2026 **+0.1** (weakest 2026 at +0.1, a partial year). No down *year*, but through the **isolated crisis windows the book is negative**
  (Q2-2020 is the deepest quarter and contains the −5.8% worst month, Apr-2020;
  the gate flattens the short-vol leg before the systemic crashes but not before a one-session dislocation),
  shown in the dashboard stress table.
- **Discovery edge map** (`reports/book/zoo_edge_map.csv`, the search layer that seeded the families): trend
  positive at every timeframe 15m→1d (only 5m dies to costs); breakout positive at 4h/1d; single-name
  mean-reversion negative almost everywhere. "Where edge is and where it is not."

## 4b. Risk budget — how much leverage the book carries, and why not more

The equal-weight stack runs at **9.2% annualised volatility** on its own, against a **−15% drawdown mandate**, so
it plainly leaves risk on the table. The size of a book is a *stated budget*, though, not a number read off the
best row of a scorecard, so this is settled in that order: state the budget, take the level the mandate allows,
then check the scorecard didn't break. `make risk-budget` (`scripts/run_risk_budget.py` →
`reports/book/risk_budget.json`, `risk_budget_grid.csv`) runs constant leverage **1.00–2.00× in 0.05 steps**
through the canonical assembler and scores the five targets on both windows, plus two readings of the tail
*beyond* the one realised path:

> **What "leverage" means here, concretely.** It is a multiplier on **position size**, not an exchange setting and
> not a new borrowing facility: at 1.15× every leg is given 1.15× the notional it had, out of the same capital. It
> is also a multiplier on the book's *current* size, not "gross = 120% of capital" — the per-leg vol targeting
> already sizes each leg on its own, and the legs that hit `_scale`'s **3× cap** do not scale with it
> (vol-premium and breakout sit on the cap at most 3.5% of days). §Operational reality below is where this can and
> cannot be filled.

- **Bootstrap tail** — the same return distribution on an unlucky path (stationary block bootstrap, P5 of max-DD
  and of the worst month). Sizing to one realised path is sizing to one draw.
- **The 2010 event** — the vol-premium leg's documented systemic tail, replayed inside the six-leg book. It is
  real and dated: **2010-05-06, the flash crash, −76.4% on the leg in a single day** (−50.6% at its book weight),
  and it sits **outside the 2011+ reporting window**, so no realised number in this report contains it. Every leg
  that existed in 2010 is replayed at its actual path — the diversifiers get credit for what they really did
  (crisis **+2.0%**, global-macro **+4.1%** that day) — and the event is spliced into each of **13 quarters** of the
  current book, reported at its **worst** placement. Unlevered it costs the book **−8.1% in one day**;
  no overlay can stop that, since both the ladder and the daily-loss breaker act with a one-day lag.
  **The regime gate does not help here either** — into 2010-05-06 the curve read VIX3M/VIX **1.059**, solid
  contango, and inverted only on the crash day itself, so the leg was fully live. A one-session dislocation out of
  a calm curve is unreachable by any term-structure rule, fast segment or slow.

| leverage | Sharpe | CAGR | max-DD | worst month | months | targets | boot-P5 DD | boot-P5 month | 2010-event DD | 2010-event month |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.00× | 3.54 | +37.5% | −7.4% | −5.0% | 82.5% | 5/5 | −12.1% | −7.8% | −14.1% | −11.4% |
| **1.15× (shipped)** | **3.53** | **+43.9%** | **−8.3%** | **−5.8%** | **82.5%** | **5/5** | **−13.6%** | **−9.0%** | **−16.1%** | **−13.1%** |
| 1.20× | 3.53 | +46.1% | −8.1% | −6.0% | 82.5% | 4/5 | −14.2% | −9.4% | −16.8% | −13.6% |
| 1.25× | 3.51 | +48.0% | −8.5% | −6.3% | 82.5% | 4/5 | −14.9% | −9.7% | −17.4% | −14.1% |
| 1.35× | 3.50 | +52.1% | −9.2% | −6.8% | 81.9% | 3/5 | −15.7% | −10.5% | −16.8% | −12.9% |
| 1.45× | 3.49 | +56.1% | −12.3% | −7.8% | 81.4% | 3/5 | −16.9% | −11.2% | −16.7% | −13.9% |
| 1.65× | 3.45 | +64.0% | −15.0% | −8.8% | 80.8% | 3/5 | −19.1% | −12.9% | −19.6% | −15.7% |
| 2.00× | 3.41 | +78.8% | −20.8% | −11.4% | 79.8% | 1/5 | −22.7% | −15.2% | −22.9% | −18.9% |

**Sharpe is flat in leverage** (3.54 → 3.41 across the whole grid — the small decay is the §8 overlay, not the
scaling; this is why the comparative studies elsewhere in this report can stay on the unlevered stack and still
quote the unlevered figure), so leverage buys CAGR and pays in drawdown, exactly as it should. **The worst
month breaks first** — and it breaks immediately: the rung above the shipped one already reads
−6.0% against the −6% floor, while the realised max-DD never breaks across the whole grid
(above ~1.5× the ladder starts cutting and caps it) and the full window's losing streak is
**2–3 at every rung**. The exact ceilings:

| constraint | largest leverage that still holds |
|---|---|
| realised max-DD | 1.65× |
| realised worst month | 1.15× |
| realised months-in-profit | 1.90× |
| bootstrap-P5 max-DD | 1.25× |
| bootstrap-P5 worst month | fails already at 1.00× |
| 2010-event max-DD | **1.05×** |
| 2010-event worst month | fails already at 1.00× |

**The frozen OOS block scores 5/5 at 20 of the grid's 21 rungs, dropping to 4/5 only at 1.50×** and never sets the choice: Sharpe stays in
**[2.97, 3.08]** and the streak **2** all the way, and only the risk metrics
scale — max-DD **−4.9% → −11.1%**, worst month **−2.6% → −5.3%** from 1.00× to 2.00×
(CAGR 29.5% → 58.6%). That block is a calm two years; it is shown, not used.

**The chosen level: 1.15×** (`BOOK_LEVERAGE`, ≈10.7% realised book vol), and on the
six-family book it sits **exactly on** its binding realised constraint rather than inside it: the
realised worst month reads −5.8% here and −6.0% one rung up, so
1.15× is the last level that clears the −6% floor. The tail readings bind harder still — the
**2010-event max-DD** allows only **1.05×**, and both tail worst-month readings fail at 1.00×.
**It stays there, and the reason is the tail rather than the
scorecard.** Two of the seven constraints are already violated at 1.00×, before any leverage is added: the
bootstrap puts a worst month past −6% inside its 5th percentile, and the 2010 flash-crash replay does the same.
Neither improves by spending the headroom the realised metrics appear to leave — both get proportionally worse,
and the event that produces them is the one no term-structure rule reaches (the curve was in contango the
session before). Raising the book to its measured ceiling would buy CAGR against exactly that tail, so the
headroom is not treated as spare risk budget.

*Being explicit about what is measured and what is judgement:* the ceilings above are measured — each is a
constraint evaluated on the grid. Which rung inside them the book takes is risk appetite, and on the shipped
book the scorecard and the mandate agree on the same rung: 1.15× is both the last level whose
realised worst month clears −6% and the level chosen on the tail evidence. Leverage could not have rescued the
targets in any case — months-in-profit and the losing-month streak are properties of the *sign* of a month, and
scaling every return by 1.15 or 1.45 leaves a negative month exactly as negative as it was.

**The full available history says the same thing louder.** Extend the canonical assembly back to 2005 — the
window nobody reports on, because only **three** of the six legs exist there
(vol-premium, crisis, global-macro; the crypto legs have no 2005) — and the book fails the targets at **any** leverage: at 1.00× it
already runs months-in-profit **78.5%** with a **7-month** losing streak
and a **−16.8%** worst month, on a **−17.5%** drawdown that is already outside
the mandate. A window that rejects every leverage cannot choose one, and it is not the shipped configuration —
but it is the only place the 2008 GFC lives, so it is stated, not dropped: with the diversifiers missing, this
book has no leverage headroom at all. The fully-listed era (2011+) is what the mandate is measured on, and there
1.15× holds with the margins above.

**Why not spend the rest of the budget.** Two of the seven constraints are already violated at 1.00×: on an
unlucky path of its *own* return distribution the book's worst month is **−8.9%**, and a repeat of the
2010 event costs **−13.1%**. The realised worst month is not the cushion it looks like either — it reads
−5.8% against the −6% floor at the shipped level, but every extra 0.1× makes the *modelled* tail
proportionally worse while the realised one improves only on paper. Only the drawdown dimension genuinely has
room (−8.3% against −15%). The honest summary: the book is under-risked **on drawdown** and constrained by
**the monthly floor under stress** — not by the realised one, which currently has 1.15× sitting inside its
limits.

**"Then don't lever the aggressive leg" — measured, and it is worse.** The obvious refinement is to leave the
short-vol leg (which carries the tail *and* 64% of P&L) at 1.00× and give the leverage to the other
five. Judged
**vol-matched** — same book risk, since any sizing change that raises risk raises return — it loses:

| | all six legs 1.15× | five legs 1.19×, vol-prem 1.00× |
|---|---|---|
| Sharpe (full / OOS) | **3.53 / 3.07** | 3.25 / 2.90 |
| CAGR | **+43.9%** | +39.9% |
| max-DD | **−8.3%** | −8.3% |
| worst month | **−5.8%** | −5.9% |
| months-in-profit | **82.5%** | 79.3% |
| targets, full window | **5/5** | 3/5 |
| 2010-event DD / month | −16.1% / −13.1% | −13.2% / −9.6% |

The mechanism does work in the direction intended — the event's worst month softens from −13.1%
to −9.6%. But volprem is not merely the aggressive leg, it is also the highest-Sharpe and most
decorrelated one, so shifting weight to the other five (which are more correlated *with each
other* than with it) buys less risk reduction than it costs in return: −0.28 Sharpe,
−4.0pp CAGR, and 3.2pp of months-in-profit, which **breaks the ≥80% target and
drops the book to 3/5**. Pushing further — de-levering
volprem to 0.5× and running the others at 2.0× — gives Sharpe **2.21**, months-in-profit **66%**, a **5-month**
losing streak and **1/5**. This is the §6 reweighting frontier again under a
different name: any move off equal-risk buys worst-month at the cost of months-in-profit. **Leverage stays uniform.**

**Operational reality — can 1.15× actually be filled?** Yes on every leg, and the binding constraint is not
leverage at all:

- **Crypto perps** (carry, breakout, BAB, the crypto side of x-sect and trend) — USD-M margin is 0.5–5% of
  notional, so 1.15× gross is a rounding error against available margin; what actually binds is per-symbol
  position limits and depth in thin names, already charged through the √-impact cost model (§9).
- **US equities long/short** (equity side of x-sect and trend) — Reg T allows 2× gross on a margin account, more
  under portfolio margin; 1.15× is inside it. Shorts need a prime broker and borrow, charged at 50bps/yr (§9).
- **ETFs and futures** (crisis, gmacro) — 5–10% initial margin; 1.15× is not a constraint.
- **The vol-premium leg is where size actually stops**, and for a different reason: short variance is sized in
  **vega notional**, half the 18-leg book sits in thin single-name / exotic ETF-vol underlyings ($10–200k vega per
  roll), and the equal-weight construction is capped around the **low tens of $M** against the **$500k**
  demonstration book ([VOLPREM.md §3b](docs/strategies/VOLPREM.md)). Short-variance margin also rises
  non-linearly in stress — the margin call arrives on exactly the day the leg loses 76%.

So the honest bottom line on sizing is that the same leg sets both limits: vega capacity caps the *dollars*, and
its excluded systemic tail caps the *multiple*. Neither limit is the exchange's.

**The §8 limits stay absolute (in book-equity percent), not scaled with leverage.** The ladder's −6/−9/−12%
triggers, its −4% restore and the −4%/day breaker are quoted as a percentage of *something*, and leverage forces
the choice; both conventions are measured across the whole grid (`limits="book_equity"` vs `"risk_budget"` in
`run_master_book.risk_overlay`):

- Scaling the triggers with leverage keeps the exposure path leverage-invariant, so the levered book is a pure
  constant scaling — tidy, but at 1.5× the deepest "stop trading" rung would sit at **−18%**, past the −15%
  mandate it exists to enforce. A limit that only fires after the mandate is already breached is not a limit.
  Absolute triggers mean a rung always costs the investor the same, whatever the dial is set to.
- The cost of that choice is real, and it grows with the dial: at 2.00× absolute triggers de-risk **11.9%** of
  days (vs 0.5% at 1.00×) and give up 2.1pp of months-in-profit and 5.4pp of CAGR against the scaled convention.
  Worse, above ~1.35× they can *deepen* the very drawdown they police — the ladder de-risks straight into the
  recovery. Measured, in the replay that lands the event in the 2022 bear market: at **1.35×** the levered crash
  day (−7.6%) trips the second rung and the book troughs at **−13.6%**; at **1.40×** the same day (−7.9%) trips
  the *deepest* rung, goes **flat**, and — unable to earn anything back — troughs at **−16.3%**, through the
  mandate. Under scaled triggers the same step is −12.0% → −12.5%. This is not an argument for looser limits: a
  stop that has to fire is a book sized too hot for its own schedule, and it is precisely why the 2010-event
  drawdown ceiling above is **1.30×** and the shipped level sits under it.
- At the shipped 1.15× the two conventions differ where it now counts (max-DD −8.29% compounded vs
  −8.20% on the fixed $500k, worst month **−5.76% vs −5.70%**): the
  absolute-limits convention that ships is the *stricter* one on
  the worst month, and it is the only one of the two that holds the target. That is a second reason to keep it,
  beyond the original one — and it matters more if someone later turns the dial up.

**Sanity check — months-in-profit must be leverage-invariant under pure scaling, and it is.** Multiplying the
1.00× book's returns by 1.00–2.00× leaves months-in-profit at **81.38%** at every level, and the scaled-limits
convention barely moves (81.38% → 80.85% at 2.00×, its exposure path being near-static). Under the shipped
absolute limits it falls **81.38% → 80.32% → 78.72% → 79.26%**
at 1.00 / 1.50 / 1.75 / 2.00× — **that is the §8 overlay, not the scaling**: the de-risked share of days
rises 0.9% → 4.4% → 8.2% → 10.5% and the daily-loss breaker fires 0 → 3 → 9 → 14 times over the same range.
Every flattened day is a day the book does not earn back, so some marginal months flip negative.

**One knob.** The book's risk now has a single source: `BOOK_LEVERAGE` in `src/config.py`, read by the canonical
assembler and by nothing else. The per-leg risk-parity target reads the same `VOL_TARGET_ANNUAL` every sleeve in
the project uses — the assembler no longer carries a `target=0.15` of its own, which is what previously made the
vol target two constants that could disagree. The two dials are now genuinely different things and only one of
them sets book risk: the per-leg target decides how the six legs are balanced *against each
other*, and it is the wrong dial for sizing the book, because it does not scale. Raise it by the same 15% and the
two books differ on **150 of 5,042 days** — the days `_scale`'s 3× cap binds, which is where the legs are most
levered and the book's worst days live. The result is a *quieter* book at the same nominal size
(−7.4% max-DD against −8.3%, worst month −4.8% against −5.8%, identical
Sharpe), and that is precisely the reason not to use the dial: the improvement is a cap clipping the fattest few
percent of days, which is a change of construction dressed as a change of size. Book risk is set by the scalar
that is genuinely a scalar.

## 4c. Risk rules — per family, and at the book (§8)

Risk logic is **specified per family, not applied uniformly**: a leg that adds into an adverse move needs
different protection from one that rides it. Every leg shares one sizing rule (trailing 60-day vol target
to 15% annualised, lagged, capped at 3×) because that is what makes the legs combinable at equal risk;
everything else below differs by family, and the differences are the point.

| family | entry | exit / stop | max holding | max exposure |
|---|---|---|---|---|
| **trend** | EMA(50/200) cross, long-biased | **held to reversal** — no time stop, deliberately: the premium lives in the fat right tail a barrier would cut | unbounded by design | 3× leg cap × 1/8 book risk |
| **breakout** | Donchian-55 break, ML confidence gate | **chandelier ATR(3) trailing stop** + long-trend(100) alignment filter | bounded by the trail | 3× × 1/8 |
| **carry** | funding z-score (7-bar level), top-100 point-in-time names | none — the premium is a *level*, not a move; the position is re-struck each rebalance | to the next daily rebalance; funding charged at every 8h settlement | dollar-neutral, beta-hedged to BTC; 3× × 1/8 |
| **vol-prem** | always short variance, 18 Cboe underlyings at equal risk | the **VIX-term-structure gate** — the leg stands down while either curve segment is inverted | one roll | 3× × 1/8, and sized on its −78% tail rather than its Sharpe |
| **x-sect** | risk-adjusted momentum, top/bottom 30% crypto · 10% equity | none — re-struck at each rebalance | 21 bars (monthly cadence, chosen to keep turnover and cost low) | dollar-neutral; 3× × 1/8 |
| **BAB** | beta-neutral long low-β / short high-β, top 20% of the top-100 liquid names | none — re-struck at each rebalance | 21 days | beta-neutral by Frazzini-Pedersen leg scaling; 3× × 1/8 |
| **crisis-alpha** | multi-lookback time-series momentum (10/20/40, 20/40/63, 40/63/120) on liquid ETFs and FX | signal flip | to the flip | 3× × 1/8 |
| **global-macro** | the same TSMOM tranches on EM FX and commodities | signal flip | to the flip | 3× × 1/8 |

At the book: **gross cap 2.0×** against a shipped constant **1.15×**, net exposure ≈ 0 (every leg is
dollar- or beta-neutral or a spread), **per-family cap 1.5× the equal weight** (which never binds while
the weights are equal), a **daily-loss circuit breaker at −4%** that flattens the following day, and a
**drawdown-responsive ladder** that cuts gross to 0.66 / 0.33 / 0.0 of target at −6% / −9% / −12% and
restores only after recovery to −4% (hysteresis, so the book cannot oscillate across a trigger). The
deepest rung is the answer to *when does it stop trading*: at −12% the book is flat and only a recovery
to −4% turns it back on. All three are causal — they read yesterday's equity, never today's.

## 5. Validation evidence

- **Discovery multiple-testing (the full 2,129-sleeve zoo):** the placebo (shuffled-signal) arm gives a
  **false-discovery rate of 0.6%**; the best single sleeve's **deflated Sharpe is 0.00 at N = 2,129** —
  individually marginal, so the book's robustness is a diversification effect, not a lucky sleeve.
- **Probability of backtest overfitting (CSCV, §6, `scripts/run_cscv.py`):** across all
  C(16,8)=12,870 in/out splits of the trial set (641 sleeves with dense coverage on the 2021+ common window),
  **PBO = 13%**, and the in-sample-best sleeve degrades from
  **+0.088 to +0.004 Sharpe/bar** out of sample (P(selected loses OOS) = 44%) — a quantified confirmation that
  single-sleeve selection is largely overfit, which is exactly why the traded book selects nothing and stacks
  decorrelated premia instead. Deflated Sharpe + placebo-FDR + CSCV together are the mandatory multiple-testing
  triad.
- **The dose-response behind PBO — what a search budget actually buys** (`scripts/run_selection_bias.py`,
  `make selection-bias`; the 641 dense-coverage candidates on CSCV's 2021+ window, median over 7 split
  points, 2,000 random candidate subsets per budget). Sweeping the number of candidates searched from
  1 to 641 and picking the in-sample winner each time:

  | candidates searched | winner's in-sample Sharpe | its out-of-sample Sharpe | inflation | P(winner loses OOS) |
  |---|---|---|---|---|
  | 1 | −0.56 | −0.74 | +0.18 | 77% |
  | 10 | +0.77 | +0.01 | +0.76 | 51% |
  | 50 | +1.28 | +0.11 | +1.17 | 45% |
  | 100 | +1.38 | +0.15 | +1.23 | 42% |
  | 641 | +1.60 | +0.28 | +1.32 | 29% |

  The in-sample column rises **an order of magnitude** with the search budget while the out-of-sample
  column is **flat from N≈10 onward** — the gain is an order statistic, not edge, and the gap (**+0.18 →
  +1.32**) is the part of any mined backtest that belongs to the search. Selection is not *worthless*
  here — the winner does beat an unselected candidate (median OOS **−0.72**) by **+1.00** — but that
  margin is mostly avoiding the structurally cost-killed families, a call available **before** looking at
  results, and the winner still loses out-of-sample **29–51%** of the time. **The same winner, deflated at
  a range of *declared* trial counts** (identical track record, only the disclosed search budget changes):
  N=1 → **0.01**, N=5 → **0.00**, and **0.00** at every rung above it, to the full N = 2,129. The
  multiple-testing penalty is set by a number only the researcher knows, which is why the trial count is
  published here — and why the ladder now ends at whatever the zoo actually mined rather than at a
  constant pinned in the script.
- **Portfolio Monte Carlo:** block-bootstrap 5th-percentile Sharpe **+3.07** (full four-scheme table in §4).
- **Leakage:** execution is delayed to t+2 (never the signal bar's own close); funding is charged at
  every 8h settlement; costs are liquidity-aware (√-impact scaled to bar $-volume, never flat); vol
  targeting uses lagged volatility; feature computability is proven by the shift audit; fixed seeds throughout.

## 5b. Parameter selection & sensitivity (§10)

Parameters are chosen a priori (conventional defaults — MR z-lookback 20 / entry 1.5; trend EMA 50/200;
Donchian 55), deliberately *not* fitted — the point of the rigour is to avoid optimising parameters into
noise. To prove this is not a false negative, `scripts/run_wfo.py` runs a walk-forward selection plus the
full sensitivity surface per family (BTCUSDT 4h, net of costs, t+2 execution):

| Family | surface (min / median / max) | % of grid positive | in-sample best (overfit) | walk-forward OOS (honest) |
|---|---|---|---|---|
| **Trend** (12 EMA configs) | +0.58 / +0.76 / +0.91 | **100%** | +0.91 | **+0.58** |
| **Mean-reversion** (18 configs) | −1.35 / −1.04 / −0.47 | **0%** | −0.47 | **−0.86** |

- **Trend is a broad robust plateau** — every grid point is positive; the honest walk-forward number
  (+0.58) sits near the median (+0.76), modestly below the overfit in-sample peak (+0.91). That small
  peak↔walk-forward gap is the signature of a real edge, not a fitted spike.
- **Mean-reversion is dead across the whole surface** — 0% of the grid positive, and even the best in-sample
  config is −0.47 (walk-forward −0.86). So MR's poor edge-map score is not a bad-parameter artifact; single-asset z-score
  reversion has no edge here net of costs (short-term continuation dominates these trending liquid assets —
  exactly why trend wins).
- One genuine implementation weakness was found and documented (`scripts/meanrev/audit_mr.py`): the event families
  exit at a triple-barrier whose width is one bar of return-vol, so MR trades hold only ~4 bars with huge
  turnover; a proper revert-to-mean exit roughly halves the loss (−2.8 → −1.55 at 1h) but stays negative.
  The uniform triple-barrier is kept for cross-family comparability — MR does not survive under either exit.

The honest way to *use* parameters is this walk-forward loop, with the sensitivity surface reported and a
deflated-Sharpe penalty sized to the grid — never peak-picking on the full sample.

## 5c. Out-of-sample design: walk-forward vs. the final block (§10)

The brief asks for **two distinct things**, and the book has both:

- **A final out-of-sample block, held to the end and run exactly once** (§10/§11) — `OOS_START=2024-07-01`,
  the last ~2 years, never inspected until the end. **§11 scores the targets on this block and nothing else**,
  so it is the deliverable's scorecard (Sharpe **3.07**, 5/5); the 15-year window is reported
  alongside it because §10 asks for per-year/per-quarter metrics and §12 for a ceiling assessment — it is
  supporting evidence, never a second scorecard.
- **A rolling & anchored walk-forward with periodic re-fitting** (§10) at the portfolio level
  (`scripts/run_wf_book.py`), over **all available data** — the non-crypto legs reach back to 2005–2012, so
  the 2016 reporting window is not a data limit; only the crypto legs are stuck at 2020. At each
  rebalance it fits the leg weights on the training window (anchored `[start,t]` or rolling `[t−2y,t]`) and
  applies them to the next block out-of-sample; concatenating the blocks gives an **accumulated out-of-sample
  track over ~20 years (2006→2026), Sharpe 3.28, max-DD −18.6%** — the book is out-of-sample across nearly the
  whole history, not just the final block. *(That track is measured on the **unlevered** stack; at the shipped
  1.15× its drawdown would be outside the mandate. That is a property of the early, thinly-populated
  window rather than of the shipped book — see §4b, "the full available history".)* It is **invariant to the choice**: anchored vs rolling, quarterly vs
  annual re-fit all land Sharpe in **[+3.28, +3.31]** (spread 0.03). **Crisis-window
  stress** on this long track: through the **2008 GFC** the book draws down only **−4.5%** on the
  three legs live then (the crisis / managed-futures leg hedges the volprem short-vol tail),
  −1.7% through 2018 Volmageddon and −0.9% through COVID. *(Caveat: the pre-2020
  crisis/gmacro track runs on **real ETF/FX prices** (SPY/GLD/TLT/EM-FX — the instruments traded and were liquid), so it
  is a strategy-logic backtest, not a live *product* track (the 2008 result is evidence the diversification
  logic works, not a tradeable record); every Sharpe is annualised by the track's actual obs/yr, not a flat 365.)*

**Why the block is ~2 years, and what that length costs.** The brief fixes that a final block exists and is
run once; it does **not** fix its length, so this is a stated design choice. The binding consideration is not
Sharpe but the two **count-based** targets — months-in-profit ≥80% and the ≤2-month streak — which need enough
months to mean anything:

| block length | months | 1 s.e. of Sharpe | months you may lose and still hold ≥80% |
|---|---|---|---|
| 1 year | 12 | ±1.01 | 2 |
| **2 years (shipped)** | **26** | **±0.71** | **5** |
| 3 years | 36 | ±0.58 | 7 |
| 5 years | 60 | ±0.45 | 12 |

At one year a ≤2-month streak target is close to a coin flip and one bad quarter breaks months-in-profit. Going
longer is bounded from the other side: the crypto legs list only from 2020, so a 5-year block would leave a
single year of live crypto history for construction. **The honest caveat that comes with the choice: at 25
months the standard error of the OOS Sharpe is ±0.70, so 3.07 is 3.07 ±0.70**, and the block is a benign
stretch — its only real stress is the Aug-2024 yen-carry unwind (book −1.0%), while Apr-2025 was *positive*
(+1.0%). The boundary is not re-cut now that results are known: moving it after the fact is window-shopping,
which is exactly what `OOS_START`'s frozen-constant comment forbids.

**The one change made after the block existed, and how its integrity was restored.** The two-segment regime
gate (§5d) was chosen *after* the block had been created, and the gate lab initially printed the block's
metrics next to every candidate — enough to contaminate a run-once holdout even if it was not consciously
used. The selection is therefore re-run on data that **stops at 2024-06-30** (`SELECT_END`,
`run_vol_premium_gates.py`), with the block printed only afterwards as a read-out. The choice reproduces
without it:

| volprem leg, book scored 2011-01 → 2024-06 | Sharpe | max-DD | worst month | months | streak | targets |
|---|---|---|---|---|---|---|
| ungated | 3.36 | −8.3% | −5.1% | 77.1% | 3 | 3/5 |
| long segment only (the previous rule) | 3.39 | −8.3% | −4.9% | 77.1% | 3 | 3/5 |
| fast segment only | 3.62 | −7.4% | −4.4% | 79.3% | 3 | 3/5 |
| **both segments (shipped)** | **3.62** | **−7.4%** | **−4.4%** | **79.3%** | **3** | **3/5** |
| both + re-entry 5d | 3.25 | −8.3% | −4.3% | 74.5% | 3 | 3/5 |
| SHIPPED + re-entry 5d | 3.39 | −8.4% | −5.0% | 78.2% | 3 | 3/5 |

Same winner, same margin, same reason the runners-up are rejected — so the rule is recoverable from
pre-block data alone and the block's one-shot status survives in substance. The threshold surface and the
200-draw block-random null are scored on the same truncated window.

**Why the full-sample number is itself an honest OOS estimate.** The portfolio weights are a-priori equal
(1/N), so there is nothing to fit at the book level — its walk-forward *equals* its full post-burn-in track.
That a-priori choice is justified with evidence, not assertion: **re-fitting the weights out-of-sample does
not beat equal weight** — an inverse-vol walk-forward nets +2.82, and a trailing mean-variance (Sharpe-max)
allocation nets +3.68 but on a **−44% drawdown (3× the equal-weight tail)** — the classic overfit signature.

**What is and isn't fitted (the obvious question).** The task invites modelling (§5), and we fit where
fitting is *validated*: the LightGBM **meta-label** models (fit inside purged/embargoed folds, with a non-ML
baseline so incremental value is measured — consolidated per family in §5d), the per-family **parameter walk-forward** (§5b), and the
discovery-zoo **sleeve selection** (re-selected each rebalance, WF-OOS Sharpe +0.13). What is deliberately
*a-priori* is (a) the classical-rule parameters — the sensitivity surface shows the edge is a plateau, not a
fitted spike, so optimising them would only add overfit — and (b) the portfolio weights (equal, justified
above). **Nothing is fit against the final OOS block.**

## 5d. Non-ML baseline vs. ML — the model's incremental value, per family (§5)

The brief (§5) requires a **non-ML baseline per family so the model's incremental value is measurable.** It is,
and the verdict is consistent across the book: **ML is a risk-and-confidence layer, not an alpha source.** Each
portfolio family carries a parameter-light classical rule as its baseline; the ML overlay is measured against it
under purged/embargoed CV on the same held-out block, so the delta below is the model's honest incremental value —
not an assertion.

| Family | Non-ML baseline (Sharpe) | + ML overlay | Measured incremental value |
|---|---|---|---|
| **Trend** | ungated EMA-50/200 + chandelier **+0.67**, DD −14.4% (OOS +0.35) | meta-gate LightGBM **+1.00**, DD **−2.9%** (OOS +0.05); conviction-sizing +0.82, DD −1.0% (OOS +0.20) | **Risk-cut, not alpha.** DD collapses −14%→−1%, OOS return stays flat — the meta-model's OOS AUC is **0.505** (a coin-flip), so the DD cut is *mechanical* (fewer positions), not predictive. LSTM/GRU/TCN/Transformer all net-negative vs the EMA rule's +0.15. |
| **Breakout** | ungated Donchian-55 **+0.41**, DD −13.3%, prec 36% | LightGBM gate **+1.01**, DD **−2.7%**, prec 39% (OOS **0.20→0.47**) | **Risk reduction + OOS robustness** — and the one family where the gate also **lifts OOS**: filtering false breakouts pays most in chop and rescues the 1h timeframe. |
| **Cross-sectional momentum** | risk-adj-mom rule: crypto **+0.71**, equity **+0.62** | LTR-LightGBM: crypto +0.61; equity mixed-panel **+0.89** (DD −32%→−25%), broad survivorship-free +0.17 | **Conditional.** Combines factors on the rich equity panel (+0.62→+0.89), but only adds estimation noise on clean daily crypto and conjures no edge on the honest broad universe where the rule is already ~0.4. |
| **Carry** | linear funding rank **+1.21**, DD −22% (MC-P5 +0.58) | ranker best +0.61 (funding-only *inverts* −0.70); **timing gate +1.52**, DD −16%, MC-P5 **0.58→0.94** | **Signal no, risk yes.** The ranker overfits low-SNR funding features; a regime-timing overlay cuts DD and lifts the MC-P5 — again risk reduction, not new return. |
| **Short-vol / VRP** | always-short, **parameter-free +0.81** | — (none) | **No ML by design.** A parameter-free baseline *is* the strongest robustness evidence (no knob to fit); a forecaster would only add overfit to a structural premium. |
| **BAB / low-vol** | classical FP beta-neutral **+0.77** | best forecaster Ridge-on-all **+0.43** (MC-P5 −0.19, not robust); trees go negative | **No alpha.** The premium pays through the *risk channel* (rank by beta, de-lever by beta) — a beta-only forecaster scores **+0.00** — so return-prediction is structurally the wrong tool; ML adds overfit risk, not edge. |

Across six structurally distinct families the model **never manufactures out-of-sample alpha that is not already
in the cross-section**; its honest, repeatable value is drawdown control and entry precision (trend, breakout,
carry). ML was also run on two *rejected* families and **confirmed the rejection**: on-chain — a 21-trial ranker on
on-chain features nets **+0.32** while the *identical harness on price features* nets **+1.09** (the method works,
the data doesn't); and calendar-seasonality — a purged-CV pre-FOMC gate makes SPY *worse* (0.24→0.07, negative OOS
IC), because removing the beta removes the return. Full per-family model grids and leakage controls:
[TREND.md §7](docs/strategies/TREND.md), [BREAKOUT.md §6](docs/strategies/BREAKOUT.md),
[XSECT.md §7](docs/strategies/XSECT.md), [CARRY.md §3](docs/strategies/CARRY.md),
[BAB.md §3d](docs/strategies/BAB.md), [ONCHAIN.md](docs/strategies/ONCHAIN.md); artifacts under
`reports/{trend,breakout,carry,onchain,xs}/`.

**Does it help the assembled book? (measured — leg-swap through the risk-parity assembly, judged on all five
targets, not Sharpe alone).** The book already clears every target, so what an ML lever has to do is hold the
one with the least room — the **worst month**, at −5.8% against −6% — while adding something.
That is the honest bar, and no lever clears it. Swapping one family's leg for its ML variant, book otherwise
identical. **This is one of the A/B tables noted in §4: its baseline is the eight-family book those swaps were
run against, so read the delta between rows, not the level.** The **leg-standalone** column is each family's raw Sharpe re-measured inside this swap
harness (equal-weight core-10; trend held crypto-only); it lines up with the deep-dive figure above (trend
**+0.69** here ≈ **+0.67** there), so read the raw→ML *delta* within each table, not the standalone level.
Reproduce: `make ml-contribution` (`scripts/run_ml_book_contribution.py` → `reports/book/ml_book_contribution.json`):

**Sharpe alone would have mis-scored this, so return is measured too.** Every leg is vol-targeted, so an ML
gate that halves time-in-market can leave Sharpe *and* drawdown flat while cutting the money the book makes —
Sharpe is a ratio and cannot see it. `run_ml_book_contribution.py` now records **CAGR on both windows**
alongside the five targets, and the column changes how two rows read:

| book, leg swapped | Sharpe full / OOS | **CAGR full / OOS** | max-DD | worst month | months | streak |
|---|---|---|---|---|---|---|
| **baseline (shipped)** | **+3.60 / +3.39** | **34.2% / 26.7%** | **−6.5%** | **−3.8%** | **79%** | **3** |
| breakout raw (no ML) | +3.61 / +3.49 | 34.3% / 27.8% | −6.5% | −3.9% | 80% | 3 |
| breakout + ML *(shipped)* | +3.60 / +3.39 | 34.2% / 26.7% | −6.5% | −3.8% | 79% | 3 |
| trend raw | +3.56 / +3.13 | 35.9% / 26.6% | −6.9% | −4.2% | 82% | 2 |
| trend + LightGBM gate | +3.61 / +3.22 | 35.3% / 25.1% | −6.9% | −3.8% | 83% | 2 |
| trend + RF gate | +3.61 / +3.35 | 34.9% / 25.3% | −6.9% | −3.6% | 84% | 2 |
| carry + timing overlay | +3.58 / +3.39 | 33.9% / 26.6% | −6.5% | −3.8% | 81% | 3 |

**Every ML lever costs return, including the ones whose Sharpe improves.** The trend RF gate reads as the
biggest Sharpe win on the block (+3.48 → +3.74) while *losing* 1.4pp of OOS CAGR: it cuts risk faster than
return, which flatters the ratio and shrinks the P&L. On the brief's $500k that 1.4pp is ~$7k/yr paid for a
ratio. This is the sharpest single argument for keeping the book's ML footprint to the one leg where the gate
also lifts OOS *return* — and it is only visible because return is now in the table.

| ML lever | leg standalone (raw → ML) | book OOS Sharpe | book OOS months-in-profit | on the binding axis |
|---|---|---|---|---|
| **Breakout** meta-gate *(shipped)* | +0.68 → +1.06 (DD −10.8%→−3.7%) | 3.88 → 3.77 | 88% → 85% | ≈flat — small give-back, still clears |
| **Carry** timing overlay | honest leg +1.33 → +1.04 | 3.77 → 3.77 | 85% → 85% | flat on both axes |
| **Trend** meta-gate / conviction † | +0.69 → +0.57…+1.00 | 3.48 → 3.60…3.74 | 88% → 81…85% | cuts the sub-leg's risk |
| **X-sect** learning-to-rank | crypto rule +0.71 > LTR +0.61 | — (loses standalone) | — | nothing to add |

† the trend swap holds the leg crypto-only, so both rows sit *below* the shipped raw+equities trend leg; the Δ
is the ML effect within that sub-construction. Every move is ≈1 month on a 25-month block — MC noise — so the
*direction* matters more than the size. Judged on the full scorecard, the picture is exactly what the standalone
table implies: ML's value is **drawdown and consistency inside a sleeve**, and at book level every lever moves
OOS Sharpe within ~0.1 and months-in-profit by a few points — the carry and trend overlays nudge months up, the
breakout gate slightly down, and **none moves the binding full-window losing streak**. What ML does **not** do
here is lift Sharpe (it is not a return-forecaster — §7), buy DD or worst-month headroom the book does not need,
or close the structural months-in-profit / streak gap — the short-gamma legs crash together, and only the crisis /
global-macro long-gamma legs (already in the book) address that. The **carry** overlay's +1.21→+1.52 standalone
lift, by contrast, is construction-specific: it does not reproduce on the honest carry leg (+1.33→+1.04) and is
flat-to-slightly-positive on the book.

**Forecasting the risk side instead of the direction — two further tactics, both measured, both rejected.**
Arms A–C all predict a *first* moment (P(book up), family forward return), and direction is the lowest-signal
quantity on this book — the §11 scorecard does not even ask for it. So two arms aim at what actually binds
(`run_ml_portfolio_overlay.py`, arms **D** and **E**), each against a leverage-matched flat control:

| arm | what it forecasts | Sharpe full / OOS | CAGR full / OOS | max-DD | months | streak | targets |
|---|---|---|---|---|---|---|---|
| **flat control (1.04×)** | nothing | **+3.73 / +3.77** | **+37% / +32%** | **−6.7%** | 81% / 85% | 2 | 5/5 |
| D vol-target, Ridge | forward 21d realised vol | +3.67 / +3.69 | +36% / +32% | −7.6% | 80% / 81% | 2 | 5/5 |
| D vol-target, LightGBM | forward 21d realised vol | +3.70 / +3.73 | +36% / +35% | −7.2% | 81% / 85% | 2 | 5/5 |
| D trailing-vol target, **no ML** | — (60d trailing) | +3.61 / +3.61 | +38% / +37% | −8.2% | 80% / 85% | 2 | 5/5 |
| E bad-month gate, logistic | P(next 21d in bottom decile) | +3.52 / +3.77 | +29% / +30% | −6.6% | **68%** / 85% | **31** | **3/5** |
| E bad-month gate, LightGBM | P(next 21d in bottom decile) | +3.48 / +3.70 | +28% / +30% | −6.6% | **68%** / 85% | **31** | **3/5** |

- **D fails against its own control, and so does its non-ML sibling.** A second layer of volatility management
  adds nothing because the legs are *already* vol-targeted individually — the forecast is re-forecasting a
  quantity the construction has already neutralised, and pays for it in drawdown (−6.7% flat → −7.6% ML).
  Note the trailing-vol version is worse still, so this is not "the model was too weak": the tactic is empty.
- **E fails in an instructive way.** Aiming the classifier straight at the binding metric makes that metric
  *worse*: months-in-profit collapses 81% → 68% and the losing streak explodes to **31 months**. The reason is
  mechanical and general — **de-grossing to flat produces a zero month, and a zero month is not a profitable
  month**. A model that correctly sits out bad stretches still destroys months-in-profit and manufactures an
  enormous streak. Months-in-profit cannot be bought by *avoiding* anything; it can only be bought by owning
  something that pays while the rest bleeds, which is what the crisis / global-macro long-gamma legs do.

That is the deeper reason the short-vol VIX gate works where every ML tactic fails: it does not de-gross the
book, it moves *one* leg out of a regime the other five still trade through, so the flat days are not flat
months.

**Per-SLEEVE, not per-family — the finest cut, and the one where hindsight is measurable.** A family
average can hide a heterogeneous effect, so the gate is also asked "*which* sleeve", at asset × timeframe
granularity across the 21 trend sleeves that have enough trades to meta-label
(`scripts/trend/run_trend_sleeve_ml.py`). In-sample the gate lifts Sharpe on only **6 of 21** sleeves — but it
cuts drawdown on nearly all of them, and the spread is wide (BTC-4h +0.32 → +0.89; SOL-4h +0.43 → −0.19). So
the question is real. Four arms, including one that is deliberately **not** a result:

| arm | Sharpe full / OOS | **CAGR full / OOS** | max-DD | worst month | months | streak |
|---|---|---|---|---|---|---|
| gate NONE *(shipped)* | +0.69 / **+0.35** | **+6% / +3%** | −14.4% | −4.9% | 53% | 5 |
| gate ALL (a-priori) | **+1.00** / +0.05 | **+2% / +0%** | −2.9% | −1.3% | 56% | 4 |
| gate SELECTED (walk-forward) | +0.69 / +0.27 | +4% / +2% | −11.7% | −3.3% | 53% | 5 |
| *gate ORACLE (hindsight — a ceiling, not a result)* | *+0.97 / +0.59* | *+6% / +4%* | *−11.0%* | *−4.1%* | *54%* | *5* |

- **Selecting sleeves honestly does not beat selecting none.** The walk-forward arm — a sleeve is gated for the
  next year only if gating beat its own baseline strictly before that date — lands **OOS +0.27 against the
  ungated +0.35**, and on less return. It gates 7.5 of 21 sleeves on average, and the churn costs more than
  the picking earns.
- **The oracle proves the effect is real *and* unharvestable.** Choosing the same sleeves with full-sample
  hindsight does lift OOS to **+0.59**; the honest walk-forward gets **+0.27**. That gap is the hindsight
  premium, and quoting the oracle as a result is precisely the error this arm exists to price.
- **Gate-ALL is the return-blindness trap in its purest form:** Sharpe rises 0.69 → 1.00 while CAGR falls
  **6% → 2%** and OOS Sharpe collapses to +0.05. It is not making money more efficiently; it is barely
  trading.

**Selective, uniform, and objective-aligned — all measured; none lifts the book.** Three further tests close
the question. **(1) Uniform application** (the anti-cherry-pick control): fitting the *same* purged-CV confidence
gate to all six legs a-priori, hard-gating either **loses** OOS Sharpe (logistic 3.77→3.53) or gains
(boosting +0.15) — and only by **cutting OOS months-in-profit to 69–77%** (from 85%) and returning the
full-window **streak to 3**; it trades away the exact metrics that bind, so any Sharpe bump bought that way is
not an improvement. The cherry-pick (gate only where it helps standalone) reads higher full-sample but breaks
months-in-profit to **77%** — exactly the overfit it looks like. **(2) Objective-aligned sizing:**
the meta-model optimises a binary win/loss log-loss → *precision*, not Sharpe/PnL (a +0.1% and a +50% win share
the label `1`), so it is misaligned by construction — which is why it is flat on fat-tailed trend (OOS AUC 0.505).
Replacing it with magnitude-aware sizing (regress the forward return, size by expected magnitude) is also within
noise (Ridge +0.01 OOS at 81% months-in-profit, boosting −0.05). **(3) Direct-Sharpe allocation** is already tested (§5c) and overfits (trailing
mean-variance buys Sharpe on a 3× drawdown tail). The implementation is textbook AFML meta-labelling over a
**rule** primary — its correct niche (a meta-model adds no information to an ML primary; it earns its keep
filtering a high-recall rule, not manufacturing alpha) — and the literature agrees with the measurement: on
low-signal data ML's honest value is risk / regime / sizing / precision, not alpha, and the dominant failure mode
is overfitting (deflated Sharpe, CSCV/PBO). The binding gap (months-in-profit) is **structural** — the
short-gamma legs crash together — so no re-objectiving of the *per-sleeve* ML manufactures the missing consistency;
the long-gamma crisis / global-macro legs already in the book are one honest fix.

*Every A/B in the rest of §5d is measured on the **unlevered stack** (1.00×), so the arms are like-for-like and
free of the book's separate sizing dial (§4b); Sharpe is flat in leverage, and CAGR / drawdown scale with it.*

**Where regime-conditioning *does* lift the book — and a simple rule beats every ML engine.** The one place it
genuinely helps is at the **portfolio** level, on the leg that drives the tail (volprem), using a **real crash
predictor** — the **VIX term structure** (an inverted curve precedes short-vol crashes) — to flatten the
short-vol exposure before spikes. The ML gates (logistic and LightGBM; a wider six-engine sweep agrees) land at
book Sharpe **3.59–3.68** full / **3.03–3.07** OOS — the engine does not matter. **A parameter-light non-ML rule
beats them all**: flatten volprem unless **both** curve segments are in contango (VIX3M/VIX ≥ 1 *and* VIX/VIX9D ≥ 1
— the contango/backwardation boundary on each, un-fitted, causal on the prior close) nets book
**Sharpe 3.36 → 3.62** full (**OOS 2.55 → 3.39**), **months-in-profit 77.1% → 79.3%**, **worst month −5.1% → −4.4%**,
**max-DD −8.3% → −7.4%** and the **losing streak 3 → 3** — taking the leg's own scorecard from **3/5** to **3/5** (§6),
positive in all 16 years. It is the *timing*, not de-risking: a **constant** cut to the same average exposure does
nothing (OOS 2.93 → 2.90) and a **random** gate stays at full Sharpe **3.08–3.34** (20-draw placebo, below the
rule's 3.73), so the VIX signal — legitimate point-in-time macro (§9) — is doing real work. **The switching is
charged**: the same rule with the gate multiplied onto finished P&L instead of run through the sleeves reads
**4.05 full / 3.99 OOS** and would *overshoot* the ≤4.0 Sharpe band — that gap is the vega spread the gate really
crosses, ~27 switches/yr, and it is paid, not assumed away. Two curve segments beat one: the long segment alone
catches 4 of the leg's 10 worst days, both together catch 9 (`make volprem` → `volprem_gates.csv`). The honest lesson, on
an ML-graded task, cuts against the grain: **the value is the VIX signal, not the model — a rule beats the ML.**
This gate ships as part of the volprem strategy (`src/risk/vol_regime.py`, its `ret_gated` series), not as a book
overlay; the per-sleeve verdict and this
win share it — ML/regime-conditioning pays where it manages *risk/tail-timing*, never as a return forecaster.
Reproduce: `make ml-contribution`.

**And ML *on top of the whole book* does not lift it either — three tactics, six engines (`make ml-portfolio`).**
The VIX win above is *surgical*: it gates the single tail leg. Putting an ML layer on the **whole assembled book**
instead helps on no honest reading of the five targets (causal walk-forward, quarterly refit, 21-day embargo,
judged on Sharpe / CAGR / max-DD / worst-month / months-in-profit / streak, full + OOS). **(A) A whole-book regime
gate** — logistic / RF / ExtraTrees / HistGB / LightGBM / MLP predicting P(book up next 21d) — flattens 14–26% of
months, and a flat month is a non-profit month, so it *worsens the binding targets*: months-in-profit
**79.3% → 64–66%**, Sharpe **3.60 → 3.35–3.44**, CAGR
**34% → 26–28%** on the full window (compounding **98× →
35–48×** the starting capital), taking the book from **5/5 to 3/5**; the marginal OOS uptick is short-block
OOS-fit, and a **constant** cut to the same average exposure matches it (a 20-draw **random** gate spans 3.00–3.40
full — the ML adds no timing beyond de-risking). **(B) Soft exposure** (scale gross by the probability, cap 1.5×) is
just leverage — CAGR rises to **50–52%** but max-DD **−9.9/−10.0%** and worst-month **−7.6/−7.7%** break the
worst-month target. **The leverage-matched control settles that** (the arm without which "ML raised the return" is
unfalsifiable — any size dial raises return): flat gross at the *same* average exposure (1.35× / 1.41×, no model)
nets the **same** CAGR (**50% / 52%**) on a **shallower** drawdown (**−9.2% / −9.5%** vs −9.9% / −10.0%) and a
better worst month (**−7.1% / −7.3%** vs −7.6% / −7.7%) — return-per-max-DD **5.41 / 5.52 vs 5.08 / 5.26**, at the
**same 4/5**. The extra return is the size dial, not the model; out-of-sample the model's sizing edges CAGR
(49–50% vs 43–45%) on a deeper drawdown — a wash on return-per-risk (8.57 / 8.63 vs 8.35 / 8.41). That the flat
control scores well is *not* a licence to lever to 1.4×: what the size dial is worth, and where it stops, is
settled on the risk budget in **§4b** — which lands at 1.15×, not 1.41×, and every one of these arms breaks the
worst month by a wide margin.
**(C) ML allocation** (tilt the family weights off equal by predicted forward return) *collapses* the book — Sharpe
**3.73 → 1.4**, months-in-profit 41–43% — trading away the decorrelation that is the whole edge (the mean-variance
overfit signature of §5c). This is precisely *why the VIX gate is applied to one leg and not the book*:
portfolio-level ML manages risk at best, never manufactures alpha, and de-risking the *whole* book cannot lift
months-in-profit because flat months are not profits — only flattening the single tail leg (volprem) removes the
crash months while the other five families stay invested and earning. Reproduce: `make ml-portfolio`
(`scripts/run_ml_portfolio_overlay.py` → `reports/book/ml_portfolio_overlay.json`).

## 6. Ceiling assessment & honest limits

- **Reachable here:** a diversified six-family book at full-sample Sharpe ≈ **3.53** net
  (+37.0%/yr on the brief's $500k at 1.15×, drawdown −8.3%,
  block-bootstrap MC-P5 **+3.07**) that meets **all five** targets on the window the brief
  scores — the final out-of-sample block (Sharpe **3.07**) — and **all five** on the full
  15-year window (Sharpe **3.53**, missing on nothing). Vol-premium anchors the Sharpe;
  five decorrelated sources cut its tail; a VIX-term-structure regime gate times the
  short-vol leg out of the crashes that used to break the worst month and cluster the losing months; and the
  crypto cross-sectional sleeve runs on **residual (idiosyncratic) momentum**, a better-built momentum that
  steadies recent-year consistency. **What clears the scorecard is dynamic tail-timing plus a better-built
  momentum, not reweighting** (next bullet) — and, for the last two targets, dropping two families (§6d-ter),
  which is why passing is reported with its cost attached rather than as a clean result. **The worst month is
  what binds now**, at −5.8% against −6%, and the honest reading is that the five together are at
  the edge of what this data supports rather than comfortably inside it. The realistic ceiling on liquid assets
  net of honest costs is **~3.3–3.8 depending on the window**.
- **Reweighting cannot close months-in-profit — but tail-timing can.** Every *static reweighting* route
  (adaptive inverse-drawdown, inverse-vol, trailing mean-variance, per-leg dispersion caps) forces
  *over*-weighting the short-vol leg to lift months, which deepens the worst month past −6% and collapses under
  ±25% perturbation — so on the **weighting axis** ≥80% months genuinely fights ≥−6% worst-month, a real frontier
  (quantified next). What closes it is a different mechanism: a **dynamic VIX-term-structure gate** that
  *under*-weights the short-vol leg only when the curve inverts — *avoiding* the crashes rather than trading them —
  lifting months-in-profit to **79.3%**, holding worst-month at **−4.4%** and cutting the losing streak to **3** at
  once (unlevered A/B; worst month **−5.8%** at the shipped 1.15× of §4b). That the overlay beats
  every ML engine and a constant/random control (§5d) confirms it is the VIX timing, not a fitted corner.
- **The *reweighting* frontier, quantified** — it is the *weighting* axis that is capped; the VIX tail-timing
  above sidesteps it (`scripts/frontier.py`, `reports/figures/frontier.png`): a **2,000-sample random search over
  the family weights** (seed 7) reaches **5 of 5 targets in 0 of 2,000** weightings — months-in-profit ≥ 80% in
  only **1**, and months ≥ 80% *and* worst-month ≥ −6% *together* in **0**. A single-knob volprem-weight sweep
  shows why — months-in-profit only rises by over-weighting the short-vol leg, which breaks worst-month:

  | volprem weight | Sharpe | months-in-profit | worst month | targets |
  |---|---|---|---|---|
  | 0.5 | 2.54 | 72% | −5.5% | 3/5 |
  | 1.0 (equal risk) | 3.05 | 74% | −6.0% | 3/5 |
  | 1.5 | 3.45 | 76% | −6.4% | 2/5 |
  | 3.0 | 4.04 | 79% | −8.6% | 1/5 |

  Months-in-profit never reaches 80% by *reweighting*, and holding worst-month ≥ −6% caps it near 74% — the
  weighting axis genuinely cannot hit 5/5. The **dynamic VIX tail-timing gate breaks that trade-off** — it
  avoids the short-vol crashes instead of trading them — so the shipped master book reaches months
  **82.4%** *and* worst-month **−5.8%** together, which the weighting axis cannot do at
  any weight. That beats the frontier: the scorecard reads **5 of 5** on the scored block and
  **5 of 5** on the full window. The mechanism is the right one, and on its own it was still not
  sufficient — the last two targets came from the composition choice in §6d-ter, not from the gate. *(Sweep on
  the core-family book; the deliverable is the six-family master with the VIX overlay.)*
- **Every diversifier earns its place — crisis-alpha is not redundant after the VIX gate (checked).** The VIX
  gate times the *short-vol* leg out of vol spikes; the crisis / managed-futures leg hedges the *broad-market*
  crashes it cannot, and the two are complementary, not overlapping. Dropping crisis (7 families + VIX gate)
  *raises* Sharpe to **4.19 full / 4.33 OOS** — but that **overshoots the ≤4.0 target band** (both windows then fail
  the Sharpe target), breaks the worst month (**−7.8%** full), returns the streak to **3 months**, and cuts OOS
  months-in-profit to **77%** — the scorecard falls to **2/5 full and 3/5 OOS**.
  It also removes real crash protection: through 2018-Q4 the book returns **−3.2% with crisis vs −7.2% without**,
  COVID **+0.9% vs −1.4%**. (Remove the crisis family from `run_master_book.FAMILIES` to reproduce.)
- **Binding constraints:** costs/turnover kill 5m (and most 15m) sleeves; the surviving edge is
  **crypto-heavy** (no family spans both classes since trend was dropped), so the book carries crypto-regime risk (visible as the
  book's thinnest years — 2026 at +0.1 — positive but well below the
  full-sample Sharpe); individual-sleeve significance is low — the edge is in the decorrelation.
- **What would extend it (honest next steps):** cross-sectional momentum on a **broad small/mid-cap
  universe** (it was weak on 20 mega-caps but is a documented edge with breadth); the meta-label
  confidence gate applied across all sleeves to lift entry precision (already built and demonstrated, §2);
  alternative data (order-flow, funding-basis term structure) for signal not present in price alone. **On-chain was tested and is dead on free data** ([docs/strategies/ONCHAIN.md](docs/strategies/ONCHAIN.md),
  §7) — including the exchange-flow series that turned out to be free for BTC/ETH after all, which beats
  buy-and-hold only by collecting beta and loses to random timing of its own position path. What is left
  is paid **entity-level** flow labelling, the illiquid small-cap tail the tradable funnel excludes, or —
  free and now also tested — chain fundamentals (fees/revenue/TVL), where crypto inverts the value
  premium and neither direction moves the book.

## 6b. Structurally distinct sources built to diversify the trend premium

The "one dominant source" limitation above is addressed by adding sources that are *not* price-trend,
each evaluated in its own deep-dive with the same rigour (vol-target 15%, t+2, liquidity-aware costs,
placebo, walk-forward, correlation to this book):

- **Cross-sectional dollar-neutral carry** (funding, [docs/strategies/CARRY.md](docs/strategies/CARRY.md)) — Sharpe **+1.21**
  (refined +1.47, walk-forward OOS +1.40), correlation to this book **−0.04**. Funding/positioning,
  not price trend.
- **Short-vol / variance risk premium** (Cboe VIX/VXN/RVX/… vs realised, [docs/strategies/VOLPREM.md](docs/strategies/VOLPREM.md)) —
  short gamma vs the book's long gamma, placebo-confirmed, correlation ~0. Deployed as a diversified
  **18-underlying Cboe book** (equity-index / single-name / international / commodity incl. gold-miners /
  rates vol indices; crypto DVOL and FX EVZ excluded on frozen ex-ante rules): standalone Sharpe **+3.58**,
  but Sharpe overstates — the honest metrics are skew **−18** and a **−78% systemic-vol tail** that
  diversification softens but cannot remove. In a momentum+carry+VRP blend it peaks **1.77 → 1.84 (10%
  weight) → 1.58 (30%)** (`reports/volprem/volprem_marginal.csv`) — a modest lift that reverses past ~10% as
  its tail dominates. In the master book it sits at equal weight (1/6) and anchors the Sharpe; must be
  sized with a tail hedge.

- **Betting-against-beta / low-vol (BAB)** (beta-neutral, [docs/strategies/BAB.md](docs/strategies/BAB.md)) —
  the leverage-constraint premium: long low-β / short high-β with Frazzini-Pedersen leg-scaling. **Clears the
  bar in crypto** (beta-neutral walk-forward OOS +0.67 top-100 / **+1.52 top-25**, MC-P5 +0.90, deflated 1.00);
  standalone ~1.29 rescaled, ~uncorrelated to the rest (corr ≈ +0.17 to the book). The concentrated **top-25
  crypto** book is the leg in §4. Equity BAB's signal is gone (post-2010 crowding) and FX is dead — a clean
  crypto-alive / equity-gone / FX-dead gradient; capacity-limited to the liquid majors.

Together with **trend**, **breakout**, **cross-sectional momentum**, **crisis-alpha**, **global-macro** and
**betting-against-beta / low-vol**, these give **eight structurally distinct sources** that survive validation;
**six of them are traded** (§6d-ter drops trend and carry, and prices what that costs), and
those six are what §4 assembles. Cross-sectional **reversal**, stat-arb **pairs**, the
**calendar/session** family (both **overnight** and **pre-FOMC/turn-of-month**, H4) and **skewness/lottery
(MAX)** families were tested for this role and rejected (§7, all real-but-beta or negative).

## 6c. The search for a second long-gamma source (§12)

§5d's iron law — de-grossing turns a losing month into a flat one, and a flat month is still not a
profitable month — means the only way to buy months-in-profit is to **own something that pays while the
short-gamma legs bleed**. Two such legs are already in (crisis-alpha, global-macro); both are trend, so
both need a crash to last long enough to trend into. `scripts/run_longgamma_search.py` searches for a third
that is convex a different way, scoring each as one more equal-risk family through the canonical assembler on
the **selection window** (pre-OOS), with the frozen block as a read-out only. *(Run against the eight-family
book — see the A/B note in §4; the levers and their verdicts are unchanged by the composition.)*

| candidate | standalone Sharpe / skew | COVID crash | yen unwind 2024 | as a full 9th family |
|---|---|---|---|---|
| A term-structure-timed long VIX | +0.29 / **+5.3** | **+84%** | +1% | 3/5 — months 79%, streak 3 |
| B vol-timed haven basket (gold/duration/JPY) | +0.19 / +1.0 | +19% | +20% | 3/5 — months 80%, streak 3 |
| C **long** crypto variance | +0.64 / **+14.9** | — (2021+) | +22% | 5/5 but −0.17 Sharpe, no gain |
| D long correlation (long index var, short single-name var) | **−3.58** / −7.6 | +11% | +6% | **0/5** — it is short the *expensive* leg |
| E long variance, only while the curve is inverted | +0.64 / **+29.3** | **+224%** | **+141%** | 3/5 — worst month −7.8% |

**C is the one genuinely new source found**, and it is the mirror of a result the volprem deep-dive already
owns: under the honest OHLC realised leg crypto short-vol is *negative* (BTC −0.41) because the intraday
path is unhedgeable for a short — so the **long** side of that same swap is a paid long-gamma leg, skew
**+14.9**. **D is dead on arrival** and worth stating: long-correlation sounds like crisis alpha but it is
structurally short the single-name variance premium, which is the richest premium in the whole vol complex.

**At full parity every candidate fails, and they fail the same way** — the hedge is handed an earner's risk
budget and pays for it every calm month. That is a sizing error, not a verdict on the source, so the size is
swept (share of one equal-risk slot):

| E, share of one slot | selection window: Sharpe / CAGR / worst month / months | frozen block: Sharpe / CAGR |
|---|---|---|
| 0 (shipped) | +3.60 / +45.5% / **−5.8%** / 83% | **+3.07** / +34% |
| 0.15 | +3.61 / +44.8% / −5.6% / 84% | +2.71 / +40% |
| 0.25 | +3.58 / +44.2% / −5.5% / 84% | +2.26 / +43% |
| 0.40 | +3.50 / +43.4% / −5.4% / 85% | +1.80 / +48% |

On the selection window the hedge does the right thing and does very little of it: the worst month gains
**0.3pp** of headroom (−5.8% → −5.5% at a quarter slot) and months-in-profit ticks 83% → 84%, for **1.3pp
of CAGR**. Month by month the effect is the same size — the book's worst month is Apr-2020 at −5.76%, and
a quarter slot of E takes it to −5.53%; the largest single improvement anywhere in the record is Nov-2011,
−2.88% → −1.99%. An earlier version of this section claimed 1.9pp of headroom off a −3.90% worst month.
That was measured on a book two data fixes and a composition change ago, and it overstated the case for
the hedge by a factor of six; the table above is now rendered from the search's own artifact so it cannot
drift again.

**It is not shipped, and the reason is the one the brief fixes.** §11 scores the targets on the frozen block,
and there the hedge costs Sharpe monotonically (**3.07 → 2.71 at the smallest size tested, → 2.26 at 0.25
and 1.80 at 0.40**) while raising CAGR — it added volatility across a block that contained no crash big
enough to pay for it. The book already clears all five targets on that block, so paying a fifth of its
scored Sharpe to buy 0.3pp on a *supporting*-window metric that already passes is the wrong trade. Recorded as a measured option, not taken:
if a future window puts the worst month back on its limit, E at 0.15–0.25 of a slot is the lever with the
evidence already attached.

## 6c-bis. The −78% tail is hedgeable after all — priced, not shipped (§12)

The book's largest single risk is the short-vol leg's **−78% tail**, and every prior section treats it as
irreducible because "a real tail hedge needs the live option smile — paid data". **That claim is retracted.**
The obstacle was never the price of data; it was that no part of this project had ever looked at an option
quote. historicaldata.net publishes **Jan–Jun 2013 free**, full chain with bid/ask, greeks and IV on 3,800
underlyings — enough to price the wing directly ([VOLPREM.md §4c](docs/strategies/VOLPREM.md),
`scripts/volprem/run_wing_cost.py`).

The wing's price *is* a truncation of the variance strip — cap at `var_cap·K²` and you give up its far tail,
so `wing = K²(full) − K²(truncated)`, both legs from the same quoted chain. Measured over **615 chain-days**
on the five deep legs it costs **12.0% of sold variance**; scaled through the cycle by Cboe's free **SKEW**
index it is **16.2%**. The load-bearing surprise is the sign of that scaling in a crash: **×0.74 through the
2008 GFC**, i.e. *below* the calm-window calibration, because at-the-money variance explodes faster than the
tail strip. Tail protection gets relatively **cheaper** exactly when it pays, which is what makes a
permanently-held cap affordable at all.

| construction (18-leg book, VIX-gated) | Sharpe | max-DD | worst day | skew |
|---|---|---|---|---|
| naked — **what ships** | +4.42 | **−77.6%** | **−76.4%** | −26.3 |
| capped 2.5×, wing unpaid — the known trap | +10.74 | −36.4% | −7.1% | −0.9 |
| capped 2.5×, wing paid at the through-cycle 16.2% | **+6.89** | **−43.9%** | **−6.4%** | **−0.8** |

**It is not shipped, and the reason is the margin.** Break-even is **~3× the calibration**: at 2× the leg
still returns +4.3 but on a −64% drawdown, at 3× it is worthless. A 2.2× margin is thinner than every other
cost sensitivity in this report (the book breaks even at 5×, the sleeve's vega spread at 22×), and the 12%
comes from five legs in one calm half-year extended by an index proxy rather than by quotes. The scaling is checked inside the free window rather than assumed — VIX 11.3–20.5 there gives **×1.30**
against SKEW's independent **×1.35** — but the window tops out near VIX 20 while a crisis is 40–80, so the
relation is extrapolated four-fold. One crisis year of chains (~$99) would settle **that**, not the price
level, which is now measured twice. Until then the
deliverable keeps the naked book and its disclosed tail, because a headline resting on a 2.2× margin from a
proxy is not a headline.

## 6d. The last hindsight universes: the trend leg's crypto core and equity names (§12)

The report's own standard is that a family trades a **survivorship-free** universe — the x-sect deep-dive's
headline is that a curated 50-coin list scores **+1.06 against +0.70** honest, and the carry leg deliberately
ships the *weaker* point-in-time construction (+1.33) over the curated one (+1.47) on that principle. The trend
leg was the exception: its crypto half traded a hard-coded `CORE10` — BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX,
LINK, LTC — which is the majors **as they look today**. SOL and AVAX did not list until Sep-2020, so for the
first half of the window the list describes the future.

Two things were suspect, not one: the leg's *level*, and the deep-dive's "for crypto, fewer instruments is
better" finding — which could be the same bias in disguise, since a list of today's winners naturally beats a
broad universe containing what died. `scripts/trend/run_trend_pit_universe.py` prices both by rebuilding the
crypto half on a **point-in-time top-10 by trailing dollar volume** — the identical membership rule the
breakout and carry legs already use.

The two universes are materially different: **78 distinct names** are ever in the honest top-10, and today's
CORE10 hold only **~63% of member-days**.

The equity half was the same bug, sharper: seven hand-picked mega-caps — AAPL, MSFT, **NVDA**, AMZN, GOOGL,
**META**, JPM — where META did not IPO until May-2012 and NVDA is the era's best-performing large cap. Index
ETFs (SPY/QQQ/IWM) stay, since nothing was picked there. Out of the full 1,639-name local panel the honest
top-7 draws on **44 distinct names**, and the hand-picked seven hold only **45% of member-days**.

| | leg Sharpe / CAGR | book (selection window) | book (frozen block) |
|---|---|---|---|
| hindsight lists, both halves | +1.31 / +11.1% | 3.72 · worst −5.6% · months 80% · **5/5** | 3.77 · months 85% · 5/5 |
| PIT crypto only | +1.13 / +9.2% | 3.67 · worst −5.2% · months 81% · 5/5 | 3.75 · months 81% · 5/5 |
| **PIT crypto + PIT equity (shipped)** | **+0.91 / +6.2%** | **3.61 · worst −5.1% · months 81% · 5/5** | **3.72 · months 81% · 5/5** |

**The survivorship premium in this leg is ~0.40 Sharpe** — 0.18 crypto plus 0.22 equity, about 30% of its
headline. The crypto part is smaller than the x-sect case (−0.36), which makes sense: trend is long-only
time-series, so it never ranks survivors against each other, and the bias enters only through *which* names
exist. At book level the full fix costs **0.10 Sharpe** and the worst month *improves* at every step
(−5.6% → −5.2% → −5.1%). It is shipped, because a universe chosen with hindsight is not a level you can defend
at any price, and this one is close to free.

**What it costs, stated:** months-in-profit falls to **80.3% full / 80.8% OOS** — still clear of
the ≥80% target, but with 0.3pp and 0.8pp of headroom against 4.6pp before. That is now by far the thinnest of
the five, and it is the honest floor: it is what this book's consistency looks like once no universe anywhere is
chosen in hindsight.

## 6d-bis. Seven routes to the streak, all measured, all closed — the frontier before the composition changed (§12)

**Read this section against the eight-family book, not the shipped one.** It is the frontier this report
found while all eight families traded, when a 3-month losing streak broke the full window. The shipped
six-family book does not have that streak (2 months) — but it clears it by
dropping two families (§6d-ter), which is a composition choice, not one of the levers below. Those levers
are what a book *keeps* its composition and still tries to fix the streak with, and every one of them
fails; that is why the section stays, and why §6d-ter is stated as a cost rather than as a solution.

The brief anticipates this case: *"If the targets are not reachable under honest validation, submit your
best result with the trade-off frontier you found... Do not tune against the final out-of-sample block to
reach a number."* The eight-family window's 3-month streak (Dec-2021 −2.5%, Jan-2022 −0.8%, Feb-2022 −1.7%)
is that case, and this is the frontier. Every route below was run, not reasoned about.

Where the streak comes from: over those three months **carry −16.4%, x-sect −13.0%, trend −11.2%**, while
BAB **+0.6%**, breakout **+1.4%**, crisis **+2.8%** and vol-prem **+9.5%** — three short-gamma legs falling
together through the crypto unwind, with the long-gamma legs already in the book not large enough to offset.

| route | result | why it cannot work |
|---|---|---|
| **change the leverage** | streak **3 at every level 1.00×–2.00×** | a streak is a property of a month's *sign*, and leverage is a positive scalar |
| **narrow the crypto universe** (`scripts/xs/run_xs_universe_sweep.py`) | streak 3 at top-10/25/50/75; **2** only at top-100 and wider | narrowing cuts the loss's *depth* (top-10 loses 1.3% over the three months vs top-50's 5.3%) but loses all three; breadth is what flips one of them |
| **de-risk into the regime** (§5d) | months-in-profit 81% → **68%**, streak → **31** | a flat month is not a profitable month — a classifier aimed at this exact target made it far worse |
| **re-weight the families** (§6) | **0 of 2,000** random weightings reach 5/5 | months-in-profit only rises by over-weighting short-vol, which breaks the worst month |
| **add long-gamma** (§6c) | only long-crypto-variance flips it, and only at *full* parity weight | it buys the supporting window by paying the **scored** one (OOS Sharpe 3.39 → 2.85, 5/5 → 4/5) — and it only lists from **2021-03**, nine months before the streak |
| **trade the trend family SHORT** (`scripts/trend/run_trend_short_leg.py`) | earns in the window (**+1.3%**, two of three months positive) and makes the book **worse**: months-in-profit 79.0% → **76.5%**, streak 3 → **4** | it is the one candidate with *full* history, so the "only exists where the miss is" objection does not apply — it simply is not big enough, and its fifteen-year standalone is **−0.17**: it pays for those two months in every other month |
| **trade the trend family BOTH ways** (same script) | still loses the window (**−4.5%**); months-in-profit 79.0% → **77.2%**, streak still 3 | the long side outweighs the short, so the leg keeps most of the loss and gives up the long-only edge (standalone +0.92 → +0.84). It does reach 5/5 on the frozen block, but the choice is made on the selection window, where it is worse |

That last row is worth stating plainly, because it is the shape every "fix" takes here: **a lever that
exists only where the target is missed will always look like it fixes it.** Long crypto variance is a real
source and it does earn through the unwind — **+30.2% over the three months, positive in all three** — but
sizing it to flip the streak means judging a fifteen-year scorecard on a leg that exists for five of those
years, and paying for it where the brief actually scores.

**The arithmetic underneath all seven rows.** A ninth family enters at 1/9 risk, so to flip a month it has
to earn roughly nine times what the book lost: **+19.3% in Dec-2021, +6.6% in Jan-2022, +12.9% in
Feb-2022**. Nothing in the tested set comes close except long crypto variance (+21.9% in December, and
only there). That is not a gap in the search — it is what the number demands: a ~20% month from one leg
is a **convexity** payoff, and convexity is paid for out of months-in-profit the other 95% of the time. A
slow directional signal like short trend cannot produce it by construction, which is why it earns the
right sign and the wrong size.

So the honest position is the brief's own: **the streak is a property of that book, not a defect left
untuned.** It is short-gamma and crypto-heavy, and a sustained crypto unwind takes three months off it. What
removes it is not a better lever but a narrower book — which is what §6d-ter measures, and prices.


## 6d-ter. The shipped composition is six families, and it was chosen against the scorecard (§12)

**This is the one choice in the book not made on a-priori grounds, so it is stated first and in full.**
Every other decision here — equal weight, the frozen universes, the regime gate's thresholds, the
leverage — is fixed before its result is seen. The *composition* is not: the book trades six families
because that is what clears all five targets, and this section is the search that produced it.

With all eight families the book scores **3/5 on the full window**
(months 78.7%, streak 3) and **4/5 on the frozen block** (months 76.9%).
Running every single- and double-removal — 37 configurations, `make composition`
(`scripts/run_composition_search.py` → `reports/book/composition_search.json`) — gives:

| configuration | full window | frozen block |
|---|---|---|
| all eight | 3/5 — Sharpe 3.58, months 78.7%, streak 3 | 4/5 — Sharpe 3.32, months 76.9% |
| drop vol-premium | 1/5 — Sharpe 1.58 (under 2.5), months 61.2%, streak 9, worst month −7.3% | 3/5 — Sharpe 2.09 (under 2.5), months 69.2% |
| drop x-sect | 3/5 — Sharpe 3.64, months 79.8%, streak 3 | 4/5 — Sharpe 3.16, months 73.1% |
| drop breakout | 3/5 — Sharpe 3.53, months 79.3%, streak 3 | **5/5** — Sharpe 3.52 |
| drop crisis | 3/5 — Sharpe 4.08 (over 4.0), streak 3 | **5/5** — Sharpe 3.79 |
| drop global-macro | 3/5 — Sharpe 3.66, streak 3, worst month −13.5% | 4/5 — Sharpe 3.17, months 73.1% |
| drop BAB | 4/5 — Sharpe 3.58, streak 3 | **5/5** — Sharpe 3.17 |
| drop trend | **5/5** — Sharpe 3.60 | 4/5 — Sharpe 3.26, months 76.9% |
| drop carry | 3/5 — Sharpe 3.51, months 79.8%, streak 3 | 4/5 — Sharpe 3.12, months 76.9% |
| drop BAB + trend | **5/5** — Sharpe 3.58 | **5/5** — Sharpe 2.98 |
| **drop trend + carry** *(shipped)* | **5/5** — Sharpe 3.53 | **5/5** — Sharpe 3.07 |
| *(26 further pairs)* | — | fail at least one |

Two rows deserve a caveat rather than a footnote: a removal can also *shorten* the book's window, because
the assembler needs two live legs, and "drop global-macro" loses its early months that way — which is why
its tail reads worse than the leg's own hedging value would suggest.

**Two of 37 pass, and that ratio is the point.** A 37-way search that returns two survivors is
weak evidence by construction, and this report spends §6 measuring exactly that failure mode: of 2,000
random re-weightings of the same eight legs, **none** reaches 5/5, and CSCV puts the probability of
backtest overfitting at 13%. A composition picked because it passes is the same mechanism seen from the
other side. It is disclosed rather than presented as a design.

**Neither removed leg is weak on its own terms.** Trend's standalone (0.89) sits between
global-macro (0.93) and x-sect (0.85); carry is **1.22**, the fourth-highest of the eight. What
singles them out is their correlation to the rest through Dec-2021→Feb-2022 — a property of that window, not of
the strategies. Dropping trend fixes the full window and leaves the block untouched at
76.9% months-in-profit; dropping carry as well is what lifts the block to
80.8%.

**What it costs, measured:**

| | eight families | six families *(shipped)* |
|---|---|---|
| targets, full / block | 3/5 · 4/5 | **5/5 · 5/5** |
| Sharpe, full / block | 3.58 / 3.32 | 3.53 / **3.07** |
| vol-premium share of P&L | 56% | **64%** |
| asset-class breadth | trend spans crypto **and** US equities | no leg spans both; three of six are crypto-only |

So the scorecard is bought with **−0.25 Sharpe on the scored block**, a **more concentrated**
book (the short-vol leg goes from 56% to 64% of P&L, against its own −78%
tail), and the loss of the only family trading both asset classes. Cross-asset breadth now rests on
vol-premium's US underlyings and global-macro's EM-FX rather than on a leg that trades both.

**The brief's own instruction points the other way**, and that is worth recording next to the result:
*"If the targets are not reachable under honest validation, submit your best result with the trade-off
frontier you found... Do not tune against the final out-of-sample block to reach a number."* The
eight-family book with §6d-bis's frontier is that submission; the six-family book is the one that passes.
Both are in this repository — `FAMILIES` in `scripts/run_master_book.py` carries the two removed legs as
comments, so restoring either is one line, and every artifact behind the eight-family result is still
published.

## 6e. The five hardest questions, answered with the measurement

Every objection below is one this report invites. Each is answered from an artifact, not from prose, and
where the answer is "yes, that is a real weakness" it says so.

**1. "A net Sharpe near 3.53 is not credible for a real book."** It would not be for a *sleeve*, and no sleeve here
earns it: the best single sleeve's **deflated Sharpe is 0.00 at N=2,129 trials**, and the same
selection walk-forwarded out-of-sample gives **+0.13**. The book's number comes from *not selecting* —
six premia at mean pairwise correlation **0.07**, each applied uniformly across its
whole universe. The check that matters: **remove the anchor leg and the remaining five
still make Sharpe +1.26**, positive every year. If the number were a mining artifact it would
collapse there.

**2. "Half the P&L is one leg with a −78% tail."** True, and it is the book's largest stated risk: vol-prem
is **64% of P&L**. Three things bound it — it is sized on the tail rather than on its Sharpe (equal risk,
never above parity), its own VIX-term-structure gate stands it down while the curve is inverted, and §6c-bis
prices a wing that would cut the worst day from **−76% to −6%** for ~16% of sold variance. That hedge is
**not shipped**, because its margin over break-even is 2.2× on a level measured in one calm half-year. The
honest position is a disclosed tail, not a hedged one.

**3. "A two-year out-of-sample block proves nothing."** At 25 months the standard error of the OOS Sharpe is
**±0.70**, stated in §5c — 3.07 is 3.07 ± 0.70. The length is a trade: at one year the ≤2-month streak
target is close to a coin flip, and the crypto legs only list from 2020 so a five-year block leaves one year
to build on. The wider evidence is the book-level walk-forward, which runs out-of-sample **2006→2026 at
Sharpe 3.28** and pays for that history in drawdown (−18.6%).

**4. "It is a crypto book with a hedge bolted on."** Three of six families are crypto-only and
**no family spans both classes** now that trend is dropped (§6d-ter) — stated in the first screen. But the equity and FX absences are *measured deaths*, not
gaps: equity BAB's beta ranking sits at the **14th percentile of shuffled rankings** (random does better),
FX carry nets **+0.39** because the price leg offsets the accrual, and breakout is negative on equities and
FX under every construction. The book is crypto-heavy because that is where the edge survived costs.

**5. "You miss targets."** Yes, and more than when this section was written. The scored out-of-sample block
is **5 of 5** — missing on nothing — and the full window is **5 of 5**, missing
on nothing. Both misses are the same underlying thing: months that are flat-to-slightly-negative in a
crypto unwind, clustered rather than scattered (Dec-2021 → Feb-2022 is the run that sets the streak). Neither
is a knob left untuned. **A streak is a sign property and leverage is a positive scalar**, so the grid shows
the same streak at *every* level from 1.00× to 2.00× — sizing cannot touch it. Nor can either be de-risked
away, because a flat month is not a profitable month: §5d measures a classifier aimed at exactly this target
and it pushes months-in-profit *down*, from 79.3% to 64–66%. What would fix them is a
source that **earns** through a crypto unwind rather than sitting out of it; §6c searched for one and found
nothing that did not break another target. That is the honest ceiling of this book, not a tuning gap.

## 7. What did not survive (kept, not hidden)

- **Naive single-name mean-reversion** and **naive large-cap cross-sectional momentum** (a curated 20-name
  panel — in-sample marginal, MC 5th-percentile below zero) cleared the bar essentially nowhere; at 5m/15m the event families are
  destroyed by turnover × cost. These *specific naive constructions* do not survive realistic costs.
  **But the families themselves survive in their honest forms** — cross-sectional momentum on a broad
  survivorship-free top-100 universe ([docs/strategies/XSECT.md](docs/strategies/XSECT.md)) and **carry** as
  cross-sectional funding ([docs/strategies/CARRY.md](docs/strategies/CARRY.md)) both survive and are in the master book. The lesson is
  *construction and universe*, not "the family has no edge": the naive single-name / curated-list
  construction fails realistic costs while the honest survivorship-free form of the same family survives.
- **Cross-sectional reversal** (short-horizon 1–5d, dollar-neutral top/bottom-30%, vol-targeted) was
  the honest broad-universe test of reversion that 3-asset single-name MR cannot settle — 40 equities
  and 30 crypto perps (`scripts/meanrev/run_mr_universe.py`). Walk-forward OOS net of costs: **equities −0.13**
  (DD −46%), **crypto −0.49** (DD −39%). Both **beat their shuffled-signal placebos** (−0.74, −2.50),
  so a faint reversal signal is real — but turnover × cost eats it and it nets negative, the same
  failure mode as single-asset MR. Not a viable family here at the daily horizon; the intraday horizon
  is data-blocked (no Twelve Data key). It is **not** a cheap decorrelated source — the honest next
  source is one structurally orthogonal to the trend book (short-vol / volatility risk premium).
- **Pairs stat-arb basket** (cointegration selected on a formation window, traded OOS): **equities
  +0.05** (121/780 pairs), **crypto 1d −1.18** (192/435 pairs). Crypto cointegration is largely
  spurious out-of-sample and actively loses; equity pairs net ≈ zero — confirming the edge map's
  near-zero equity-pairs score. Stat-arb pairs are not a source here.
- **Calendar / session (overnight vs intraday)** — the brief's required §4 "Calendar and session
  effects" family, run through the full funnel ([docs/strategies/OVERNIGHT.md](docs/strategies/OVERNIGHT.md)). The genuine
  overnight premium is **real but is beta, not timing** (SPY overnight Sharpe 0.72 vs intraday 0.45;
  IWM 0.99 vs intraday −0.11 — small-cap intraday is negative): long-only market-directional exposure,
  a slice of the trend book, not a decorrelated alpha. The market-neutral cross-sectional book is
  **dead** (Sharpe −3.13, break-even at 0.1× cost) — isolating the overnight leg forces a full daily
  round-trip (~2× gross/day) that no cross-sectional signal survives; held 24h the overnight signal
  (−0.57) ≈ a plain reversal (−0.39). A naive first build looked marginally positive (+0.18) purely
  from two data bugs — 543 split-adjustment artifacts and a union-calendar misalignment, both caught
  and fixed. Decorrelated (+0.07) but negative — documented, not traded.
- **Calendar seasonality — pre-FOMC drift + turn-of-month (H4)** — the event-based half of the §4
  calendar family, held-through-window so cost is charged only at the edges (not overnight's daily
  round-trip), full funnel ([docs/strategies/SEASONAL.md](docs/strategies/SEASONAL.md)). Both effects are **real but beta,
  not alpha**. The pre-FOMC drift has a clean shape (SPY +8.7bps day-before / +7.5bps announce, then
  −16.9/−15.8bps the two days after; in-window Sharpe +1.25) yet the timing book nets only **+0.05–0.13**
  across SPY/QQQ/IWM/DIA and **fails the shuffled-calendar placebo** (63rd–74th pctile — a 1-day hold
  pays a full round-trip for ~8 events/yr). **Turn-of-month is beta by construction**: net Sharpe rises
  monotonically as the window widens toward buy-&-hold (SPY (−1,+1) 0.08 → (−4,+5) 0.77 ≈ B&H 0.76), the
  classic (−1,+3) window (0.29) **underperforms buy-&-hold**, the stock book is flat at ~0.29 across
  top-50…500 (no cross-sectional signal), and crypto ToM is dead (−0.01). Combined SPY sleeve +0.32
  (MC-P5 −0.11, deflated 0.31), decorrelated (+0.18) but sub-bar → **drags the book** (3.47→3.16 @30%),
  excluded. **One genuine edge-map find:** BTC's exact 24h→2pm-ET pre-FOMC window returns **+102bps,
  t=+2.4** — a significant crypto risk-on drift, located for future event-study work, not a levered sleeve.
  **Neither a market-neutral cross-asset long/short nor ML rescues it** (`run_seasonal_xasset_ml.py`):
  the dollar-neutral seasonal-momentum book between names is negative or sub-bar (best is crypto ToM +0.36,
  96th placebo pctile), a purged-CV pre-FOMC ML gate (VIX/10y-2y-slope/drift) makes it worse (SPY
  0.24→0.07, negative OOS IC), and a cross-sectional LGBM ranker is worse in-window than all-days —
  removing the beta removes the return.
- **Cross-sectional skewness / lottery (MAX)** — H2, short high-skew / long low-skew, the retail
  lottery-mispricing bet ([docs/strategies/LOTTERY.md](docs/strategies/LOTTERY.md)). **Inverted in crypto** (skew-short −0.38,
  MAX-short −0.67; all 24 window×tail cells negative, walk-forward OOS −0.43): the monthly-horizon
  **momentum** premium dominates, so the same recently-exploded memecoins the lottery bet shorts are the
  ones momentum longs — and momentum wins (the only positive side, long-high +0.15/+0.50, is re-labelled
  momentum). The tilt is **not sign-stable across timeframes** (crypto skew-short 1d −0.38 → 4h +0.20 →
  1h +0.04, flipping to weak intraday reversal, never clearing 0.5) — the fingerprint of a momentum/
  reversal term structure, not a mispricing premium. **Real but sub-bar in equity** (broad skew-short
  +0.25, mid/small +0.11; MAX-short −0.95): the top-100-liquid cut excludes the low-priced retail names
  the anomaly needs. **Not independent of low-vol** — regressed on a low-vol/BAB proxy (and on H1's
  crypto BAB book) the residual Sharpe is
  **0.00** (consistent with H1's converse: BAB has alpha beyond skew, skew has none beyond BAB — BAB is
  the real effect, skew the re-label). The driving extreme returns are *real* (zero data-glitches;
  winsor Δ −0.02), so this is an economic verdict, not an artifact; perp funding charged at 8h
  settlements is a further −5.5%/yr headwind (→ −0.57). Decorrelated (−0.17) but negative — drags the
  book (1.62→1.43 at 30%), documented not traded.
- **Betting-against-beta / low-vol (BAB)** — H1, long low-β / short high-β, the leverage-constraint
  premium ([docs/strategies/BAB.md](docs/strategies/BAB.md)). **The crypto leg is in the book (§4/§6b); the equity and FX
  legs did not survive.** **The construction is the finding:** the naive dollar-neutral book
  carries a large *residual market-beta tilt* (−0.6 crypto, −1.2 equity) that in a bull market hides the
  premium and even flips the apparent sign in equity (long-high-β "wins" +0.33); **beta-neutralising**
  (Frazzini-Pedersen leg-scaling) removes it and is worth ~+0.6 Sharpe. Beta-neutral **crypto +0.77**
  full-sample (MC-P5 +0.17, deflated 0.55, cost-robust past 8×, **beats its shuffled-ranking placebo at the
  99th pctile** so the edge is the signal not the tilt), decorrelated (corr ≈0 to carry, ≈0.20 to the master
  book) and it **lifts the carry sleeve 1.47→2.10** (risk-parity with the top-25 book) — and it is **beta, not lottery** (corr to a −skew book 0.07, alpha survives skew
  control t=+2.0; the low-vol proxy has no alpha beyond skew). **Robust across universe size (top-10→200,
  +0.46→+1.51) and every timeframe 5m→1d (+0.75→+0.87, bar-frequency-invariant), strongest concentrated** (top-25 +1.51);
  a rebalance×beta-lookback sweep is monotone **slow-is-optimal** (90d beta × monthly = corner +0.81; hourly
  rebal is +0.56 gross → −0.17 net) — **no intraday BAB**. **Clears the robust bar:** holding the a-priori
  beta-neutral construction fixed and selecting only its parameters OOS, walk-forward is **+0.67 (top-100)
  and +1.52 (top-25)**, MC-P5 +0.17/+0.90, deflated 0.90/1.00 — the concentrated top-25 book is the standout
  (Sharpe +1.51, DD −14%, positive 6/7 years incl. 2025 +1.83). Holding the a-priori beta-neutral
  construction fixed and selecting only its parameters out of sample is what clears the bar; a pooled
  walk-forward that also re-picks the dollar-neutral construction reads a falsely pessimistic +0.32. **Equity BAB's signal
  is gone:** the +0.38 full-sample is the construction's mechanical net-long tilt — the real ranking sits at
  the **14th** placebo percentile (below random), WF-OOS +0.24, deflated **0.03**, dead 2023-26 (the
  documented post-2010 crowding). **FX is dead too** (12 USD-major pairs, beta-neutral −0.18, MC-P5 −0.57 —
  FX majors are deeply institutional with no retail-leverage story, and their cross-sectional premium is
  carry, not BAB), so BAB spans a clean **crypto-alive / equity-gone / FX-dead** gradient. A **data-integrity catch** (same trap as overnight): 24 crypto + 12 equity
  split/delisting artifact name-days winsorised — left in, 18 outlier days faked a **0.97** beta≈lottery
  correlation that collapses to **0.00** clean. The concentrated **top-25 crypto** book (Sharpe +1.51, DD
  −14%) is the BAB leg in the master book (§4); the equity and FX legs are dead — capacity-limited to the
  liquid majors ([docs/strategies/BAB.md](docs/strategies/BAB.md)).
- **On-chain / network signals (H3)** — the brief's "alternative data" family and the one information
  source *not* derived from price ([docs/strategies/ONCHAIN.md](docs/strategies/ONCHAIN.md)). **Two data
  facts had to be corrected before the verdict could be trusted, both of which had flattered it.** First,
  exchange net-flows and exchange-held supply are **free** on the Coin Metrics community tier for BTC and
  ETH — an earlier pass recorded them as pay-walled because it inferred entitlement from group 403s rather
  than reading the vendor catalog (a multi-metric call 403s whole if any one metric is Pro). Second, four
  of the 37 names were measuring **dead ERC-20 shells** (VET/ZIL/QTUM/LRC report zero active addresses on
  44-79% of days), which parks them at the expensive extreme of a per-address value rank on a pure
  artifact; excluding them leaves **33 names**. What is genuinely Pro-walled is narrow — adjusted transfer
  value and entity-adjusted supply bands (realized cap is recoverable as mktcap÷MVRV). Top-50/100 remains
  impossible: SOL/SUI/TON/APT publish market data only, no network metrics.

  On clean data the a-priori headline — on-chain **value** (market cap per active address), top-20 — nets
  **+0.15** (was +0.40 with the shells) and **fails every OOS gate** (MC-P5 −0.52, placebo 72nd pctile,
  deflated 0.07). **Exchange flows were then tested as a BTC/ETH timing overlay** (two names cannot form a
  cross-section): the best, exchange-supply-trend long/flat, nets **+0.96** against buy-and-hold +0.85 —
  but rotating the *same* position path to random dates gives a 95th percentile of **+1.01**, so none of
  the four flow overlays beats its own random-timing control, and 7 of 8 HAC predictive regressions find
  no forecasting power. The flow thesis is now a **tested negative**, not an untested excuse. **One signal
  does survive:** adoption momentum (active-address growth, top-20) nets **+0.73** with MC-P5 +0.08,
  placebo 98th pctile, walk-forward OOS +0.74 with construction held fixed, and alpha over price momentum
  + reversal of **t=+2.04**, stable across every top-N. It is still excluded — post-hoc, deflated 0.50 at
  the family's 36 trials, **+0.32 correlated with price momentum**, and the book does not move
  (3.828→**3.831** at a 15% weight). Dilution, ownership and fee-yield factors fail outright. **ML confirms
  it** — a 21-trial ranker (ridge/RF/extra-trees/hist-GBM/LightGBM + classifiers × on-chain/price/both
  features × horizon × top-N, purged CV): on-chain features best **+0.32**, the same harness on *price*
  features **+1.09** (so the method is sound), and adding on-chain to price *degrades* it. **Equities/FX
  have no on-chain analogue** (crypto-only by nature).

  **The chains that free network data cannot see were tested on a different axis entirely.** DefiLlama
  publishes daily fees, revenue and TVL for 28 chains — SOL, SUI, TON, APT, SEI, TIA, ARB, OP — free and
  without a key, and Coin Metrics supplies the market cap, so a genuine fee yield (the crypto earnings
  yield) is computable where address counts are not. **Crypto inverts it:** buying the chains that look
  cheap on their own cash flows nets **−0.82** over 2022-06→2026-07 on 27 chains, with the placebo at the
  **6th percentile** — the cross-section carries real information with the sign reversed, exactly as it
  did for the lottery factor. The reason is visible in the legs: fee yield ranks Bitcoin permanently
  expensive (almost no fees against a trillion-dollar cap) and the L2s permanently cheap, at 0.014
  turnover per bar — a standing structural tilt, not a valuation that closes. The post-hoc sign flip
  (+0.86) is *not* the BTC-dominance trade (hedging that spread out leaves +0.81) but fails every other
  gate: placebo 94th percentile against its own 95th, deflated 0.54, alpha over price t=+1.66, all P&L in
  the final two years, and book lift 3.777 → 3.799. Excluded. The remaining honest paths are paid
  entity-level flow labelling and a wide small-cap on-chain panel.
- **Residual / idiosyncratic momentum (H5)** — momentum on each name's market-beta *residual*, vol-standardised
  (Blitz-Huij-Martens 2011), the fix for the weak equity leg ([docs/strategies/RESIDMOM.md](docs/strategies/RESIDMOM.md)). **It is a
  better-built momentum, not a new source:** ~0.8 correlated with the raw momentum already in the book and with
  no significant alpha over it (t = +0.1…+1.1), so it **refines the existing sleeve rather than joining as a
  family** — corr to book 0.18–0.35, dilutes at every weight. Held head-to-head (identical execution, signal
  swapped), it **beats raw momentum outright on crypto** (+0.45 → **+0.61** standalone, walk-forward incremental
  **+0.25**, 94% of the grid positive, placebo-clean 93rd pctile, positive 6/7 years) — and the construction
  reproduces the literature to the parameter (formation ≈ 20–30d = the 1–4-week crypto horizon, EW-panel factor
  beats BTC, and — unlike equities — the skip does *not* help). On **equity** the canonical decoupled form is
  *below* raw at top-100 (+0.41 vs +0.48; single-window ties it +0.49) and wins only at full 692-name breadth
  (+0.70 vs +0.56), but it **halves the momentum-crash bleed** (raw's worst-5-months −12.3% → −5.0%) and its
  walk-forward +0.45 edges raw's +0.39. The **"lower-beta" premise does not bind** — the books are already
  dollar-neutral (β ≈ −0.005 crypto / −0.05 equity), so there is no market beta to strip (the one large strip,
  FX +0.35 → −0.15, is where there is no edge). **FX dead.** The actionable win is a **drop-in sleeve upgrade**
  (`risk_adj_mom` → `idio_mom` lifts the crypto x-sect sleeve +0.45 → +0.61) and a tail-safer equity leg — the
  highest-certainty modest win, lowest diversification value, exactly as H5 was ranked.

## 8. Reproduce

```bash
make setup          # python3.12 venv + pinned deps  (macOS: brew install libomp)
make master         # the headline, offline from the committed series (~seconds)
make composition    # §6d-ter: the 37-way composition search behind the six-family book
make risk-budget    # §4b: the leverage grid, the bootstrap tail and the 2010 event
make lint           # every figure in this report re-resolved from the artifacts; fails on drift
make reproduce      # run_book -> make_report ; writes reports/* and the dashboard
```
Fixed seeds throughout; no held-out block was tuned against. **The prose here is a template
(`scripts/report_assets/report.md`) and every figure in it is a named placeholder resolved from a committed
artifact by `scripts/report_numbers.py`** — so a number in this report cannot disagree with the run that
produced it, and `make lint` fails the build if the two drift apart.
