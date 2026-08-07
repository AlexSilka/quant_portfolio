# Task A — Cross-Asset Alpha Discovery & Portfolio Assembly
## Approach and rationale

> The strategic frame, the verified data/tooling facts the architecture rests on, the
> end-to-end pipeline, and a walkthrough of how each requirement in the brief is met.
> Results live in [../REPORT.md](../REPORT.md); this document is the *why*.

---

## 0. Strategic frame

The targets are demanding. Net Sharpe 2.5–4.0 with ≤15% drawdown and ≥80% profitable months, on a
single out-of-sample block run once, net of realistic costs, is at the edge of what is honestly
achievable for a cross-asset systematic portfolio — public multi-strategy funds live nearer Sharpe
1–2. A submission that "shows" net Sharpe 3+ on the held-out block has, more often than not, leaked or
overfit. The brief itself makes the point: *"the map of where edge was found, and where it was not,
matters to us equally,"* and *"if the targets are not reachable under honest validation, submit your
best result with the trade-off frontier."*

So the deliverable is two things at once:

1. **An assembled portfolio** that meets as many targets as it honestly can, and
2. **An honest edge map** — where edge is real, where it is not, what the ceiling is, and what the
   binding constraint is — backed by a pipeline that **measures its own false-discovery rate.**

The headline is not a backtest number; it is trust in the numbers. Three design principles follow.

- **Pipeline-as-product.** The primary diagnostic is not a strategy's Sharpe but the pipeline's own
  error rate: how much "edge" the identical machinery finds on deliberately empty data. Research is run
  like a clinical trial with a permanent placebo arm — every discovery is reported as *real minus
  placebo*, never in isolation.
- **Meta-labelling as the ML layer.** Each sleeve is built as `primary rule → ML confidence gate`: a
  rule-based signal sets the side (and doubles as the non-ML baseline the brief requires), and a
  secondary model predicts `P(the trade wins)` and gates/sizes entries. ML's incremental value is then
  measurable by construction — sleeve-with-gate minus baseline-without.
- **Cross-asset as a structural diversifier.** Crypto (24/7, driven by funding and liquidity) and US
  equities (session-bound, driven by macro and earnings) are decorrelated for structural reasons, not
  by fitting. That decorrelation is the portfolio's main source of robustness, and it is verified out
  of sample (§7), not asserted.

---

## 1. Verified facts: data and tooling

The architecture rests on directly measured data depth, not assumptions. Everything below was probed
against the live sources.

### 1.1. History-depth matrix — the key architectural fact

| Class | 1d | 1h | 5m / 15m | Covers Q4-2018 intraday | Source |
|---|---|---|---|---|---|
| Equities / ETF | 2006+ | deep | **from ~2020** | no (1d only) | Twelve Data Pro |
| Crypto spot | BTC since 2017-08 | full | **full** | yes (spot) | data.binance.vision |
| Crypto perp + funding | since 2020-01 | full | **full** (funding 8h) | no (perps didn't exist) | data.binance.vision (um) |

Direct depth measurements: BTCUSDT spot 5m spans 2017-08 → present (108 months), SOLUSDT spot 5m from
2020-08; BTC perp 5m + `fundingRate` from 2020-01, SOL perp from 2020-09. The ingestion path is verified
end-to-end (a real month, BTCUSDT-1h-2024-01, parses to 744 clean OHLCV+trades+taker bars).

Loader gotchas handled up front, not discovered in production:
- (a) the CSV header row is inconsistent — spot files are headerless, some futures files carry a header
  → the parser sniffs the first line;
- (b) spot timestamps switched to **microseconds** from 2025-01-01 (16 digits) while futures stay
  milliseconds → normalised by magnitude;
- (c) prebuilt futures/um klines begin 2019-12-31 platform-wide, so BTC/ETH carry a 1–3.7-month
  post-listing gap fillable only from `trades`/`aggTrades`, while SOL and everything later are complete
  from day one;
- (d) COIN-M inverts `volume`/`quote_volume` semantics — only USD-M and spot are used.

**Design consequence.** Equity intraday (5m/15m) is available from ~2020, so the short-timeframe
research covers COVID-2020, 2021, 2022 and 2023→present on both asset classes. The one window equity
intraday does not reach is **Q4-2018**; the §11 short-timeframe isolation over Q4-2018 is served by
crypto **spot** (since 2017), and equities are handled on daily bars there. Crypto stays on Binance
bulk (deeper, plus funding, no credit limits); equities use Twelve Data. Both loaders sit behind a
single bar interface, switched by config.

### 1.2. Point-in-time macro (against look-ahead)

Macro comes from FRED's public `fredgraph.csv` endpoint, applied with a **1-month release-lag proxy**
(`src/data/rates.py`) — not an ALFRED first-release vintage feed. This matters only for *revised*
series. The macro the live sleeves actually consume — 3-month interbank rates (IR3TIB01), the
broad trade-weighted dollar (DTWEXBGS) and VIX (VIXCLS) — is published once and essentially not
revised, so the 1-month lag is an exact, slightly conservative point-in-time proxy rather than an
approximation. Where a truly revised series would be needed, the lag is stated and defended.

| Series | Revised → true vintage needed? |
|---|---|
| CPIAUCSL, PAYEMS, GDPC1, PCEPI, UNRATE | **Yes** — point-in-time mandatory (not used as live features) |
| VIXCLS, DGS2/DGS10/T10Y2Y, FEDFUNDS/DFF, DTWEXBGS | No (published once) |
| ISM PMI (NAPM) | Removed from FRED since 2016-06 — unavailable on FRED |

Macro is used as features for equities and as a regime filter; for crypto, rates/DXY are optional.

### 1.3. Stack

| Purpose | Library | Notes |
|---|---|---|
| Data / speed | pandas, numpy, pyarrow | pinned in `requirements.txt` |
| Crypto data | in-house `data.binance.vision` loader (bulk zip → parquet) | end-to-end verified |
| Equities / FX | Twelve Data Pro (split-adjusted daily 2006+ and intraday) | depth measured |
| Calendars | pandas_market_calendars | `scripts/validate_sessions.py` enforces NYSE sessions/half-days |
| Features / TA | in-house vectorised engine (82 features) | look-ahead audit passes |
| Feature selection | in-house IC / stability / hierarchical-redundancy report | `scripts/feature_report.py` |
| Models | lightgbm, scikit-learn | macOS needs `brew install libomp` |
| Portfolio | in-house equal-weight risk parity + §8 drawdown-ladder overlay | `run_master_book.py`, `src/risk/overlay.py` |
| Validation / overfit | in-house deflated Sharpe + in-house CSCV/PBO | `src/metrics.py`, `src/validation/cscv.py` |
| Labelling | in-house triple-barrier + meta-labelling | — |
| Monte Carlo | arch (StationaryBootstrap) + in-house trade-order / entry-jitter / random-start | `src/validation/monte_carlo.py` |
| Reporting | in-house matplotlib PNGs + self-contained SVG dashboard | `make_figures.py`, `make_report.py` |
| Macro | FRED `fredgraph.csv` + 1-month release-lag proxy | `src/data/rates.py` |

The overfit/validation stack (deflated Sharpe, purged+embargoed CV, CSCV/PBO) is written **in-house**
against `arch` + `scipy` + `scikit-learn` primitives. This is a deliberate choice: `mlfinlab` went
commercial and was removed from PyPI, `pandas-ta` (the original) is compromised, and several convenient
backtest/portfolio libraries carry restrictive licences (AGPL, Commons Clause, GPL). Keeping the core
in-house and permissive keeps `requirements.txt` minimal and every reported number auditable. The
environment is pinned (locked `requirements.txt`, fixed seeds throughout).

---

## 2. Pipeline architecture

```
data/            ingest → align calendars → parquet, point-in-time
  ├─ crypto:  data.binance.vision bulk (spot+perp klines, funding, aggTrades)
  ├─ equity:  Twelve Data (daily + intraday), point-in-time S&P membership
  └─ macro:   FRED fredgraph.csv + 1-month release-lag proxy (non-revised series)
      │
features/        82 candidates, each computable-at-bar, PIT normalisation
      │
labels/          triple-barrier (side + meta-label win/lose), sample weights
      │
sleeves/         asset × timeframe × family × feature-subset × model
  │  primary rule (baseline)  →  meta-label ML gate (confidence score)
      │
screening/       pre-registered criteria → deflated Sharpe, PBO/CSCV,
      │          FDR on shuffled/synthetic → survival funnel (counts)
      │
portfolio/       equal-weight risk parity + §8 DD-ladder overlay, correlation
      │          matrix + its stability, marginal-contribution curve
      │
risk/            per-family logic + portfolio limits + drawdown ladder
      │
validation/      purged+embargo CPCV · walk-forward · single OOS (run once)
      │          · Monte Carlo (P5/P50/P95) · per-year/quarter · crisis windows
      │
report/          edge map · funnel · marginal contribution · cost sensitivity ·
                 charts · tables · trade log · what didn't work · ceiling
```

A thin end-to-end slice (one crypto sleeve through the full validation) gives `make reproduce` from the
start; sleeves and asset classes are then added into the same harness. Peripheral work (a results
dashboard) reads the pipeline's outputs and has no logic of its own.

---

## 3. §2 Universe and instruments

**Universe-entry rule (frozen before any portfolio result is evaluated):**
- Crypto: top-N Binance perps/spot by median 30-day dollar volume above a threshold at the entry date,
  listed ≥ a minimum period before the test start. The threshold and N are declared and fixed by rule,
  not by cherry-picking. Where a survivorship-free breadth test is needed, the universe is rebuilt on a
  **point-in-time** basis each rebalance (delisted names included) — the curated-list versions score
  higher and that bias is quantified, not hidden.
- Equities/ETF: a liquid core of large caps + ETFs (delisting risk ≈ 0 over the test window); for a
  broad cross-section, point-in-time S&P 500 membership.

**Spot vs perp.** Perp is the default for crypto sleeves — it is required for shorts, leverage and the
funding realism the brief mandates — with spot used where pre-2020 depth is needed (Q4-2018).

**Direction and leverage.** Long-short. Limits are **enforced in the assembly** (`run_master_book.py`),
not merely declared: gross ≤ 200%, net ≈ 0 by dollar-neutral leg construction, a per-family weight cap,
a −4% daily-loss circuit breaker and the §8 drawdown ladder. They are set from a risk budget, not fitted.

**Survivorship / short histories / delistings.** Large caps + ETFs make delisting risk negligible, and
the residual bias (a current-membership cross-section inflates annual returns by ~1–2 pp) is quantified
and any breadth test built on point-in-time membership. A sleeve is not admitted to screening until its
instrument has enough bars for its label horizon plus embargo. Any instrument admitted because it
backtested well is named and its contribution counted separately.

---

## 4. §3 Timeframes and calendars

- Sleeves freely mix 5m / 15m / 1h / 4h / 1d.
- **Equities respect sessions, gaps and half-days** — enforced in `scripts/validate_sessions.py`,
  which pulls the NYSE schedule from pandas_market_calendars and asserts every intraday bar falls inside
  a real session. It also caught a genuine data issue: the raw 15m vendor feed carries full-length bars
  on three half-days (the day-after-Thanksgiving / Christmas-Eve 1pm closes), which the session filter
  removes. The traded equity legs are daily (one bar per session), so the overnight gap is a break, not
  a bar, and half-days shorten the session.
- **Crypto is 24/7.** The two calendars are reconciled on a common UTC timeline: cross-asset features
  (relative strength vs BTC, beta) are compared only on the intersection of open windows; crypto and
  equity sleeves are computed on their own grids and joined at the portfolio level on daily P&L.
- **Bar-close → execution delay.** A bar-`t` signal executes no earlier than the open of bar `t+1`
  (with extra lag for slow sleeves — execution is delayed to t+2 in the book). No sleeve executes at
  the price of the bar that generated its signal. The lag is declared and uniform in code.

---

## 5. §4 Feature library (80–120 candidates)

Built before any modelling, across every family the brief lists (each with several lookbacks):
trend/MA structure, momentum/ROC, mean-reversion (z-score, distance from anchor, spreads), volatility
(realised, Parkinson, Garman-Klass, ATR, vol-of-vol, regime), range/breakout/channel/consolidation,
volume/flow (OBV, VWAP distance, dollar volume, imbalance proxy from Binance taker-buy/sell),
oscillators, statistical structure (Hurst, autocorrelation, variance ratio, entropy), higher moments
(rolling skew/kurtosis, tail measures), cross-asset (relative strength vs index/sector/BTC, rolling
beta, correlation stability, dispersion), calendar/session, macro. **82 features shipped.**

**Discipline:**
- Every feature is **computable-at-bar**: the value at bar `t` uses only data ≤ `t`, enforced by a
  shift audit — recomputing on a truncated series must not change historical values (it changes zero).
- **Normalisation is point-in-time** (rolling z / percentile), never full-sample (full-sample leaks the
  future distribution).
- The per-feature report gives predictive strength (IC and IC t-stat in CV), stability over time
  (sign-of-IC stability across years) and redundancy clusters (hierarchical clustering on correlation).
- **Reduction** (`scripts/feature_report.py`): keep a feature iff |IC·t| ≥ 2 **and** sign-consistency
  ≥ 0.6 **and** it is its cluster's representative. On the crypto panel 47/82 features are individually
  significant, only 4 survive the full stability + redundancy reduction (27 clusters), and
  volatility/calendar carry no univariate signal — reported per family, and evidence that the edge is in
  construction, not any single feature.
- Feature selection is fit **inside training folds only** (otherwise selection leakage; §9).

---

## 6. §5 Sleeves, families, models, meta-labelling

**Sleeve = asset × timeframe × family × feature-subset × model.** The final portfolio spans **eight
structurally distinct families** — trend, carry, short-vol / variance risk premium, cross-sectional
momentum, breakout, crisis-alpha (managed-futures), global-macro (EM-FX + commodities trend) and
betting-against-beta / low-vol (§4 of the report).

**Every sleeve = primary rule + meta-label gate:**
- **Primary (side).** The family's rule — this is also the **non-ML baseline** the brief requires, tuned
  for recall (catch opportunities).
- **Meta-model (gate/size).** A binary classifier `P(primary wins)` on features; trade only when
  `P > threshold`. This trades recall for **precision** — fewer false entries, which is what non-zero
  costs demand. ML's incremental value = sleeve metrics with the gate minus the baseline without it.
- **Model choice: LightGBM.** With a small effective sample (the number of independent trades and
  regimes, not bars) and low signal-to-noise, gradient boosting beats deep nets here. Neural nets
  (TCN/LSTM/GRU/transformer) are only used if measured to add over GBM — they were not, and that is
  reported in "what didn't work."

**Targets (declared per sleeve): triple-barrier** — upper `+pt·σ_t`, lower `−sl·σ_t`, vertical at
`t+h` (max holding). The label is the sign of the first barrier touched; the meta-label is whether the
primary's bet won. `σ_t` is trailing volatility; sample weights account for label overlap in time. The
total candidate count generated and evaluated is reported (the 1,279-sleeve discovery zoo, §6).

---

## 7. §6 Robustness and multiple-testing control

**Most candidates die here.** Acceptance criteria are frozen before screening (positive net Sharpe
after base costs, OOS deflated Sharpe > 0, PBO < 0.2, sign stability across years, a minimum trade
count per regime).

- **Deflated Sharpe (DSR)** — the probabilistic Sharpe relative to the expected maximum under the null
  at the actual number of trials, using the non-normal Sharpe variance estimator (skew and kurtosis) and
  `E[max SR]`. `N` is the **effective** number of independent trials (correlated sleeves clustered),
  else over-deflation.
- **PBO via CSCV** — a T×N performance matrix, S disjoint blocks, all `C(S, S/2)` in/out combinations,
  logit of the OOS rank of the best in-sample configuration. PBO is the fraction of combinations where
  the best in-sample falls below the OOS median; ≳ 0.5 means selection is a coin flip.
- **FDR at the true N** — Benjamini-Hochberg-Yekutieli (robust to dependence) plus a haircut Sharpe; the
  significance bar for a new factor is |t| > 3, not 2.
- **Placebo arm (the pipeline's own FDR).** The identical pipeline is run on shuffled series and on
  synthetic (GBM / stationary block-bootstrap) series. How much "edge" it finds there is its error rate,
  reported as the real − placebo delta.
- **Survival funnel (counts):** generated → passed in-sample → passed walk-forward → passed Monte Carlo
  → entered the portfolio, alongside how much the placebo arm found.

---

## 8. §7 Portfolio assembly

- **Allocation: genuine equal-weight risk parity.** Each family is vol-targeted to ~15% on trailing
  (lagged) vol, then weighted 1/N — no performance-based selection — with the §8 drawdown-ladder overlay
  on top. Equal weight is deliberate over HRP / mean-variance: on eight already-decorrelated, vol-matched
  legs a covariance optimiser mostly fits in-sample noise, and equal weight is the honest no-selection
  baseline. That choice is justified with evidence (§5c of the report): re-fitting the weights out of
  sample does **not** beat equal weight — a mean-variance allocation lifts Sharpe but triples the
  drawdown, the classic overfit signature.
- **Correlation matrix and its stability** — computed out of sample and over rolling windows.
  Diversification that exists only in-sample is not diversification, so the split-half and OOS-block
  correlation is reported (mean pairwise ≈ 0.06, stable across halves).
- **Marginal-contribution curve** — families added in order of contribution, plotting portfolio Sharpe,
  drawdown and months-in-profit, showing where the curve flattens.
- **P&L share per family** and the portfolio with the top contributor removed (a test of single-source
  dependence — volprem is ~half of P&L, and the book still stands at Sharpe 1.81 without it).
- **Competition for capital** — on simultaneous signals, allocation by risk budget; on opposing
  positions in the same asset, netting at the portfolio level (double costs forbidden), with the event
  logged. The book combines return series rather than positions, a stated limitation.

---

## 9. §8 Risk management

Risk logic is **per-family, not uniform** — a mean-reversion sleeve adding into an adverse move needs
different protection from a trend sleeve riding it:

| Level | Rules |
|---|---|
| Sleeve | sizing (vol-target / meta-confidence-scaled), stop/exit (MR by time and spread widening; trend trailing), max holding = vertical barrier, max exposure |
| Portfolio | gross/net limits, per-asset and per-family caps, daily loss limit |
| Drawdown ladder | triggers by drawdown depth → stepwise gross reduction (−6/−9/−12% → 0.66/0.33/0.0), restore at −4% with hysteresis |
| Stop / restart | halt on daily-loss breach / drawdown floor; restore on recovery |

---

## 10. §9 Costs, data integrity, leakage

**Capital: USD 500,000** for sizing and cost calculations.

**Cost model — liquidity-aware, never a flat constant** (which the brief forbids):
```
cost_per_trade ≈ commission + half_spread + k · σ_bar · sqrt(Q / ADV_bar)
```
- **Commission** — crypto: the published Binance schedule (spot taker 0.1% / 0.075% with BNB; USD-M
  futures taker 0.05% / maker 0.02%), order type justified; equities: a realistic commission **plus
  stock-borrow on shorts** (`EQUITY_BORROW_BPS_ANNUAL = 50` bps/yr general-collateral on short gross).
- **Half-spread** — from a bid/ask proxy (crypto from depth/aggTrades).
- **Slippage — square-root market impact** (Almgren-style), scaled to bar volume and order size. At
  $500k against the majors' ADV the impact is small — shown honestly — but the model still penalises
  illiquidity.
- **Funding charged at every settlement** = `notional_at_mark × funding_rate` at the settlement instant
  (not annualised, not amortised). Rate > 0 → longs pay shorts. BTC/ETH/SOL settle on **8h** (00/08/16
  UTC); the rate is the premium index plus a clamp, capped per-symbol. Measured carry runs ~13%/yr for a
  long in contango — a material line item, not a rounding term.

**Three cost levels** (base, ~2× base, ≥ 3× base) plus the **break-even cost** at which the portfolio
stops making money; **turnover and cost as a share of gross P&L, per sleeve** (which sleeves are
cost-fragile).

**Leakage audit — for every transform, why it cannot see the future:**

| Transform | Leakage protection |
|---|---|
| Feature computation | data ≤ bar only; shift audit |
| Scaling / normalisation | rolling / PIT percentile; fit inside the training fold only |
| Labels | triple-barrier with an explicit horizon; embargo = label horizon |
| Feature selection | inside the training fold (selector re-fit per fold) |
| Hyperparameters | tuned inside train only; the meta-gate threshold set on CV, not train |
| Universe selection | rule frozen before portfolio evaluation |
| Macro | FRED + 1-month release-lag proxy (non-revised series); no forward-fill of the future |

No forward-fill that propagates unavailable information. Fixed seeds throughout.

---

## 11. §10 Validation

- **Purged + embargoed CV** — purging drops training observations whose labels overlap the test;
  **embargo = the sleeve's label horizon** plus feature memory. The primary tool is combinatorial
  purged CV (CPCV): N groups, k test, `C(N,k)` splits reconstructed into a distribution of OOS paths,
  whose Sharpes feed the DSR.
- **Walk-forward** — rolling and anchored, with periodic re-fit; the cadence and window policy are
  declared and shown not to drive the result (anchored vs rolling, quarterly vs annual re-fit all land in
  a narrow band).
- **A single final OOS block** — frozen to the end, run **exactly once**, never tuned against.
- **Monte Carlo** — stationary/block bootstrap (Politis-Romano, block length from `optimal_block_length`),
  trade-order resampling, entry jitter ±1–3 bars, randomised starts — reporting **P5 / P50 / P95** of
  Sharpe, max drawdown and monthly hit rate.
- **Parameter sensitivity** across the surface around the chosen settings, not just the peak.
- **Per-year / per-quarter** metrics plus isolated crisis windows: Q4-2018, Feb–Mar 2020, 2021, 2022,
  2023→present.

---

## 12. §11–12 Targets, edge map and ceiling

Targets are computed on the portfolio, net of all costs, on the final OOS block. The honest position is
stated up front (the brief's "assessment of the ceiling"):

- The upper end of net Sharpe 2.5–4.0 cross-asset after realistic costs is rarely achievable honestly;
  the realistic zone for a disciplined multi-strategy book is lower. Where a target is hit, it is shown
  not to be an artifact (DSR / PBO / placebo); where it is not, the **trade-off frontier** is delivered —
  where Sharpe stops improving as monthly consistency is pushed, where costs cap turnover, where adding
  sleeves stops paying.
- **Edge map** — performance by asset class × timeframe × family: where edge is found and where it is
  not. Edge is denser on crypto and on the mid timeframes (1h/4h); at 5m/15m it is thinner and eaten
  faster by costs.
- **Binding constraints**, stated plainly: for crypto carry, funding (~13%/yr); for 5m,
  costs/turnover; for equity intraday before 2020, data (Q4-2018); and for the book as a whole, the
  ceiling on cross-family decorrelation.

---

## 13. §13–14 What is produced, and the honest position

- **Report** — approach, results, validation evidence, leakage audit, known weaknesses.
- **Code** — runnable, with README, a locked `requirements.txt` and fixed seeds. One command reproduces
  the headline (`make reproduce`; `make master` rebuilds the portfolio offline in seconds).
- **Charts** — portfolio and per-sleeve equity, drawdown, monthly-return heatmap, rolling-12m Sharpe,
  exposure and turnover, sleeve correlation matrix, edge map.
- **Tables** — portfolio and per-sleeve metrics, per year and per quarter.
- **Trade log** for the OOS period.
- **What did not work and why it was set aside** — an explicit section (neural nets with no increment
  over GBM; families that are real-but-beta or cost-killed; a naive construction that looked good at
  mid-price but not after costs).

Any library or model is permitted, free data is sufficient, and the final OOS block is never tuned
against. If the targets are not reachable under honest validation, the best result is submitted with the
trade-off frontier. These are the spine of the submission, not caveats on it: the deliverable is a
portfolio a reader can trust, and an honest map of the edge behind it.

---

## Sources

- Binance public data: `data.binance.vision` (spot / futures / um klines + `fundingRate`),
  `api.binance.com` / `fapi.binance.com`.
- Equities / FX: Twelve Data Pro — equity intraday from ~2020, split-adjusted daily from 2006.
- Validation methodology: Bailey & López de Prado, *Deflated Sharpe Ratio* (SSRN 2460551); Bailey et
  al., *Probability of Backtest Overfitting / CSCV*; López de Prado, *Advances in Financial Machine
  Learning* (purged CV, triple-barrier, meta-labelling); Harvey & Liu, *haircut Sharpe / t > 3*.
- Macro: FRED `fredgraph.csv` with a 1-month release-lag proxy (`src/data/rates.py`); ALFRED
  first-release vintage is the ideal for revised series where a key is available.
- Survivorship-free equity universe: point-in-time S&P 500 membership; SEC EDGAR (Forms 25/15) for
  delisted prices.
