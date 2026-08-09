# Strategy hypotheses — research ledger

> **Canonical-book note.** This is the hypothesis ledger behind the edge map: the candidate families
> considered as diversifiers, with the verdict on each. The shipped portfolio is the eight-family
> equal-weight master book in [REPORT.md](../REPORT.md) (Sharpe **3.72** full / **3.77** OOS), assembled by
> `scripts/run_master_book.py`. Any master-book Sharpe quoted below is a snapshot at the time a
> hypothesis was evaluated, not the current headline.

Economically-grounded candidate families evaluated as potential diversifiers for the book, each run
end-to-end through the same funnel with the verdict recorded either way. Ranked at the outset by
(documented-edge strength × decorrelation from the then-current book × data-in-hand × likelihood of
surviving cost).

**Shared acceptance bar** (same funnel as every other family — do not deviate):
- Construction: dollar-neutral cross-sectional long/short unless stated, **vol-targeted to 15%**,
  **t+2-style execution delay** (never fill at the signal bar's close), **liquidity-aware costs**
  (commission + half-spread + √-impact, never flat), funding charged at every 8h settlement for crypto.
- Validation: **shuffled-signal placebo** (real must beat ~95th pct of shuffles), **purged/embargoed
  walk-forward OOS**, **block-bootstrap MC** (P5 > 0), **deflated Sharpe** at the true trial count,
  **correlation to `reports/master_book.parquet`** + does-it-lift-the-book curve.
- Robust bar = OOS Sharpe > 0.5 **and** MC-P5 > 0; portfolio-admission = also corr-to-book < ~0.3 and a
  positive marginal contribution. Report the verdict either way (a rigorous "dead" is a valid result).
- **Reuse the existing engine.** `src/sleeves/xsect.py::xs_backtest` already does dollar-neutral
  top/bottom-quantile long/short with liquidity-aware cost + vol-target on a price panel — most of these
  are a **new ranking signal swapped into it**, ~30–50 lines, not a new harness. Data ready:
  `data/cache/xs/{crypto,stocks_broad}_1d_close.parquet` (+ `_adv`), `crypto_1h_close` for intraday,
  830 funding series under `data/raw/futures/um/fundingRate/`, `src/features/engine.py` already computes
  `skew/kurt/beta/hurst`.

---

## H1 — Betting-against-beta / low-volatility (BAB)  ★ top pick

**Thesis.** Leverage-constrained and lottery-seeking investors overbid high-beta assets, so low-beta
earns positive risk-adjusted alpha (Frazzini-Pedersen 2014). One of the most robust factors; works in
equities **and** crypto, and crypto's acute retail-leverage demand should make it strong there.

**Construction.** Each bar estimate trailing beta βᵢ (rolling 60–90d regression of name return on the
equal-weight panel / BTC), or use trailing vol as the simpler proxy. Rank; long the low-β quintile,
short the high-β quintile, **beta-neutral** (lever the long leg up / short leg down to net-zero beta) —
or dollar-neutral as the v1. Vol-target 15%. Run separately on crypto (300-name panel) and equity (692).

**Data.** Ready — close panels + the `beta_` feature. **Reuse** `xs_backtest`, signal = `−beta` (or `−vol`).

**Survive/kill.** Survives if WFO OOS > 0.5, beats placebo, corr-to-book < 0.3. Prior: US-equity BAB has
decayed/crowded post-2010 → expect equity-weak; **crypto BAB is the honest bet**. Orthogonalize the vol
proxy vs H2 to confirm it is beta, not lottery.

**Fit.** Decorrelated (leverage-constraint premium ≠ momentum/carry/trend). Likely crypto-tilted.

---

## H2 — Cross-sectional skewness / lottery (MAX) factor  — ❌ TESTED, DEAD

> **Verdict — tested end-to-end, dead ([LOTTERY.md](strategies/LOTTERY.md), `make lottery`).** Not a tradable
> decorrelated premium in any liquid universe here. Crypto (the primary bet) is **inverted**: skew-short
> **−0.38**, MAX-short −0.67, all 24 window×tail cells negative, walk-forward OOS −0.43 — the
> monthly-horizon momentum premium dominates, so the lottery *short* is the momentum *long* inverted (the
> only positive side is re-labelled momentum). Equity real-but-**sub-bar** (+0.25, below 0.5) because the
> top-100-liquid cut excludes the low-priced retail names the anomaly needs. **Not independent of low-vol**
> (residual Sharpe 0.00 vs both a BAB proxy and H1's BAB book), sign-flips across timeframes, funding is a
> further −5.5%/yr headwind, and it drags the book (1.62→1.43) — excluded. The spec below is kept for
> reference / to prevent a re-run.

**Thesis.** Investors overpay for lottery-like assets (high idiosyncratic skew, high recent max return)
so they underperform (Bali-Cakici-Whitelaw 2011 "MAX"; Kumar 2009). Crypto is the extreme case — retail
lottery demand for high-skew memecoins — so **short high-skew / long low-skew** should pay in crypto.

**Construction.** Signal = trailing return skew (20–60d) **or** MAX = mean of the top-5 daily returns
over the past month. Short high, long low, dollar-neutral, vol-target 15%. Crypto panel is the primary
test; equity secondary.

**Data.** Ready (`skew` feature exists; MAX is a one-liner). **Reuse** `xs_backtest`, signal = `−skew`.

**Survive/kill.** Survives if OOS > 0.5, decorrelated, beats placebo. Watch the overlap with H1
(high-skew names are often high-vol) — regress skew-book returns on the BAB book to show it is an
independent lottery effect, not re-labelled low-vol.

**Fit.** Decorrelated, crypto-native; pairs naturally with H1 as a "retail-mispricing" sub-book.

---

## H3 — On-chain / exchange-flow signals (crypto)  — ❌ TESTED, DEAD (free-data)

> **Verdict — tested end-to-end, dead ([ONCHAIN.md](strategies/ONCHAIN.md), `make onchain`).**
> **Exchange flows are free for BTC/ETH and were tested; they do not work.** An earlier version of this
> verdict called them pay-walled — wrong, and wrong in the flattering direction: entitlement was inferred
> from group 403s instead of read from the vendor catalog, and a multi-metric call 403s whole if any one
> metric is Pro. Two names cannot form a cross-section, so flows are a BTC/ETH timing overlay: the best,
> BTC exchange-supply-trend long/flat, nets **+0.96** vs buy-and-hold +0.85 — but **rotating the same
> position path to random dates has a 95th percentile of +1.01**, so none of the four flow overlays beats
> its own random-timing control, and 7 of 8 HAC predictive regressions show no forecasting power (only
> BTC net-flow→7d, t=−2.04, which is the chance rate for eight tests). What *is* genuinely Pro-walled is
> narrow: adjusted transfer value, entity-adjusted supply bands, USD fees (realized cap is recoverable as
> mktcap÷MVRV). **Four names were also measuring dead ERC-20 shells** (VET/ZIL/QTUM/LRC, zero active
> addresses on 44-79% of days) — excluding them cuts the universe to **33** and the headline value book
> from +0.40 to **+0.15** (MC-P5 −0.52, placebo 72nd pctile, deflated 0.07): a third of the old headline
> was that artifact. **One signal survives the clean-up:** adoption momentum (active-address growth,
> top-20) nets **+0.73**, MC-P5 **+0.08**, placebo **98th pctile**, WF-OOS **+0.74** with construction
> held fixed, alpha over price **t=+2.04** — stable across every top-N, unlike `divergence` (+0.80 at
> N=20 only). It is still **excluded**: post-hoc, deflated 0.50 at the family's 36 trials, **+0.32
> correlated with price momentum** (a better-built momentum, like H5), and the book does not move
> (3.828→**3.831** at 15%). The value axis is subsumed exactly as published (Liu-Tsyvinski-Wu JF2022;
> Cong et al MgmtSci2024); the new axes fail outright — dilution is *negative*, ownership and fee-yield
> flat. Non-flow BTC/ETH overlays still underperform buy-and-hold (+0.85; best SSR +0.55). **ML changes
> nothing** (21-trial ranker, purged CV, now with holder/issuance/fee features): on-chain best **+0.32**,
> the same harness on *price* features **+1.09** (proving the method), adding on-chain to price degrades
> it. **top-50/100 impossible on free data** (SOL/SUI/TON/APT carry market data only → 33-name ceiling).
> The remaining honest upgrade paths are paid **entity-level** flow labelling, a wide small-cap panel, or
> — untested and free — **protocol fundamentals** (fees/revenue/TVL via DefiLlama), which reach exactly
> the chains Coin Metrics cannot. Spec kept below.

**Thesis.** Exchange net-flows, stablecoin-supply changes, active-address growth and miner flows carry
information **not present in price** — the one genuinely new information source for crypto. The main
report already names on-chain as the honest next step.

**Construction.** Time-series and cross-sectional signals, e.g. exchange-inflow spike → sell pressure
(short); stablecoin-supply expansion → risk-on (long BTC/majors); address-growth momentum → long. Start
with BTC/ETH time-series, then cross-sectional across majors.

**Data.** **NOT in the repo — confirm free access FIRST** (Coin Metrics community API, blockchain.com
charts, Glassnode free tier, Dune). This is the one hypothesis that needs a new ingestion loader; scope
that before committing. Everything must be point-in-time (no revised/backfilled aggregates).

**Survive/kill.** Survives if it adds decorrelated OOS edge over price-only sleeves. Higher effort,
higher upside. Beware look-ahead in on-chain aggregates (many are revised) — align to first-availability.

**Fit.** Genuinely orthogonal to every price/funding sleeve. Best diversifier if the data holds up.

---

## H4 — Calendar seasonality done right: pre-FOMC drift + turn-of-month  — ❌ TESTED, real-but-beta

> **Verdict — tested end-to-end, not deployable ([SEASONAL.md](strategies/SEASONAL.md), `make seasonal`).** Both
> effects are real in the data but net of cost they are **beta-timing, not alpha** — H4 joins the
> overnight/session family in the "real-as-beta, not viable market-neutral" pile. The **pre-FOMC drift**
> has a clean signature (SPY +8.7bps day-before / +7.5bps announce, then −16.9/−15.8bps the two days
> after; in-window Sharpe +1.25) but the standalone timing book nets only +0.05–0.13 across SPY/QQQ/IWM/
> DIA and **does not beat a shuffled-calendar placebo** (63rd–74th pctile; a 1-day hold pays a full
> round-trip for ~8 events/yr). **Turn-of-month is beta**: the net Sharpe *rises monotonically as the
> window widens* toward buy-&-hold (SPY (−1,+1) 0.08 → (−4,+5) 0.77 ≈ B&H 0.76), the classic (−1,+3)
> window (0.29) **underperforms buy-&-hold** and sits at the 57th placebo pctile, the stock book is flat
> at ~0.29 across top-50…500 (no cross-sectional signal), and crypto ToM is dead (−0.01). The combined
> SPY sleeve nets +0.32 (MC-P5 −0.11, deflated 0.31), is decorrelated (+0.18) but sub-bar so it **drags
> the book** (3.47→3.16 at 30%). **One genuine result kept for the edge map:** BTC's exact 24h→2pm-ET
> pre-FOMC window returns **+102bps, t=+2.4** — a significant crypto risk-on drift, not a levered sleeve.
> **Trading it market-neutral *between* assets and with ML does not rescue it either:** a dollar-neutral
> long/short across names (in-window only) is negative or sub-bar (crypto pre-FOMC −0.47, stocks ToM −0.41
> below-random, crypto ToM +0.36 at the 96th placebo pctile — one marginal near-miss); a conditional
> pre-FOMC ML gate (VIX/10y-2y-slope/drift, purged CV) makes it worse (SPY 0.24→0.07, negative OOS IC) and
> a cross-sectional LGBM ranker is worse in-window than all-days. Removing the beta removes the return.
> The spec below is kept for reference / to prevent a re-run.

**Thesis.** (a) **Pre-FOMC announcement drift** — a large share of the equity premium accrues in the
~24h before scheduled FOMC announcements (Lucca-Moench 2015); (b) **turn-of-month** — returns
concentrate in the last + first few trading days (flows). Both are calendar-deterministic → decorrelated.
Unlike the just-killed overnight sleeve, these are **event-based and low-turnover**, so cost will not
automatically kill them.

**Construction.** Long the equity index (SPY/QQQ) only inside the pre-FOMC window / turn-of-month window,
flat otherwise; optionally a cross-sectional tilt. Also test the crypto analogue (turn-of-month, weekend).
FOMC dates are deterministic; take them from the Fed calendar / FRED.

**Data.** Ready (price panels; FOMC dates deterministic). **Do not** reuse the daily-round-trip cost model
that killed overnight — these hold for days, so charge only entry/exit.

**Survive/kill.** Survives if net Sharpe > 0.5 **and** it is not merely beta-timing (compare vs
buy-and-hold, as with overnight). Prior: real but low-capacity and partly beta — map it honestly.

**Fit.** Decorrelated, adds equity breadth the book lacks; low capacity.

---

## H5 — Residual / idiosyncratic momentum (equity)  — ✅ TESTED, in-family refinement (not a new source)

> **Verdict — tested end-to-end ([RESIDMOM.md](strategies/RESIDMOM.md), `make residmom`).** Residual momentum is a
> real, literature-consistent improvement to momentum *construction*, but a better-built momentum, **not a
> decorrelated new source** — so it upgrades an existing sleeve rather than joining the book as a family.
> It **beats raw momentum outright on crypto** (+0.45 → **+0.61** standalone, walk-forward incremental
> **+0.25**, 94% of the grid positive, positive 6/7 years) and **halves the momentum-crash bleed on equity**
> (raw's worst-5-months −12.3% → residual −5.0%). But it is **~0.8 correlated with raw momentum** and adds
> **no significant alpha over it** (t = +0.1…+1.1), so it **does not lift the master book** (corr 0.18–0.35,
> dilutes at every weight). The **"lower-beta" premise does not bind** — the book is already dollar-neutral
> (β ≈ −0.005 crypto / −0.05 equity), so there is no market beta to remove; the only large beta-strip (FX
> +0.35 → −0.15) is where there is no edge. The thesis was equity, but the win is **crypto** — where the
> construction reproduces the literature to the parameter (formation = the 1–4-week horizon, EW-panel factor,
> no skip). **On equity the canonical decoupled form is *below* raw at top-100** (+0.41 vs +0.48; single-window
> ties it +0.49) and only wins at full breadth (+0.70 vs +0.56) — a crash-hedge and quality improvement, not a
> return upgrade. **FX dead.** Honest role: a **drop-in upgrade to the crypto momentum sleeve** (`risk_adj_mom`
> → `idio_mom`) and a tail-safer equity leg — the highest-certainty modest win, lowest diversification value,
> exactly as ranked. Spec kept below.

**Thesis.** Momentum on market-beta-**residualized** returns (Blitz-Huij-Martens 2011) is higher-Sharpe
and lower-beta than raw momentum, especially in equities. The book's equity legs are its weakest; this
could lift them without adding a new source of drawdown.

**Construction.** `resid_mom` **already exists** in `src/sleeves/xsect.py` (regress out the equal-weight
market, rank on the residual's mean/vol). Test standalone on the equity panel vs the raw x-sect momentum
already in the book; measure the incremental OOS Sharpe and the beta reduction.

**Data.** Ready (`resid_mom` implemented). Smallest lift-to-effort of the five.

**Survive/kill.** Survives if OOS Sharpe > the raw-momentum equity leg **and** lower beta.

**Fit.** Momentum-adjacent — improves equity breadth, but does **not** diversify the source (it is still
a momentum premium). Lower diversification value than H1–H3; highest certainty of a modest win.

---

## Also worth one measured shot (not top-5)

- **Pure ML cross-sectional return forecaster** (the §5 "ML forecast" the brief asks for, as *alpha* not
  a meta-gate): GBM on the full 82-feature panel → forward-return rank → dollar-neutral book, with strict
  purged CV and feature-selection **inside** folds. Measures whether any ML alpha exists beyond the
  classical premia. High overfit risk — the repo already found an ML *ranker on carry* destroyed value;
  frame this as a measurement ("is there ML alpha left?"), and expect the honest answer may be "no".

## Do NOT retry (already tested here — dead or already in the book)

Overnight/session ([OVERNIGHT.md](strategies/OVERNIGHT.md)), **calendar seasonality — pre-FOMC drift + turn-of-month
([SEASONAL.md](strategies/SEASONAL.md) — H4, real-but-beta / fails the shuffled-calendar placebo)**,
**skewness/lottery-MAX ([LOTTERY.md](strategies/LOTTERY.md) — H2, inverted in crypto / sub-bar in equity)**,
**on-chain / network signals ([ONCHAIN.md](strategies/ONCHAIN.md) — H3, free-data dead: value is a static
coin-type tilt, BTC/ETH exchange flows fail their random-timing control, and the one live signal is
re-labelled momentum that leaves the book flat)**, single-asset mean-reversion, daily cross-sectional reversal,
pairs stat-arb, volume-spike ([VOLSPIKE.md](strategies/VOLSPIKE.md)), FX carry, equity dividend carry,
funding-**momentum** (MACD on funding — already −1.42, the *level* is the carry signal). Trend, breakout,
cross-sectional momentum, funding carry and short-vol/VRP are already surviving streams — refine, don't
re-discover.

## Priority call

**H2 (skew/lottery) is tested — dead ([LOTTERY.md](strategies/LOTTERY.md)):** in a liquid tradable universe the
lottery short collides with the momentum premium (inverting it in crypto) and what skew signal remains is
not independent of low-vol. **H4 (calendar seasonality) is tested — real-but-beta, not deployable
([SEASONAL.md](strategies/SEASONAL.md)):** the pre-FOMC drift and turn-of-month are genuine effects (crypto's 24h
pre-FOMC drift is significant, t=2.4) but net of cost they are beta-timing that fails the shuffled-calendar
placebo and drags the book — mapped alongside overnight. **H1 (BAB)** is the live version of the
retail-mispricing thesis — the leverage-constraint premium, tested separately ([BAB.md](strategies/BAB.md)).
**H3 (on-chain) is tested — dead on free data ([ONCHAIN.md](strategies/ONCHAIN.md)):** exchange flows turned out to be
free for BTC/ETH and were tested — they beat buy-and-hold only by collecting beta, and lose to random
timing of the same position path; the free valuation axis degenerates to a static coin-type tilt; and the
one signal that clears the gates, adoption momentum (+0.73, alpha t=+2.04), is +0.32 correlated with
price momentum and leaves the book at 3.83 either way (free cross-section caps at 33 names — no top-50/100). **H5 (residual
momentum) is tested — an in-family refinement, not a new source ([RESIDMOM.md](strategies/RESIDMOM.md)):** it beats raw
momentum on crypto (+0.45 → +0.61, walk-forward incremental +0.25) and halves equity momentum-crash bleed,
but is ~0.8 correlated with raw momentum (no alpha over it), so it upgrades the existing crypto momentum
sleeve rather than joining the book — the highest-certainty modest win, as ranked. **All five top-5
hypotheses are now tested end-to-end**; the surviving decorrelated additions are H1-BAB (crypto satellite)
and the short-vol/VRP already in the book — the momentum family is refined by H5, not extended.
