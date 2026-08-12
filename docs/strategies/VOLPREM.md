# Vol premium — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the equal-weight master book assembled by `scripts/run_master_book.py`; its composition, scorecard, leverage and target verdict live in [REPORT.md](../../REPORT.md), which is RENDERED from the artifacts and so cannot disagree with the run. Restated here they would go stale the next time the book is re-run, which is exactly what happened to the numbers this line used to carry — so this page quotes none of them. Any master-book figure below is a snapshot from when this family was evaluated, and is labelled as one.

**Scope.** Build and honestly evaluate a **short-volatility / variance-risk-premium (VRP)** sleeve —
the source structurally *orthogonal* to the trend book (short gamma vs long gamma), chosen after
cross-sectional reversal was confirmed dead. Implied vol is Deribit **DVOL** (free public API, BTC &
ETH, from 2021-03-24); the realised leg is the Binance perp bars. All numbers vol-targeted to 15%,
net of vega-spread costs, executed at t+2, over 2021-03-24 → 2026-08. Reproduce: `python
scripts/volprem/run_vol_premium.py`. Artifacts: `reports/volprem/volprem_*.{csv,parquet}`.

---

## 0. TL;DR

- **The VRP is a real, decorrelated return source — but honest accounting is everything.** Short-vol
  harvests the variance risk premium (implied > realised) and is ~0-correlated to the momentum and
  carry books (+0.03 / −0.02) — a structurally new source. Its danger is a fat left tail, and *how you
  measure the realised leg decides whether you even see it*: on the naive close-to-close leg crypto
  looked positive (BTC +1.36), but under the honest OHLC leg (intraday path + gap) **crypto short-vol
  is negative** and the premium lives in the equity-index / commodity / rates cross-section (§2c).
- **Where it is real (honest OHLC leg):** **EM +3.11, Dow +2.86, Russell +2.84, S&P +2.79, gold +2.41,
  bonds +2.38, oil +2.21, gold-miners +2.06** carry the book on Sharpe — but half of them earn *and*
  blow up: VIX→SPY, VXD→DIA, VXN→QQQ, OVX→USO and VXEWZ→EWZ each carry a −99/−100% single-day tail, so
  a high Sharpe here is never a reason to hold one alone. **Crypto is negative** (BTC −0.41, ETH −0.86)
  and FX (EVZ) is ruinous. Equity/commodity/rates give 2005–2011 histories vs crypto's 2021.
- **It is deployed as a diversified BOOK, never one asset.** Across **18 Cboe underlyings** with clean
  OHLC (VIX/VXN/RVX/VXD/VXEFA, VXAPL/AZN/GOG/GS/IBM, VXEEM/EWZ/FXI, OVX/GVZ/VXSLV/VXGDX, VXTLT), the equal-risk
  book nets **Sharpe +3.58, maxDD −78%, skew −18, 84% profitable months** (15 of the 18 live through
  2022-24 — see the publication gap below). Crypto and FX are excluded on
  frozen structural / data-quality rules (below), *not* on backtested Sharpe. Diversification is the whole
  game: several single-name sleeves **hit −99/−100% ruin** standalone; at 1/18 risk each the book absorbs them.
  (18 Cboe legs are deployed, incl. VXGDX gold-miners; crypto DVOL and FX EVZ are considered but excluded for unhedgeable / corrupt data — below.)
- **The realised leg is measured from OHLC (intraday path + gap), not close-to-close — the honest tail.**
  A delta-hedged short-gamma book pays the intraday path, which close-to-close nets away. Correcting it
  (Rogers-Satchell + overnight) barely moves Sharpe but nearly doubles the tail (maxDD
  −50%→**−78%**, skew −8.7→−18). Verified on genuine events — the 2010 Flash-Crash alone was a −47%
  book day (SPY closed −3% but ranged 11% intraday). The premium is real; the *risk* was understated.
  In the master (equal-weight risk parity, 8 families) the honest book lands the portfolio at **DD ≈ −8%** — inside
  the 15% mandate, up from the flattered ≈ −6% the close-to-close book showed.
- **The premium is the source, proven by placebo.** Re-strike the same swap at *trailing realised
  vol* instead of DVOL — stripping the implied premium — and it flips to **−0.82**. The edge lives
  in implied-minus-realised (the VRP), not in the short-variance machinery.
- **It harvests best by simply being short; DVOL timing does not add.** Gating to "short only when
  implied is rich" (+0.64) *underperforms* always-short (+0.81) — the premium is a persistent
  **level**, so timing it throws away good days. Same lesson as the carry book (level, not momentum).
- **Deployed as a family in the master at equal weight (1/8 ≈ 12.5%), watched — not maxed.** It is the top
  marginal contributor but also drives the portfolio tail (master DD ≈ −8%, vs ≈ −6% on the flattered
  close-to-close book); with a −78% own-tail it should sit at or **below** its parity share, never above.
- **Honest hazards:** a severe left tail (**skew −18, −78% DD**) verified on real flash crashes; the
  standalone Sharpe is not tradeable at face value (a real tail hedge needs the paid option smile); and
  a naive cap looks like Sharpe 8 only by truncating the crash **for free** — the honest book is uncapped.

---

## 1. Construction

A discrete **variance swap**, re-struck weekly (`src/sleeves/vol_premium.py`). Short the implied
variance `K² = (DVOL/100)²` fixed at the decision bar, pay the perp bars' own realised variance
`365·r²`; the short profits in calm (realised < implied) and loses in a spike. Per-option historical
marks are not free, so the P&L is a variance-swap **replication**, not an executed option chain; the
option spread is modelled in vega terms (`dVar ≈ 2K·dVol`) and charged on each roll. Two assets
(BTC, ETH), each vol-targeted to 15% and equal-risk averaged.

- **Non-ML baseline (§5):** always-short (side = −1). Parameter-free — nothing to overfit.
- **Family rule:** short only when `DVOL > k·RV_trailing` (implied rich vs recent realised).
- **Leakage:** the strike and side are decided at the bar and **shifted t+2** before they multiply
  any squared return, so no leg sees its own or an earlier day's realised move. The realised leg the
  short pays is the *outcome*, not a peek.
- **Risk control (§8), family-specific:** short vol's defining hazard is the left tail, so the honest
  version leans on **position (vol-target) de-risking after a spike** and, optionally, a capped swap —
  *not* the uniform portfolio stop that suits a trend sleeve. The cap is left off the headline because
  capping is not free (see §5).

## 2. Standalone profile (vol-targeted 15%, net, t+2)

> **Scope — the crypto-only (BTC+ETH) construction study on the *close-to-close* realised leg.** It
> establishes the machinery (always-short baseline, placebo, the capping trap); the **deployable** form is
> the OHLC cross-asset book in §2c, where under the honest realised leg crypto short-vol turns *negative*
> (BTC −0.41, ETH −0.86) and the premium lives in the equity / commodity / rates cross-section. §3
> (sensitivity/exec-lag/cost) is measured on this crypto construction.

| book | Sharpe | MC-P5 | maxDD | worst day | skew | months+ |
|---|---|---|---|---|---|---|
| **always-short baseline** (parameter-free) | **+0.81** | +0.04 | −29% | −16% | **−6.35** | 63% |
| short-when-rich (timed) | +0.64 | −0.12 | −33% | −16% | −6.67 | 63% |
| BTC leg (timed) | **+1.11** | +0.17 | −32% | −22% | −7.57 | 62% |
| ETH leg (timed) | +0.07 | −0.63 | −58% | −29% | −10.2 | 55% |
| **placebo — struck at realised vol (no premium)** | **−0.82** | — | −69% | −16% | −6.36 | 51% |
| *capped 2.5× (wing cost NOT modelled)* | *+8.21* | *+7.33* | *−6%* | *−4%* | *−0.99* | *89%* |

The always-short baseline being parameter-free is the strongest robustness evidence — there is no
knob to fit. The capped row is shown only to flag the trap: truncating the crash tail for free
manufactures a Sharpe-8 illusion.

**Per-year (baseline):** 2021 **+6.20**, 2022 +0.37, 2023 +1.19, 2024 −0.04, 2025 +0.62, 2026 −0.80.
The premium is strong in calm regimes and flat-to-negative in 2024/2026; 2021 is a partial-year
low-vol-launch artifact. Regime-dependent, exactly as short vol should be.

## 2c. Where and how it is run — the diversified book

The universe is the underlyings with a free implied-vol index **and clean data** — **18 Cboe indices**:
VIX/VXN/RVX/VXD/VXEFA (equity), VXAPL/VXAZN/VXGOG/VXGS/VXIBM (single names), VXEEM/VXEWZ/VXFXI
(international), OVX/GVZ/VXSLV/VXGDX (commodities), VXTLT (rates); realised legs are the matching ETF/spot
(Twelve Data). **Crypto (DVOL) and FX (EVZ) are excluded on frozen ex-ante rules, not on backtested
Sharpe:** crypto's 30%-intraday-range days are unhedgeable for a short-vol delta-hedge (so BTC −0.41,
ETH −0.86 on the honest leg), and the free EURUSD OHLC carries corrupt prints (a −13% "daily" move at a
1% range) with EVZ discontinued 2025-03. Dropping the weak-but-clean single names would be overfitting,
so they stay.

**Three of the eighteen are not continuously live, and until 2026-08 the loader hid it.** Cboe stopped
publishing **VXSLV, VXFXI and VXGDX in Feb-2022** — the same month it discontinued VXXLE, which the
universe rule already excludes for exactly this reason — and resumed them in 2025. The implied series was
reindexed onto the underlying's trading calendar with an *unbounded* forward fill, so through 2022-2024
those three legs kept selling variance at their frozen February-2022 strike against live realised vol.
Three years of P&L out of a number that had stopped moving, and the book counted eighteen live sleeves the
whole time when it had **fifteen**. The fill is now capped at five bars (enough to cross a holiday
mismatch, not enough to invent a strike) and an unquoted bar drops the leg from the frame instead of
holding it flat, so the equal-weight average renormalises onto what is genuinely quoted. Everything below
is measured after that repair; it cost the book **0.14 Sharpe on the full window and 0.59 out-of-sample**,
where the manufactured legs had been padding the calm stretch that the OOS block mostly is.

**Run it as a book, not one asset.** Under the honest OHLC realised leg the per-sleeve picture is brutal
(`reports/volprem/volprem_book_sleeves.csv`): the legs that carry the book are **EM +3.11, Dow +2.86, Russell +2.84,
S&P +2.79, gold +2.41, bonds +2.38, oil +2.21, gold-miners +2.06**, while several of those same legs
(VIX→SPY, VXD→DIA, VXN→QQQ, OVX→USO, VXEWZ→EWZ) carry **−99/−100% single-day drawdowns** on flash-crash
days — earning and ruinous are the same legs, which is the case for the book and against any one of them. The deployable form is the equal-risk **book** across
the 18, which stays positive only because these catastrophes fall on *different* dates:

| | Sharpe | maxDD | skew | months+ | span |
|---|---|---|---|---|---|
| **diversified book (18 legs; 15 live 2022-24)** | **+3.58** | **−78%** | **−18** | 84% | 2005–2026 |
| average single sleeve | ~+1.8 | ~−81% | — | — | — |
| placebo book (fair strike, no premium) | **−1.73** | — | — | — | — |

- **Diversification is the whole game.** Book Sharpe +3.58 vs a single-sleeve average ~+1.8, and the
  book's −78% tail still beats most single sleeves' −99% (mean pairwise sleeve corr +0.26). Placebo
  −1.73 — the premium, not the basket, is the source.
- **Crypto short-vol does not survive honest accounting.** A reversal from the close-to-close view
  (which showed BTC +1.36): crypto's intraday path is too violent for a delta-hedged short. This is why
  the crypto-only baseline (§2) is superseded by this cross-asset book.
- **Cross-asset breadth is what works.** EM / gold / bonds / gold-miners earn without the −99% tail; the
  equity-index legs earn a comparable Sharpe but bring that tail with it, and crypto brings only tail.
  Breadth over *uncorrelated* crashes — not equity breadth, whose
  vol spikes are systemic (2008, Mar-2020 hit every equity leg at once) — is what keeps it alive.
- **The tail is the true risk, and it is −78%, not the −50% close-to-close showed.** Verified on real
  events: the 2010 Flash-Crash alone was a −47% book day (SPY closed −3% but ranged 11% intraday).
  Sharpe cannot see this; treat the book as a *decorrelated diversifier whose tail must be respected*,
  never a Sharpe-3 engine. A real tail hedge needs the live option smile (paid data, §4b).
- **Timeframe: daily is the right frame — tested, not assumed** (`scripts/volprem/run_vol_premium_tf.py`, on
  BTC/ETH intraday DVOL — the only free intraday implied vol, and crypto is now excluded anyway). There,
  headline Sharpe *rises* intraday (1d +0.81 → 4h +3.39 → 1h +5.15) but it is a mirage: skew collapses
  **−6.4 → −23.8**, part is a sampling artifact (aggregating the 1h book to daily drops Sharpe 5.15 →
  3.42), risk is not held at 15% (the 1h book runs ~40% vol as the vol-target lags fine bars), and the
  cost model omits the intra-bar delta-hedging that actually erodes intraday short-vol. The book's equity
  underlyings have **no free intraday implied vol at all**. Verdict: daily, by both the premium's 30-day
  horizon and data availability.

## 3. Robustness

- **Sensitivity:** 100 configs (k_rich × RV-lookback × restrike) — **96% positive**, median +0.72,
  range [−0.79, +1.21]. A broad plateau, not a spike (`reports/volprem/volprem_sensitivity.csv`).
- **Exec-lag ladder:** Sharpe +0.52 → +0.64 → +0.74 → +0.89 → +1.00 at lag 1/2/3/5/8 — **stable-to-
  rising**, the opposite of a leak's collapse. The premium is slow-moving; precise entry timing is
  not the alpha, which is why it survives (and even prefers) delay.
- **Cost:** base +0.64, 3× +0.57, **break-even ≈ 16 vol pts/roll (~22× base)**. The premium dwarfs a
  realistic vega spread — cost is not the binding constraint.
- **Book cost robustness — the shipped number is NET of option cost, not gross** (18-leg book,
  `reports/volprem/volprem_cost_robustness.csv`). Every leg is charged a per-underlying vega half-spread
  (`COST_BY_CLASS`: index **1.0**, single-name **2.5**, EM/commodity **2.0** vol-pts/roll — at/above the
  published ~0.5-vega index / 1–2.5-vega single-name range, J.P. Morgan / Risk.net). Those spreads already
  cost **~0.38 Sharpe** (gross ×0 **3.96** → shipped ×1 **3.58** @252), and the edge **survives far wider
  execution**: ×2 **3.19**, ×3 **2.79**, ×5 **2.00**. So `var_cap=1e9, wing_markup=0` (**"naked"**) means
  *no bought tail hedge* — the book eats the full −78% tail — **not** that costs are unmodelled. The
  binding constraints are the systemic tail and single-name variance-swap **capacity**, not per-trade cost.

## 3b. Capacity — the real binding constraint

Cost is not the limit (above); **capacity is**. A short-variance book is sized in **vega notional** ($
P&L per 1 vol-point), and the 18 legs sit in very different markets:

| leg class | legs | vega depth per roll (order of magnitude) | source |
|---|---|---|---|
| equity **index** | VIX / VXN / VXD / RVX / VXEFA | **deep** — >$2 bn index-variance vega outstanding ($1.5 bn S&P); a desk quotes ~$1–5 M | [CFTC / Mixon](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_volderivatives.pdf) |
| **single name** | VXAPL / VXAZN / VXGOG / VXGS / VXIBM | **thin** — "very few participants outside dispersion"; ~$50–200 k | [J.P. Morgan](https://derivativesacademy.com/storage/uploads/files/modules/resources/1702207867_allen_einchcomb_granger_jpm_2006_variance_swaps.pdf) |
| **exotic ETF-vol** | VXEEM / VXEWZ / VXFXI, OVX, GVZ, VXSLV, **VXGDX**, VXTLT | **barely a variance market** — replicated with listed ETF options, ~$10–50 k, wider | — |

Because the book is **equal-weight**, it needs ~equal vega in every leg, so capacity is set by the
**thinnest** legs — and half the book (single-name + exotic ETF-vol) sits in that thin-to-nonexistent
tail. Order of magnitude, at the modelled spreads and assuming the thin legs absorb ~$10–50 k vega/roll:
the ~5 deep **index** legs alone would scale to **$100 M+**, but the **full 18-leg equal-weight
construction is capped at roughly the low tens of $M** before the thin legs can no longer be filled —
against a **$500 k** demonstration book (`CAPITAL_USD`), so there is real but *not* institutional-scale
headroom. Past that you must **drop to the deep index legs**, which shrinks the very diversification that
softens the −78% tail: a *bigger* tail at *bigger* size.

The ×3-spread stress (§3) buys headroom for "the thin legs cost more than modelled"; it does **not** buy
headroom for "the market isn't there at size." Honest bottom line: the standalone Sharpe is a
**research-scale** number — the deployable form at institutional size is *fewer, deeper legs plus a real
option-smile tail hedge*, i.e. materially lower Sharpe, stated not hidden.

## 4. Portfolio value-add

VRP is cleanly orthogonal — short gamma pays when trend bleeds in calm, and trend's long gamma hedges
VRP's crash. Correlation to the momentum and carry books is **~0** (+0.03 / −0.02). That decorrelation,
not the standalone Sharpe, is what earns it a slot.

In the canonical master (`scripts/run_master_book.py`, equal-weight risk parity over the traded families), the
honest 18-leg volprem book is the **top marginal contributor** (removing it drops the equal-weight stack from
3.78 to **1.73**). But it also drives the portfolio's tail — the book's honest (jump-to-open) drawdown is **≈ −8%
with volprem vs ≈ −6% on the flattered close-to-close accounting** (§4b). So it is a genuine co-engine *and*
the family that most needs its weight watched, exactly because its own tail is −78%.

## 4b. Deployable form — fitting the 15% mandate

The naked book's **−78%** standalone drawdown does not fit a 15% portfolio DD limit, and **the
instrument-level fix — an option wing that caps the tail — cannot be credibly priced from free data**
(`scripts/volprem/run_vol_premium_deploy.py`): leave the cap unpaid and vol-targeting inflates a fake Sharpe
(15+); price it off the trailing realised tail and it over-/mis-charges into ruin (−100%). A real tail
hedge needs the live option smile — which turns out to be obtainable free, and is now priced in §4c. A sleeve-level P&L stop and an ex-ante implied-spike
de-gross were both tested and **do not help** *reactively*: the crashes are too fast to de-risk into once vol is
already spiking. (A **leading** signal is different — a VIX-term-structure gate fires *before* the
crash, §5 below and [REPORT.md](../../REPORT.md) §5d/§6, and times the book's exposure — the scorecard it reaches is reported there, not restated here.)

**The mandate is on the portfolio, and sizing meets it.** In the canonical master (equal weight, 1/8 to
volprem), trend's long gamma structurally hedges VRP's vol-spike crashes, so the portfolio lands at
**DD ≈ −8%** — inside the 15% limit, though with less margin than the flattered close-to-close book
showed (**≈ −6%**). VRP is deployable at ≈ its risk-parity share, no tighter; its −78% own-tail means it
should if anything sit **below** parity (a per-family tilt), not above. The trustworthy claim is the
drawdown behaviour, not the Sharpe; the offset it leans on is structural (a vol spike is a big move
trend rides) but not guaranteed in a whipsaw regime.

## 4c. What the tail hedge actually costs — priced from real quotes, for free

§4b says the instrument-level fix cannot be credibly priced from free data. **That was wrong, and this
section retracts it.** The obstacle was never the price of data; it was that nobody here had ever looked
at an option quote. historicaldata.net publishes **Jan–Jun 2013 free**, full chain with bid/ask, greeks
and IV on 3,800 underlyings — enough to price the wing directly (`scripts/volprem/run_wing_cost.py`).

**Method: the wing's price *is* a truncation of the variance strip.** A variance swap's fair strike is
the model-free integral over the whole OTM strip (the VIX construction). Capping the swap at
`var_cap · K²` means, in replication terms, giving up the far tail of that strip — so the cost is simply

    wing = K²(full strip) − K²(strip truncated below the crash strikes)

Both sides come from the same quoted chain, so this is a market price, not a model. Measured over 615
chain-days on the book's five deep legs:

| leg | ATM implied vol | wing cost, share of sold variance | 90th pct |
|---|---|---|---|
| IWM | 17.6% | **4.6%** | 9.7% |
| QQQ | 14.7% | 6.2% | 11.8% |
| SPY | 13.7% | 6.5% | 11.9% |
| GLD | 18.1% | 16.1% | 37.9% |
| TLT | 13.5% | 26.3% | 71.8% |
| **mean** | — | **12.0%** | — |

**Through the cycle it is ~16%, and — the load-bearing surprise — it does *not* spike when you need it.**
2013 is calm, so the level is scaled by Cboe's **SKEW** index (free, 1990+), which measures exactly the
tail strip's weight relative to the at-the-money one: ×1.35 over 2005–2026, ×1.20 through COVID, and
**×0.74 through the 2008 GFC** — *below* the calibration. In a crash at-the-money variance explodes
faster than the tail strip, so tail protection gets relatively *cheaper* at the moment it pays. That is
what makes a permanently-held cap affordable at all.

**The result, with the wing paid at the through-cycle 16.2%** (book of 18, VIX-gated, everything else
unchanged):

| construction | Sharpe | max-DD | worst day | skew | months+ |
|---|---|---|---|---|---|
| naked (shipped today) | +4.42 | **−77.6%** | **−76.4%** | −26.3 | 89% |
| capped 2.5×, wing **unpaid** — the known trap | +10.74 | −36.4% | −7.1% | −0.9 | 93% |
| **capped 2.5×, wing paid through-cycle** | **+6.89** | **−43.9%** | **−6.4%** | **−0.8** | 90% |

The −78% one-day annihilation becomes a bounded −44% grind: **worst day −76.4% → −6.4%, skew −26 → −0.8**.
That is the single largest risk reduction available anywhere in this project.

**The regime scaling is validated inside the free window, not assumed.** Jan–Jun 2013 spans VIX 11.3–20.5
(the June taper-tantrum), so the wing's own stress sensitivity is measurable from the quotes themselves:
**10.3% at low VIX → 13.4% at high VIX, ×1.30** — against the ×1.35 the SKEW index says independently over
20 years. Two free estimates, one from quotes inside the window and one from a two-decade index, agree.
Per-leg correlations are weak and mixed (SPY −0.05, QQQ +0.32, GLD −0.26), so read ×1.30 as a coarse
bucket average rather than a per-leg predictor.

**What would break it, stated:** the whole result rests on the wing costing ~16% of sold variance.
Measured break-even is **~3× that (36%)** — at 2× the leg still returns +4.3 but on a −64% drawdown, and
at 3× it is worthless. So the margin is ~2.2×, which is thinner than every other cost sensitivity in this
project (the book breaks even at 5×, the sleeve's vega spread at 22×). The 12% level is measured on five
deep legs in one calm half-year and extended by an index proxy rather than by quotes; buying a single
crisis year of chains (~$99) would settle **one specific thing** — not the price level, which is now measured
twice, but whether the stress relation holds *outside* its measured range: the free window tops out near
VIX 20 and a real crisis is 40–80, so ×1.30 is extrapolated four-fold. **Until then this is not shipped** — the deliverable stays the naked book with its
disclosed −78% tail, because a headline resting on a 2.2× margin from a proxy is not a headline.

## 5. Honest limits & ceiling

- **The systemic tail is the binding constraint; breadth cannot remove it, but *timing* sidesteps it for the
  portfolio.** Even across 18 underlyings the book draws down **−78%** in a systemic vol event (measured on the OHLC
  realised leg — the honest number; close-to-close showed only −50%). Cross-asset breadth over *uncorrelated*
  crashes softens it but does not defuse it, and an instrument-level tail hedge needs the live option smile (paid).
  What this strategy does instead is **time** the exposure: a VIX-term-structure regime gate flattens the leg
  whenever *either* curve segment inverts (VIX3M/VIX < 1 or VIX/VIX9D < 1), *before* the systemic crash — the
  strategy's own signal (`src/risk/vol_regime.py`, published as
  the `ret_gated` series the book consumes; [REPORT.md](../../REPORT.md) §5d/§6) that lifts the master to 5/5 on
  both windows. It catches **9 of the leg's 10 worst days** (the long segment alone catches 4), and its switching
  is charged the vega spread at the sleeve — ~27 round trips a year — so the timing is not free: on the finished
  P&L it would read a full 0.3 of book Sharpe higher, straight through the ≤4.0 target band. It cuts the tail's
  *portfolio* clustering (the losing-month driver), not the standalone −78%, and it cannot reach a one-session
  dislocation out of a calm curve: into 2010-05-06 the curve stood at VIX3M/VIX **1.059** and inverted only on the
  crash day (VIX9D does not even list until 2011, so that day cannot test the fast segment either).
  Gated standalone: Sharpe **+3.91**, months-in-profit **87%**, DD still **−77.7%** — the tail is the flash crash.
- **Sharpe overstates; skew and drawdown are the honest metrics.** Correcting the realised leg from
  close-to-close to OHLC (path + gap) barely moved the Sharpe but nearly doubled the tail (DD −50% →
  −78%, skew −8.7 → −18). Trust the −18 skew and −78% DD, never the Sharpe.
- **Crypto and FX are excluded, not in the book.** Crypto short-vol is structurally negative under the
  honest leg (BTC −0.41, ETH −0.86 — its intraday path is unhedgeable), and the free EURUSD OHLC is
  corrupt (EVZ discontinued 2025-03). The book is carried by EM / gold / bonds / gold-miners plus the
  equity-index legs, and it is those equity legs that bring the −99% tail with their return.
- **Universe is gated by free implied-vol indices with clean data** (18 Cboe underlyings deployed). No free
  per-name-beyond-Cboe or altcoin IV, and crypto/FX excluded per above.
- **What did not work (kept, not hidden):** single-asset deployment (several sleeves ruin at −99/−100%);
  DVOL *timing* (gating underperforms always-short — the level is the premium); the free-cap
  construction and the priced-wing/de-gross tail hedges (all methodology traps, caught and discarded).

## 6. Reproduce

```bash
make volprem     # crypto deep-dive + cross-asset edge map + the diversified 18-sleeve book
```
Or individually: `run_vol_premium.py` (crypto BTC/ETH deep-dive), `run_vol_premium_xasset.py`
(per-class edge map), `run_vol_premium_book.py` (the diversified book — the deployable form).
Artifacts: `reports/volprem/volprem_results.csv`, `reports/volprem/volprem_sensitivity.csv`, `reports/volprem/volprem_execlag.csv`,
`reports/volprem/volprem_marginal.csv`, `reports/volprem/volprem_xasset.csv`, `reports/volprem/volprem_book_sleeves.csv`,
`reports/volprem/volprem_book.parquet`. Implied-vol sources (all free): **Deribit DVOL** (crypto, cached
`data/raw/deribit/`), **Cboe** VIX/VXN/RVX/VXD/VXEFA/VXAPL/VXAZN/VXGOG/VXGS/VXIBM/VXEEM/VXEWZ/VXFXI/
OVX/GVZ/VXSLV/VXTLT/EVZ (equity/FX, cached `data/raw/cboe/`). Realised legs: perp bars + Twelve Data
ETF/spot. Twelve Data Pro does not carry Cboe indices on this plan — hence Cboe's own CSV, not yfinance.
