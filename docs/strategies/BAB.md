# Betting-against-beta / low-volatility — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.76** full / **3.61** OOS). The **crypto beta-neutral top-25** book is the BAB family in it (§5); the equity and FX legs did not survive (§3b).

**Scope.** H1 of the research backlog ([HYPOTHESES.md](../HYPOTHESES.md)): the leverage-constraint
premium (Frazzini-Pedersen 2014) — long low-beta, short high-beta — run through the same funnel as every
other family (vol-target 15%, t+2-style delay, liquidity-aware costs, block-bootstrap MC, shuffled-signal
placebo, purged/embargoed walk-forward OOS, deflated Sharpe, cost sensitivity, correlation to the
deliverable book + lift curve). All numbers net of costs. Tested separately on **crypto** (300-name
point-in-time panel, top-100-liquid each bar, 2020→2026), **US equity** (692-name PIT panel, top-100-liquid,
2010→2026) and **FX** (25 pairs, 2012→2026; §3b). Figure: [reports/figures/bab.png](../../reports/figures/bab.png).
Reproduce: `make bab`.

---

## 0. TL;DR

- **BAB is a real, decorrelated leverage-constraint premium that clears the robust bar in crypto**
  (beta-neutral walk-forward OOS **+0.67** at the a-priori top-100, **+1.52** at the concentrated top-25;
  MC-P5 +0.17 / +0.90), is **signal-gone in US equity** and **dead in FX**. The deployable form — the
  **top-25 liquid-majors** beta-neutral book — is the BAB family in the master book (§5): ≈uncorrelated to
  the other legs (corr ≈ +0.17 to the book), and it also pairs well with carry in isolation (a 2-leg blend
  lifts carry 1.47 → 2.10, §3c). *(The a-priori beta-neutral construction is held fixed through the
  walk-forward; the dollar-neutral variant carries a residual short-market tilt and reads far weaker — §3c.)*
- **The construction is the whole story.** The naive dollar-neutral book is dominated by a large
  *residual market-beta tilt* (−0.61 crypto, −1.23 equity: long low-β minus short high-β nets short the
  market), which in a rising market makes it look dead — and in equity even makes the *wrong* sign
  (long-high-β) "win" (+0.33 vs −0.40). **Beta-neutralising** (Frazzini-Pedersen leg-scaling: de-lever the
  high-β leg so the two legs' betas cancel) removes the tilt and reveals the premium — worth **~+0.6
  Sharpe** in crypto. Dollar-neutral is a v1 trap; beta-neutral is the honest number.
- **Crypto beta-neutral clears the robust bar** (Sharpe +0.77 full-sample, MC [P5 **+0.17**, P50 +0.78],
  maxDD −23%, cost-robust past 8×, beats its shuffled-ranking placebo at the 99th percentile). The honest
  **walk-forward OOS is +0.67** at the a-priori top-100 and **+1.52** at the concentrated top-25 (deflated
  0.90 / 1.00, MC-P5 +0.17 / +0.90) — **both pass OOS > 0.5 and MC-P5 > 0** (§3c). *(A walk-forward that
  lets the pool also pick the inferior **dollar-neutral** construction reads +0.32; holding the a-priori
  beta-neutral construction fixed and selecting only its parameters OOS — the correct test, since the
  construction is theory-fixed, not fitted — gives +0.67.)*
- **The concentrated top-25 is the strong config, not a split-half fluke:** WF-OOS **+1.52** (in-sample
  +1.65, a tiny gap), deflated **1.00**, MC-P5 +0.90, maxDD −14%, **positive 6 of 7 years including 2025
  +1.83** — it did *not* decay the way top-100 did (the top-100 softness is largely breadth-dilution).
- **Robust across universe size** (top-10→200, +0.46→+1.51) **and every timeframe 5m→1d** (+0.75→+0.87,
  bar-frequency-invariant) — the a-priori top-100 is a conservative pick (§3b).
- **A strong carry diversifier** (§3c): corr(BAB, carry) ≈ **0.00**, and a vol-matched risk-parity blend
  lifts carry **1.47 → 1.60** (top-100) / **→ 2.10** (top-25) — the decorrelation math, a genuine second
  source, not re-labelled carry.
- **It is *beta*, not lottery.** The BAB book is ~uncorrelated with a −skew (lottery / MAX) book (crypto
  **0.07**, equity **0.00**) and its alpha **survives controlling for skew** (t = **+2.0** crypto, +1.7
  equity). The plain low-vol proxy is the weaker cousin — it carries the same market tilt and has **no**
  alpha beyond skew (t ≈ 0). So the premium is the leverage-constraint effect, not re-labelled H2.
- **US-equity BAB: the signal is gone.** Beta-neutral reads +0.38 full-sample, but that number is the FP
  construction's **mechanical net-long tilt** — the real beta ranking sits at the **14th percentile** of
  shuffled-ranking books (random rankings do *better*), WF-OOS +0.24, **deflated 0.03**, dead 2023-26. The
  documented post-2010 crowding, sharpened by the placebo to "the beta signal adds no value." The honest
  bet was always crypto; the equity leg confirms it.
- **A data-integrity catch (the same trap that faked overnight's +0.18):** 24 crypto + 12 equity
  split/delisting artifact name-days (a prior close rounding to zero → ∞ return; a mis-adjusted day →
  ±hundreds-of-percent) are winsorised to flat. Left in, **18 outlier days manufacture a spurious 0.97
  correlation** between the beta and lottery books — which collapses to **~0.00** once cleaned. Reported,
  not hidden (§2).

---

## 1. Construction — why *where you put the neutrality* is the whole story

The ranking signal is a trailing **panel beta** βᵢ: a rolling 90-day regression of each name's return on
the equal-weight panel "market", fully vectorised and computable at bar t (`src/sleeves/bab.py`, the same
machinery as `xsect.resid_mom`). Rank low→high; long the low-β quintile, short the high-β quintile,
top-100-liquid each bar, monthly rebalance, delayed t+2 so a bar-t signal never fills at its own close.
Two honest constructions — and the gap between them is the finding:

| construction | legs | residual market beta | what it earns |
|---|---|---|---|
| **dollar-neutral** (v1) | long $1 low-β, short $1 high-β | **−0.6 to −1.2** (net short the market) | the premium **plus** a short-beta tilt — in a bull market the tilt dominates |
| **beta-neutral** (Frazzini-Pedersen) | long low-β, **de-lever** short high-β by β̄_low/β̄_high | **≈ 0** (net-β +0.03 realised) | the leverage premium with the market tilt removed — the honest BAB |

The dollar-neutral book is the literal signal-swap into `xsect.xs_backtest` (signal = −beta). The
beta-neutral book is what that engine cannot express — leg-scaling to net-zero beta — built in
`bab.bab_weights(neutral="beta")` and run through `bab.bab_backtest`, which shares `xs_backtest`'s exact
cost model (commission + half-spread on turnover + √-impact, never flat) so the two are directly
comparable. The measured lift from neutralising the tilt is **+0.18 → +0.77** (crypto) and **−0.40 →
+0.38** (equity): the market tilt is not a detail, it is the difference between "dead" and "real".

## 2. Data integrity — artifacts winsorised (the caught trap)

The survivorship-free broad panels carry split/delisting prints. On the top-100-liquid universe:

- **Equity:** 6 name-days with |return| > 50% **plus 6 ∞** (a prior close rounding to zero) — e.g. a
  single +9,730% print. **Crypto:** 24 name-days with |return| > 100% (liquidation/print artifacts).
- `bab.winsorize_panel` drops ±inf and treats |bar-return| beyond the floor (50% equity, 100% crypto —
  a-priori, above the 99.99th percentile of real moves) as flat, so the print can neither be earned nor
  dominate a signal. The absolute price level is irrelevant here (dollar-neutral book, ADV a separate
  panel); only the cleaned returns matter, and pre-listing NaNs are preserved.
- **Why it matters:** left in, a handful of these name-days land in every book's tail and dominate the
  vol-target — making otherwise-orthogonal books look identical. The beta-book↔skew-book return
  correlation reads a spurious **+0.97** on the raw equity panel and **+0.00** on the cleaned one (the
  signals' own cross-sectional rank-correlation is +0.04). The raw-vs-clean headline delta (crypto
  +0.67 → +0.77, equity +0.27 → +0.38) is reported by the driver, not hidden. This is the overnight
  sleeve's lesson repeated: a plausible cross-sectional "edge" that is actually a few rows of bad data.

## 3. Results — crypto strong-but-fading, equity decayed

Chosen **a-priori** config (90-day beta, quintile tails, monthly rebalance, top-100 liquid) — *not* the
surface peak (crypto's peak is lb90/decile at +0.95; the a-priori lb90/quintile is +0.77):

| metric | **crypto** | **equity** |
|---|---|---|
| dollar-neutral −beta (residual β) | +0.18 (−0.61) | −0.40 (−1.23) |
| dollar-neutral −vol | +0.08 | −0.44 |
| **beta-neutral (FP)** | **+0.77** | **+0.38** |
| MC [P5, P50, P95] | [**+0.17**, +0.78, +1.43] | [+0.02, +0.39, +0.76] |
| max drawdown | −23.2% | −39.9% |
| **walk-forward OOS** (purged, 90-bar embargo) | in-sample +0.95 → **+0.67** | in-sample +0.53 → **+0.24** |
| **deflated Sharpe** | **0.90** (N=9) | **0.03** (N=18) |
| cost 1× / 2× / 3× · break-even-to-0.5 | 0.77 / 0.74 / 0.71 · **> 8×** | 0.38 / 0.36 / 0.33 · ~0.5× |
| beta vs BTC market (robustness) | **+0.51** | — |

- **Surface (§10 sensitivity):** beta-neutral is a **broad positive plateau** — every one of the 9
  (lookback × quintile) cells is positive (crypto +0.42→+0.95, equity +0.28→+0.53), while dollar-neutral
  is negative in most (the tilt drag). A real signal, not a single-cell spike; the a-priori config is
  near the plateau, not its peak.
- **Per-year:** crypto positive 5 of 7 years (+2.2 / +1.6 / −0.3 / +1.1 / +1.0 / **+0.1 / −1.9** in
  2025-26) — the factor **worked strongly 2020-24 and faded in the 2025-26 crypto drawdown**, which is
  exactly what the walk-forward's +0.95 → +0.67 decay captures. Equity positive 11 of 17 years but
  **dead 2023-26** (−1.5 / −0.3 / −0.3 / −0.2) — the documented crowding.
- **Placebo (shuffle the beta ranking, rebuild the beta-neutral book — keeps the FP construction's
  mechanical net-long tilt, randomises *which* names are long/short):** crypto real **+0.77 beats the
  99th percentile** of 100 shuffles (shuffle mean +0.09, p95 +0.60) — the edge is the *signal*, clearing
  the 95th-percentile bar. **Equity is the opposite: real +0.38 sits at the 14th percentile** (shuffle
  mean +0.53) — the beta signal adds **nothing**; the +0.38 is the mechanical tilt earning bull-market
  drift, and *random* rankings do better. So crypto BAB is a real signal; the equity number is a
  construction artifact, not a factor. (The raw dollar-neutral cross-section is weak either way — crypto
  +0.18 at the 71st percentile, equity −0.40 below noise.)

## 3b. Robustness — universe size, timeframe, and FX as a third asset

Both dimensions the main run fixes a-priori (top-100-liquid, 1d) are swept on the crypto beta-neutral (FP)
book, and BAB is also run on **FX** (`scripts/bab/run_bab_robust.py` → `reports/bab/bab_robust.csv`); full-sample
and first/second-half Sharpe (a split-half temporal-stability check, lighter than the §3 purged WF):

| universe (1d) | full | 1st½ | 2nd½ | | timeframe (top-100) | names | full | 1st½ | 2nd½ |
|---|---|---|---|---|---|---|---|---|---|
| top-10 (2/leg) | +1.23 | +1.57 | +0.87 | | 1d | 300 | +0.77 | +1.06 | +0.44 |
| **top-25 (5/leg)** | **+1.51** | +1.61 | **+1.38** | | 4h | 199 | +0.75 | +0.97 | +0.54 |
| top-50 | +1.01 | +1.54 | +0.43 | | 1h | 174 | +0.81 | +0.74 | +0.88 |
| top-100 (a-priori) | +0.77 | +1.06 | +0.44 | | 15m | 149 | +0.87 | +0.64 | +1.09 |
| top-200 | +0.46 | +0.83 | +0.03 | | 5m | 129 | +0.77 | +0.91 | +0.64 |

- **Positive at every setting** — top-10→200 (+0.46 → +1.51) and the entire **5m→1d** grid (+0.75 → +0.87).
  The +0.77 is **not a top-100/1d artifact**; it is a broad plateau, and the a-priori top-100 is a
  *conservative* pick.
- **Strongest concentrated:** the premium *rises* as the universe tightens to the most-liquid majors
  (top-25 +1.51) — exactly where retail leverage/lottery demand is most acute — and dilutes with breadth
  (top-200 +0.46). Caveat: top-25 is 5 names/leg (concentrated idiosyncratic risk), and the concentrated
  book was checked only on this split-half, not the full purged-WF / placebo / MC funnel.
- **Bar-frequency-invariant (5m→1d), the signature of a genuinely slow factor:** full-sample Sharpe is flat
  at +0.75→+0.87 across every timeframe because the signal (90-day beta) and the trade (monthly rebalance)
  are slow — finer bars only re-sample the same monthly-turnover book, they do not add signal (nor, at 5m,
  does the short vol-target window distort it). Intraday universes thin with liquidity (300→129 names) but
  stay positive. The informative axis for "an intraday BAB" would be **rebalance** frequency, not bar
  frequency — not pursued, as the monthly-slow premium is the documented one.
- **The 2025-26 "decay" is partly breadth-dilution, not pure factor death:** the second half is weak on
  top-100 (+0.44) but *holds* on the concentrated (top-25 **+1.38**) and finer-bar (1h +0.88, 15m **+1.09**)
  books — so top-100's softness is largely dilution; the concentrated book's walk-forward clears the bar
  decisively (+1.52, §3c).
- **No intraday BAB — the factor is fundamentally slow (rebalance × beta-lookback grid, crypto 1h).**
  Net Sharpe is monotone in *both* directions: the corner-optimum is the **slowest** cell (90-day beta ×
  monthly rebalance, **+0.81**), and every step toward a shorter beta *or* a faster rebalance lowers it,
  into deep negatives (**−6.25** at 1-day beta × hourly). Faster loses two ways — the cost-check at 90-day
  beta × hourly rebalance reads **+0.56 gross → −0.17 net** (turnover × cost eats **0.73**), *and* that
  +0.56 gross is itself below the monthly number, because a slow beta rebalanced fast just churns on noise.
  So the monthly a-priori is the corner-optimum, not an arbitrary pick, and there is no faster BAB to harvest.
- **FX — dead (a third asset class, not just crypto/equity).** On the 12 USD-major pairs oriented to a
  clean dollar-factor market (2012-2026, 1 bp, no impact), beta-neutral BAB nets **−0.18** (MC-P5 −0.57,
  both halves negative); the raw dollar-neutral −β is +0.10 (noise), and the muddled all-25-pair version is
  −0.17. FX majors are **deeply institutional with no retail-leverage/lottery story** (the crypto driver),
  and the FX cross-sectional premium that *does* exist is **carry** (already in the book), with which BAB
  overlaps (high-yield ≈ high-beta risk-on currencies). So BAB is strongest where leverage demand is acute
  (crypto), signal-gone in equity, and **absent in FX** — a clean three-asset gradient.

## 3c. Deep-dive — the concentrated book through the full funnel, and a carry overlay

`scripts/bab/run_bab_deep.py` (→ `reports/bab/bab_deep_summary.json`) runs the two variants §3b flagged.

**(1) Full funnel on the beta-neutral book — does it clear the robust bar standalone?** The walk-forward
here selects only the free parameters (β-lookback × quintile) *within* the a-priori beta-neutral
construction — the correct test, because the construction is theory-fixed, not fitted (letting the
pool also pick the inferior dollar-neutral construction reads +0.32):

| cut | Sharpe | MC-P5 | **WF-OOS** (in-sample) | deflated (N=9) | placebo | maxDD | per-year positive | robust bar |
|---|---|---|---|---|---|---|---|---|
| a-priori top-100 | +0.77 | +0.17 | **+0.67** (+0.95) | 0.90 | 100th | −23% | 5/7 (dies 2025-26) | **PASS** |
| **concentrated top-25** | +1.51 | **+0.90** | **+1.52** (+1.65) | **1.00** | 100th | **−14%** | **6/7, 2025 +1.83** | **PASS** |

Both clear **OOS > 0.5 and MC-P5 > 0**. The top-25 book is the standout — a tiny in-sample↔OOS gap
(+1.65 → +1.52) is the signature of a real plateau, deflated Sharpe ≈ 1.0 survives the multiple-testing
haircut, drawdown is only −14%, and it stays positive in 2025 (+1.83) where top-100 died — so top-100's
"decay" is largely breadth-dilution, not the factor dying. Caveat: top-25 = 5 names/leg (concentrated
idiosyncratic risk); size it accordingly.

**(2) BAB + carry overlay — a genuine second source.** BAB is near-uncorrelated with the carry sleeve
(`carry_refined`): corr **−0.02** (top-100) / **+0.00** (top-25). Vol-matching both to 15% and blending:

| blend (carry + BAB) | carry-only | +20% BAB | +35% BAB | 50/50 risk-parity |
|---|---|---|---|---|
| with top-100 BAB | +1.47 | +1.62 | +1.67 | +1.60 |
| with **top-25** BAB | +1.47 | +1.79 | +2.00 | **+2.10** |

Because the two legs are ~zero-correlation, the blend Sharpe ≈ (S_carry + S_bab)/√2 — so even the weaker
top-100 BAB *lifts* carry (1.47 → 1.67 at ~35%), and the top-25 book lifts it to **2.10**. This is the
cleanest portfolio statement: BAB is a **decorrelated source that improves the carry sleeve**, independent
of the master book, against which BAB reads corr ≈ 0.20 (< 0.3, still admissible)
but adds little because that book is already very high-Sharpe on this window.

## 3d. Is there ML alpha beyond the linear beta? — no (and why)

`scripts/bab/run_bab_ml.py` asks whether a learned cross-sectional forecaster beats the single trailing beta:
predict each name's forward 21-day return from a factor-feature panel (beta 60/90/120, vol, downside-vol,
skew, momentum, reversal — all computable-at-bar from close/ADV), rank on the **purged/embargoed-CV** OOS
prediction, build the **same FP beta-neutral book**, and compare to the classical +0.77. Feeding −beta
through the identical book path reproduces **+0.80 ≈ +0.77**, so the comparison is apples-to-apples.

| OOS Sharpe (FP beta-neutral, N=12 trials) | beta-only | +risk | +all features |
|---|---|---|---|
| Ridge (linear) | +0.00 | −0.04 | **+0.43** |
| Lasso (sparse-linear) | +0.12 | −0.14 | +0.31 |
| Random Forest | −0.12 | −0.33 | −0.05 |
| HistGradientBoosting | −0.41 | −0.53 | −0.02 |

**No model beats the classical +0.77.** Best is Ridge-on-all-features **+0.43** (deflated 0.47, **MC-P5
−0.19** — not robust); the nonlinear trees overfit to *negative* OOS Sharpe. Two honest takeaways:

- **The edge is in the construction, not in return-prediction.** A linear model given *only beta*, asked
  to forecast forward returns, scores **+0.00** — it cannot rediscover BAB, because the factor pays through
  the **risk channel** (rank by beta, de-lever by beta) rather than beta being a good *return* predictor.
  A forecasting model is structurally the wrong tool for this premium. Adding momentum features lets ML
  reach +0.43 (it picks up a little cross-sectional momentum) but never the direct factor's +0.77.
- **Confirms the repo's prior** (an ML ranker on carry destroyed value; §5's honest finding is ML's role
  here is *risk reduction, not a Sharpe boost*). So for BAB the classical FP construction is the right
  tool; ML adds overfit risk, not alpha. (An ML meta-gate for drawdown control is a separate, untested
  question — but the top-25 book's DD is already only −14%, so the need is small.)

## 4. Is it *beta*, or *lottery*? — the orthogonalisation (H1 vs H2)

High-β names are often high-skew "lottery" names (H2), so the two premia must be disentangled. Books for
−beta (beta-neutral), −vol and −skew, then correlations and regressions:

| | corr(β, vol) | corr(β, skew) | BAB α net of skew (t) | low-vol α net of skew (t) |
|---|---|---|---|---|
| **crypto** | +0.14 | **+0.07** | +13.3%/yr (**t = +2.0**) | +2.4%/yr (t = +0.4) |
| **equity** | +0.16 | **+0.00** | +6.1%/yr (t = +1.7) | −5.7%/yr (t = −1.6) |

- The BAB book is **near-orthogonal to the lottery (−skew) book** in both asset classes, and its alpha
  **survives** regressing out the skew book (crypto t = +2.0, equity t = +1.7). So the premium is the
  **leverage-constraint / beta** effect, **not** re-labelled skewness — in *this* universe H1 and H2 are
  independent sub-books, not the same trade (contra the "beta anomaly = lottery demand" reading, which is
  a *cleaned-data* finding here — before winsorising, the two looked 0.97 identical, §2).
- The **low-vol proxy is the weaker cousin**: it carries the same market tilt as −beta and has **no alpha
  beyond skew** (t ≈ 0 crypto, *negative* in equity). Use trailing beta, not trailing vol.

## 5. Portfolio value — decorrelated, clears the bar, and lifts carry

The crypto beta-neutral book is a genuine decorrelated source: corr **≈ 0.00 to the carry sleeve** and
**≈ 0.20 to the current master book** (< 0.3, still admissible). The cleanest portfolio statement is the
**carry overlay in §3c** — a vol-matched risk-parity blend lifts carry **1.47 → 1.60** (top-100) / **→ 2.10**
(top-25), because two ~zero-correlation legs combine at ≈ (S₁ + S₂)/√2.

**In the master book, BAB is the crypto beta-neutral top-25 leg** (`scripts/run_master_book.py` is the
authoritative assembler; `scripts/bab/run_bab_portfolio.py` is the exploratory view). BAB top-25 is
~uncorrelated with every existing leg (carry +0.05, **x-sect −0.01**, breakout +0.13, volprem +0.11; most
correlated with trend +0.36; **≈ +0.17 to the assembled book**) — a genuinely new source, admitted at
equal risk alongside the other seven:

| family | standalone Sharpe (rescaled-in-book) | corr to book | share of book PnL |
|---|---|---|---|
| volprem (anchor) | 4.57 | — | 52% |
| BAB top-25 | **1.29** (+1.51 standalone top-25 full-sample, §3b) | **+0.17** | **6%** |

- **Decorrelated breadth, not a headline lift.** The book is **volprem-dominated** — volprem's standalone
  4.57 anchors it (removing volprem drops the book from **3.52 to 1.75**), so the equal-risk *average*
  Sharpe is set by volprem and no single modest-Sharpe family moves the headline much. BAB's role in the
  book is a decorrelated, robust source — the same as carry, x-sect, crisis and gmacro (all standalone
  ≈ 0.5–1.3): they broaden the book, they do not inflate its Sharpe.
- **Sized for concentration.** top-25 = 5 names/leg, so it enters at the same family risk cap as the
  others; the a-priori beta-neutral construction is held fixed through the walk-forward rather than
  re-discovered OOS. The top-100 variant is the weaker build (its 2025-26 softness is largely
  breadth-dilution, not the factor dying), so the concentrated top-25 is the one carried.

## 6. Honest verdict & ceiling

- **Reachable here:** a real, decorrelated leverage-constraint premium in **crypto** that **clears the
  robust bar** — beta-neutral WF-OOS **+0.67** (a-priori top-100) / **+1.52** (concentrated top-25), MC-P5
  +0.17 / +0.90, deflated 0.90 / 1.00, and confirmed **beta, not lottery**. Deployable form: the top-25
  liquid-majors book (Sharpe +1.51, DD −14%, positive 6/7 years), which pairs well with carry in isolation
  (a 2-leg vol-matched blend lifts carry 1.47 → 2.10, §3c), sized for its 5-name-per-leg concentration.
  In the master book it is the **crypto beta-neutral top-25 leg** (§5) — a decorrelated, equal-risk source
  that broadens the book to eight structurally distinct premia; like every non-volprem family its role is
  breadth and robustness, not lifting the volprem-anchored headline Sharpe.
- **Binding constraints:** (1) **construction** — the dollar-neutral v1 is a trap (its market tilt hides
  the premium and even flips the apparent sign in equity); the FP beta-neutralisation is mandatory, and a
  walk-forward must hold it fixed rather than re-discover it OOS (re-discovering it reads +0.32). (2) **breadth** — the edge concentrates in the most-liquid majors (top-25 ≫ top-200), so it is
  capacity-limited; and the top-100 book's 2025-26 softness is largely dilution, not the factor dying.
- **What did not work (kept, not hidden):** the dollar-neutral construction (both asset classes), the
  trailing-vol proxy (no alpha beyond skew), the **entire equity leg** — whose +0.38 the shuffled-ranking
  placebo exposes as the construction's mechanical tilt (real ranking at the 14th percentile, below
  random), not a signal — and **FX entirely** (beta-neutral −0.18, MC-P5 −0.57: no retail-leverage story,
  and the FX cross-sectional premium is carry, not BAB). So the honest span is **crypto viable (esp.
  concentrated) / equity signal-gone / FX dead**, plus the methodology (a caught data-integrity artifact,
  the dollar-vs-beta-neutral decomposition, the construction-placebo, and the WF-pool correction above).

## 7. Reproduce

```bash
make bab     # run_bab.py        -> bab_{summary.json,grid.csv,orthogonal.csv,returns.parquet} + figures/bab.png
             # run_bab_robust.py -> bab_robust.csv   (universe × timeframe × rebalance × FX, §3b)
             # run_bab_deep.py   -> bab_deep_summary.json  (top-25 full funnel + carry overlay, §3c)
             # run_bab_ml.py     -> bab_ml.csv  (ML forecaster vs classical beta, purged-CV, §3d)
             # run_bab_portfolio.py -> bab_portfolio_summary.json  (in-book vs standalone, §5)
```

Fixed seed (7) throughout; the winsor floors (50% equity / 100% crypto) and the beta-neutral leg-scaling
are in `src/sleeves/bab.py`. Sources: Frazzini & Pedersen, "Betting Against Beta" (JFE 2014); Bali, Brown,
Murray & Tang, "A Lottery-Demand-Based Explanation of the Beta Anomaly" (JFQA 2017) — the H1↔H2 link this
sleeve tests and, on cleaned data, finds to be *independent* here.
