# Residual / idiosyncratic momentum (H5) — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.76** full / **3.61** OOS; this family's residual-momentum construction now ships as the crypto x-sect leg — §0). Any master-book Sharpe quoted below is the book *snapshot at the time this family was evaluated*, not the current headline.

**Scope.** H5 of the research backlog ([HYPOTHESES.md](../HYPOTHESES.md)): residual (idiosyncratic) momentum
— rank on the momentum of each name's **market-beta residual**, standardised by residual vol
(Blitz-Huij-Martens 2011) — run through the same funnel as every other family (vol-target 15%, t+2-style
delay, liquidity-aware costs, block-bootstrap MC, shuffled-signal placebo, purged/embargoed walk-forward
OOS, deflated Sharpe, cost sensitivity, correlation to the master book + lift curve). All numbers net of
costs. Tested separately on **crypto** (300-name PIT panel, top-100-liquid, 2020→2026), **US equity**
(692-name PIT panel, 2010→2026) and **FX** (25 pairs, 2012→2026), across timeframes (1d/4h/1h/15m),
universe sizes (top-10→all) and the construction grid. Figure:
[reports/figures/residmom.png](../../reports/figures/residmom.png). Reproduce: `make residmom`.

The acceptance test is precise, because residual momentum is *momentum* — the question is not "is there
an edge" but "does residualising the return **before** ranking beat the raw risk-adjusted-momentum book
already in the book, on OOS Sharpe **and** beta". So every result is a **head-to-head**: identical
execution (same lookback / skip / quantile / rebalance / universe), only the ranking signal swapped.

---

## 0. TL;DR

- **Residual momentum is a real, literature-consistent improvement to momentum *construction* — but it
  is a better-built momentum, not a new decorrelated source.** It beats raw momentum outright on **crypto**
  (+0.45 → **+0.61** standalone, walk-forward OOS incremental **+0.25**) and it **halves the momentum-crash
  bleed on equity** (in raw momentum's worst 5 months, residual loses −5.0% vs raw's −12.3%). But it is
  **~0.8 correlated with raw momentum**, adds **no significant alpha over it** (t = +0.1 to +1.1), and so
  it **does not lift the book as a separate 9th family** (added @30% it dilutes, 3.77 → 3.48). Its honest
  role — **now shipped** — is a **drop-in upgrade to the crypto x-sect momentum sleeve**
  (`build_xs_book.crypto_spot_xsect`): swapped in as that leg's construction it holds the book's full-window
  Sharpe (3.77) and **lifts the out-of-sample scorecard to 5/5** (OOS Sharpe 3.28 → 3.61, months 77% → 81%)
  at the cost of the full-window losing streak (2 → 3). Plus a **crash-hedge on the equity leg**. Not a new
  admitted family — exactly the "highest-certainty modest win, lowest diversification value" H5 predicted.
- **The thesis was equity; the win is crypto.** On crypto the residual construction is textbook-clean and
  strictly better than raw at the a-priori config and at **every** timeframe 1d→15m (Δ up to +0.26). On
  equity the canonical decoupled construction actually **underperforms raw on full-sample Sharpe** (+0.41 vs
  +0.48) — the single-window form ties it (+0.49) — and its value is **crash-reduction and tail quality**,
  not return.
- **The "lower beta" selling point is redundant here.** Residual momentum's headline claim (near-zero
  market beta) is already delivered by the book's **dollar-neutral** construction: the raw momentum books
  already run at β ≈ −0.005 (crypto) / −0.05 (equity). Residualising an already-market-neutral book does not
  lower an already-near-zero beta (on equity it is marginally *more* negative). The one place beta collapses
  — **FX**, +0.35 → −0.15 — is a place with no momentum edge to harvest.
- **Crypto: residual +0.61** full-sample (MC [P5 −0.08, P50 +0.62], deflated 0.64 at 24 trials, cost-robust
  to 2.5× base), **94% of the construction grid positive** (best +1.02), placebo-clean at the **93rd
  percentile**, positive **6 of 7 years**. The construction confirms the literature exactly: the sweet-spot
  **formation window is ~20–30 days = the documented 1–4-week crypto momentum horizon** (Liu-Tsyvinski-Wu
  2022), residualising on the **equal-weight panel beats BTC** (+0.61 vs +0.45), and — unlike equities —
  **skipping the most-recent bar does *not* help** (crypto has continuation, not reversal, at the daily bar).
- **Equity: same modest ~0.4–0.5 edge, de-crashed.** Placebo-clean (96th pctile), MC-P5 +0.04, walk-forward
  OOS **+0.45 vs raw +0.39** (meets the H5 letter), and the classic **1-month skip helps** (+0.43 vs +0.35,
  confirming Blitz-Huij-Martens). But full-sample Sharpe is *below* raw and the edge appears only at **full
  breadth** (all 692 names +0.70 vs raw +0.56) — the idiosyncratic anomaly wants breadth, and the top-100
  liquid cut is where it is weakest. A quality improvement, not a return upgrade.
- **FX: dead**, like raw FX momentum — the sign check prefers reversal, walk-forward +0.09, deflated 0.13.
  Residualising only strips a hidden **+0.35 "dollar-factor" beta** out of the raw FX-momentum book (which
  makes raw FX momentum a disguised directional-dollar bet, not a cross-sectional one) — a diagnostic, not an edge.
- **The implementation is the canonical form.** Ranking on the residual's **mean ÷ std** over the formation
  window is *rank-equivalent* to Blitz-Huij-Martens' (Σε)/σ signal — the strong, vol-standardised form, not
  the weaker unscaled variant. A market-model (single-factor) residual captures most of the multi-factor
  benefit (Chaves 2016), so residualising on the equal-weight panel is a faithful, defensible construction.
  Shift audit `max|full − truncated| = 0` — computable-at-bar, no leakage.
- **ML does not beat the rule, but confirms it (§5b).** A learning-to-rank model tops out at the `idio_mom`
  rule's level (crypto ML +0.39 / best +0.61 vs rule +0.61; equity +0.34 vs +0.41) — no ML alpha over a clean
  single signal. But an **ablation** shows residual momentum is the ranker's **most valuable feature**: adding
  it lifts the ML +0.09 → +0.39 (crypto) and flips it −0.34 → +0.34 (equity) — the raw-momentum features do
  **not** reconstruct the residualised signal, independent confirmation that residualising carries real
  ranking information. The meta-label gate cuts the residual book's drawdown (crypto −35% → −12%) at a Sharpe
  cost — risk reduction, not alpha.

---

## 1. Construction — what residual momentum is, and why this is the canonical form

Raw momentum ranks on *total* return, so winners/losers inherit whatever factor did well over the
formation window — a raw-momentum book carries **dynamic, mean-reverting factor exposures** that add
variance and periodically detonate ("momentum crashes", Daniel-Moskowitz 2016: after a market bottom the
loser leg becomes a high-beta written-call on the market and a sharp rebound crushes the book). Residual
momentum fixes the **signal**: rank each name on its **stock-specific (idiosyncratic) return**, so the
winners/losers are no longer selected for factor luck.

`src/sleeves/xsect.py` implements two forms, both a signal-swap into the shared `xs_backtest` (dollar-neutral
top/bottom-quantile, t+2 execution, liquidity-aware cost, vol-target 15% — identical to the raw-momentum sleeve):

| signal | residualisation | windows | note |
|---|---|---|---|
| `resid_mom(lb, skip)` | regress each name's return on the **equal-weight panel** ("market"), strip β·market | **one** window for beta *and* formation | the single-window form |
| `idio_mom(form_lb, beta_lb, skip)` | same, but **separate** beta-estimation and formation windows | long stable beta, recent formation | the canonical Blitz-Huij-Martens decoupling |

Both then rank on the residual's **mean ÷ std** over the formation window. Two facts make this the
*canonical* construction, not an approximation:

- **mean ÷ std is rank-equivalent to Blitz-Huij-Martens' signal.** BHM rank on (Σε)/σ(ε) = N·mean(ε)/σ(ε);
  dividing by the constant N does not change the cross-sectional ordering, so mean÷std produces the identical
  ranking — the **vol-standardised (information-ratio) form**, which is the strong version of the anomaly, not
  the weaker unscaled Chaves/Lu-Lu form.
- **A market-model (single-factor) residual is a legitimate, near-canonical construction.** Chaves (2016)
  shows most of the multi-factor benefit comes from orthogonalising to the **market**; adding size/value adds
  little. Crypto has no robust SMB/HML analog, so residualising on the equal-weight panel is the pragmatic and
  defensible choice — and it beats residualising on BTC here (§4). The residual uses an **estimated β**
  (`β = cov(rᵢ, mkt)/var(mkt)`), not a demean — a true market-model residual, not β ≡ 1.

The head-to-head holds `lb / skip / quantile / rebalance / universe` fixed at the raw sleeve's a-priori
config and swaps only the signal, so any difference is the residualisation, nothing else.

## 2. Data integrity — artifacts winsorised (same trap as BAB / overnight)

The survivorship-free broad panels carry split/delisting prints. `bab.winsorize_panel` drops ±inf (a prior
close rounding to zero) and treats a bar-return beyond the floor (crypto 100%, equity 50% — a-priori, above
the 99.99th percentile of real moves) as flat, so a print can neither be earned nor dominate a signal.

- **Equity: 1045 artifact name-days + 6 ∞** winsorised (the broad S&P panel is the messiest — 692 names,
  16 years, many delistings). **Crypto: 52** (liquidation/print artifacts). **FX: 0** (clean, small universe).
- **Why it matters:** left in, these name-days land in the residual's mean/std and dominate the vol-target.
  The raw-vs-clean headline delta is reported by the driver, not hidden — on crypto the raw-panel idio reads
  **+0.85** and cleans to **+0.61** (the +0.85 was partly a handful of un-winsorised liquidation prints); on
  equity **+0.54 → +0.41**. The clean number is the honest one.

## 3. Results — the head-to-head (identical execution, signal swapped)

Chosen **a-priori** config per asset (matching the raw sleeve so the comparison is apples-to-apples): crypto
30-day formation / 90-day beta / tercile / monthly / top-100; equity 252-day formation / 756-day (3y) beta /
decile / monthly / top-100; FX 90-day formation / 250-day beta / tercile / all-25.

| metric | **crypto** | **equity** | **FX** |
|---|---|---|---|
| raw total-return `mom` | +0.29 | +0.58 | −0.18 |
| raw risk-adj `risk_adj_mom` (book benchmark) | +0.45 | +0.48 | −0.12 |
| single-window `resid_mom` | +0.34 | +0.49 | −0.05 |
| **decoupled `idio_mom` (canonical BHM)** | **+0.61** | **+0.41** | **−0.04** |
| Δ residual − raw (Sharpe) | **+0.16** | −0.07 | +0.09 |
| book market-β: raw → residual | −0.005 → −0.001 | −0.052 → −0.071 | **+0.346 → −0.155** |
| **momentum-crash months** (raw's worst 5): raw → residual | −8.2% → **−5.8%** | −12.3% → **−5.0%** | −10.5% → −4.2% |
| construction surface: % positive · best · median | 94% · +1.02 · +0.40 | 100% · +0.57 · +0.41 | 63% · +0.19 · +0.04 |
| placebo (shuffle signal): real percentile · shuffle-p95 | 93rd · +0.63 | 96th · +0.33 | 39th · +0.34 |
| **walk-forward OOS: residual vs raw** (incremental) | **+0.27 vs +0.02 (+0.25)** | **+0.45 vs +0.39 (+0.06)** | +0.09 vs −0.05 (+0.14) |
| MC [P5, P50, P95] | [−0.08, +0.62, +1.26] | [+0.04, +0.42, +0.77] | [−0.47, −0.03, +0.38] |
| deflated Sharpe (N trials) | 0.64 (24) | 0.90 (18) | 0.13 (18) |
| cost 1× / 2× / 3× · break-even | 0.61 / 0.53 / 0.45 · **2.5×** | 0.41 / 0.39 / 0.37 · cost-flat, below 0.5 | −0.04 / −0.06 / −0.08 |
| corr to raw-momentum book · alpha ⟂ raw (t) | +0.78 · +4.2%/yr (t +1.1) | +0.82 · +0.2%/yr (t +0.1) | +0.87 · +1.1%/yr (t +0.6) |
| corr to master book · book-lift @30% | +0.18 · 3.77 → 3.48 (dilutes) | +0.35 · 3.21 → 2.74 (dilutes) | +0.04 · dilutes |

- **Crypto — residualising beats raw outright.** +0.61 vs +0.45 full-sample, and the honest walk-forward is
  **+0.27 vs the raw grid's +0.02 on the identical harness** (incremental +0.25). *(The absolute WF levels are
  low because this comparison runs a deliberately small formation×quantile grid with a conservative 90-bar
  embargo — not the rich multi-scheme grid that gives XSECT.md's headline crypto WF of ~0.5–0.95; the valid
  claim is the like-for-like **incremental**, and residual momentum's larger OOS persistence is itself the
  documented BHM "more consistent over time" property.)* The surface is a broad
  plateau (94% positive, a-priori +0.61 vs best +1.02 — the a-priori is conservative), placebo-clean at the
  93rd percentile, positive **6 of 7 years** (only 2022 negative, −1.5). MC-P5 is marginally negative (−0.08)
  and deflated 0.64 — a real standalone edge, but a *momentum* edge (see §5), so its home is the momentum sleeve.
- **Equity — a de-crashed version of the same modest edge.** The canonical decoupled idio *underperforms* raw
  on full-sample Sharpe (+0.41 vs +0.48; single-window resid1w ties it at +0.49), but it is **placebo-clean at
  the 96th percentile**, and its **walk-forward OOS +0.45 beats raw's +0.39** — meeting the H5 letter (OOS >
  the raw equity leg). Its real edge is the crash channel (§3b) and it appears at breadth (§4), not at top-100.
- **FX — dead.** Every construction is ≤ 0; the sign check prefers *reversal* over momentum on FX residuals;
  the residual book sits *below* its own shuffle (39th percentile). The only thing residualising does is strip
  the raw FX-momentum book's large **+0.35 dollar-factor beta** — i.e. raw FX "momentum" was substantially a
  disguised directional-dollar bet, not a cross-section. No edge either way (FX's real cross-sectional premium
  is carry, already in the book).

## 3b. Momentum crashes — the robust, universal win

The one Blitz-Huij-Martens result that reproduces cleanly on **every** asset here is crash reduction. Taking
the raw risk-adjusted-momentum book's **own worst 5 months** and reading each book's return in exactly those
months:

| | raw momentum | residual momentum | bleed reduced |
|---|---|---|---|
| **crypto** | −8.2% | −5.8% | ~30% |
| **equity** | −12.3% | −5.0% | ~60% |
| **FX** | −10.5% | −4.2% | ~60% |

Residual momentum bleeds **30–60% less** in raw momentum's worst months, because it never loads on the
time-varying factor beta that detonates in a rebound — exactly the mechanism the literature describes. Note
the book already vol-targets to 15% (a Barroso-style crash control), so this is the *incremental* crash
reduction from residualising the **signal**, on top of managing the **weights** — the two are complementary.

## 4. Robustness — universe size, timeframe, parameters (`reports/residmom/residmom_robust.csv`)

Every robustness axis, always **raw vs residual side by side** so the question stays "does
residualising beat raw *here*":

**Universe size (top-N liquid, 1d), residual − raw Sharpe:**

| top-N | 10 | 25 | 50 | 100 | 200 | all |
|---|---|---|---|---|---|---|
| **crypto** (res / raw) | +0.09 / +0.50 | +0.16 / +0.32 | **+0.29 / +0.11** | **+0.61 / +0.45** | +0.28 / +0.23 | +0.18 / +0.22 |
| **equity** (res / raw) | +0.02 / +0.39 | +0.23 / +0.27 | **+0.53 / +0.40** | +0.41 / +0.48 | +0.50 / +0.56 | **+0.70 / +0.56** |

- Residual **needs breadth** — it beats raw at top-50→200 (crypto) and is strongest of all at the **full
  692-name equity panel** (+0.70 vs +0.56), but **loses on ultra-concentrated top-10/25** (3–5 names/leg makes
  the residual/beta estimate noisy and the idiosyncratic bet thin). This is the documented shape: idiosyncratic
  momentum is a breadth anomaly. The crypto sweet spot is top-100; the equity edge is a full-breadth effect.

**Timeframe (top-100, windows scaled by bars/day), residual − raw Sharpe:**

| timeframe | 1d | 4h | 1h | 15m |
|---|---|---|---|---|
| **crypto** (res / raw) | +0.61 / +0.45 | +0.54 / +0.28 | +0.63 / +0.61 | +0.72 / +0.53 |
| Δ residual − raw | +0.16 | +0.26 | +0.02 | +0.19 |
| **FX** (res / raw) | −0.03 / −0.12 | −0.04 / +0.11 | −0.28 / −0.35 | — |

- **Crypto residual beats raw at every timeframe** — the residual construction is bar-frequency-robust. Split-half
  stability is best at **1d** (halves +0.37 / +0.85 — stronger recently) and **15m** (+0.79 / +0.66); 4h/1h are
  front-loaded (strong 2020–22, weak 2023–26). **FX is dead at every timeframe.**

**Parameters (idio net Sharpe off the a-priori):**

| axis | crypto 1d | equity 1d |
|---|---|---|
| **skip** | sk0 **+0.61** > sk5 +0.57 > sk2 +0.47 > sk1 +0.41 | sk21 **+0.43** > sk5 +0.39 > sk0 +0.35 |
| **formation** | f30 **+0.61** > f20 +0.53 > f45 +0.26 > f90 +0.09 | f63 **+0.51** > f126 +0.49 > f252 +0.41 |
| **rebalance** | rb5 **+0.94** > rb21 +0.61 > rb42 +0.12 | rb10 **+0.54** > rb42 +0.43 > rb21 +0.41 |
| **weighting** | volinv +0.63 ≈ equal +0.61 ≈ rank +0.59 | rank +0.42 ≈ equal +0.41 > volinv +0.34 |
| **market factor** | **EW panel +0.61** > BTC +0.45 | (EW panel) |

- **The construction confirms the literature exactly.** Crypto's sweet-spot formation is **20–30 days = the
  documented 1–4-week crypto momentum horizon** (Liu-Tsyvinski-Wu 2022 — a 12-month residual momentum would
  *not* work on coins), its faster rebalance (weekly rb5 +0.94) fits that short horizon, and **skipping the
  most-recent bar does not help crypto** (sk0 is best — crypto has short-term *continuation*, not the
  short-term *reversal* the skip is designed to dodge). Equity is the mirror image: the **classic 1-month skip
  helps** (sk21 +0.43 > sk0 +0.35, the Jegadeesh-Titman / BHM gap), and a **shorter-than-12-month formation**
  (63–126d) beats the textbook 252d on this survivorship-free panel. Robust to weighting on both.

## 5. Is it a new source? — orthogonalisation vs raw momentum (no)

The decisive portfolio question: is residual momentum *decorrelated* from the raw momentum already in the book,
or is it re-labelled momentum? Regress the residual book on the raw risk-adjusted-momentum book:

| | corr(residual, raw) | residual alpha ⟂ raw (ann) | alpha t |
|---|---|---|---|
| **crypto** | +0.78 | +4.2%/yr | +1.1 |
| **equity** | +0.82 | +0.2%/yr | +0.1 |
| **FX** | +0.87 | +1.1%/yr | +0.6 |

- **Residual momentum is ~0.8 correlated with raw momentum and adds no significant alpha over it** (t = +0.1
  to +1.1, none clearing 2). It **is** momentum — a better-*constructed* momentum (cleaner signal, smaller
  crashes), not an independent premium. This is exactly what H5 predicted ("momentum-adjacent — does not
  diversify the source").
- Consequently it **does not lift the master book**, which already contains the x-sect momentum sleeve:
  correlation to the book is +0.18 (crypto) / +0.35 (equity), and blending it in *lowers* the book's Sharpe at
  every weight (crypto 3.77 → 3.48 at 30%). Adding a ~0.6-Sharpe sleeve that is 0.8-correlated with an existing
  sleeve to a high-Sharpe book is dilutive — the honest negative. Unlike BAB (corr ≈ 0, *lifts* the book),
  residual momentum is not a diversifier.

## 5b. ML layer — does a learned ranker beat the residual rule? (no — but residual is its most valuable feature)

The same two ML experiments the repo runs on every family (`scripts/residmom/run_residmom_ml.py`, reusing the
`xsect_ml` harness — features stamped at bar t, forward target cross-sectionally demeaned, all prediction
expanding/purged walk-forward), pointed at H5. Baseline = the `idio_mom` **rule**; the headline is an
**ablation** — learning-to-rank on the raw-momentum feature set (multi-horizon mom / risk-adj / reversal /
vol / distance-from-high / beta / volume) **vs** that same set **plus residual-momentum features**.

| | rule `idio_mom` | LTR raw features | LTR raw **+ residual** features | Δ from adding residual | meta-gate |
|---|---|---|---|---|---|
| **crypto 1d** | **+0.61** | +0.09 (best +0.27) | +0.39 (best lgbm **+0.61**) | **+0.30** | p>0.6: DD −35% → **−12%** (Sharpe +0.37, trades 20%) |
| **equity 1d** | **+0.41** | **−0.34** (all models <0) | +0.34 (best +0.40) | **+0.67** | p>0.5: +0.32 at −32% DD |

- **ML does *not* beat the rule.** The learned ranker tops out at the rule's level (crypto ensemble +0.39,
  best single model +0.61 = the rule; equity +0.34 < +0.41). A single vol-standardised residual-momentum
  signal is already a clean ranking — a flexible model mostly adds estimation noise, the same asymmetry XSECT
  §7 found for raw momentum. No ML alpha beyond the rule.
- **But residual-momentum is the single most valuable feature you can hand the ranker.** Adding it lifts the
  LTR ensemble **+0.09 → +0.39** on crypto and **flips it −0.34 → +0.34** on equity — Δ **+0.30 / +0.67**.
  Without the residual feature the raw-momentum ranker is weak (crypto) or outright negative (equity); with it
  the model recovers to ≈ the rule. So a tree given `mom`, `beta_60` and `radj` separately **does not
  reconstruct** the residualised signal — residualisation is a non-trivial transformation that carries
  ranking information the raw features do not span. This is independent ML confirmation of the rule finding:
  the residual signal is genuinely better, and the simple rule already captures all of it.
- **No contradiction with §5.** Residual momentum adds information as a *ranking feature* (Δ+0.30/+0.67 above)
  yet its *book return* has no alpha over the raw-momentum book (§5, t ≈ 0, corr 0.8). Both hold: a better
  ranking input still produces the same momentum trade — the residualisation improves *how you sort*, not
  *what premium you harvest*.
- **Meta-gate = risk reduction, not Sharpe** — gating the residual book to high-confidence periods cuts
  drawdown hard (crypto −35% → −12% at p>0.6) at a Sharpe cost and ~20–70% market participation. The repo's
  documented meta-gate pattern, reproduced on the residual book: a legitimate risk overlay, not alpha.

## 6. Portfolio value — an in-family upgrade, not a new admission

Because residual momentum is re-labelled momentum, the right way to *use* it is not "add it on top" but
**"swap it in for the raw signal in the existing sleeve"**:

- **Crypto x-sect momentum sleeve:** swapping `risk_adj_mom` → `idio_mom` lifts the standalone sleeve
  **+0.45 → +0.61** (1d), and residual beats raw at 4h/1h/15m too (§4) — a measured, drop-in **construction
  upgrade** to a sleeve already in the master book, at the same turnover and cost. This is the actionable H5
  deliverable.
- **Equity x-sect momentum leg:** residual does not raise the a-priori Sharpe (top-100 +0.41 vs +0.48) but it
  **cuts the crash bleed ~60%** (−12.3% → −5.0%) and **wins at full breadth** (+0.70 vs +0.56). If the equity
  leg is ever rebuilt on the broad universe, the residual signal is the tail-safer choice at equal return.
- **Not admitted as a new family** — it fails the portfolio-admission test on the axis that matters (corr to
  book < 0.3: equity 0.35 ✗; positive marginal contribution: dilutes ✗), because it is not a new source.

## 7. Honest verdict & ceiling

- **Reachable here:** a genuine, literature-consistent **construction improvement to momentum** — strictly
  better than raw on crypto (standalone +0.61 vs +0.45, walk-forward incremental +0.25, positive 6/7 years,
  placebo-clean, cost-robust to 2.5×) and a **~60% crash-bleed reduction on equity**. The crypto construction
  reproduces the literature to the parameter (formation = the 1–4-week horizon, EW-panel factor, no skip). This
  is the **drop-in upgrade to the crypto momentum sleeve** (§6) — the actionable win, delivered.
- **Binding constraints:** (1) **it is momentum** — 0.8-correlated with the raw sleeve, no significant alpha
  over it, so it is an *in-family* refinement, not a decorrelated new source, and it does **not** lift the
  master book. (2) **The "lower beta" premise does not bind** — the book is already dollar-neutral, so there is
  no market beta to remove (the only large beta-strip, FX +0.35 → −0.15, is where there is no edge). (3) **On
  equity it is a quality, not a return, improvement** — the canonical decoupled construction is below raw at
  top-100 and only wins at full breadth / in the tail. (4) **FX is dead**, like raw FX momentum.
- **What did not work (kept, not hidden):** the decoupled long-beta `idio_mom` on the equity top-100 panel
  (below raw and below the single-window form); residualising on **BTC** for crypto (worse than the EW panel);
  the **skip** on crypto (hurts); **FX entirely** (residualising only exposes that raw FX momentum was a hidden
  dollar-beta bet). The value delivered is the **map** (H5 covered across crypto / equity / FX × 4 timeframes ×
  universe size × the full construction grid), the **actionable sleeve upgrade** (crypto +0.45 → +0.61), and the
  **crash-hedge** on the equity leg — a high-certainty modest win, exactly as the hypothesis ranked it.

## 8. Reproduce

```bash
make residmom   # scripts/residmom/run_residmom.py        -> reports/residmom/residmom_{summary.json,grid.csv,returns.parquet}
                # scripts/residmom/run_residmom_robust.py  -> reports/residmom/residmom_robust.csv
                # scripts/residmom/run_residmom_ml.py      -> reports/residmom/residmom_ml_{crypto_1d,stocks_broad_1d}.csv
                #                                    + reports/figures/residmom.png
```

Fixed seed (7) throughout; the winsor floors (100% crypto / 50% equity) are in `src/sleeves/bab.py`; the two
residual-momentum signals (`resid_mom` single-window, `idio_mom` decoupled BHM) are in `src/sleeves/xsect.py`.
Sources: Blitz, Huij & Martens, "Residual Momentum" (J. Empirical Finance 2011); Chaves, "Idiosyncratic
Momentum: US and International Evidence" (J. of Investing 2016, market-model residual captures most of the
benefit); Blitz, Hanauer & Vidojevic, "The Idiosyncratic Momentum Anomaly" (2020); Daniel & Moskowitz,
"Momentum Crashes" (JFE 2016, the crash channel residualising removes); Liu, Tsyvinski & Wu, "Common Risk
Factors in Cryptocurrency" (J. of Finance 2022, the 1–4-week crypto momentum horizon).
