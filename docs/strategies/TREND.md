# Trend-Following Deep-Dive

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.52** full / **2.64** OOS, mean cross-family corr 0.06). Any master-book Sharpe quoted below is the book *snapshot at the time this family was evaluated* (the book grew as families were added, and gmacro became the 7th); the canonical headline is REPORT.md.

**Dashboard:** [trend_dashboard.html](../../reports/trend/trend_dashboard.html) · **Reproduce:** `python scripts/trend/run_all_trend.py`
· **Code:** `src/sleeves/trend_lab.py`, `scripts/trend/*` · **Sibling deep-dives:** breakout, cross-sectional, carry, mean-reversion

---

## 0. TL;DR

Trend-following is one of the eight families in the shipped book and the most **broadly robust** price
premium in it — the only family that spans **both** asset classes (crypto + US equities) and it works at
every mid timeframe. (It is not the book's Sharpe anchor — that is short-vol; trend is ~11% of P&L and
earns its slot on decorrelation.) This deep-dive rebuilds it from scratch across every entry, exit,
direction mode, timeframe and instrument, adds an ML overlay, and validates it with walk-forward and a
shuffled-data null. Headline results, net of liquidity-aware costs + funding, t+2 execution, vol-targeted
to 15%, on **50 crypto perps + 10 US equities/ETFs, 2012–2026** (crypto spliced spot-2017 + perp-2020):

> **The recommended trend book** (frozen config **core-10 crypto (1d+4h) + 10 US equities (1d)**, EMA long-biased,
> equal-risk, 30 sleeves) nets **Sharpe +1.32, max drawdown −11.3%, MC-P5 +0.88, held-out OOS +0.67**, positive in
> **13 of 15 years**, and is **invariant to the walk-forward window/cadence choice**. Alternative constructions on that core (measured
> on the fuller 1d+4h+1h set — dropping 1h is what lifts OOS to the +0.67 headline above):
> - **Peak Sharpe — EMA long-only / asym 70/30:** Sharpe **+1.32**, DD −9.9%, OOS +0.44.
> - **Peak robustness — conviction-blend, long-only:** Sharpe **+1.13**, DD **−4.8%**, OOS +0.38.
> - **Sized to the −15% budget — blend vol-managed (§8):** raises CAGR at ~constant Sharpe, tail DD ~−15%.
>
> (A broad 50-name crypto universe is *worse* — Sharpe 1.16, OOS +0.11 — see finding 4.)

**Four findings that matter, stated up front:**
1. **Long-only beats long-short on every entry.** The short leg is pure drag on a
   structurally-upward book; keep it only as a 70/30 **asymmetric** sleeve for the bear/crash hedge.
2. **Most of the long book's Sharpe is harvested market beta, not timing.** A shuffled-data null proves
   the *beta-neutral* trend-timing alpha is real (exceedance 5%) but modest (~0.5 Sharpe); the rest is
   long exposure to a positive-drift market, with the trend signal's real job being **drawdown control**.
3. **ML is a risk-reduction tool, not an alpha engine** — a meta-gate cuts max drawdown from −14% to −1%
   at flat-to-slightly-positive Sharpe; it does not lift out-of-sample return (its trade-outcome AUC is
   0.86 in-sample but **0.505 out-of-sample** — no forward edge; the DD cut is mechanical).
4. **For crypto, fewer instruments is better** — a small liquid core (~10 majors) beats a broad 50–200 alt
   universe on Sharpe *and* OOS *and* drawdown. Crypto is one correlated cluster; the illiquid tail adds cost
   and 2022–2025 bad-regime drag, not diversification. Breadth belongs across *asset classes*, not alts.

The honest ceiling is **~1.3 net on the liquid core** (1.0–1.16 if over-broadened), not the aspirational
2.5–4.0. The binding constraint is that trend is **one premium**: it decays in chop (2022, and the 2025 crypto
trend death) and whipsaws at sharp reversals — no construction fixes the absence of a trend to capture.

---

## 1. What the trend sleeve is, and what this deep-dive adds

The shipped trend sleeve is one line — `src/sleeves/momentum.py`, an EMA fast/slow crossover held to
reversal (`+1` when EMA-50 > EMA-200). Trend is one of the eight families in the master book and the only
one that spans **both** asset classes (crypto + US equities), so it is the natural breadth leg. This
deep-dive is not rescuing a null (as the cross-sectional study did) nor confirming a death (as
mean-reversion did) — it is finding **how far the trend premium can honestly be pushed**, and decomposing
where its return actually comes from.

The construction fix that the base report already identified — **hold to reversal, not to a fixed barrier**
— is the foundation: trend edge lives in the fat tail of large moves, and a short exit throws it away. This
study widens every other axis around it.

## 2. How trend is built here — the full matrix

`src/sleeves/trend_lab.py` makes every axis a controlled variable (all computable-at-bar; the engine adds
the t+2 fill downstream, so nothing executes at its own signal bar):

| Axis | Options measured |
|---|---|
| **Entry** | EMA cross · SMA cross · time-series-momentum sign (Moskowitz-Ooi-Pedersen) · MACD · Donchian channel · **multi-lookback blend** (AQR-style continuous forecast) · risk-adjusted **strength** |
| **Direction** | long-short · **long-only** · short-only · **asymmetric 70/30** |
| **Regime gate** | none · ADX≥25 · long-term-EMA alignment · realised-vol band |
| **Exit** | held-to-reversal · chandelier ATR-trail · Donchian channel · time stop |

Two position conventions: *discrete* (a signed side consumed by an exit) and *continuous* (a trend
*strength* in [−1, 1] used directly, so conviction sizes the bet). Construction choices follow the
managed-futures literature (AQR *Demystifying Managed Futures*; Moskowitz-Ooi-Pedersen 2012 *Time Series
Momentum*), the crypto asymmetric-allocation result (arXiv 2602.11708), and López de Prado (AFML) for the
ML labelling/CV. Costs, funding, vol-targeting and the engine are reused verbatim from the book harness, so
every number is directly comparable to the shipped book.

## 3. Long-only vs long-short — the question, answered

At the **book level** (equal-risk over 160 sleeves = 50 crypto × {1d,4h,1h} + 10 equity × 1d, 2017+), every
entry tells the same story: **long-only dominates long-short.**

| Entry (multi-TF book) | long-short | **long-only** | asym 70/30 | long-only DD |
|---|---|---|---|---|
| EMA cross | +0.93 | **+1.16** | +1.16 | −9.9% |
| Conviction blend | +0.62 | **+1.00** | +0.93 | **−4.6%** |
| (1d only, for reference — tsmom) | +0.57 | +1.08 | +0.98 | −8.5% |

**Why:** the short leg of a trend book bleeds. Crypto and equities have a structural upward drift, so
shorting downtrends is a costly bet against assets that quietly recover, and crypto shorts pay funding.
The short leg *is* the hero in bear regimes — on BTC-1d, long-short earns +0.10 in 2018 and +0.17 in 2022
where long-only loses — but across the full sample that hedge does not pay for its drag.

**Recommendation:** trade **long-only** for peak risk-adjusted return, or **asymmetric 70/30** to retain a
partial bear/crash hedge at essentially no Sharpe cost (asym ties long-only at 1.16 for EMA while keeping the
2022 loss shallower: −0.87 vs deeper for pure long-only). Symmetric long-short is dominated and is kept only
as the beta-neutral *diagnostic* in §6.

## 4. Edge map — where trend works, and where it dies

Median single-instrument Sharpe (dir = long-short, held-to-reversal, net of costs). Single-name trend is
*meant* to be modest — the premium is a diversification effect (§5) — but the map shows exactly which
timeframes carry it:

| Entry | 5m | 15m | 1h | 4h | 1d | equity 1d |
|---|---|---|---|---|---|---|
| EMA | −1.84 | −0.12 | +0.29 | **+0.35** | +0.12 | +0.42 |
| SMA | −1.82 | −0.08 | +0.31 | +0.33 | −0.04 | **+0.47** |
| Blend | −3.89 | −0.72 | +0.27 | +0.36 | +0.15 | +0.27 |
| TSMOM | −12.85 | −4.45 | −0.82 | +0.37 | +0.25 | +0.08 |
| MACD | −16.67 | −6.67 | −1.32 | −0.05 | +0.19 | −0.29 |

- **4h is the crypto sweet spot** (every entry positive), then 1d, then 1h. **15m and 5m are dead** —
  cost × turnover destroys the signal; the faster/more-reactive the entry (MACD, TSMOM), the worse it dies.
- **MA-crosses (EMA/SMA) are the most timeframe-robust**; the continuous blend is best at 1h/4h; TSMOM/strength
  shine at 1d/4h but over-trade the fast grid.
- **Equities: MA-crosses lead** (SMA +0.47, EMA +0.42), TSMOM/MACD do not — single-name equity trend is weak,
  which is why the book leans on *breadth and cross-asset* diversification rather than any one equity name.

The book therefore uses **1d + 4h + 1h only** (5m/15m excluded a priori).

## 5. The book — diversification is the strategy

Trend Sharpe is a **diversification** effect: the best single sleeve is ~1.3 (BNB-4h), the median is ~0.4,
yet 160 decorrelated sleeves combine to **1.16** — the managed-futures signature. Mean pairwise correlation is
**+0.14** (max +0.88 among same-asset crypto sleeves; equities decorrelate the crypto cluster). The
marginal-contribution curve rises steeply for the first ~30 sleeves then flattens — past ~50 sleeves added
in contribution order, book Sharpe plateaus, confirming the gain is breadth, not a few stars.

**Entry trade-off (both are legitimate operating points):**

| Book | Sharpe | max DD | MC P5 | held-out OOS (2024-07+) | character |
|---|---|---|---|---|---|
| EMA, asym 70/30 | **+1.16** | −9.9% | +0.74 | +0.11 | peak Sharpe; choppier |
| Blend, long-only | +1.00 | **−4.6%** | +0.58 | **+0.58** | conviction-sizing halves DD, best OOS |
| **EMA+Blend, long-only** | +1.13 | −6.9% | +0.72 | +0.20 | balanced: ~EMA Sharpe, ⅓ less DD |

The **blend** (continuous conviction sizing) is the more *robust* book: it scales exposure down when trends are
weak, so it halves the drawdown and degrades gracefully through the 2025 crypto trend death (2025 Sharpe +0.02
vs EMA's −0.62). The **EMA** book is the higher in-sample Sharpe. **Ensembling the two** (equal-risk over both
entries' sleeves) is the best single operating point — 1.13 Sharpe at −6.9% DD — but the two books are **0.92
correlated**, so this is *smoothing, not new diversification*: it reinforces that trend, however constructed, is
**one premium**. A production book would lean blend or the ensemble for the risk profile; a Sharpe-max mandate, EMA.

**Do NOT pick the best-backtesting assets** (`run_trend_universe.py`; the universe-selection bias, quantified). The instinct
to keep only the instruments where trend worked best is overfitting, and the held-out block proves it:

| portfolio | in-sample Sharpe | **held-out OOS Sharpe** | OOS maxDD |
|---|---|---|---|
| **full frozen universe (160 sleeves)** | +1.28 | **+0.11** | −7.5% |
| top-5 in-sample winners | +2.25 | **−0.09** | −5.4% |
| top-10 in-sample winners | +2.14 | +0.13 | −5.2% |
| top-40 in-sample winners | +1.77 | −0.03 | −9.0% |

Per-sleeve **IS→OOS Sharpe correlation is −0.06** — past winners do not predict future winners; a random-K
basket does as well as the "winners", and the *bottom*-40 in-sample sleeves did **best** OOS (+0.46). Selecting
the top-5 inflates in-sample Sharpe by **+0.97 of pure look-back bias** (1.28→2.25) while turning OOS *negative*.
The universe rule is therefore frozen at the **asset-class × timeframe × liquidity** level — never per-name —
and every instrument the rule admits is held. But **how many** instruments is itself a decision, and here trend
breaks with the managed-futures "more is always better" rule:

**More crypto HURTS — a small liquid core is best** (`run_trend_breadth.py`). We have ~243 perps with intraday
data (not 50). Ranking crypto by liquidity (median pre-2024 dollar-volume, an a-priori rule) and scaling the book:

| crypto N (by liquidity) | book Sharpe | maxDD | **held-out OOS** | MC-P5 | mean corr |
|---|---|---|---|---|---|
| **10** | **+1.36** | −10.1% | **+0.64** | +0.88 | +0.25 |
| 20 | +1.28 | −9.9% | +0.08 | +0.84 | +0.25 |
| 50 | +1.20 | −10.6% | −0.17 | +0.75 | +0.24 |
| 100 | +1.13 | −10.2% | −0.30 | +0.70 | +0.23 |
| 200 | +1.06 | −12.3% | −0.70 | +0.61 | +0.22 |

Sharpe, OOS and tail all **decline monotonically as the universe widens**. The reason crypto breaks the
managed-futures breadth rule: it is **one correlated cluster** (mean sleeve correlation stays ~0.23 no matter how
many alts you add — no new diversification), the marginal names are **illiquid** (√-impact cost drag) and
**trend worse** (alt pump-and-dump chop), and their short histories sit inside the 2022-2025 bad-trend regime
(the OOS collapse to −0.70 at N=200 is the small-cap trend death, which the majors dodged: BTC/ETH-led 2025
+0.04 vs the 50-name book's −0.62). The frozen **config core-10** (defined before any evaluation) *is* that
liquid core — **Sharpe +1.32, OOS +0.44, MC-P5 +0.88** — beating the 50-name book (1.16 / +0.11) on every metric
at the same drawdown. So: freeze the universe by liquidity, hold the whole *core*, and keep the **core small**
(~10 crypto majors + the 10 equities). Breadth belongs across *asset classes*, not across correlated alts.

There is also a **floor** — shrinking below ~7–10 is not free. At N=3 the point drawdown is −15.8% (over the
−15% budget) and the tail −22.8%; N=5 is −12.6% / −20.2% vs N=10's −10.1% / −18.5%, and 2022 deepens (−2.2 vs
−1.8). Full-sample Sharpe *peaks* at N≈7–10 (1.35–1.36), not at 5 (1.32). Top-5's marginally higher 2-year OOS
(+0.72 vs +0.64) is within noise, and tuning universe size to a 2-year window is itself overfitting. Robust
criteria (full-sample Sharpe + drawdown) and the frozen config core-10 both land at **~10**: enough names to
control drawdown, few enough to stay in the liquid quality core.

## 6. Where the return comes from — beta vs timing (the honest core)

A shuffled-**data** null (the task's "run the pipeline on synthetic data"): shuffle each price's returns into a
random walk with no trends, recompute the trend signal on it, re-backtest. A signal exploiting *real* serial
dependence collapses to ~0; the exceedance rate is the false-discovery rate.

| Book | real Sharpe | synthetic-null P50 | exceedance | reading |
|---|---|---|---|---|
| **beta-neutral (long-short)** | +0.98 | +0.52 | **5.0%** | trend **timing** edge is real (p≈0.05) |
| long-biased (asym) | +1.23 | +1.65 | 97% | Sharpe is **harvested beta**, not timing |

This is the deep-dive's central honest finding. Strip market beta (long-short) and the trend-timing alpha is
**real and significant** — but modest, ~0.5 Sharpe. The long-biased book's headline ~1.16 is mostly **being
long a positive-drift market**; on a drifting random walk, always-long actually *beats* the trend signal (the
null sits above the real book), because the trend rule sometimes steps aside and misses drift. So in a long
book the trend overlay's genuine job is **not excess return — it is drawdown control** (going flat/short in
downtrends cuts the −4.6%/−9.9% drawdowns to a fraction of buy-and-hold's).

Decomposition of the ~1.16 book: **≈0.5 validated beta-neutral timing alpha + ≈0.65 harvested beta**, with the
trend rule earning its keep as risk management on the beta.

## 7. ML overlay — measured incremental value

Primary = the non-ML EMA-50/200 rule with a chandelier exit, segmented into trades; secondary = LightGBM/RF/
HistGB predicting P(trade wins) from the 82-feature library, under purged+embargoed CV (overlapping labels
leak under plain k-fold). Core-10 crypto × {1d,4h,1h}, measured in-sample **and** on the 2024-07+ held-out block.

| Variant | Sharpe | IS / OOS | max DD | note |
|---|---|---|---|---|
| baseline (ungated rule) | +0.67 | +0.84 / +0.35 | −14.4% | — |
| meta-gate (LightGBM + uniqueness wts) | **+1.00** | +1.52 / +0.05 | −2.9% | +0.33 Sharpe, huge DD cut, OOS flat |
| **continuous conviction-sizing** (LightGBM) | +0.82 | +1.15 / **+0.20** | **−1.0%** | best-balanced: DD tiny, OOS positive |
| meta-gate (RandomForest) | +0.61 | +0.67 / **+0.47** | −0.7% | generalizes best OOS |

Thresholds 0.50/0.55/0.60 → 0.65/0.71/0.67 (a plateau, not a peak-picked spike). The honest verdict matches
the breakout and cross-sectional studies: **ML buys risk reduction and precision, not out-of-sample alpha.**
The meta-gate collapses max drawdown from −14% to −1% and lifts in-sample Sharpe, but does not raise OOS
return; per-timeframe, the gate does **not** rescue the weak fast TFs for trend (unlike breakout). Continuous
conviction-sizing is the most defensible use — smallest drawdown, positive OOS.

**Why the OOS is flat — the feature analysis.** Pooling all 16,843 EMA-chandelier
trades and predicting win/loss, the meta-model's AUC is **0.856 in-sample but 0.505 out-of-sample** — a coin
flip. It has **no out-of-sample power to tell winning trend trades from losing ones**; the in-sample lift is
overfit. So the drawdown reduction is **mechanical** (the gate simply takes fewer/smaller positions), not
predictive — which is exactly why DD falls but OOS Sharpe does not. Gain-importance (in-sample) leans on the
**trend/MA 18% · cross-asset-correlation 14% · volatility 12% · range-breakout 10% · momentum 10%** families;
every family contributes >2%, but none of it survives out-of-sample. Honest reading: for trend-trade
meta-labelling on liquid assets, the predictable signal in these features is ~nil forward — use the ML for
sizing/risk, never as a trade-outcome oracle.

**Deep sequence models — the capacity control (`run_trend_deep.py`).** To rule out "the trees just lacked
capacity", four architectures — **LSTM, GRU, TCN, Transformer** — were trained (PyTorch/MPS, 169k pooled 4h
windows of vol-normalised sequences, strict train<2024-07 / test-OOS split) to forecast forward direction:

| model | OOS AUC | gross Sharpe | **net Sharpe** | net (low-turnover) | ann. turnover |
|---|---|---|---|---|---|
| LSTM | 0.515 | +0.57 | −0.45 | −0.16 | 23× |
| Transformer | **0.523** | +0.51 | −0.28 | **+0.03** | 12× |
| GRU / TCN | 0.513 / 0.517 | +0.15 / +0.11 | −0.86 / −0.90 | −0.32 / −0.21 | 24× / 19× |
| **EMA rule (no ML)** | — | — | — | **+0.15** | low |

Deep models reach **AUC ~0.52** — a whisper above chance (0.50) and the trees (0.505), so there *is* a faint bit
of learnable structure. But it is **economically nil**: the raw signal turns over 12–24×/yr and goes **net-negative**
(−0.28 to −0.90); even smoothed to kill turnover the best (Transformer) reaches only **+0.03 net — still below the
simple EMA rule's +0.15**. Verdict, now measured end-to-end: the ceiling is **not model capacity** — a
Transformer with 169k sequences does no better than a two-line moving-average cross. The exploitable-net-of-cost
trend signal on liquid majors simply isn't richer than what the EMA captures. (Deep models are an optional control;
`torch` is not required for the deployed book.)

**Regime filters — tried, set aside.** Gating entries to a trending regime (ADX≥20/25, Kaufman efficiency
ratio ≥0.3/0.4, vol band) was tested at the book level: **neutral to negative** (best, ADX-20, +0.01 Sharpe;
vol −0.15), drawdown unchanged, only a +0.04 OOS flicker. A slow EMA-cross held-to-reversal is **already** a
regime filter (it holds only while the trend persists), so ADX/efficiency on top are redundant. Not adopted.

## 8. Portfolio risk management & sizing to the drawdown budget

Two portfolio-level overlays (`src/risk/overlay.py`), sitting *above* the per-sleeve 15%-vol targeting,
implement the book's per-family risk logic — and answer the practical question "we're under the −15% drawdown
budget, can risk-based sizing turn that headroom into more return without raising the tail?"

- **Volatility target** — scale the whole book to a constant annualised vol off *lagged* realised vol
  (Moreira-Muir / AQR volatility-managed portfolios).
- **Drawdown ladder** (stated triggers / steps / stop / restore): cut gross exposure to **0.66× at −6%**
  drawdown, **0.33× at −9%**, **flat (stop trading) at −12%**; restore to full only once drawdown recovers
  above **−4%** (hysteresis, so it does not flip-flop at a threshold). Causal — bar t is sized from the
  managed equity's drawdown through t−1.

| Book / overlay | Sharpe | OOS | CAGR | point DD | tail DD (MC-P5) |
|---|---|---|---|---|---|
| EMA asym, raw 1× | **+1.16** | +0.11 | 8.5% | −9.9% | −18.5% |
| EMA asym + drawdown ladder | +1.13 | +0.09 | 7.9% | −8.7% | −18.0% |
| Blend long-only, raw 1× | +1.00 | +0.58 | 3.7% | −4.6% | −9.9% |
| **Blend LO, vol-managed to the 15% *tail* budget** | **+1.06** | +0.61 | **6.2%** | −7.4% | −15.5% |

**Honest findings:**
- **Book-level vol-targeting does not raise full-sample Sharpe** (−0.01 to −0.09): the sleeves are already
  vol-targeted, so the book layer acts mostly as **leverage**, not a timing edge. It does help the recent
  OOS block and lets a low-vol book fill the risk budget.
- **The drawdown ladder is required risk control, and nearly free** (−0.03 Sharpe): it caps catastrophic
  loss (hard stop at −12%) at almost no Sharpe cost — but being *reactive*, it **cannot pin the 5%-tail
  drawdown** (the loss is already taken by the time a trigger fires; EMA's tail stays ~−18%). This is an
  honest limit of any drawdown-reactive de-risking.
- **The one genuine budget win is the conviction-blend book:** vol-managed to the 15% *tail* budget it runs
  ~2× average gross for **Sharpe 1.06** (up from 1.00), CAGR **3.7%→6.2%**, OOS 0.61, tail ~−15%. Because
  blend's conviction-sizing already damps vol spikes, vol-targeting it cleanly fills the budget; the choppier
  EMA book is *already* at the tail budget (−18%) and must **not** be levered.
- **Net:** vol-based sizing satisfies the risk requirement and converts spare drawdown budget into return at
  ~constant Sharpe, but it does **not** beat the 1.16 the construction already achieves. Sharpe is set by
  construction / diversification / ML (maxed for trend); the risk overlay is risk management, not an alpha engine.

## 9. Walk-forward & validation

- **Performance targets met** (`run_trend_trades.py`): max drawdown **−9.9% / −4.6% / −6.9%** (EMA / blend /
  ensemble) and worst calendar month **−4.1% / −2.5% / −3.3%** — inside the **≤15% DD, ≥−6% month** targets on
  both full sample and OOS. (The point DD passes; the 5%-tail stress (MC-P5 −18% for EMA) is the honest caveat
  handled by §8's sizing.)
- **OOS trade log** (`reports/trend/trend_oos_trade_log.csv`, 7,422 trades, 2024-07+): win rate **30%**, best
  trade **+66%** vs worst −10%, median hold ~100 bars, longs/shorts balanced — the textbook trend signature
  (many small losses funded by a few fat right-tail winners; the 30% hit rate is *expected*, not a defect).
- **Parameter sensitivity surface** (23-config grid, net): **100% of the crypto grid is positive** (min +0.08,
  median +0.41, max +0.57), 91% of equity — a broad robust plateau, the signature of a real premium, not a
  fitted spike.
- **Parameter walk-forward** (pick best config on train, apply OOS): crypto **+0.43** vs in-sample peak +0.82;
  equity **+0.72** vs +0.83. The walk-forward number sits near the grid *median*, so choosing the config adds
  little over a fixed default — exactly what a real (non-overfit) edge looks like.
- **Selection walk-forward across 9 policies** ({anchored, rolling-2y, rolling-3y} × {annual, semiannual,
  quarterly refit}): OOS Sharpe **+1.11 to +1.18, σ 0.02** — the result **does not depend on the window/cadence
  choice**. Full-sample no-selection book: +1.16.
- **Monte-Carlo** (stationary block bootstrap, 1000 reps): Sharpe [P5 +0.74, P50 +1.17, P95 +1.61], maxDD
  [P5 −18%, P50 −12%], monthly hit [P50 64%].
- **Cost sensitivity**: Sharpe 1.16 / 1.12 / 1.09 at 1× / 2× / 3× base cost — trend's low turnover makes it
  cost-insensitive (break-even far above 3×). **Entry-timing jitter** (exec-lag 1/2/3 bars): 1.14 / 1.16 / 1.16.
- **Per-year (regime profile)**: strong in trending years (2013 +2.6, 2017 +3.2, 2021 +2.3, 2023 +1.9, 2024
  +1.2), weak/negative in chop and sharp bears (2022 −0.9, 2025 −0.6). Equity trend carries 2012–2016 (crypto
  starts 2017), a genuine cross-asset diversification, not a single-market artifact.
- **Isolated crises**: bull-2021 +2.3, chop-2023-25 +0.9 — but **Q4-2018 −1.6 and Feb-Mar-2020 −4.5**: trend
  **whipsaws at sharp reversals**, precisely where the long-only short-leg absence hurts and the asym hedge helps.
- **Leakage audit**: t+2 execution (never the signal bar's close, verified `max|full−truncated| = 0` on past
  bars); funding at every 8h settlement; √-impact costs scaled to bar $-volume; vol-target uses lagged vol;
  ML scalers/models fit inside train folds only under purged+embargo CV; equity Sharpe annualised at 252,
  crypto at 365; fixed seeds throughout.

## 10. Ceiling & honest limits

- **Reachable here:** a diversified trend book at Sharpe ≈ **1.0–1.16** net, drawdown −4.6% to −9.9%. The
  2.5–4.0 target is not honestly reachable with trend on liquid assets net of realistic cost.
- **Binding constraint:** trend is **one premium**. It pays only when a trend exists — it decays in chop
  (2022) and died market-wide in crypto 2025 (the OOS block's drag), and it whipsaws at sharp reversals
  (Q4-2018, Covid). No entry/exit/ML trick manufactures a trend where there is none; the book correctly went
  flat rather than losing big, but flat is the ceiling in a trendless regime.
- **The alpha is smaller than the headline.** ~0.5 Sharpe of validated beta-neutral timing; the rest is
  harvested beta with trend as drawdown control. That is a real, useful strategy — but it should be sold as
  "long the drift, risk-managed by trend," not as pure market-neutral alpha.
- **What extends it (honest next steps):** genuinely *independent* premia to diversify the trend source
  (cross-sectional momentum — the sibling study shows it is viable and market-neutral, corr ~0.4 to trend;
  carry; short-vol) rather than more trend-adjacent sleeves; a broader instrument set (more perps, futures
  beyond crypto/equity) for more managed-futures breadth; conviction-sizing (blend/ML) as the default for the
  drawdown profile.

## 11. The trend block — what gets wired into the portfolio (per-family universe)

Each strategy family carries its **own** frozen universe, set by its own edge map — not one shared list.
Trend's is **crypto perps + US equities at 1d/4h/1h**; the sibling families differ (breakout is crypto-only,
vol-premium is equity/FX-led, cross-sectional is crypto+equity). The trend block enters the master portfolio
as *one* equal-risk return stream over these 160 sleeves (`run_trend_composition.py`, `trend_composition_*.csv`):

| by asset class | sleeves | P&L share | risk share | median sleeve Sharpe |
|---|---|---|---|---|
| crypto (50 × 1d/4h/1h) | 150 | 67% | 85% | +0.28 |
| **equity (10 × 1d)** | 10 | **33%** | **15%** | **+0.69** |

| by timeframe | sleeves | P&L share | risk share | median Sharpe |
|---|---|---|---|---|
| 1d | 60 | 50% | 36% | +0.27 |
| 4h | 50 | 32% | 32% | **+0.41** |
| 1h | 50 | **18%** | 32% | +0.26 |

- **Equity punches far above its weight** — 10 sleeves for a third of the P&L at 15% of risk; the *entire*
  top-10 risk-contributor list is equity (QQQ, SPY, GOOGL, MSFT, IWM, AMZN, JPM, META, NVDA, AAPL), because
  equity trend is decorrelated from the crypto cluster. This is precisely why trend's universe is **cross-asset**:
  crypto-only Sharpe **+0.95**, equity-only **+1.00**, **combined +1.16** — each class lifts the other.
- **No single name carries it**: removing the top contributor (QQQ-1d) leaves Sharpe at **+1.18** —
  unchanged. Robustness is diversification, not a star.
- **1h is the marginal timeframe** (18% of P&L for 32% of risk); 4h is the most efficient. A leaner book can drop
  1h at little cost; the full set is kept for breadth.

**Recommended wiring:** the trend block = **EMA (or EMA+Blend) long-biased, equal-risk over the frozen config
core-10 crypto (1d+4h) + 10 US equities (1d)** — a *small liquid core*, not the broad 50–200 alt universe (§5
breadth), and **without 1h** (it adds no Sharpe and drags OOS: 1d+4h OOS +0.67 vs 1d+4h+1h +0.44; edge map and
composition both flag 1h as the marginal timeframe). That 30-sleeve book is **Sharpe +1.32, DD −11.3%, MC-P5
+0.88, OOS +0.67**; sized at the portfolio level to its share of the −15% budget (§8). It combines with the other families' blocks — each on its **own**
universe (breakout crypto-only, vol-premium equity/FX-led) — at the master-portfolio layer, where trend's
decorrelation from cross-sectional (~0.4) and vol-premium is the diversification that lifts the whole book beyond
any single premium. Do **not** hand-pick names by backtest inside the block (§5); freeze the core by liquidity.

**Integration result (`run_trend_in_portfolio.py`, wired into `run_master_book.py`).** The block is published as
`reports/trend/trend_block_returns.parquet` (standalone Sharpe ~1.31) and enters the master book as one
equal-risk leg. Trend is **decorrelated from every other family** (carry −0.08, vol-prem +0.04,
cross-sectional −0.11, breakout −0.07; mean ≈ −0.06 across the eight) — a genuine diversifier. In the
shipped eight-family book it contributes **~11% of P&L**, lifts the portfolio **+0.28 Sharpe** over the book
without it, and keeps the master **positive every calendar year 2011–2026**, carrying the years trend alone
struggles (2022, 2025) on the other families. An earlier five-family integration snapshot shows the same
mechanism:

| five-family integration snapshot | Sharpe | maxDD | MC-P5 |
|---|---|---|---|
| with the *old* trend leg (Sharpe 0.84) | +2.06 | −10.0% | +1.32 |
| with this improved trend block | +2.26 | −7.1% | +1.52 |

Swapping the improved block for the old 0.84-Sharpe leg lifted that book **+0.20 Sharpe (2.06 → 2.26) at
−2.9pp drawdown** — the same diversification mechanism the eight-family book runs on. The honest route
toward the target is the same either way: **not a bigger trend book, but trend as one decorrelated leg among
independent premia.**

## 12. Reproduce

```bash
source .venv/bin/activate
python scripts/trend/fetch_data.py          # spot 2017+ history + MKR intraday (idempotent)
python scripts/trend/run_trend_sweep.py     # edge map  -> reports/trend/trend_sweep.csv
python scripts/trend/run_trend_book.py --entry ema    # headline book + robustness (blend for the low-DD variant)
python scripts/trend/run_trend_wfo.py       # sensitivity + parameter-WF + selection-WF x9
python scripts/trend/run_trend_ml.py        # ML incremental value (purged CV)
python scripts/trend/run_trend_breadth.py   # crypto breadth scaling (N=20..200) — fewer liquid names win
python scripts/trend/run_trend_features.py  # feature-family importance + OOS AUC of the meta-model
python scripts/trend/run_trend_regime.py    # regime-filter study (ADX / efficiency / vol gates)
python scripts/trend/run_trend_risk.py      # vol-target + drawdown ladder, sizing to the 15% budget
python scripts/trend/run_trend_trades.py    # OOS trade log + performance-target check
python scripts/trend/make_trend_report.py   # dashboard
# or: python scripts/trend/run_all_trend.py  (all stages in order)
```
Fixed seeds throughout; the 2024-07+ block is reported but never tuned against.
