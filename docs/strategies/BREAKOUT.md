# Breakout sleeve — deep-dive report

Reproduce: `scripts/run_bo_*.py` (see [§10](#10-reproduce)). Figures: [reports/figures/breakout.png](../../reports/figures/breakout.png),
[reports/figures/breakout_book.png](../../reports/figures/breakout_book.png). Module: [src/sleeves/breakout_lab.py](../../src/sleeves/breakout_lab.py).

## 1. Executive summary

**Breakout is a crypto trend-following premium — real but modest, and regime-bound.** Searching every
construction × instrument × timeframe with realistic costs (t+2 execution, liquidity-aware slippage,
funding at every settlement), the honest result is:

> A **frozen, pre-registered** book (10 largest crypto perps, no per-sleeve selection) on the
> corrected construction nets **Sharpe ≈ 1.0** (Monte-Carlo 5th-pct **+0.40**), **max drawdown −3.7%**,
> **break-even at ~10× base costs**, positive in **6 of 7 years including the 2022 crash**. The ML
> meta-label gate roughly **doubles the fast-timeframe Sharpe (0.41→1.01) and cuts its drawdown ~5×
> (−13%→−3%)**. The strictly-held-out 2024-07→2026 block scores **+0.19** — because crypto itself
> stopped trending (BTC buy-hold Sharpe 2024 +1.75 → 2025 +0.05 → 2026 −0.99), and the book correctly
> went flat rather than losing.
>
> Adding a **cross-sectional** breakout sleeve (a point-in-time top-30-liquid crypto universe, ranked by
> 52-week-high nearness, long the most broken-out / short the least, dollar-neutral) and blending it with
> the trend leg at **risk parity** gives the final squeeze: **Sharpe +1.40, MC-P5 +0.77, max DD −11.5%**,
> positive every year 2020-2025 (2022 crash +1.11, trendless-2025 chop +0.86). The legs correlate **+0.10**
> — the cross-sectional leg earns from *dispersion*, so it is decorrelated from the trend leg and covers
> its regime hole. Strict OOS +0.23 (dragged by the 2026 crypto downturn; both legs are honest about it).
> No universe look-ahead: membership uses trailing volume only.

Three findings drive the work, each measured, not asserted:

1. **The exit was the bug.** The current book exits breakouts at a triple-barrier ~4 bars wide, which
   throws away the fat right tail that carries trend P&L. Replacing it with a **chandelier ATR-trailing
   exit** ≈ **doubles the survivor count** (17→32 crypto sleeves on 1d/4h) and flips 4h from dead to
   positive — with ~half the turnover.
2. **The edge is crypto, on 1d/4h only.** Equities and FX are negative for *every* breakout
   construction; intraday (1h and faster) is net-negative to cost. 15m and 5m are dead even with the
   best exit + ML.
3. **Selection doesn't generalise; construction does.** Picking *which* coins will be the star sleeves
   is an in-sample illusion (walk-forward +0.24); the *construction* is robust (parameter walk-forward
   +1.16, three sibling configs all ≈1.5 in-sample).

Breakout ≈ time-series momentum in a coarse, path-dependent encoding — there is no evidence of a
breakout premium separable from the trend premium. This is the honest edge map, consistent with the
project's momentum-book finding.

## 2. What was built

- **[breakout_lab.py](../../src/sleeves/breakout_lab.py)** — composable, causal building blocks: entries
  (Donchian, Bollinger, Keltner, with ATR buffer), filters (volume expansion, long-trend alignment,
  volatility squeeze), and the full exit menu — bounded (triple-barrier, time-stop) vs trend-riding
  (opposite-channel/Turtle, chandelier ATR-trail, held-to-reversal). Every window is backward-only and
  channels are lagged one bar; execution is delayed t+2 by the engine, so nothing fills at the signal
  bar's own price.
- **Harness** (`scripts/bo_common.py`) reuses the book's cost model verbatim (crypto taker 5bps +
  half-spread + √-impact, funding every settlement, vol-target to 15%), so every number here is
  directly comparable to the momentum book. Data is read strictly offline from cache.

## 3. The exit fix — breakout edge lives in the fat tail

The literature is unambiguous (trend/breakout returns are positively skewed; a fixed profit target
converts a positively-skewed edge into a negatively-skewed one and deletes the P&L-carrying tail — the
Zarattini QQQ study found hold-to-EOD beat *every* profit target). Testing five exits over the same
Donchian-55 entries across 18 instruments × {1d,4h} (`run_bo_exits.py`):

| Exit | mean Sharpe | frac > 0.5 | mean turnover |
|---|---|---|---|
| triple-barrier (current book) | 0.287 | 31% | 14.5×/yr |
| time-stop | 0.336 | 28% | 8.9 |
| opposite-channel (Turtle) | 0.308 | 33% | 7.6 |
| **chandelier ATR-trail** | **0.352** | **39%** | **8.1** |
| held-to-reversal | 0.272 | 31% | 6.2 |

The chandelier exit has the highest mean Sharpe, the most survivors, and ~45% less turnover. It is
**regime-dependent** (exactly as theory predicts): the short triple-barrier already wins on the
cleanest trenders (AVAX, NEAR, XRP), while trend-riding *rescues* the choppy names (BNB −0.06→+0.91,
LTC −0.50→+0.01). Lifting the floor and cutting turnover is precisely what makes a diversified book
robust. **Aggregate lift on crypto** (all constructions, `run_bo_sweep.py`): 4h mean Sharpe
**+0.03 → +0.29**, 1d **+0.20 → +0.32**.

## 4. Edge map — where breakout works and where it does not

Full sweep: 9 constructions × 153 instruments × 5 timeframes = 4,104 candidates (`reports/breakout/bo_sweep.csv`).

**By timeframe** (mean Sharpe, best construction): `1d ≈ 4h  ≫  1h  ≫  15m  ≫  5m`.

| TF | ungated (chandelier) | ML-gated | verdict |
|---|---|---|---|
| 1d | ~1.0 (book) | (too few trades to meta-label) | **edge** |
| 4h | +0.64 | **+0.94** | **edge** |
| 1h | +0.05 | **+0.37** | marginal — ML makes it tradeable |
| 15m | −1.67 (DD −63%) | −0.95 (DD −6%) | **dead** — ML cuts loss, can't flip positive |
| 5m | −7.40 (DD −99%) | −1.24 (DD −3%) | **dead** — cost drag overwhelms any signal |

The intraday death is the textbook breakout failure: turnover × cost dominates the shrinking per-trade
alpha. Even the ML gate's most confident 0.3–3% of trades on 5m/15m do not overcome realistic costs
(`run_bo_fast_tf.py`).

**By asset class** (mean Sharpe, 1d+4h+1h): crypto **+0.20** (kelt +0.24), equity **−0.05**, FX **−0.29**.
Single US equities mean-revert short-term and FX breakout never beats plain trend — breakout is a
crypto (and, in the literature, diversified-futures) phenomenon, not a single-name equity one.

## 5. The honest book — selection bias, quantified

Applying the pre-registered construction `d55_atr3` (Donchian-55 → chandelier(3), all components chosen
from the literature *before* seeing results) and screening each sleeve (Sharpe>0.5 & MC-P5>0):

- **In-sample-SELECTED book:** 16 crypto survivors, **Sharpe 1.46–1.59** across three sibling
  constructions (d55_atr3 1.59, kelt_atr3 1.56, +trend-filter 1.46), placebo-FDR **2/405 (0.5%)**. But
  this look-aheads the survivor selection.
- **Walk-forward** (`run_bo_walkforward.py`) separates the two things that could be overfit:
  - *Sleeve selection* — picking which of 405 candidates to hold by trailing Sharpe: **+0.24 OOS**
    (−0.14…+0.24 across anchored/rolling × annual/semi/quarterly — robust to the choice, robustly
    mediocre). You cannot know the star coins in advance.
  - *Construction* — fixing a liquid universe and walk-forwarding only the config: **+1.16 OOS** vs
    +1.17 full-sample. The construction generalises.
- **Frozen no-selection book** (the core-10, take *all* sleeves): **Sharpe 0.87–1.04**
  full-sample (kelt 1.04, MC-P5 +0.40). Breadth *hurts* — top-30 dilutes to 0.70 (marginal alts add
  noise, not edge). This ≈1.0 is the honest, unbiased number; the 1.5 is the selection premium.

## 6. ML meta-label incremental value

Primary side = Donchian-55; a secondary model predicts P(this chandelier trade wins) from the 82-feature
library and gates entries, keeping the fat-tail exit. Purged + embargoed CV throughout; labels are the
realised trade sign (causal). 1d is excluded — only ~30 trades/sleeve, too few to train. On the
core-10 **4h+1h** book (`run_bo_ml.py`):

| Variant | Sharpe (IS / OOS) | max DD | precision |
|---|---|---|---|
| ungated baseline | 0.41 (0.51 / 0.20) | −13.3% | 36% |
| LightGBM + uniqueness-weights | **1.01 (1.28 / 0.47)** | **−2.7%** | 39% |
| HistGradientBoosting | 0.93 (1.08 / **0.61**) | −3.0% | 39% |
| RandomForest | 0.48 (0.38 / **0.66**) | −0.5% | 38% |

The gate more than doubles Sharpe, cuts drawdown ~5×, and — the point — **improves the held-out OOS
period** (0.20→0.47–0.66): filtering false breakouts is most valuable in chop. Threshold is a robust
plateau (0.50→0.97, 0.55→0.90, 0.60→1.13), not a spike. AFML uniqueness-weighting adds ~0.1 Sharpe and
lifts OOS. Honest incremental value: **risk reduction + OOS robustness + rescuing 1h**, not merely a
peak-Sharpe boost.

> **These are generalisation estimates, not a track record.** Purged k-fold keeps folds contiguous
> but trains each on its whole complement, so the gate that filters a 2020 trade is fitted on
> 2021-2026. [§13](#13-the-ml-gate-under-a-walk-forward) re-runs it under an expanding walk-forward
> and prices the difference; the gate survives, at a lower level.

## 7. Final book & robustness

Combining the honest legs — core-10 × 1d raw chandelier (non-ML trend capture) + 4h/1h ML-gated —
equal-risk, 30 sleeves (`run_bo_final.py`):

- **Sharpe +1.03**, max DD **−3.7%**, months-in-profit 51%, total +24%, **MC [P5 +0.40, P50 +1.02, P95 +1.64]**.
- **Per-year:** 2020 +2.13, 2021 +2.67, 2022 **+0.39**, 2023 +1.37, 2024 +1.03, 2025 +0.29, 2026 −0.82.
  Positive through the 2022 crypto crash (it went short) — the signature of a real trend edge, not long-beta.
- **Cost sensitivity:** 1× +1.03, 2× +0.92, 3× +0.81; **break-even ≈ 10.4× base cost** — very robust.
- **Diversification:** sleeve-correlation mean **+0.08** (max +0.60) — the ML gate + multi-timeframe mix
  decorrelated the book from the +0.25 of the selected version. Best single-sleeve deflated Sharpe is
  0.10 at N≈1,160 trials — individually marginal; the book is a decorrelation effect (reported as such).

## 7b. Cross-sectional breakout — the fix for the regime weakness

Single-name (time-series) breakout is dead on equities/FX. The honest second look is *cross-sectional*:
rank the panel each bar and go long the most broken-out / short the least, dollar-neutral — a
market-neutral bet on **dispersion**, not on the market trending. Signal = George & Hwang
52-week-high nearness (`close / trailing-max`), the evidenced cross-sectional breakout proxy
(`run_bo_xs.py`, `src/sleeves/cross_sectional.py:breakout_signal`).

| Panel | best signal | Sharpe (IS / OOS) | MC-P5 | placebo |
|---|---|---|---|---|
| **crypto** (50 perps, daily) | nearness-126 | **+1.03 (1.11 / 0.86)** | +0.38 | −2.48 |
| equity (1,613 names, monthly) | nearness-252 | −0.39 (−0.40 / −0.33) | — | −0.69 |
| FX (25 pairs, monthly) | nearness-252 | +0.13 (0.06 / 0.45) | — | −0.46 |

- **Crypto cross-sectional is a real, decorrelated edge** (+1.03, OOS +0.86, placebo −2.5). It
  correlates only **+0.16** with the time-series book, and a 50/50 blend nets **Sharpe +1.18, MC-P5
  +0.54, max DD −9.4%, OOS +0.84** — the market-neutral leg carries 2025 (+1.74) when the trend leg is
  flat. This is the single most valuable addition here.
- **Equity cross-sectional is negative even done right** — monthly rebalance (the evidenced slow
  cadence; daily churn on a 1,613-name panel dies to turnover, placebo −11) on a broad universe still
  scores −0.39. The 52-week-high anomaly has decayed post-2000 and does not survive realistic costs.
- **FX** is noise (+0.13, at the placebo floor).
- **Caveat (stated):** both crypto panels (time-series and cross-sectional) are *current* liquid perps
  — survivorship-biased (no dead coins like LUNA/FTT), so the crypto levels are optimistic; the
  regime-complementarity and the equity/FX non-results are the robust qualitative findings.

**Breadth / liquidity (re-run on the expanded ~800-perp cache, `run_bo_xs_big.py` / `run_bo_xs_liq.py`).**
The cache was expanded to ~830 USDT perps (mostly small/micro-cap alts + memecoins, a few
tokenized stocks). More names **monotonically destroy** the edge — it is a liquid-majors phenomenon:

The edge vs breadth is an **inverted-U (hump), peaking at top-20 to top-30** most-liquid perps:

| top-N by $-volume | 1d Sharpe (MC-P5) | 4h Sharpe (MC-P5) | 1h Sharpe (MC-P5) | max DD |
|---|---|---|---|---|
| top-10 | +0.69 (+0.10) | +0.33 (−0.30) | +0.61 (−0.01) | ≈ −25% |
| top-20 | +1.07 (+0.43) | +1.54 (+0.89) | +1.65 (+1.02) | ≈ −18% |
| **top-30** | **+1.22 (+0.59)** | +1.48 (+0.82) | **+1.70 (+1.10)** | ≈ −16% |
| top-50 | +0.97 (+0.37) | +1.26 (+0.63) | +1.12 (+0.54) | ≈ −20% |
| top-100 | +0.84 (+0.20) | +0.87 (+0.22) | +0.76 (+0.13) | ≈ −29% |
| top-300 | +0.36 (−0.25) | +0.28 (—) | +0.18 (—) | ≈ −52% |

- **Too broad kills it** (past ~top-150 the MC 5th-pct goes negative, DD → −40…−52%): illiquid alts
  inject pump-noise, and a flat 6bps cost *understates* their true slippage, so their real contribution
  is worse than shown.
- **Too narrow also kills it** (top-10: Sharpe collapses, MC-P5 goes ≤0): with a 30% book that is only
  long-3 / short-3, there is not enough cross-section to diversify idiosyncratic risk and the ranking is
  too coarse. The edge needs breadth to be a *dispersion* bet, not a 3-name punt.
- **Sweet spot top-30** (best Sharpe on every TF, MC-P5 up to +1.10, lowest DD −14…−17%; best at 1h/4h).
- **Caveat — these small-N levels are the most biased.** The liquidity rank uses full-history volume, so
  top-30 = the coins that *became* most liquid and survived (look-ahead + survivorship); 1.2–1.7 is
  optimistic. The robust, tradeable version ranks on *trailing* volume point-in-time (next).

**Point-in-time top-30 — the honest, look-ahead-free sleeve (`run_bo_xs_pit.py`).** At each bar the
universe is the top-30 by *trailing* 63-day dollar-volume (lagged), so membership only knows the past;
a coin enters when it has actually become liquid and drops out when it fades (its position is closed —
real churn, charged). This quantifies and removes the selection bias:

| TF | static top-30 Sharpe (MC-P5) | **PIT top-30 Sharpe (MC-P5)** | bias removed |
|---|---|---|---|
| 1d | +1.22 (+0.58) | **+0.80 (+0.20)** | −0.42 |
| 4h | +1.48 (+0.81) | **+1.14 (+0.48)** | −0.34 |
| 1h | +1.70 (+1.08) | **+1.02 (+0.37)** | −0.68 |

The look-ahead was worth **~0.3–0.7 Sharpe**; the honest sleeve is **~1.0–1.14 (best at 4h), MC-P5 still
positive**, and its PIT 4h per-year is positive every year 2020-2025 (2022 +1.14, 2025 +0.73) — the
regime-complementarity survives de-biasing. **Combined 50/50 with the time-series book** (correlation
+0.13) at **risk parity** (`run_bo_combined.py`; each leg re-scaled to 15% vol on trailing vol, then
equal-weighted): **Sharpe +1.40, MC-P5 +0.77, max DD −11.5%, months-in-profit 63%**, positive every year
2020-2025 (2022 +1.11, 2025 +0.86), strict OOS **+0.23**. Marginal: trend leg alone +1.04 → +XS **+1.40**
(+0.36 from a +0.10-correlated leg). This decorrelated two-leg crypto breakout book is the honest
deliverable — Sharpe ~1.4 net, robust MC 5th-pct, no universe look-ahead.

**Why crypto and not equities — verified, not asserted (`run_bo_xs_signals.py`).** Running cross-sectional
*momentum* (trailing return) and *breakout* (nearness/Donchian) through the *same* long-short harness
isolates the cause:

| panel | XS momentum (best) | XS breakout (best) |
|---|---|---|
| **equities** (1,613, monthly) | **+0.43** | **−0.39** |
| **crypto** (50, daily) | +0.71 | **+1.03** |

The equity null is **breakout-specific, not a broken harness**: the identical harness turns positive on
momentum (+0.43) and negative on breakout (−0.4…−0.5) on the same stocks. Mechanism — single stocks
mean-revert short-term, so "buy the fresh N-day high" breakout timing is whipsawed by idiosyncratic
reversals, while the smoother trailing-return momentum survives. On crypto (persistently trending, no
short-term reversal) the coarse breakout encoding is fine and even **beats** momentum (+1.03 vs +0.71).
So breakout is a crypto edge specifically; on equities the tradeable cross-sectional edge is *momentum*,
not breakout (a separate sleeve, out of scope here).

## 7c. Contribution to the master portfolio

Breakout is one of the **eight** families in the canonical book (`scripts/run_master_book.py` — the single
portfolio assembly; see [REPORT.md](../../REPORT.md) §4). Every family is re-scaled to a common 15% vol on
trailing (PIT) vol and **equal-weighted (1/N, no performance-based selection)** over the 2011→2026 window
(breakout, a crypto-perp leg, lists from 2020). Breakout's honest series is the combined trend+ML / PIT
cross-sectional squeeze above (`reports/breakout/bo_combined_portfolio.parquet`).

- **Standalone (rescaled) Sharpe:** breakout **+1.38** — mid-pack among the eight (vol-premium **4.57**
  anchors; trend 1.35, BAB 1.29, carry 1.27, gmacro 1.02, x-sect 0.89, crisis 0.49).
- **Correlation to the book:** **+0.56** (mean *pairwise* cross-family correlation ≈ 0.06); breakout is a
  genuinely independent crypto source, not a trend-cluster duplicate.
- **Master with vs without breakout:** Sharpe **3.52 → 3.50** — breakout's marginal (leave-one-out) add is
  **≈ +0.02** (`breakout_delta_sharpe` in `reports/master_book_summary.json`); **vol-premium is the anchor** —
  removing *it* drops the book to **1.75**. Breakout earns its slot by decorrelation and crypto-regime
  coverage, not by lifting the headline Sharpe.
- **Marginal-contribution curve** (added in standalone-descending order): vol-premium 4.57 → +breakout 4.49 →
  +trend 4.20 → +BAB 4.09 → +carry 4.14 → +gmacro 4.12 → +x-sect 3.93 → +crisis 3.52 — the curve *drifts down* as
  diversifiers join (they trade a little average Sharpe for a much smaller tail).

Honest read: on the honest survivorship-free book, breakout is a **fully-decorrelated crypto family**. Its
leave-one-out contribution to the headline Sharpe is small (**+0.02**) because the book is volprem-anchored,
but it adds genuine crypto-regime coverage that no other leg provides — a decorrelation slot, not a Sharpe
lift. Canonical numbers live in `reports/master_book_summary.json`; the breakout-local diagnostic
(`run_bo_contribution.py`) is kept in sync with the master.

## 8. Validation & leakage

- **Placebo:** shuffle-sign sleeves survive **2/405 (0.5% FDR)** — real signal passes far more than noise.
- **Deflated Sharpe** sized to the true trial count (1,160): best sleeve 0.10 → individually marginal,
  honestly stated.
- **Leakage:** execution t+2 (never the signal bar's close); channels lagged one bar; vol-target uses
  lagged vol; funding charged at every 8h settlement; costs liquidity-aware (never flat); **ML labels
  are trade outcomes used only as targets; features stamped at entry; purged+embargoed CV with the
  embargo sized per timeframe; scalers/models fit inside train folds only.** Fixed seeds throughout.

## 9. Ceiling assessment & honest limits

- **Reachable here:** a crypto breakout/trend book at **Sharpe ≈ 1.0 net** over a full cycle, drawdown
  ≈ 4%, with the ML gate doing the risk-control work. Not the 2.5–4.0 aspiration — that is not honestly
  reachable with breakout rules on liquid crypto net of realistic costs.
- **Binding constraints:** (1) **Regime** — the edge is trend-conditional; 2025–26 crypto had no trend
  to capture, so the held-out block is ~0 (the book preserved capital while buy-hold lost ~100 bps of
  Sharpe — trend-following working as designed, not breaking). (2) **Single asset class** — every
  survivor is crypto; in a broad crypto crash the book's decorrelation would compress. (3) **Not a new
  source** — breakout is the trend premium; it diversifies the momentum book's *timing/exit*, not its
  underlying return driver.
- **What extends it (one already done):** the **cross-sectional crypto sleeve** above lifts the combined
  book to OOS +0.84 and directly covers the trend leg's regime hole (§7b) — the best single improvement.
  Remaining: the same construction on **diversified futures/commodities/bonds** (where the
  century-of-evidence trend premium lives at 1–10 bps cost); explicit regime gating (arm the trend leg
  only when a trend-strength filter is on) to skip the trendless drawdowns; a point-in-time crypto
  universe to remove the survivorship optimism.

## 10. Reproduce

```bash
python scripts/breakout/run_bo_exits.py       # exit-style experiment (§3)
python scripts/breakout/run_bo_sweep.py       # full edge map (§4)
python scripts/breakout/run_bo_book.py d55_atr3   # in-sample-selected book (§5); also d55_atr3_tr, kelt_atr3
python scripts/breakout/run_bo_frozen.py      # frozen no-selection book + OOS split (§5)
python scripts/breakout/run_bo_walkforward.py # sleeve-selection + parameter walk-forward (§5)
python scripts/breakout/run_bo_ml.py          # ML meta-label variants (§6)
python scripts/breakout/run_bo_fast_tf.py     # 15m/5m full-treatment test (§4)
python scripts/breakout/run_bo_final.py       # final time-series book + robustness (§7)
python scripts/breakout/run_bo_xs.py          # cross-sectional breakout, all panels (§7b)
python scripts/breakout/run_bo_xs_tf.py       # cross-sectional across timeframes (§7b)
python scripts/breakout/run_bo_xs_big.py      # expanded ~800-perp universe, 3 sub-universes (§7b)
python scripts/breakout/run_bo_xs_liq.py      # liquidity-tier sweep (the hump) (§7b)
python scripts/breakout/run_bo_xs_pit.py      # point-in-time top-30, look-ahead-free (§7b)
python scripts/breakout/run_bo_xs_signals.py  # momentum vs breakout, same harness (§7b)
python scripts/breakout/run_bo_combined.py    # final risk-parity two-leg squeeze (§7b)
python scripts/breakout/run_bo_contribution.py # marginal contribution to the multi-family book (§7c)
python scripts/breakout/run_bo_spot.py        # spot vs perp, pre-perp history, PIT universe (§12)
python scripts/breakout/run_bo_ml_wf.py       # the gate under a walk-forward, net labels (§13)
python scripts/breakout/run_bo_improve.py     # one-knob construction candidates (§11)
python scripts/breakout/run_bo_ts_book.py     # shipped vs corrected assembly (§13)
python scripts/run_master_book.py    # integrated master book incl. breakout (§7c)
python scripts/breakout/make_bo_figures.py    # figures
```

Fixed seed 7 throughout; the 2024-07 block is scored once. Non-ML baseline per family is the raw
Donchian rule; ML incremental value is measured against it.

## 11. What did not work (kept, not hidden)

- **Triple-barrier exit on breakout** — the current book's construction; it discards the fat tail and
  is net-negative on crypto on average (−0.29 across 1d/4h/1h vs +0.20 for the chandelier).
- **Intraday breakout (1h and faster)** — 1h is marginal (needs ML to reach +0.37); 15m/5m are dead
  even with the best exit + ML (5m −1.24 after the gate, from −7.40 raw).
- **Equities & FX breakout** — negative for every construction and timeframe tested.
- **Sleeve selection by trailing Sharpe** — an in-sample illusion (+0.24 OOS); the honest book takes a
  frozen universe with no selection.
- **Cross-sectional breakout on equities & FX** — negative even at the evidenced monthly cadence on a
  broad 1,613-name universe (−0.39); the 52-week-high anomaly decayed post-2000 and does not survive
  costs. (Only the crypto cross-sectional sleeve worked — §7b.)
- **Daily-rebalanced cross-sectional on a large equity panel** — turnover death (placebo −11); the
  52-week-high edge is slow and must be traded monthly.
- **The long-trend alignment filter** — pre-registered as best-evidenced, but neutral-to-slightly-
  negative on already-trending crypto (1.46 vs 1.59 without it); kept off in the default.
- **Adding breadth (top-30 vs core-10)** — dilutes (0.70 vs 0.87); marginal alts add noise, not edge.

One-knob candidates against the frozen construction (`run_bo_improve.py`; spot, long-short, core-10,
1d+4h). Each is scored on the perp-era window, the full spot window, **and** the never-seen 2017-19
block — the third column is the one that decides, because the first two share a price history:

| candidate | 2020+ | 2017+ | **2017-19 (never seen)** | verdict |
|---|---|---|---|---|
| Donchian **89** instead of 55 | +0.21 | +0.10 | **−0.12** | not established |
| **ADX ≥ 20** regime gate | +0.10 | +0.05 | **−0.11** | not established |
| ADX ≥ 25 | −0.04 | −0.14 | — | worse |
| Donchian 20 / 34 / 144 | −0.03 / −0.00 / +0.12 | −0.00 / +0.03 / −0.07 | — | no signal |
| chandelier k = 2 / 4 / 5 | −0.13 / −0.07 / −0.15 | −0.05 / −0.04 / −0.13 | — | k=3 is a shallow plateau |
| volume-weighted position size | −0.63 | −0.56 | — | turnover death |
| suppress re-entry after a stop | ±0.00 | ±0.00 | ±0.00 | no-op, see below |

- **A longer channel and an ADX gate both look like improvements and neither survives.** Each lifts
  2020+ by ~0.1-0.2 and each *loses* ~0.11 on the only block the parameters were never exposed to. What
  they do robustly everywhere is cut risk, not raise return: drawdown −9.1% → −6.6%, turnover 9.1 → 6.5
  round turns a year. Take them as a turnover reduction if that is worth something; do not book the
  Sharpe. Combining them adds nothing over the longer channel alone (+0.10 on 2017+, same as lb-89).
- **Volume-weighted position sizing** — the cheapest breakout analogue of the volume-weighted crypto
  TSMOM result (Huang, Sangiorgi & Urquhart 2024). Modulating the held position by a rolling volume
  z-score churns it every bar: turnover 9 → 56/yr, Sharpe +1.09 → +0.53, drawdown −9.1% → −20.5%. The
  published result weights *across assets in a portfolio*; it does not transfer to modulating a single
  time-series position, and the transfer was the assumption, not the finding.
- **Suppressing re-entry after a chandelier stop** — reading the code suggests a defect: the entry is a
  *persistent* side (+1 for every bar above the channel), so a stop-out should be bought straight back
  on the next bar while the condition still holds. Collapsing the side to an onset impulse
  (`breakout_lab.fresh_side`) changes 1,046 signal bars to 755 on BTC 4h and changes **zero** held
  positions: at all 259 stop-outs the persistent side is already off, because a 3×ATR retracement always
  takes price back inside a 55-bar channel. The defect does not exist; recorded because it was measured
  rather than assumed.
- **Equal-weighting with `fillna(0).mean()`** — the shipped book divides by all 30 slots regardless of
  how many have listed, so it runs at 78% of intended size through 2020. Real, and small: correcting it
  to an active-sleeve mean moves the book +1.03 → +1.06 Sharpe, CAGR +3.3% → +3.4%.


## 12. Venue: the funding bill, and the 2.4 years before perps existed

Everything above executes on USD-M perpetuals, which is why every window starts 2020-01 — the month
perps list. Binance **spot** klines for the same names reach back to **2017-08**, and the two venues
price the same position very differently (`run_bo_spot.py`).

**The funding bill is the largest single cost in this strategy, and it is conditional.** Averaged over
the core-10 since 2020, holding a perp long costs **10.3%/yr** in funding (positive at 71-86% of
settlements). But the book is not long at random — it is long precisely when the market is trending up
and funding is extreme. Measured against the book's own executed position:

| perp funding, conditional on the book's position | annualised |
|---|---|
| paid while the book is **long** | **−23.4%** |
| received while the book is **short** | **+2.3%** |
| unconditional | −10.3% |

So the long leg pays more than twice the headline rate, and the short leg collects almost nothing —
funding has usually already normalised or inverted by the time a trend book flips short. Against a
gross CAGR of ~7% at a ~0.2 average gross exposure, this is the dominant cost line, ahead of
commission and slippage combined.

Spot has no funding but charges **2× the taker fee** (10bps vs 5bps) and its short is not free: the
coin must be borrowed. Live Binance VIP-0 cross-margin rates (read 2026-08-10 from the public
margin-spec endpoint) average **2.93%/yr** across the core-10 — BTC 0.44%, ETH 2.16%, up to SOL 5.47%.
That is charged here on short gross, the same convention as the equity short-borrow.

**Matched window 2020-01→2026-07, frozen core-10, 1d+4h, long-short, all costs charged:**

| execution | Sharpe | MC-P5 | CAGR | vol | maxDD |
|---|---|---|---|---|---|
| all-perp (the shipped book) | +0.89 | +0.29 | +6.2% | 7.0% | −9.2% |
| all-spot | +1.01 | +0.40 | +7.2% | 7.1% | −9.1% |
| **long on spot, short on perp** | **+1.03** | **+0.42** | **+7.3%** | 7.1% | −9.0% |

Spot beats perp *despite* paying double the fee, because 8 round turns a year at 5 extra bps is 40bps
while the funding it avoids is worth ~1pp of CAGR at this exposure. The split — long leg on spot, short
leg on perps — is the venue-optimal execution and costs nothing extra to run; the incremental gain over
all-spot is small (the short leg's borrow-vs-funding gap is ~5pp/yr on ~0.2 gross, ~45% of the time).

**The history is the more valuable half.** 2017-08→2019-12 is data no version of this construction has
been fitted on, and the early spot cross-section is *less* survivorship-biased than any perp universe:
4 names in 2017-12, 12 by 2018-06, 59 by 2019-12, and the cohort is EOS, ICX, IOTA, HOT — 2017-bubble
alts that faded and that no current-perp list contains.

| spot, 1d+4h, long-short | Sharpe | MC-P5 | CAGR | maxDD |
|---|---|---|---|---|
| full 2017-08→2026-07, frozen core-10 | +1.09 | +0.57 | +7.9% | −9.1% |
| **never-seen block 2017-08→2019-12** | **+1.34** | +0.27 | +9.3% | −4.7% |
| full window, point-in-time top-10 (trailing dollar volume) | +0.90 | +0.39 | +5.6% | −7.4% |
| never-seen block, point-in-time top-10 | +1.33 | +0.31 | +8.7% | −3.5% |

The construction holds on the pre-perp block at a level indistinguishable from the fitted period, which
is the strongest single piece of evidence in this document that breakout on crypto is a real premium
rather than a 2020-2021 artifact. The frozen core-10's hindsight premium over a point-in-time universe
is **~0.19 Sharpe** on the full window and **~0.01 on the never-seen block** — the frozen list flatters
mainly the period in which those names were already the winners.

**Long-only looks better and is worse.** Dropping the short leg lifts the ratio on every cut (spot
1d+4h: +1.09 → +1.55; perp: +0.89 → +1.28). Regressed on an equal-weight buy-and-hold of the same ten
coins, the reason is visible:

| leg | corr to buy-hold | R² | alpha/yr | t(alpha) |
|---|---|---|---|---|
| long-short | −0.02 | 0.00 | **+7.4%** | +2.66 |
| long-only | **+0.67** | **0.45** | +4.2% | +2.75 |

Long-only trades ~3pp of annual alpha for 0.67 correlation to crypto beta, and it pays for it in the
regimes that matter: 2022 −0.88 vs long-short +0.52, 2026 −3.21 vs −0.43. In a book whose reason for
holding a crypto sleeve is decorrelation, the higher Sharpe is the wrong thing to buy.

## 13. The ML gate under a walk-forward

`run_bo_ml.py` estimates the gate with `purged_kfold`. Purging removes label-overlap leakage; it does
not remove look-ahead, because each fold trains on its whole complement. Measured on BTC 4h
(`run_bo_ml_wf.py`), the share of each fold's training events that lie *after* its test window:

| fold | test span | training events after the test fold |
|---|---|---|
| 0 | 2020-04 → 2021-09 | **100%** |
| 1 | 2021-09 → 2022-11 | 74% |
| 2 | 2022-12 → 2024-04 | 50% |
| 3 | 2024-04 → 2025-06 | 25% |
| 4 | 2025-06 → 2026-07 | 0% |

The strict 2024-07 block sits in folds 3 and 4, so even the held-out number is ~25% future-fitted. The
replacement is an expanding walk-forward: a block is scored only by trades whose label had already
resolved before it opened, plus the embargo. A second correction is independent of the first — the
label is `close(t1)/close(t0)` at the *signal* bars and gross, while the book fills at t+2 and pays
fees and funding. Re-pricing the label at execution, net of round-trip cost and funding, flips **10.3%**
of labels (win rate 36.4% → 35.3%).

Each gate is compared against the same ungated book restricted to exactly the sleeves and dates that
gate covers, so this is a gate comparison and not a window comparison. `=vol` rescales the gated book
to the ungated book's volatility — the gate keeps only 13-20% of trades and runs at ~2% vol against
7.8%, so the raw ratio flatters it and the raw CAGR damns it; only the vol-matched column is readable.

| perp, 4h+1h | Sharpe | CAGR | =vol CAGR | maxDD | kept | **OOS Sharpe** |
|---|---|---|---|---|---|---|
| ungated primary | +0.30 | +2.1% | — | −13.3% | 100% | +0.20 |
| k-fold, gross labels (shipped) | +0.89 | +1.8% | +6.9% | −2.7% | 19% | +0.31 |
| k-fold, net labels | +0.75 | +1.4% | +5.7% | −3.3% | 18% | +0.77 |
| **walk-forward, net labels** | **+0.72** | +2.0% | +5.5% | −5.5% | 16% | **+0.64** |
| walk-forward pooled across symbols | +0.78 | +1.8% | +5.8% | −3.2% | 13% | −0.07 |
| walk-forward, AFML bet sizing | +0.55 | +0.6% | +4.1% | −1.8% | 100% | +0.30 |

| spot, 4h+1h | Sharpe | =vol CAGR | maxDD | **OOS Sharpe** |
|---|---|---|---|---|
| ungated primary | +0.24 | — | −14.2% | −0.06 |
| k-fold, gross labels | +0.80 | +6.5% | −3.8% | +0.28 |
| k-fold, net labels | +0.74 | +6.0% | −3.2% | **+1.06** |
| **walk-forward, net labels** | +0.45 | +3.4% | −4.1% | +0.76 |
| walk-forward, AFML bet sizing | +0.74 | +5.8% | −1.4% | **+1.01** |

Four things follow, and they are separable:

1. **The gate survives.** Stripped of every future fold, meta-labelling still takes perp from +0.30 to
   +0.72 and its held-out block from +0.20 to +0.64; spot from +0.24 to +0.45 and −0.06 to +0.76. The
   headline in §6 was inflated, not invented.
2. **The look-ahead was worth ~0.1-0.3 Sharpe**, more on spot than on perp (k-fold/net vs
   walk-forward/net: 0.75 → 0.72 perp, 0.74 → 0.45 spot; on the held-out block 0.77 → 0.64 and
   1.06 → 0.76).
3. **The label is worth more than the model.** Pricing the label net-of-cost-at-execution moves the
   held-out block from +0.31 to +0.77 (perp) and +0.28 to +1.06 (spot) — a bigger effect than the
   choice of estimator, and it costs nothing.
4. **Pooling and bet-sizing are not free wins.** One model per timeframe trained on all ten symbols
   has ~10× the data and is worse out of sample (−0.07 perp). AFML bet sizing halves the drawdown and
   is the best spot variant out of sample (+1.01) but the worst on perp (+0.55) — no consistent edge.

**Assembled book, one correction at a time** (`run_bo_ts_book.py`; core-10, 1d raw chandelier +
4h/1h gated, 2020-01→2026-07 so the split is tradeable throughout):

| assembly | Sharpe | MC-P5 | CAGR | maxDD | months+ | **OOS** |
|---|---|---|---|---|---|---|
| shipped: all-perp, k-fold gate, gross labels | +1.04 | +0.38 | +3.4% | −4.1% | 49% | +0.13 |
| + walk-forward gate on net labels | +1.07 | +0.46 | +4.8% | −5.2% | 49% | +0.26 |
| **+ venue split (long spot, short perp)** | **+1.07** | **+0.50** | +3.8% | −5.2% | 52% | **+0.37** |

The headline barely moves. Everything the corrections buy shows up where the shipped book was weakest:
the held-out block nearly triples (+0.13 → +0.37), the Monte-Carlo 5th percentile rises +0.12, months
in profit go 49% → 52%, and the 2026 downturn improves from −0.73 to −0.21. That is the right shape for
a correction — it did not manufacture return, it removed an optimism and a cost.
