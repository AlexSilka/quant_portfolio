# Volume spike — deep-dive findings

**Scope.** Port an existing live Rust bot (**spike_bot / VolumeSpike**) into the framework and test
it honestly as a candidate sleeve: long-only altcoin perps, 15m, entry on an anomalous volume burst.
The bot's own edge is discretionary — it per-coin-optimises and positions are entered manually on a few
"pump" names (FHE, PIPPIN, JELLYJELLY). That cannot be walk-forward validated, so here **one global
parameter set is applied uniformly to a frozen alt universe and the results are pooled** — the
portfolio question is "does the volume-spike edge exist *systematically*, net of cost?". All numbers
vol-targeted to 15%, net of liquidity-aware costs, executed at t+2, over 2020-01 → 2026-07. Reproduce:
`python scripts/volspike/run_volspike.py all`. Artifacts: `reports/volspike/volspike_wf_oos.parquet`.

---

## 0. TL;DR

**Not viable as a systematic portfolio sleeve — it does not clear the acceptance bar, on any honest
cut.** Walk-forward OOS Sharpe is **−0.5 to −1.0** across every window×cadence scheme; Monte-Carlo P5
**−1.79**; deflated Sharpe **0.00** at N=10 trials; it barely beats a random-entry placebo; break-even
cost is **≈1.02× base** (zero cost cushion). It *is* decorrelated from the book (|corr| ≤ 0.13) — but a
break-even-gross, negative-net sleeve adds nothing however uncorrelated.

The honest edge map underneath the verdict:
- On **liquid large alts** a volume spike is followed by a **real but tiny** upward drift (+0.55% over
  24h, t ≈ 8) that survives honest t+2 execution — but it is **smaller than the liquidity-aware
  round-trip cost**, so it nets to zero.
- On **small-cap alts — the bot's actual target — the drift is negative** (t −2…−5): buying the spike
  is buying exhaustion / a local top. So the bot's live wins are **survivorship / selection**, not a
  systematic edge.

This confirms and sharpens the book's prior (REPORT §7): event families at 15m are destroyed by
turnover × cost. Here there is barely any gross edge to destroy where the strategy claims one.

## 1. Construction

Faithful to the bot (`docs/STRATEGY_VIABILITY.md`, `lib/strategy/src/signal.rs`), simplest form only
(the plain volume-spike entry — no fractal / BTC-alignment gates):

- **Entry (long, at the close of bar t), fires iff both:**
  1. volume spike — `quote_volume[t] ≥ k_vol · SMA(quote_volume, vol_win)`, the SMA taken over the
     `vol_win` bars **strictly before** t (trigger excluded), on quote (USDT) volume. A ratio
     threshold, not a z-score.
  2. non-falling price — `(close[t]/close[t-1] − 1)·100 ≥ pcp`.
- **Exit — triple barrier** (vol-scaled take-profit / stop / vertical) or a fixed **time-stop**, reusing
  the framework's event-sleeve machinery. The bot's **trailing stop is dropped**: on 15m bars without
  ticks the intrabar peak is unobservable, so a trailing exit would be self-deception. Barriers are
  detected on the close (no intrabar-path assumption) — the same convention as breakout/mean-reversion.
- **Execution t+2** — the bot fills ~5ms after the close; on bars the honest fill is the next bar's
  open. The event study (§2) confirms the edge is a multi-hour drift, **not** a first-bar latency
  artefact, so the delay does not distort the measurement.
- **Costs** liquidity-aware crypto perp taker (5 + 1 bps + √-impact scaled to bar $-volume) + funding
  at every 8h settlement; sizing vol-targeted to 15% annualised; equal-risk book across the universe.
  Identical to every other sleeve, so the number is comparable to the book.
- **Universe (frozen before evaluation):** the 30 most-liquid USD-M alt perps, **ex BTC/ETH** (the
  strategy targets alts, not majors), ≥36 months history, ranked by full-sample median $-volume.
- Code: [src/sleeves/volume_spike.py](../../src/sleeves/volume_spike.py) (entry only) +
  [scripts/volspike/run_volspike.py](../../scripts/volspike/run_volspike.py) (evaluation).

## 2. Does the edge exist? — event study by liquidity tier

Mean forward return after a spike event, vs a same-count random-bar baseline (15,118 events on the
liquid tier; 70,337 on small-caps). Measured from the signal close **and** from the honest open[t+1]:

| horizon | liquid top-15, excess vs baseline | t-stat | small-cap (120–200), excess | t-stat |
|---|---|---|---|---|
| 8 bars (2h)  | **+0.090%** | 3.8 | −0.062% | −4.4 |
| 32 bars (8h) | **+0.173%** | 5.7 | −0.104% | −3.9 |
| 96 bars (24h)| **+0.551%** | 8.4 | −0.085% | −2.2 |

The liquid-tier drift is nearly identical from close and from open[t+1] (e.g. +0.72% vs +0.69% at
96 bars) — the edge is not a fill-timing artefact. The small-cap sign is **negative and significant**:
systematically, the spike marks a top there.

## 3. Net of costs — walk-forward + robustness

Parameter walk-forward: on each train block pick the best of a 10-config a-priori grid by train
Sharpe, apply it OOS on the next block, stitch. Run under anchored/rolling × annual/semiannual refits
to prove the result does not depend on the choice.

| test | result |
|---|---|
| OOS Sharpe (4 window×cadence schemes) | **−0.99 / −0.70 / −0.86 / −0.50** (all negative) |
| Monte-Carlo (primary, block bootstrap) | P5 **−1.79**, P50 −0.96 |
| in-sample full-sample best of grid | **+0.02** (the ceiling, with overfit) — frac of grid positive 20% |
| deflated Sharpe (N=10 trials) | **0.00** |
| placebo (entries moved to random bars) | real max +0.02 vs placebo max −0.61 — signal adds almost nothing |
| cost sensitivity ×1 / ×2 / ×3 | +0.02 / −0.45 / −0.93 |
| break-even cost | **≈1.02× base** (no cushion) |
| per-year Sharpe | 2021 **+2.5** only; 2022 −0.9, 2023 +0.3, 2024 +0.1, 2025 −0.8, 2026 −0.7 |
| correlation to book sleeves | trend −0.02, x-sect −0.13, carry −0.03, eq-trend +0.02, eq-xsect +0.05, pairs −0.06 |

The gross↔net decomposition is the whole story: the bot's toy default (k=2, vol_win=10, tight barriers)
runs **217× annual turnover** for a **−0.14 gross / −2.88 net** Sharpe; making entries rarer and holds
longer cuts turnover to 13× and net to −0.40, but gross stays ≈ +0.08 — the improvement is "trade less,
lose less to cost", not "find edge". The best liquid-only long-hold config reaches **+0.03 net** — i.e.
break-even, before the deflated-Sharpe penalty.

## 4. Why it fails — the edge map

- **Liquid alts:** a genuine short-horizon volume-momentum drift exists, but it is ~0.1–0.5% over 2–24h
  — below the liquidity-aware round-trip cost at 15m. Cost-fragile by construction.
- **Small-cap alts:** the volume spike is a **contrarian** signal (negative forward return) — the
  opposite of the long entry. The bot's success on hand-picked pump coins is the survivorship of the
  few that continued up; the systematic expectation is a loss. This is the single most important
  finding: the strategy's stated home is where it is worst.
- **Regime:** the only positive year is 2021 (the alt mania). It is a bull-beta artefact, not a
  timing edge — consistent with the trend book's own regime profile, and it does not diversify it.

## 5. Honest limits & what would extend it

- The test is of the **simplest** entry (volume + price), as scoped. A richer signal might do better,
  but the honest lead is narrow: the only real drift is liquid-alt volume-momentum, and it is
  cost-bound at 15m. The defensible next probes are a **coarser timeframe** (1h/4h, where the same
  drift pays fewer cost turns) or **taker-buy imbalance** rather than raw volume (aggressor flow, not
  gross size) — *not* small-caps, where the sign is wrong.
- What would **not** rescue it: barrier tuning (surface is flat-to-negative), or chasing the small-cap
  book (survivorship). Trailing-stop exits cannot be honestly simulated on bars and would not change
  the gross edge.

## 6. Reproduce

```bash
python scripts/volspike/run_volspike.py all      # smoke -> walk-forward -> robustness  (~3 min)
```

Fixed seed 7 throughout. Writes `reports/volspike/volspike_wf_oos.parquet` (the stitched OOS series).
