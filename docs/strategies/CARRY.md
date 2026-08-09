# Carry — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.72** full / **3.77** OOS), assembled by `scripts/run_master_book.py` (the older `build_book.py` "streams" pipeline in §6 is superseded). Carry is one leg (P&L share ~5%), **not** the book's biggest contributor — vol-premium anchors it. Any master-book Sharpe below is a snapshot at evaluation time.

**Scope.** A full rework and evaluation of the perpetual-funding **carry** family: what the existing
sleeve does, why it fails, the two constructions under which carry actually works, ML overlays,
walk-forward across window schemes, leakage audits, and portfolio value-add. All numbers are net of
liquidity-aware costs, executed at t+2 bars, funding charged at every 8h settlement. Figure:
[reports/figures/carry.png](../../reports/figures/carry.png). Reproduce: `make carry`.

---

## 0. TL;DR

- **Directional single-asset carry — the sleeve that shipped — is dead** (mean Sharpe ≈ +0.1). It
  takes full price risk in a ~60%-vol asset to collect ~13%/yr funding, so price noise swamps the
  signal. This confirms the earlier "carry did not survive" verdict — but only for that construction.
- **Cross-sectional dollar-neutral funding carry is a real, distinct edge.** Rank the 50-perp panel
  by trailing funding, long the names the market pays you to hold (low/negative funding), short the
  expensive ones (high funding). **Net Sharpe +1.21, MC-P5 +0.58, maxDD −22%, 70% profitable months**;
  walk-forward OOS **+0.88** mean across six window schemes; **58% of the parameter grid** clears the
  robust bar while the shuffled-funding **placebo is −2.4**. It survives every leakage audit and is
  **~0-correlated to the momentum family** — a genuinely new return source, not a trend proxy.
- **Refinement adds a validated ~0.5 of walk-forward Sharpe.** A **BTC-beta hedge** (the book
  structurally shorts high-beta hot coins ⇒ net-short beta ⇒ bull-market drag) plus inverse-vol
  weighting and a no-trade buffer lift it to **Sharpe +1.47, MC-P5 +0.83, maxDD −18%** and, crucially,
  **walk-forward OOS +0.88 → +1.40** — the gain *holds out-of-sample in both rolling and expanding
  schemes*, so it is real, not fitted. (Funding-momentum signals and signal-weighting were tested and
  *rejected* — the funding **level** is the right carry signal.)
- **Delta-neutral cash-and-carry (basis) harvests ~14%/yr funding at ~2.2% vol** (full 46-name panel).
  Naive daily construction dies to two-leg turnover cost; a hold-through-regime version nets **~10%/yr
  after realistic financing at 2.2% vol (−2% maxDD)** — and across 46 names the carry-crash tail
  **diversifies away (skew +2.5, not −5)**. The catch: the huge raw Sharpe (~4.5) is a low-vol artifact
  realisable only with leverage (which reintroduces gap risk in a synchronised deleveraging), and the
  harvest is capacity/crowding-limited (funding compressed toward T-bills in 2025). Real, but bounded.
- **ML: signal no, risk yes.** An ML *ranker* does not beat the linear funding rank (it overfits the
  low-SNR funding features). An ML *timing overlay* (regime gate) lifts Sharpe 1.21→1.52 and cuts DD
  22%→16% — value is risk reduction, exactly as on the momentum family.
- **Cross-asset: carry is an EDGE only in crypto.** The same machinery on FX (rate-differential carry,
  12 currencies) and equities (dividend-yield carry, 50 names) shows the carry *accrual* is real
  everywhere (+4.5%/yr FX, +5.2%/yr equity) but the price/spot leg *offsets* it — FX Sharpe **+0.39**
  (17th pct of its placebo), equity **−0.69** (20th pct). Carry works in crypto because the price leg
  *helps* (crowded-long reversal); in efficient FX/equity markets the premium is priced away
  (high-yield currencies depreciate ≈ UIP; high-yield stocks are value traps). See §7.

---

## 1. Why the shipped sleeve fails

`src/sleeves/carry.py` sides a single asset off a funding z-score: short when funding is richly
positive, long when richly negative. The problem is scale. BTC funding averages ~0.012%/8h ⇒ **~13%/yr**
carry; BTC price volatility is **~60%/yr**. A full-size directional position collects the drip but eats
the vol, so the funding signal must also predict price *direction* — which it does not reliably. Measured
across BTC/ETH/SOL/DOGE the mean directional Sharpe is **+0.11**; the book-wide edge map had carry
negative at every timeframe. Correct diagnosis, incomplete construction: carry is not a directional bet.

## 2. Cross-sectional dollar-neutral carry — the edge

**Construction** (`src/sleeves/carry_xs.py`). Each day, rank all names by EMA-smoothed daily funding;
equal-weight **long the bottom 20%, short the top 20%**, dollar-neutral, hold with a short rebalance;
vol-target the book to 15%. Market beta nets out and the book collects the cross-sectional **funding
spread** (short-basket funding minus long-basket funding — positive by construction), while the price
legs largely cancel. This matches the documented practitioner recipe (top-50 universe, quintile
long/short, EMA-smoothed funding, daily rebalance; Aperiodic/Unravel, Robot Wealth, Presto Research),
and the honest net Sharpe lands in the documented **1–2 zone**, not the inflated 4–6 of gross basis carry.

**Headline (level-7 EMA, top-20%, vol-targeted 15%, 2020–2026, 50 names):**

| Sharpe | MC-P5 | MC-P50 | maxDD | months+ | corr to momentum family |
|---|---|---|---|---|---|
| **+1.21** | +0.57 | +1.23 | −22.3% | 70% | **−0.04** |

**It is a broad plateau, not a lucky spike.** Of 62 (signal × lookback × top-fraction × rebalance)
configs, **36 (58%)** clear Sharpe>0.5 **and** MC-P5>0; the full-sample sensitivity surface is **92%
positive** (median +0.81). The shuffled-funding **placebo is −2.38** — the edge lives in the funding
information, not the long/short machinery. **Deflated Sharpe ≈ 0.76** at N=62 trials — individually
suggestive but below the 0.95 single-sleeve bar (honest), yet an order of magnitude above the momentum
book's best-sleeve DSR (~0.06); the standalone case rests on the **+0.88 walk-forward** and the
decorrelated portfolio contribution, not on one deflated sleeve.

**Robust across sampling frequency, not a daily-close artifact.** Rebuilt on **4h** bars the edge holds
(**Sharpe +1.30** at a 7-day lookback), but at higher turnover (1,837 vs ~1,200) for no Sharpe gain —
funding only updates every 8h, so faster rebalancing adds cost, not signal. **Daily is the right
frequency**; the edge appearing identically at 4h and 1d is further evidence it is real.

**Per-year Sharpe:** 2020 +0.73, **2021 +0.21** (weak — a uniform bull compresses cross-sectional
dispersion, everyone is a crowded long), 2022 +0.81, 2023 +1.02, 2024 +1.56, 2025 +2.38, 2026 +2.56.
Carry *strengthens* over 2023→26 as the perp market matures and funding dispersion widens — the
**opposite** of single-asset BTC basis carry, whose level compressed toward T-bills in 2025 (BitMEX,
BIS). Cross-sectional carry harvests *dispersion*, not the absolute level, so the two decouple.

### 2.1 Leakage audit (the result is real, not a timing artifact)

The book makes money on both a smooth funding accrual and a price leg, and funding is contemporaneously
correlated with returns — so the price leg was audited from four orthogonal angles (`scripts/carry/audit_carry.py`):

- **Leg decomposition.** funding-only Sharpe **+12.5** at −1% DD (a smooth accrual — high Sharpe only
  because its vol is tiny; **not** independently harvestable at that level without delta-hedging, see §4);
  price-only **+0.97**; total +1.21.
- **Execution-lag ladder.** Price-leg Sharpe decays **gracefully** (0.97 → 0.82 → 0.81 → 0.67 at lag
  2/3/6/8 days, dying only by lag 12) — the signature of a real multi-day predictive signal. A
  contemporaneous leak would collapse at lag 3–4.
- **Extra signal purge.** Shifting the funding signal back 1–5 more days holds price-Sharpe ≈ 0.8.
- **Long vs short leg.** Longing cheap-funding names **wins** on price (+0.65); shorting rich-funding
  names **loses** on price (−0.38) — you short them purely for the funding. A nuanced, honest picture:
  the "high funding predicts underperformance" reversal effect is real but concentrated in the long leg.

### 2.2 Incremental to momentum (fills a literature gap)

The academic literature documents that funding is *driven by* past returns (trend-chasing; R² up to
0.92) and that high carry predicts future crashes/liquidations — but **no accessible study regresses
funding-sorted returns on reversal/momentum controls**. This project measures it directly: the carry
book is **+0.03-correlated to cross-sectional momentum**, and a funding signal **residualised on
trailing return** (stripping the momentum-explained component) *still* earns Sharpe **+0.81**. Carry
carries information beyond price — it is not reversal or momentum in disguise.

### 2.3 Walk-forward (honest parameter choice, across window schemes)

`scripts/carry/run_carry_wfo.py` rolls forward, picking the best (lookback, top-fraction) on each train
window and applying it out-of-sample, across six schemes:

| scheme | OOS Sharpe | MC-P5 | maxDD |
|---|---|---|---|
| roll 365/90 | +0.93 | +0.18 | −30% |
| roll 365/180 | +0.77 | +0.10 | −37% |
| roll 545/90 | +0.92 | +0.11 | −29% |
| roll 270/90 | +0.90 | +0.26 | −45% |
| expand */90 | +0.80 | +0.14 | −38% |
| expand */180 | +0.93 | +0.28 | −34% |

**Mean OOS +0.88**, every scheme positive, vs the peak-picking (overfit) reference +1.21. The small
peak↔walk-forward gap is the plateau signature. Out-of-sample drawdown widens to −29/−45% (param choice
lags regime turns) — the honest cost of choosing parameters live.

### 2.4 Cost robustness

Break-even at **~5× base cost** (base = 6bps/side). Sharpe holds at +0.93 (2×) and +0.65 (3×). Annual
turnover ≈ 180× on a gross-2 dollar-neutral book — high, but the edge survives it comfortably.

### 2.5 Refinement (each lever a hypothesis, kept only if it survives out-of-sample)

`scripts/carry/run_carry_refine.py` tests four research-suggested levers one-factor-at-a-time against the
equal-weight baseline, then re-validates the combination — a refinement is kept only if walk-forward
OOS improves, else it is reported as in-sample overfitting.

| lever | full-sample ΔSharpe | OOS (WFO) contribution | rationale / verdict |
|---|---|---|---|
| **BTC-beta hedge** | +0.06 | **+0.22** (biggest) | book shorts high-beta hot coins ⇒ net-short beta (−0.044) ⇒ bull drag; hedge cuts it to +0.006. **Kept.** |
| **inverse-vol weighting** | +0.04 | +0.04 | risk-parity within legs; volatile alts stop dominating risk. **Kept.** |
| **no-trade buffer** | +0.02 | +0.05 | hold a name until its target drifts past a band; turnover hygiene. **Kept.** |
| vol-adjusted signal | +0.38 | (not combined) | higher Sharpe but tilts toward a low-vol anomaly; excluded to keep the book pure-carry |
| signal-weighting | **−0.60** | — | over-concentrates; **rejected** |
| MACD-funding (momentum) | **−1.42** | — | funding *momentum* is the wrong bet — the **level** is the carry signal; **rejected** |

**Validated refined book (BTC-beta hedge + inverse-vol + buffer, funding-level signal):**

| | Sharpe | MC-P5 | maxDD | months+ | walk-forward OOS (expand / roll) | placebo |
|---|---|---|---|---|---|---|
| baseline | +1.21 | +0.57 | −22% | 70% | +0.73 / +1.16 | −3.1 |
| **refined** | **+1.47** | **+0.83** | **−18%** | 68% | **+1.41 / +1.39** | −1.2 |

The refined book is **+0.77-correlated to the baseline** (it improves carry, doesn't replace it), and
the OOS gain is consistent across both walk-forward schemes — the signature of a real structural
improvement. The single biggest lever, the **beta-hedge**, is economically motivated (it removes a
measured net-short beta) rather than fitted, which is why it survives OOS. Deflated Sharpe stays
individually marginal (0.2–0.8 depending on trial-variance assumptions) — as always for one sleeve; the
standalone case rests on the walk-forward and the decorrelation, not on a deflated single-sleeve number.

### 2.6 Universe breadth — where does carry live? (`scripts/carry/run_carry_breadth.py`)

The whole Binance USD-M archive was pulled — **830 perps incl. delisted names** (LUNA, FTT, WAVES…) —
and carry re-run on a **point-in-time** universe: each date, the top-N by trailing dollar volume (a coin
is traded only while it was actually liquid, so a delisted coin simply leaves the set — survivorship-honest).

| universe (PIT top-N by $-vol) | Sharpe | MC-P5 | maxDD |
|---|---|---|---|
| top-30 (megacaps) | +0.84 | 0.15 | −34% |
| top-50 | +0.69 | 0.04 | −30% |
| **top-75** | **+1.26** | 0.64 | −25% |
| **top-100** | **+1.33** | 0.71 | −23% |
| top-150 | +1.17 | 0.59 | −29% |
| top-200 | +1.20 | 0.61 | −29% |

- **The curve peaks at ~75–100, then flattens** — breadth helps up to a point, not monotonically.
- **Carry lives in the mid-cap tier, not the megacaps.** Top-30 is *weak* (+0.84): the largest coins
  have compressed, low-dispersion funding, so there is little carry spread to harvest. The dispersion —
  the fuel — is in names ~50–100. The illiquid **150–200 tail adds cost and noise, not edge**.
- **Survivorship inflation is small but real: +0.05 Sharpe.** PIT-with-delisted nets **+1.29** against
  current-listed-only **+1.34**. An earlier version of this line claimed the *opposite* sign — that
  including dead names raised the Sharpe — and that was a measurement artifact. "Still listed" was read
  as "has a non-NaN last close", but Binance's archive keeps emitting daily bars for a **settled**
  contract at a frozen price and zero volume (FTM, BAL, AGIX, ALPACA, FTT ran on for 407 such bars;
  `exchangeInfo` reports them `SETTLING`). That put 123 dead perps into the survivors-only book and
  flattened the very comparison the test exists to make. Survivors are now defined by traded volume.
  The eligible universe was never affected — zero volume already keeps these names out of the PIT
  top-N — so only this diagnostic moved.
- **Cost / regime caveat.** The wide universe is more cost-sensitive (top-100 breaks even ~30 bps/side vs
  the curated core's ~30 bps too, but degrades faster) and more exposed to deleveraging — 2022 turns
  **negative (−0.5)** vs the curated 50's +0.8, as junk alts crash together. Gate the tail by liquidity
  (the PIT top-N does exactly this).
- **Bottom line.** Going 50→100 is a **rigor upgrade** (point-in-time, no hand-curation, survivorship-clean)
  and teaches *where* the edge sits, but it is **not a Sharpe win**: the best wide-universe book is
  *comparable to, not above,* the curated 50-name refined book (+1.47). The honest deployable universe is
  **PIT top ~75–100, mid-cap-inclusive** — wider adds tail risk without paying.
- **Note:** the Sharpes in this table predate the crypto-native purge (§6) — they include a tokenized-gold
  beta that inflated them by ~0.1. The purged, deployable numbers (band 15–90 = **+1.21**) are in §6.

## 3. ML overlays

`scripts/carry/run_carry_ml.py`, all purged/expanding-window OOS, fixed seed, measured against the linear baseline.

- **ML ranker — does not beat linear.** LightGBM/Ridge predicting each name's forward return from
  funding + price + vol features (3 feature sets × 2 models × 2 horizons) tops out at Sharpe **+0.61**
  (vs linear **+1.21**); LightGBM on funding-only features *inverts* (−0.70). The linear funding rank is
  already near-optimal, and ML overfits the low-SNR signal — corroborating the one published ML-on-carry
  result (a gradient-boosted ranker that "destroyed capital after costs", SSRN 6701738).
- **ML timing overlay — works.** A logistic regime gate on market-wide state (mean funding, funding
  dispersion, BTC vol/trend, the book's own recent P&L) predicting the up-week and scaling exposure lifts
  Sharpe **1.21 → 1.52**, cuts maxDD **−22% → −16%**, MC-P5 **0.58 → 0.94**. ML's honest value here is
  **risk reduction, not signal** — the same finding as in the trend/momentum sleeves. (The overlay's
  months-in-profit drop is a gating-to-cash metric artifact.)

## 4. Delta-neutral cash-and-carry (basis)

Long spot + short perp (or the reverse when funding is negative), delta-neutral, harvest funding
(`basis_carry_hold`, run on the 12 liquid names with spot cached; measured spot–perp basis 0.072%, so
the legs hedge well and the residual is genuine basis risk).

- **Naive daily construction dies to cost.** Collecting +16%/yr funding but paying **~−18%/yr in
  two-leg turnover** (115× turnover, ~3,800 funding sign-flips each forcing a full round-trip) ⇒ **net
  negative**. This is the documented failure mode (only ~40% of arb opportunities profitable net).
- **Hold-through-regime recovers it.** Smoothed funding + a dead-band with hysteresis (hold through the
  contango regime, don't flip on every sign-change) + weekly rebalance cuts turnover to **5–15×/yr**,
  cost to ~1%/yr, and lets the harvest through: **+13.5%/yr at 2.2% vol** on the full 46-name panel.
  After a realistic **7%/yr spot financing** it nets **~10%/yr at 2.2% vol, −2% maxDD** — matching the
  ~9.4% financed ROE industry anchor (CF Benchmarks). Breadth matters: on 12 names raw vol was 4.4% with
  −5 skew; across 46 names vol falls to 2.2% and **skew turns positive (+2.5)** — the single-name
  carry-crash tail diversifies away.

**Honest caveats (why this is not the portfolio's alpha engine):** the huge raw Sharpe (~4.5 at 2.2%
vol) is a **low-vol artifact** — deploying at size means leverage, and leverage reintroduces the gap
risk a smooth daily backtest understates (a *synchronised* multi-name deleveraging cascade, e.g.
Oct-2025's record liquidation, gaps every leg at once); and the harvest is **capacity- and
crowding-limited** — structural delta-neutral inventory (Ethena/BFUSD) compressed funding toward T-bills
in 2025, so forward funding is lower than this 2020–26 sample. Real and harvestable, but bounded
"infrastructure carry", not scalable edge.

## 5. Corrected edge map

| construction | Sharpe (net) | notes |
|---|---|---|
| directional single-asset carry (shipped) | **≈ +0.1** | price risk swamps funding — dead |
| **cross-sectional dollar-neutral carry (1d)** | **+1.21** (WFO +0.88) | the edge; decorrelated, incremental to momentum |
| **+ beta-hedge + inv-vol + buffer (refined)** | **+1.47** (WFO +1.40) | validated OOS; removes structural net-short beta |
| + ML timing overlay | +1.52 | risk-reduction overlay |
| delta-neutral basis (hold-through-regime, 46 names) | ~10%/yr @ 2.2% vol after financing | low-vol harvest; leverage/capacity-limited |

The original edge map's all-negative "carry" row was measuring the directional construction only. The
cross-sectional construction is where the funding premium lives.

## 6. Portfolio integration — carry as a sleeve, on its own universe (`scripts/run_carry_book.py`)

Carry is wired into the book as a distinct sleeve **with its own per-sleeve universe** — different
strategies work best on different coins, and carry's edge is in the mid-caps. The universe was chosen
empirically (robustness, not peak Sharpe) on the wide point-in-time set, **crypto-native only** (see the
hygiene note below):

| carry universe (PIT, crypto-native) | Sharpe | MC-P5 | maxDD | worst year |
|---|---|---|---|---|
| top-100 (incl. megacaps) | +1.29 | +0.67 | −23% | −0.51 (2022) |
| top-75 | +1.14 | +0.56 | −25% | −0.86 (2022) |
| **band 15–90 (mid-caps, no megacaps)** | **+1.21** | **+0.59** | **−21%** | **+0.01 (2022)** |

The **mid-cap band (liquidity rank 15–90, ~75 names)** wins on robustness: comparable Sharpe to top-100
but a **non-negative worst year**. Dropping the top-15 megacaps (funding compressed → no carry spread)
*and* the illiquid tail removes the 2022 deleveraging blowup. So carry trades a different coin set than
the trend sleeves, by design.

**Universe hygiene (crypto-native only).** The raw Binance USD-M archive includes non-crypto-native perps
that pollute a funding-carry cross-section — stablecoins (USDC, FRAX, USTC), **tokenized gold (PAXG, XAUT)**
and synthetic index perps (BTC-dominance, DeFi basket) — excluded via `carry_xs.NON_CRYPTO_NATIVE`. This is
not cosmetic: PAXG sat inside the traded band **361 days**, and because tokenized gold has low funding the
carry book *longed* it straight into a gold rally, **inflating the sleeve Sharpe to +1.33**. Purged, the
honest number is **+1.21** — the removed 0.12 was a gold beta, not a funding premium. (No tokenized equities
exist in the set — Binance's stock tokens were spot and delisted in 2021.)

**Integration.** Carry enters the master book (`scripts/run_master_book.py`) as one equal-weight leg on
this mid-cap universe. It is **~0-correlated to every other family** (funding/positioning, not price
trend), so it earns its slot on decorrelation rather than standalone Sharpe, and contributes **~5% of
book P&L** — vol-premium is the anchor, not carry. Its **1d and 4h expressions are only ~0.6-correlated**
(they catch funding-rank turns of different speed), so running both adds genuine timeframe
diversification, not redundancy. Full portfolio detail is in [REPORT.md](../../REPORT.md) §4.

## 7. Cross-asset carry — crypto vs FX vs equity

Carry is not one trade but a *family*, defined per asset class by whatever the market pays a holder:
crypto perp **funding**, an FX **short-rate differential**, an equity **dividend yield**. All three were
built on the *same* dollar-neutral cross-sectional machinery and the *same* validation (shuffled-signal
placebo distribution, per-year, skew, cost sensitivity), so they compare on one footing. Figure:
[reports/figures/carry_xasset.png](../../reports/figures/carry_xasset.png).

The comparable, vol-normalised metric is the **price-leg Sharpe** — whether the funded asset's own price
move helps or fights the carry. (Raw %/yr magnitudes are not cross-comparable: crypto alts run ~70% vol
vs FX ~10%, so only the *sign* of each %/yr leg compares; net Sharpe and price-leg Sharpe are the
apples-to-apples numbers.)

| asset (carry signal) | universe | carry accrual | price leg | price-leg Sharpe | net Sharpe | vs placebo | verdict |
|---|---|---|---|---|---|---|---|
| **Crypto** (funding) | 50 perps | +28%/yr | +40%/yr | **+0.97 (helps)** | **+1.21** | ~99th pct | **real edge** |
| **FX** (3m rates) | 12 currencies | +5%/yr | −1%/yr | −0.20 (fights) | +0.39 | **17th pct** | weak / non-edge |
| **Equity** (dividend yield) | 50 US names | +5%/yr | −21%/yr | **−0.95 (fights)** | **−0.69** | **20th pct** | negative |

**FX carry (`scripts/carry/run_carry_fx.py`).** Long high-rate / short low-rate currencies (MXN/ZAR/AUD vs
CHF/JPY), 3-month interbank rates from FRED (keyless), point-in-time-lagged. The rate accrual is a
smooth +4.5%/yr (Sharpe +64 in isolation) — but the high-yield currencies *depreciate* (FX leg
−1.2%/yr) and the book carries the classic carry-crash tail (skew −0.9; COVID −12%, 2015 EM −7.6%). Net
Sharpe +0.39 sits at the **17th percentile of 200 shuffled-rate placebos** — the rate signal does *worse*
than random currency selection. G10-only is weaker still (+0.16): DM rate dispersion collapsed under
ZIRP. This is the documented post-2008 breakdown of the FX carry trade, reproduced here.

**Equity carry (`scripts/carry/run_carry_equity.py`).** Long high-dividend / short low-dividend (utilities,
telecom, staples vs growth/tech), Twelve Data dividends → trailing-12m yield. Accrual +5.2%/yr, but
dividend-payers (a value tilt) were crushed by growth over 2012–26: price leg **−21%/yr**, net Sharpe
**−0.69**, 20th percentile of placebo. Dividend carry here is really the value factor, and it wore the
value drawdown (worst in 2020 and 2023 growth rallies; positive only in 2022).

**Why crypto is different.** In all three, ranking by carry correctly finds the high-carry assets — but
in FX and equity those are exactly the assets the market discounts (a depreciating currency, a value
trap), so carry is *anti-predictive* for price. Crypto is the exception: leveraged-retail demand keeps
funding rich beyond fair value, and high funding flags a *crowded long that mean-reverts*, so the price
leg **adds to** the funding leg instead of cancelling it. Carry is an edge only where the funded asset's
price does not arbitrage the premium away. The three books are near-uncorrelated (crypto↔FX −0.04,
crypto↔equity +0.09, FX↔equity −0.16), but only the crypto one is worth trading.

## 8. Honest limits & ceiling

- Standalone cross-sectional carry is **~0.9 Sharpe out-of-sample** — a genuine diversifying family, not a
  2.5–4.0 standalone strategy. Its value is as a **decorrelated diversifier**, realised at the book level.
- **Binding constraints:** funding-**dispersion** regime (carry is weak in a uniform bull — 2021 +0.2);
  **crowding/compression** (forward funding lower than the 2020–26 sample); **negative-skew tail** on the
  basis variant; breadth in early years (20 names in 2020 vs 50 by 2023).
- **What did not work (kept, not hidden):** directional single-asset carry; the ML ranker; naive daily
  basis; extreme quantiles (top-10%) and long lookbacks (30d), which degrade the cross-sectional book.
- **Spot's deeper history (2017+) cannot extend carry (`scripts/carry/run_carry_spot.py`).** Funding is a
  perp-only quantity that begins 2020, so the +29 months of extra spot history has nothing to harvest.
  Tested a spot-native reversal proxy for carry's price leg on 2017–2020: it is **~0-correlated to the
  real funding carry** and **negative on every window**, and the pre-2020 cross-section is only ~10 names
  (too thin). Carry is honestly 2020+. Silver lining: the ~0 proxy↔carry correlation is *evidence carry is
  a distinct signal, not repackaged price reversal* — funding carries positioning information price lacks.

## 9. Reproduce

```bash
make carry     # crypto: run_carry -> audit -> ml -> wfo -> refine -> portfolio -> basis
               # cross-asset: run_carry_fx -> run_carry_equity ; then both figures
```

Artifacts: `reports/carry/carry_results.csv` (grid), `reports/carry/carry_ml.csv`, `reports/carry/carry_wfo.csv`,
`reports/carry/carry_refine.csv`, `reports/carry/carry_basis.csv`, `reports/carry/carry_fx.csv`, `reports/carry/carry_equity.csv`,
`reports/carry/carry_headline.{parquet,csv}`, `reports/figures/carry.png`, `reports/figures/carry_xasset.png`.
Data added for cross-asset: FRED 3-month rates (`src/data/rates.py`, keyless) and Twelve Data dividends
(`src/data/twelvedata.py::load_dividends`) — both cached under `data/raw/`.
Sources for construction and external benchmarks: BIS WP1087 "Crypto Carry"; He/Manela/Ross/von Wachter
"Fundamentals of Perpetual Futures" (arXiv:2212.06888); Borri/Liu/Tsyvinski/Wu (arXiv:2510.14435);
Aperiodic/Unravel and Robot Wealth cross-sectional carry notes; Presto Research funding-alpha note;
SSRN 6701738 (ML-ranker net-of-cost failure); BitMEX "State of Crypto Perps 2025"; CF Benchmarks basis ROE.
