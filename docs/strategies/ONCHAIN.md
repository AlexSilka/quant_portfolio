# On-chain / network signals (crypto) — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.72** full / **3.77** OOS). Any master-book Sharpe quoted below (e.g. a 3.77 "book drag" baseline) is the book *snapshot at the time this family was evaluated*, not the current headline.

**Scope.** H3 from [HYPOTHESES.md](../HYPOTHESES.md): on-chain data (exchange flows, stablecoin supply,
active-address growth, chain cash flows, valuation ratios) carries information **not present in price** — the one
genuinely new information source for crypto. The data access is *gated*, so establishing exactly what is
free is part of the result — and getting that wrong the first time cost this family a correct verdict
(§1). Entitlement is read from the vendor catalog, a free ingestion loader was built, and the family was run through
the **same funnel as every other family**: dollar-neutral cross-sectional long/short, vol-target 15%,
t+2 execution, liquidity-aware costs, shuffled-signal placebo, purged/embargoed walk-forward OOS,
block-bootstrap MC, deflated Sharpe at the true trial count, correlation-to-book + lift, **plus an
orthogonality regression against price momentum and price reversal on the identical universe** (the
decisive "does on-chain beat price?" test). All numbers net of cost. Figure:
[reports/figures/onchain.png](../../reports/figures/onchain.png). Reproduce: `make onchain`.

---

## 0. TL;DR

- **Free-data on-chain is not a tradable decorrelated premium in the liquid universe reachable here.**
  It joins lottery/MAX, overnight, cross-sectional reversal and pairs in the "tested, real-as-something-
  else-or-sub-bar, not viable" pile. A rigorous edge-map entry, not a hidden gap.
- **Exchange flows are free for BTC/ETH, they were tested, and they do not rescue the thesis (§1, §5).**
  The metric the whole "coins leaving exchanges = accumulation" story rests on — `FlowInEx*`/`FlowOutEx*`
  plus exchange-held supply — is published on the Coin Metrics **community** tier for BTC and ETH, back
  to 2011/2015. Two of 33 names is a time series, never a cross-section, so it is tested as a BTC/ETH
  timing overlay. Best result, BTC exchange-supply-trend long/flat, nets **+0.96** against buy-and-hold
  **+0.85** — and that gap is **beta, not timing**: rotating the *same* position path to random dates
  gives a mean of +0.38 and a **95th percentile of +1.01**, above the real +0.96. Not one of the four
  flow overlays clears its own random-timing bar, and 7 of 8 predictive regressions show no forecasting
  power (only BTC net-flow → 7d return, t=−2.04, direction-correct, and one hit in eight is what chance
  pays). **Free exchange-flow data is now a tested negative rather than an untested excuse.**
- **What is genuinely Pro-walled is narrower than the flow story:** adjusted transfer value (the NVT
  numerator), realized cap, USD fees, miner revenue, entity-adjusted supply and address-balance bands.
  Realized cap is recoverable anyway as market cap ÷ MVRV. Free and used here: activity, holders,
  supply, market cap, MVRV, issuance (31 names), native-unit fees (14), plus BTC transaction-value and
  miner revenue from blockchain.com.
- **The top-N answer is a hard data ceiling: top-10/20/30 are testable, top-50/100 are not.**
  The newer high-liquidity chains a top-100 would need (**SOL, SUI, TON, APT, SEI, TIA…**) carry market
  data only on the community tier — no network metrics at any price short of Pro. Stated, not skipped.
- **Four of the original 37 names were measuring dead ERC-20 shells and are now excluded (§1).**
  VET, ZIL, QTUM and LRC report **zero active addresses on 44-79% of days** — holders migrated to native
  chains whose on-chain data is not free — and a day when a dead shell shows one address makes market-cap-
  per-address astronomical, parking the name at the expensive extreme of the value rank on a pure
  artifact. Excluding them cuts the universe to **33** and cuts the headline value book from +0.40 to
  **+0.15**: a third of the old headline was that artifact.
- **The a-priori headline — on-chain value (market cap per active address, "CVALUE"), top-20 — nets
  Sharpe +0.15 and fails every out-of-sample gate:** MC-P5 **−0.52**, placebo **72nd percentile**
  (shuffle p95 **+0.63** > real), deflated Sharpe **0.07** at N=36 trials. It is a **static coin-type
  tilt** (turnover 0.03/bar, most names never flip side), and on clean data it is not even positive
  enough to argue about.
- **One signal does survive the clean-up, and it is adoption momentum, not value.** Active-address growth
  (`adr_mom30`) top-20 nets **+0.73** with **MC-P5 +0.08**, placebo **98th percentile** (p95 +0.54),
  walk-forward OOS **+0.74** with the construction held fixed, and alpha over price momentum + reversal
  of **t=+2.04**. It is stable across every top-N (+0.71…+0.76), which is what separates it from
  `divergence` (+0.80 at N=20 but +0.24/+0.34/+0.39 elsewhere — a single-cell spike). **It is reported as
  a post-hoc candidate, not a discovery:** it surfaced only after the dead-shell fix, its deflated Sharpe
  at the family's true 36-trial count is **0.50**, and it is **+0.32 correlated with price momentum** —
  a better-built momentum, in the same sense residual momentum was ([RESIDMOM.md](RESIDMOM.md)).
- **And it still does not earn a slot: the book does not move.** Correlation to the deliverable book
  +0.13, and on the **full scorecard** — not Sharpe alone, which was never what binds this book — a 15%
  weight reads Sharpe 3.71 → 3.70, **CAGR 45.6% → 39.3%**, max-DD −8.3% → **−6.8%**, worst month
  −5.76% → **−4.51%**, months-in-profit 81% → **77%**: **5/5 → 4/5**. The ratio holds while the money
  falls by six points — which is exactly what a Sharpe-only reading cannot show. The tail improvement is real arithmetic and it is not
  the signal's doing. Rotating the same sleeve to random dates — identical vol, skew and
  autocorrelation, alignment destroyed — gives Sharpe 3.78, max-DD −6.9%, worst month −4.87%, months
  **81%** and keeps **5/5**. The control beats the real thing on every axis, so what looks like a tail
  hedge is dilution, and the signal's actual timing costs four points of months-in-profit.
- **ML changes nothing — no non-linear on-chain alpha either (§4b).** A 21-trial ML ranker ({ridge, RF,
  extra-trees, hist-GBM, LightGBM} + classifiers × {on-chain / price / both} features × {5,21,63}d horizons ×
  top-N, all purged-CV OOS), now including holders, issuance and fee-yield features: ML on **on-chain**
  features fails (best **+0.32**, most ≤ 0); the *same harness* on **price** features finds a strong edge
  (**+1.09**) — so the failure is the data, not the method — and appending on-chain to price degrades it.
  Best on-chain-ML alpha over price is t=+0.80.
- **The other BTC/ETH overlays remain worse than holding.** MVRV-z long/flat **+0.07** (long/short −0.99),
  NVT +0.18, Puell −0.49, stablecoin-SSR +0.55, against BTC buy-and-hold +0.85 — and with 2-3 cycles of
  history the whole arm is near-unbacktestable regardless (§5).
- **The chains Coin Metrics cannot see were tested too, on cash flows instead of addresses (§5b).**
  DefiLlama gives free daily fees/revenue/TVL for 28 chains — SOL, SUI, TON, APT, SEI, TIA, ARB, OP —
  and Coin Metrics supplies the market cap, so a real fee yield is computable. **Crypto chain-value is
  inverted:** buying the chains that look cheap on their own cash flows nets **−0.82**, and the placebo
  sits at the **6th percentile**, meaning the cross-section is genuinely informative with the sign
  reversed — the same thing crypto did to the lottery factor. Fee yield ranks Bitcoin permanently
  expensive and the L2s permanently cheap at 0.014 turnover/bar, so it is a standing structural tilt,
  not a valuation that closes. The post-hoc flip (+0.86) is not the BTC-dominance trade (hedging that
  spread leaves +0.81) but fails everything else: placebo 94th < 95th, deflated 0.54, alpha over price
  t=+1.66, all P&L in the last two years, and on the full scorecard a 30% weight scores 4/5 against its
  rotated control's 5/5 — same dilution, same verdict. **Excluded.**
- **Decorrelated (corr to book +0.06) but it drags** (blended Sharpe 3.83 → 2.73 as weight rises).
  Correctly **excluded**. Documented, not traded.

---

## 1. The data gate — what is free, what is pay-walled, and why it matters

H3 is the only hypothesis needing a new ingestion loader, so the free-access probe *is* part of the
result — and the first version of that probe got the answer wrong, in the direction that flattered the
verdict. **Entitlement is now read from the vendor's own catalog** (`catalog-v2/asset-metrics` publishes a
`community: true` flag per asset × metric), not inferred from HTTP 403s. The distinction is not academic:
a multi-metric request 403s in its entirety if *any* single metric is Pro, so 403-inference blacklists
free metrics for the crime of sharing a call with a paid one. That is exactly what happened to exchange
flows, which were recorded as pay-walled and are not.

| source | free & used | genuinely Pro-walled |
|---|---|---|
| **Coin Metrics community** v4 | PriceUSD, CapMrktCurUSD, **AdrActCnt**, TxCnt, TxTfrCnt, **AdrBalCnt**, SplyCur, **CapMVRVCur** (all ~33 names); **IssTotNtv/USD** (31); **FeeTotNtv** (14); BlkCnt (16); HashRate (8); **FlowInEx\*, FlowOutEx\*, SplyEx\*** (**BTC + ETH only**) | TxTfrValAdj\* (NVT numerator), CapRealUSD, FeeTotUSD, RevUSD, SplyAct1yr, AdrBalUSD\*Cnt, NVTAdj — **0 assets free** |
| **blockchain.com** charts (BTC only) | estimated-transaction-volume-usd, miners-revenue, hash-rate | — (BTC only; no altcoins) |

So the sharpest signal the thesis names — **exchange inflow = sell pressure** — *is* free, for the two
assets that matter most, with 15 and 11 years of history. It is tested in §5 as a time-series overlay
(two names cannot form a cross-section) and it does not survive. What remains behind the Pro wall is the
transfer-value / realized-cap / entity-adjusted block; realized cap is recoverable regardless as
`CapMrktCurUSD ÷ CapMVRVCur`, so the only irreducible gaps are adjusted transfer value and entity-level
supply bands.

**Universe — 33 names, after excluding four dead ERC-20 shells.** The tradable set is
`free-CM-coverage ∩ repo crypto price panel`, 2020-2026, liquidity-ranked. On-chain `PriceUSD` daily
returns correlate **≥0.95** with the repo's Binance returns for every name (only KNC at 0.949, a known
token-migration artifact), so the coin→asset mapping is sound. But mapping-correct is not
measurement-correct: **VET, ZIL, QTUM and LRC are only free as their bridged ERC-20 contracts**, and
holders left those contracts for the native chains years ago — they post **zero active addresses on 48%,
44%, 79% and 64% of days**. The native asset codes exist on Coin Metrics but carry market data only, so
there is no mapping fix. They are screened out mechanically (`live_universe()`: >20% zero-activity days),
because a shell that shows one address on a stray day makes market-cap-per-address astronomical and
parks the name at the expensive extreme of the value rank on nothing at all. Removing them takes the
headline value book from +0.40 to **+0.15** — the deleted third was the artifact. Residual zero-count
days elsewhere (134 in AdrActCnt, 137 in TxCnt) are treated as missing rather than as zeros, since a live
chain with no active addresses is an indexing outage.

Coverage after the screen: AdrActCnt 33/33, MVRV 32, holders 32, issuance 31, fees 14, exchange flows 2.
**Top-N ceiling = 33**; a top-50/100 cross-section is impossible on free data because the newer
high-liquidity chains (SOL, SUI, TON, APT, SEI, TIA, …) publish market data but no network metrics below
the Pro tier — a genuine constraint, not a shortcut.

## 2. Construction — the on-chain signal family

Thirteen signals (each declared before fitting, all reported, never peak-picked) in
`src/sleeves/onchain.py`, each swapped into `src/sleeves/xsect.py::xs_backtest` — the identical
dollar-neutral / liquidity-aware-cost / vol-target engine the momentum, carry and BAB books use, so the
numbers are directly comparable. Daily counts are 7-day-smoothed first (kills weekday seasonality), and
every transform is trailing (no look-ahead; the engine adds the t+2 delay). Two economic angles:

```
ADOPTION MOMENTUM   adr_mom30, tx_mom30   log growth of active-addresses / tx-count over 30d  → long fast-growing
ON-CHAIN VALUE      nvm_val               −(market cap / active addresses)   = price-per-user, long cheap  ← HEADLINE
                    nvm_z_val             self-relative (per-asset 365d z-score of log NVM) — removes the coin fixed-effect
                    metcalfe_val          −(market cap / active-addresses²)  (Metcalfe fair value ∝ users²)
                    mvrv_val / mvrv_z_val −MVRV level / self-relative z       (price vs realized cost basis)
OWNERSHIP           holder_mom30          log growth of non-zero-balance addresses — accumulation breadth (a stock, not a flow)
                    holder_val            −(market cap / holders) = price-per-owner, the value twin built on ownership
DILUTION            low_inflation         −(annualised issuance ÷ supply) — long the names paying holders least dilution
NETWORK EARNINGS    fee_yield_val         fees × price ÷ market cap = the crypto earnings yield (14-name subset)
DIVERGENCE          divergence            z(address growth) − z(price return): activity outrunning price (orthogonal by build)
EXCHANGE FLOW       exchange_netflow_z    (inflow − outflow) ÷ exchange balance, z-scored — BTC/ETH time series only (§5)
                    exchange_supply_trend 30d change in the share of supply held on exchanges — the stock counterpart
```

**Headline = `nvm_val`** — chosen a-priori on the *literature*, not the fit: the one on-chain factor with
hard academic backing is cross-sectional on-chain **value**, market cap per active/new address, LOW = cheap
= long (Liu-Tsyvinski-Wu; Cong et al.). `top-20` is the round mid-cross-section (deliberately **not** the
sweep-maximal N=30), so the headline is not peak-picked; the full top-N × signal surface is in §3.

## 3. Results — a static value tilt that dies out-of-sample

**Top-N × signal sweep** (net Sharpe, monthly rebalance) — the top-10/20/30 question, on the clean
33-name universe:

| signal | N=10 | N=20 | N=30 | N=33 |
|---|---|---|---|---|
| **adr_mom30 (adoption mom)** | **+0.76** | **+0.73** | **+0.71** | **+0.74** |
| tx_mom30 | +0.35 | +0.55 | +0.70 | +0.62 |
| **nvm_val (value, headline)** | −0.11 | **+0.15** | +0.37 | +0.18 |
| nvm_z_val (self-relative) | +0.08 | +0.10 | +0.44 | +0.36 |
| metcalfe_val | −0.14 | +0.14 | +0.42 | +0.24 |
| mvrv_val / mvrv_z_val | −0.45 / −0.35 | −0.15 / −0.48 | −0.22 / −0.13 | −0.24 / −0.07 |
| holder_mom30 / holder_val | −0.22 / +0.07 | +0.07 / −0.20 | +0.37 / −0.08 | +0.35 / −0.14 |
| low_inflation | −0.50 | −0.27 | −0.33 | −0.26 |
| fee_yield_val (14 names) | −0.07 | −0.10 | +0.08 | +0.09 |
| divergence | +0.24 | +0.80 | +0.34 | +0.39 |
| blend (value+divergence) | −0.08 | +0.11 | −0.05 | +0.06 |

**Stability across N is the discriminator here, and only one signal has it.** `adr_mom30` sits in a
+0.71…+0.76 band at every breadth; every other positive is a single cell — `divergence` posts +0.80 at
N=20 and +0.24/+0.34/+0.39 either side of it, which is the shape of noise, not of an effect. The value
family is sign-unstable and weak throughout, and the three new axes fail outright: ownership adds nothing
over activity, **low-inflation is consistently negative** (the high-issuance names outperformed — dilution
lost to momentum over this window, the same collision that inverted the lottery factor in
[LOTTERY.md](LOTTERY.md)), and fee yield is flat on the 14 names that have fees.

**The a-priori headline `nvm_val` top-20 monthly** nets:

| metric | value |
|---|---|
| net Sharpe | **+0.15** |
| MC [P5, P50] | **[−0.52, +0.18]** |
| max drawdown | −34% |
| turnover / bar | **0.03** |
| per-year | 2020 −1.41 · 2021 −0.04 · 2022 +0.61 · 2023 −0.46 · 2024 +0.24 · 2025 **+1.55** · 2026 +0.77 |

Gates, all failed:

- **Placebo (200 name-shuffles):** real +0.15 sits at the **72nd percentile** (shuffle mean −0.01,
  p95 **+0.63**). A random re-assignment of the same signal to different names clears +0.15 routinely.
- **MC-P5 −0.52 < 0** — the block-bootstrap 5th percentile is deeply negative.
- **Deflated Sharpe 0.07** at the family's N=36-trial count.

The family-level walk-forward now reads **in-sample best +0.80 → OOS +0.78** (9 refits), a sign flip from
the −0.64 this document previously reported. That is not the value book improving: with the dead shells
gone the walk-forward pool simply stops selecting broken value configurations and starts selecting
`adr_mom30`, whose OOS behaviour is genuine. The honest way to read it is that the *pool* contains one
signal that works, which is the §3b finding — the headline itself is still dead.

**Why it fails — it is a static tilt, not a signal.** Turnover **0.03/bar**, and the median name **never
changes side** (100% side-persistence; 21 of 32 names never flip). The book is permanently:

```
LONG  (cheap per address):  LTC +0.17, ETC +0.16, BCH +0.14, DOGE +0.10, XLM +0.07, ALGO +0.07   ← old PoW coins
SHORT (rich per address):   LINK −0.19, UNI −0.16, SHIB −0.11, XRP −0.10, AAVE −0.07, LDO −0.06   ← newer tokens
```

Market-cap-per-active-address structurally separates legacy PoW coins (many cheap on-chain transfers per
dollar of cap) from smart-contract-era tokens (high cap on a smaller address base) — a **fixed coin
characteristic**, not a time-varying mispricing. So ranking on it is ~ranking on coin generation. What is
left after the shell fix is a bet on old-coin-vs-new-coin relative performance over 2020-2026 (positive in
2025, negative in 2020/2023) dressed as on-chain value — which is exactly why the placebo cannot
distinguish it. The **self-relative** form (`nvm_z_val`, which removes the coin fixed-effect) is no better
(+0.10), and its z-window is unstable (z180 +0.63, z365 +0.10, z540 −0.09) — the shorter the window the
better, the fingerprint of fitting noise. **Cost is not the binding constraint** (turnover is tiny; +0.15
at 1×, +0.12 at 3×) — the book is simply far below the 0.5 bar.

### 3b. The one signal that survives — adoption momentum, reported as post-hoc

`adr_mom30` (30-day growth in active addresses, 7d-smoothed, top-20, monthly) is the only member of the
family that clears the repo's robustness gates:

| gate | result | bar |
|---|---|---|
| net Sharpe | **+0.73** | — |
| MC-P5 (block bootstrap) | **+0.08** | > 0 ✅ |
| placebo (200 name-shuffles) | **98th pctile** (p95 +0.54) | > 95th ✅ |
| walk-forward OOS, construction held fixed | **+0.74** (9 refits) | > 0 ✅ |
| alpha over price mom + reversal | **t = +2.04** | > 2 ✅ |
| deflated Sharpe at the family's 36 trials | **0.50** | ~0.9 ❌ |
| per-year | 2020 +0.47 · 2021 +0.76 · 2022 +0.99 · 2023 +0.35 · 2024 **+1.92** · 2025 +1.27 · 2026 −1.78 | 6 of 7 positive |
| **book at 15% weight (full scorecard)** | Sharpe 3.71→3.70, DD −8.3%→−6.8%, worst −5.76%→−4.51%, months 81%→77% — **5/5 → 4/5**, and its rotated control keeps 5/5 | ❌ |

Three things keep this a finding rather than a sleeve. **It is post-hoc** — it became interesting only
after the dead-shell fix, and the walk-forward is scored with the construction held fixed precisely so
the pool is not credited for re-discovering it (the lesson recorded in [BAB.md](BAB.md)). **It is
momentum** — +0.32 correlated with the price-momentum book on the identical universe, so the t=+2.04
alpha buys a better-built version of a premium the book already runs, in the same way residual momentum
did ([RESIDMOM.md](RESIDMOM.md)). And **the portfolio does not move**: +0.003 Sharpe at a 15% weight is
noise, and 30% costs 0.15. The 2026 partial-year −1.78 is worth naming too — the newest data is the worst
data for it.

The correct summary is that free on-chain contains one honest, modest, momentum-shaped signal, and the
book already owns that premium through cheaper channels.

**And it does not upgrade that channel either** (`scripts/onchain/run_onchain_blend.py`). A signal that
cannot carry a sleeve can still be worth something *inside* one — residual momentum earned its place that
way — so `adr_mom30` was rank-blended into the crypto x-sect sleeve's own idiosyncratic-momentum signal,
on the 33 names with coverage, at three doses declared in advance:

| dose | blended | shuffled-name control (mean / p95) | percentile | |
|---|---|---|---|---|
| 0.10 | +0.626 | +0.601 / +0.673 | 72nd | inside the noise |
| 0.25 | +0.516 | +0.548 / +0.631 | 30th | inside the noise |
| 0.50 | +0.485 | +0.422 / +0.606 | 72nd | inside the noise |

against a raw sleeve of **+0.573**. Not one dose clears its own control, and the control is what makes the
table readable: at a 0.10 dose the blend nets **+0.626**, comfortably above the raw sleeve — and so does
the *shuffled* arm, at **+0.601**. Perturbing the ranks of 33 of 300 names lifts this sleeve on its own.
Read without the control arm, that cell is a +0.05 upgrade; read with it, there is nothing there.

**Timeframe.** On-chain metrics are **daily aggregates** — there is no free intraday on-chain (and block-
level counts are daily by nature), so the 5m/1h/4h axis the price sleeves sweep does not exist here. The
holding-period axis was swept instead (rebal ∈ {1,7,21,63} bars): daily +0.21, weekly +0.16, monthly
+0.15, quarterly +0.04; a true weekly-resampled book nets +0.29. The edge, such as it is, lives at the
daily-to-monthly horizon and decays by the quarter — consistent with a slow characteristic tilt, not a
timing signal.

## 4. The decisive test — does on-chain add edge *over price*?

The survive/kill bar is "adds decorrelated OOS edge **over price-only sleeves**". Each on-chain book
was regressed on a price-momentum book (`mom 30d`) and a price-reversal book (`−mom 12m`, the price
"value" proxy) built through the *same* engine on the *identical* 33 names — alpha = the intercept:

| on-chain book | α (ann) | **α t-stat** | corr price-mom | corr price-rev | verdict |
|---|---|---|---|---|---|
| **adr_mom30 (adoption mom)** | **+0.122** | **+2.04** | **+0.32** | +0.03 | clears the bar — and it *is* momentum |
| **divergence** | **+0.126** | **+2.55** | **−0.66** | −0.04 | clears on t, fails on stability (§3) |
| nvm_val (value, headline) | +0.026 | +0.42 | +0.07 | −0.12 | subsumed by price |
| nvm_z_val | +0.014 | +0.24 | −0.22 | +0.30 | subsumed by price |
| mvrv_val / mvrv_z_val | −0.027 / −0.087 | −0.48 / −1.42 | −0.24 / −0.39 | +0.32 / +0.12 | subsumed / negative |
| holder_mom30 / holder_val | +0.010 / −0.032 | +0.16 / −0.51 | −0.07 / +0.04 | +0.02 / +0.03 | subsumed by price |
| low_inflation | −0.047 | −0.67 | +0.06 | +0.05 | negative |
| fee_yield_val | −0.020 | −0.28 | −0.07 | −0.14 | subsumed by price |
| blend | +0.012 | +0.21 | −0.50 | +0.15 | subsumed by price |

**Two signals now clear t > 2 — and the earlier version of this document, run on a universe containing
four dead ERC-20 shells, wrongly reported that none did.** Adoption momentum (t=+2.04) is the real one,
and its +0.32 price-momentum correlation says plainly what it is: a better-built momentum, not a new
premium. `divergence` posts the higher t (+2.55) but is built as z(address growth) − z(price return), so
its −0.66 correlation to the price-momentum book means the regression is largely hedging a short-momentum
position against a long-momentum control; combined with a Sharpe that exists only at N=20, it does not
earn a claim. Note also the multiplicity: these are 11 tests, and a Bonferroni-honest bar here is
|t| ≈ 2.8, which neither clears. The value axis is subsumed exactly as published: the
crypto factor zoo collapses to market + size + momentum, with on-chain *value* the one separately-priced
addition — and that value premium is documented on **4,000+ coins including the illiquid small-cap tail,
using price-to-new-address**. The free, liquid, active-address universe reachable here is precisely the
wrong universe for it: it excludes the small caps where the premium concentrates and must use active
(not new) addresses. So the honest reconciliation mirrors lottery/MAX — **the documented premium lives in
the illiquid tail the tradable funnel excludes; in the liquid universe it degenerates to a static
coin-type tilt with no OOS edge and no alpha over price.**

## 4b. ML — does a model find non-linear on-chain alpha the linear books missed? No

The §4 orthogonality test is linear, so the honest follow-up is whether a model finds a non-linear
on-chain→forward-return relationship. Ran a full **ML ranker** grid (`scripts/onchain/run_onchain_ml.py`, reusing
the repo's leakage-controlled harness `src/sleeves/xsect_ml.py`): a model predicts each name's
cross-sectionally-demeaned forward return from a feature panel, long the top / short the bottom, through
the *same* dollar-neutral book. **21 trials** = {ridge, random-forest, extra-trees, hist-GBM, LightGBM}
regressors + {logistic, RF, LightGBM} classifiers, × {on-chain / price / both} feature sets, × {5d, 21d,
63d} horizons × {top-20, top-37} — every one **out-of-sample** under purged/embargoed expanding CV
(embargo ≥ the forward horizon, so overlapping targets cannot leak). Net OOS Sharpe (21d, top-20):

The feature panel now carries the newly-available metrics too — holder-count momentum, holders-per-active
address, cap-per-holder, measured issuance and fee yield — so the model gets every free on-chain axis,
not just activity and valuation. Net OOS Sharpe (21d, top-20):

| model | on-chain features | **price features** | both |
|---|---|---|---|
| ridge | −0.11 | +0.03 | −0.45 |
| random-forest | −0.34 | +0.47 | −0.07 |
| extra-trees | +0.18 | **+1.09** | +0.23 |
| hist-GBM | −1.05 | +0.86 | +0.15 |
| LightGBM | −1.03 | +0.63 | −0.28 |

Three findings, all pointing the same way:

- **ML on on-chain features fails** — best **+0.32** (LightGBM classifier), most ≤ 0 and several deeply
  negative (hist-GBM −1.05, LightGBM −1.03); every other-horizon/other-N robustness cell is negative.
  No model beats the linear on-chain baseline, and none clears the bar OOS.
- **The harness works — on PRICE features it finds a strong edge** (+0.47 to **+1.09**). This is the key
  control: the pipeline is not broken or over-conservative; it recovers the known price cross-sectional
  premium. The on-chain failure is the **data**, not the method.
- **Adding on-chain to price DEGRADES it** — extra-trees +1.09 → +0.23, LightGBM +0.63 → −0.28 when the
  on-chain block is appended. On-chain features are **noise to the model**: they add nothing and cost
  generalisation.

**Decisive, again:** best on-chain-ML **+0.32** vs best price-ML **+1.09** → on-chain features **do not
beat** price. (That on-chain figure is now scored across regressors *and* classifiers; an earlier version
of this comparison quietly excluded the classifier arm from the on-chain side, understating it. The
asymmetry runs the other way — classifiers were only ever run on on-chain features — so the comparison is
generous to on-chain and it still loses.) The best on-chain-ML book's alpha over price momentum+reversal
is **t=+0.80** (no edge — same verdict as the linear §4). Feature importance on the combined model does
put on-chain features at 58% of total gain, led by nvm_z, fee_yield_90 and holder_mom_90 — but the models
that *use* them lose to the models that do not, so importance-within-a-losing-model is not evidence of
signal. Even the best *price*-ML book fails the multiple-testing haircut (MC-P5 **−0.31**, deflated
Sharpe **0.02** at N=21 trials) — it is the selection-inflated winner of 21 tries, not a robust sleeve.

**Meta-gate** (an ML model predicts P(the value book is up next month) from
panel-regime state and gates exposure) does what it does everywhere in this repo — **buys risk reduction**:
the value book's −34% max-DD falls to **−15.6%** and Sharpe rises +0.15 → +0.29, in market only 28% of
the time. On a book this weak that is a statement about the −34% drawdown, not about alpha; halving the
exposure of something with no edge mostly halves the damage. Consistent with the repo's standing result
("ML buys risk reduction and precision, not out-of-sample alpha", [docs/TREND.md](TREND.md)).

**So ML changes nothing: there is no on-chain alpha, linear or non-linear.** Figure:
[reports/figures/onchain_ml.png](../../reports/figures/onchain_ml.png); `scripts/onchain/run_onchain_ml.py`.

## 5. Time-series overlays (BTC/ETH) — regime context, not backtestable alpha

The classic on-chain *timing* indicators (MVRV z-score, NVT, Puell, stablecoin-SSR) plus the two
**exchange-flow** series are BTC/ETH macro overlays with only **2-3 non-overlapping cycles** of history —
near-unbacktestable by construction. Long/flat (risk-on when the metric is below its own trailing mean),
t+2, vs vol-targeted buy-and-hold (BTC +0.85, ETH +0.91):

| overlay | Sharpe | maxDD | | overlay | Sharpe | maxDD |
|---|---|---|---|---|---|---|
| **BTC buy-and-hold** | **+0.85** | — | | **BTC exch-supply-trend long/flat** | **+0.96** | −20% |
| BTC MVRV-z long/flat | +0.07 | −32% | | BTC exch-supply-trend long/short | +0.31 | −37% |
| BTC MVRV-z long/short | −0.99 | −74% | | **ETH exch-netflow long/flat** | +0.78 | −25% |
| ETH MVRV-z long/flat | +0.17 | −36% | | ETH exch-netflow long/short | +0.26 | −27% |
| BTC NVT long/flat | +0.18 | −45% | | BTC exch-netflow long/flat | +0.60 | −24% |
| BTC Puell long/flat | −0.49 | −48% | | BTC exch-netflow long/short | +0.41 | −30% |
| BTC stablecoin-SSR growth | +0.55 | −13% | | ETH exch-supply-trend long/flat | +0.65 | −28% |

The exchange-flow overlays are the best-looking rows in the table, and one of them — BTC
exchange-supply-trend — **beats buy-and-hold** (+0.96 vs +0.85) with less than two-thirds of the
drawdown. That is exactly the result the thesis predicts, and it is not real.

**The control that kills it: random timing of the same position path.** A long/flat rule is invested only
part of the time (here 25-29% on average), so in a market that rose it collects a slice of beta by
construction and its Sharpe says more about the slice than about the timing. Rotating the realised
position path by a random offset preserves average exposure, switching frequency and on/off persistence
exactly, and destroys only the alignment with returns — 500 rotations per overlay:

| overlay | real | random-timing mean | random-timing p95 | percentile | avg exposure |
|---|---|---|---|---|---|
| BTC exch-supply-trend long/flat | +0.96 | +0.38 | **+1.01** | 93rd | 0.29 |
| ETH exch-netflow long/flat | +0.78 | +0.33 | **+0.87** | 89th | 0.25 |
| BTC exch-netflow long/flat | +0.60 | +0.34 | +0.98 | 78th | 0.27 |
| ETH exch-supply-trend long/flat | +0.65 | +0.39 | +0.99 | 77th | 0.26 |
| BTC MVRV-z long/flat (reference) | +0.07 | +0.42 | +1.17 | 28th | 0.37 |

**Not one clears its own 95th percentile.** Random dates, same exposure, beat the actual signal more than
5% of the time in every case — and with four overlays tried, a best-of at the 93rd percentile is what
noise pays. The +0.96 was beta wearing a flow signal's clothes.

**The prior question, asked directly: does the flow series forecast returns at all?** Newey-West
regressions of forward return on the flow z (overlapping windows, lag = horizon):

| series | 7d fwd | 30d fwd |
|---|---|---|
| BTC net-flow | β −0.0062, **t = −2.04** | β +0.0013, t = +0.11 |
| BTC exchange-supply trend | β −0.0002, t = −0.06 | β −0.0008, t = −0.05 |
| ETH net-flow | β −0.0006, t = −0.17 | β −0.0107, t = −0.86 |
| ETH exchange-supply trend | β −0.0054, t = −1.39 | β +0.0093, t = +0.46 |

One of eight crosses |t| = 2, in the theorised direction (inflows to exchanges precede weakness) — and
one in eight at the 5% level is precisely the chance rate. Note also which one: the *net-flow* series at
7 days, while the exchange-supply-trend series that produced the headline +0.96 Sharpe has **t = −0.06**,
no forecasting content whatsoever. A trading rule that scores well on a signal with zero predictive power
is a rule that is not trading the signal.

**So the exchange-flow thesis is now tested rather than deferred.** The data was free the whole time; the
answer is that on BTC and ETH daily aggregates it does not beat holding the asset once the beta is
controlled for. What paid vendors sell on top of this is *entity-level* labelling — which wallets, which
exchange, whale versus retail — and that remains untested here.

**Every non-flow overlay still underperforms simply holding BTC.** The best, stablecoin-SSR-growth (+0.55,
and a tidy −13% max-DD from sitting out risk-off), rests on one structural stablecoin-supply uptrend.
MVRV long/short (−0.99) is actively destructive — fading "expensive" MVRV shorts the biggest bull legs.

## 5b. Chain fundamentals — the other half of crypto, valued on cash flows

Everything above runs on Coin Metrics' free network data, which covers 33 mostly-legacy coins and has
**no network metrics at all** for the chains that carry modern activity. That is a real hole in a
verdict about "on-chain": SOL, SUI, TON, APT, SEI, TIA, ARB and OP are simply absent. **DefiLlama
fills it** — free, no key, daily **fees, revenue and TVL** per chain, Ethereum back to 2018 — and
market cap comes from Coin Metrics (`CapMrktEstUSD` is free for 27 of the 28 mapped chains, TON being
the exception), so these are real valuation ratios rather than raw counts.

This is a different economic axis from anything above: not *how busy* a chain looks, but **what it
earns**. `src/data/defillama.py`, `src/sleeves/fundamentals.py`, `scripts/onchain/run_fundamentals.py`.

```
VALUE     fee_yield    annualised fees ÷ market cap      ← HEADLINE (the crypto earnings yield, inverse P/F)
          rev_yield    annualised revenue ÷ market cap   (the slice that accrues to the token)
          tvl_yield    TVL ÷ market cap                  (price-to-book)
GROWTH    fee_growth   trailing-quarter fees vs the quarter before — contains no price at all
          tvl_growth   capital arriving on the chain
QUALITY   fee_margin   fees ÷ TVL — what the parked capital actually does
```

**Two caveats stated before the numbers.** The panel is young and narrow: fees for the modern chains
begin 2022-2024, so it runs **2022-06 → 2026-07, 27 chains, breadth growing 10 → 26**. And DefiLlama
*backfills* protocol adapters, so history is revised — growth signals are the most exposed, and a
positive result here is an upper bound. Tezos is dropped automatically: its fee series stops in
Jan-2025, and a stalled series held forward becomes a silent constant tilt — the same failure the
dead ERC-20 shells caused in §1.

**The result is a clean inversion.** Net Sharpe, monthly rebalance:

| signal | N=10 | N=15 | N=20 | all |
|---|---|---|---|---|
| **fee_yield (headline)** | +0.55 | +0.42 | −0.14 | **−0.82** |
| rev_yield | +0.03 | +0.08 | −0.59 | −0.62 |
| tvl_yield | +0.31 | +0.28 | −0.19 | −0.60 |
| fee_growth | +0.06 | +0.03 | +0.19 | −0.29 |
| tvl_growth | −0.19 | −0.16 | +0.39 | +0.34 |
| fee_margin | +0.16 | −0.21 | −0.41 | −0.72 |
| value_blend | +0.38 | +0.27 | −0.35 | −0.97 |

Buying the chains that look cheap on their own cash flows **loses money**: the headline nets **−0.82**
(MC-P5 −1.70, deflated 0.00), and every valuation ratio agrees. Critically the placebo sits at the
**6th percentile** — the real cross-section is *worse* than 94% of random name-shuffles, which means
the structure carries genuine information with the sign reversed. This is crypto doing to value what
it already did to the lottery factor ([LOTTERY.md](LOTTERY.md)): the textbook direction is inverted.

**Why the sign flips is visible in the legs.** Fee yield ranks **Bitcoin permanently expensive** — it
collects almost nothing in fees against a cap in the trillions — and the high-throughput L2s
permanently cheap. Mean percentile rank over 2025-26: long ARB (1.00), SOL (0.90), POL (0.90), Sonic
(0.90), OP (0.84); short BTC (0.05), TIA (0.11), FIL (0.17), LTC (0.18), XLM (0.24). Turnover is
**0.014/bar** — the legs essentially never move. So this is not a valuation signal that opens and
closes; it is a standing structural bet on high-fee chains against Bitcoin and the legacy L1s, and
over 2025-26 that bet was on the wrong side.

**The inversion, measured rather than asserted.** Flipping a sign after seeing the result is post-hoc,
so the flip goes through the same funnel and stays labelled:

| gate | inverted book (long "expensive" chains) | bar |
|---|---|---|
| net Sharpe | **+0.86** | — |
| MC-P5 | +0.01 | > 0, barely ✅ |
| placebo | **94th pctile** (p95 +0.90) | > 95th ❌ |
| walk-forward OOS | +0.33 | > 0 ✅ |
| deflated Sharpe (N=14) | **0.54** | ~0.9 ❌ |
| alpha over price mom + reversal | t = +1.66 | > 2 ❌ |
| per-year | 2022 −0.96 · 2023 +0.17 · 2024 +0.03 · 2025 **+2.01** · 2026 **+2.81** | — |
| **book at 15% (full scorecard)** | 4/5, against a rotated control that also scores 4/5 | ❌ |

The obvious suspicion — that "long expensive chains" is just the BTC-dominance trade — was tested and
**does not hold**: correlation to the BTC-minus-equal-weight-alts spread is +0.25, and hedging that
spread out leaves **+0.81 of the +0.86**. (Hedging means removing β×spread. Subtracting the fitted
intercept as well would zero the mean by construction and "prove" any book is explained by anything.)

So the inversion is not a beta artifact — but it fails on every other count: it does not beat its own
placebo, its deflated Sharpe is 0.54 against a ~0.9 bar, its alpha over price is t=+1.66, and all of
the P&L is in the last two years of a panel that only starts in 2022. Trading it would mean trading a
sign flip discovered after the fact, on 27 chains, four years, and revised data. **Excluded, and the
value here is the map: crypto chain-value is inverted, the effect is real enough to be measurable, and
neither direction moves the book.**

## 6. Cross-asset — why this is crypto-only (the stocks / FX question)

On-chain data is **intrinsically crypto**: it is the public-ledger record of a blockchain. **Equities and
FX have no on-chain ledger**, so the H3 signal has no literal analogue there — testing "on-chain stocks"
is not a meaningful backtest, it is a category error. The nearest *conceptual* analogue is
**flow / positioning** data, a different data class:

- **Equities:** fund flows (EPFR), ETF creation/redemption, short interest, 13F holdings, insider &
  options flow. All either non-free, low-frequency, or not point-in-time — none in the repo.
- **FX:** CFTC Commitments-of-Traders positioning, TIC capital flows, central-bank reserve flows. Weekly
  at best, heavily revised — a positioning signal, not a network signal.

These are worth their own hypothesis (a "flow/positioning" family), but they are **not H3** and not free
point-in-time here, so they are named as the honest analogue and left out of scope rather than faked.
The cross-asset breadth H3 was hoped to add is therefore structurally unavailable on free data.

## 7. Portfolio value — decorrelated but it drags

- **Correlation to the deliverable book: +0.06** for the value headline, **+0.10** for adoption momentum
  (per stream: trend −0.13, breakout +0.12, carry +0.02, x-sect-mom +0.03, volprem +0.03, BAB +0.07).
  Genuinely decorrelated, as a new-information source should be — but decorrelation only pays on a sleeve
  that earns something:

| on-chain weight | 0% | 15% | 30% | 50% |
|---|---|---|---|---|
| blended Sharpe — value headline | 3.71 | 3.61 | 3.35 | 2.60 |
| blended Sharpe — adoption momentum | 3.71 | 3.70 | 3.54 | 2.95 |
| targets met — adoption momentum | 5/5 | **4/5** | 4/5 | 4/5 |
| targets met — its rotated control | 5/5 | **5/5** | 5/5 | 4/5 |

The value book monotonically lowers the blend. Adoption momentum, the one signal with real
out-of-sample behaviour, is **flat**: +0.003 Sharpe at a 15% weight is indistinguishable from zero, and
past that it costs. **Marginal contribution is negative or nil → exclude.**

## 8. Honest verdict & ceiling

- **Reachable here:** nothing portfolio-admissible. The free on-chain axis is a **static coin-type tilt**
  on the value side (+0.15, fails placebo/MC/DSR), **negative** on the dilution side, **flat** on
  ownership and fee yield, and on the one axis that works — adoption momentum, +0.73 with MC-P5 +0.08,
  placebo 98th and alpha t=+2.04 — it is **momentum the book already owns** (+0.32 correlated to price
  momentum, deflated 0.50, and worth +0.003 book Sharpe at a 15% weight).
- **Binding constraints:** (1) the documented on-chain *value* premium lives in the **illiquid small-cap**
  universe the tradable funnel excludes; (2) the free cross-section caps at **33 names** (top-50/100
  impossible — newer chains publish market data only), too thin for a robust cross-sectional book;
  (3) **exchange flows are free but only for BTC/ETH**, which makes them a two-asset timing overlay, and
  as an overlay they fail the random-timing control; (4) what genuinely remains Pro-walled is narrow —
  adjusted transfer value and entity-adjusted supply bands — and it is *entity labelling*, not raw chain
  data, that the paid vendors actually sell.
- **What did not work (kept, not hidden):** all 13 cross-sectional signals including the four built on
  newly-recovered metrics (holders, cap-per-holder, issuance, fee yield), both value normalisations, the
  full top-10/20/30/33 × signal surface, the holding-period and z-window sweeps, the weekly resample, the
  placebo, the purged walk-forward OOS, three cost levels, the price-orthogonality regressions, **14
  BTC/ETH time-series overlays with a random-timing control and HAC predictive regressions (§5)**, and a
  21-trial ML ranker plus meta-gate (§4b). The value delivered is the **map** — H3 covered with rigour,
  the free-vs-paid boundary re-drawn correctly, and the exchange-flow thesis converted from an untested
  excuse into a tested negative.
- **Two corrections this document previously carried, both in the flattering direction.** The exchange-flow
  metrics were reported as pay-walled when they are free for BTC/ETH — an artifact of inferring
  entitlement from group 403s instead of reading the vendor catalog. And four names in the cross-section
  were measuring dead ERC-20 shells, inflating the headline value book from +0.15 to +0.40. Both are fixed
  in code (`live_universe()`, catalog-driven fetch), not just in prose.
- **Where an edge might still be** (the honest upgrade path, not free): **entity-level** flow data — which
  wallet, which exchange, whale versus retail (Glassnode / CryptoQuant / Nansen, all paid) — is the part
  of the flow story this test could not reach, since aggregate BTC/ETH net-flow is now shown not to work.
  The documented **price-to-new-address value** premium needs a **wide small-cap** on-chain panel
  (hundreds of coins), not the liquid-33 cut. The free cash-flow axis is no longer on this list — it
  was built and tested (§5b) and crypto inverts it. What is left unexplored is **protocol**-level
  fundamentals below the chain level (per-DEX, per-lender fee splits), which needs a hand-curated
  token→protocol mapping rather than a new data source.

## 9. Reproduce

```bash
python -m src.data.onchain      # build the free on-chain cache (Coin Metrics community + blockchain.com)
make onchain                    # scripts/onchain/run_onchain.py       -> reports/onchain/onchain_{summary.json,returns.parquet}
                                # scripts/onchain/run_onchain_ml.py    -> reports/onchain/onchain_ml_{summary.json,returns.parquet}
                                # src.data.defillama                   -> data/cache/fundamentals/
                                # scripts/onchain/run_fundamentals.py  -> reports/onchain/fundamentals_{summary.json,returns.parquet}
                                #                              + reports/figures/onchain{,_ml}.png
```

Fixed seed (7) throughout; the final walk-forward block runs once. `run_onchain_ml.py` reads its linear
baseline from `run_onchain.py`'s summary, so run them in that order. The universe map, the catalog-driven
free-metric resolution, the dead-shell screen and the point-in-time alignment are in
`src/data/onchain.py`; signals in `src/sleeves/onchain.py`.
Sources: Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency" (JF 2022); Cong, Karolyi, Tang &
Zhao, "Crypto Wash-Trading… Value, Factor Pricing & Market Segmentation" (Mgmt Sci 2024); Kalichkin,
"Rethinking NVT Ratio" (2018); Glassnode "MVRV Z-Score"; Coin Metrics community API.
