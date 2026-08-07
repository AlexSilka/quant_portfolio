# On-chain / network signals (crypto) — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.66** full / **2.64** OOS). Any master-book Sharpe quoted below (e.g. a 3.77 "book drag" baseline) is the book *snapshot at the time this family was evaluated*, not the current headline.

**Scope.** H3 from [HYPOTHESES.md](../HYPOTHESES.md): on-chain data (exchange flows, stablecoin supply,
active-address growth, valuation ratios) carries information **not present in price** — the one
genuinely new information source for crypto. The data access is *gated* — free access had to be
confirmed first. Confirmed live (2026-08), a free ingestion loader was built, and the family was run through
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
- **The data gate is half-open, and that shapes the whole result (§1).** The high-information metrics the
  thesis leans on — **exchange net-flows, adjusted transfer value (the NVT numerator), fees, miner
  revenue, realized cap** — are **all pay-walled**. What is free is **network-activity & valuation**:
  active addresses, transaction counts, supply, market cap, and MVRV, across **37 liquid names**
  (free on-chain ∩ the repo price panel), plus BTC transaction-value & miner-revenue from blockchain.com.
  So this honestly tests *on-chain value + adoption momentum*, not *exchange-flow* on-chain.
- **The top-N answer is a hard data ceiling: top-10/20/30/37 are testable, top-50/100 are not.**
  Free on-chain covers ~37 tradable names; the newer high-liquidity chains the cross-section would need
  for a top-100 (**SOL, SUI, TON, APT, SEI, TIA…**) are Pro-walled on Coin Metrics. Stated, not skipped.
- **The best in-sample book — on-chain value (market cap per active address, "CVALUE"), top-20 — nets
  Sharpe +0.40 and fails every out-of-sample gate:** MC-P5 **−0.28** (< 0), placebo **80th percentile**
  (< 95th; shuffle p95 **+0.65** > real), and **purged walk-forward OOS −0.64** (in-sample best +0.70).
  Deflated Sharpe at N=24 trials **0.22**. Three independent gates agree: no real edge.
- **It is a *static* tilt, not a signal.** Turnover **0.03/bar**; **median name never flips side (100%
  side-persistence, 24/36 names)**. The book is permanently **long old PoW coins** (LTC, ETC, BCH, DOGE,
  XLM, ALGO — cheap per address) and **short newer tokens** (LINK, UNI, SHIB, XRP, AAVE — rich per
  address). Market-cap-per-address separates coin *generation/type*, a fixed characteristic — which is
  exactly why the placebo can't distinguish it and it dies out-of-sample.
- **The decisive test — does on-chain add edge OVER price? No.** Regressed on price-momentum + price-
  reversal books built on the *identical* 37 names, **no on-chain signal clears t > 2**: on-chain value
  α t=**1.04**, adoption momentum α t=**1.91** (and it is the one most correlated with price momentum,
  +0.33), everything else t < 1. Precisely the literature's verdict — on-chain *value* ≈ a value factor,
  on-chain *momentum* ≈ price momentum (Liu-Tsyvinski-Wu, JF 2022; Cong-Karolyi-Tang-Zhao, Mgmt Sci 2024).
- **ML changes nothing — no non-linear on-chain alpha either (§4b).** A 21-trial ML ranker ({ridge, RF,
  extra-trees, hist-GBM, LightGBM} + classifiers × {on-chain / price / both} features × {5,21,63}d horizons ×
  top-N, all purged-CV OOS): ML on **on-chain** features fails (best +0.20, most ≤ 0); the *same harness* on
  **price** features finds a strong edge (**+1.02**) — so the failure is the data, not the method — and
  **appending on-chain to price degrades it** (+0.89 → −0.50). Feature importance is led entirely by price
  features. The ML meta-gate cuts the value book's DD (−23% → −17.7%) but halves Sharpe (+0.40 → +0.08):
  risk-reduction, not alpha.
- **BTC/ETH time-series overlays all underperform buy-and-hold.** Vol-targeted BTC buy-hold nets **+0.85**
  on the window; MVRV-z long/flat **+0.07** (long/short −0.99), NVT +0.18, Puell −0.49, stablecoin-SSR
  +0.55. On-chain *timing* destroys value versus simply holding — and with only 2-3 cycles it is
  near-unbacktestable regardless (§5).
- **Decorrelated (corr to book +0.07) but it drags** (blended Sharpe 3.77 → 2.85 as weight rises).
  Correctly **excluded**. Documented, not traded.

---

## 1. The data gate — what is free, what is pay-walled, and why it matters

H3 is the only hypothesis needing a new ingestion loader, so the free-access probe *is* part of the
result. Two no-key sources, each probed per-(asset, metric) live before wiring in (`src/data/onchain.py`):

| source | free & used | pay-walled (not used) |
|---|---|---|
| **Coin Metrics community** v4 | PriceUSD, CapMrktCurUSD, **AdrActCnt**, TxCnt, TxTfrCnt, SplyCur, **CapMVRVCur** | **Flow\*ExNtv (exchange net-flows)**, **TxTfrValAdjUSD (NVT numerator)**, fees, miner revenue, realized cap, 7d/30d active-addr |
| **blockchain.com** charts (BTC only) | estimated-transaction-volume-usd, miners-revenue, hash-rate | — (BTC only; no altcoins) |

The economically sharpest signal the thesis names — **exchange inflow = sell pressure** — is exactly the
one that is pay-walled, and for a structural reason (the edge is the proprietary exchange-wallet
*labelling*, not the raw chain). Stablecoin supply (USDT/USDC/DAI/BUSD) *is* free, so the risk-on tilt is
testable; the flow signal is not. **This is scoped honestly: what follows tests the free on-chain axis —
network-activity & valuation — and states the flow gap rather than approximating it from raw chain data.**

**Universe.** The tradable set is `free-CM-coverage ∩ repo crypto price panel` = **37 names**, 2020-2026,
liquidity-ranked. Integrity-checked: on-chain `PriceUSD` daily returns correlate **≥0.95** with the repo's
Binance returns for every name (only KNC at 0.949, a known token-migration artifact) — so the
coin→asset mapping is correct, no mismatched series. Coverage: AdrActCnt 37/37, MVRV 36/37 (TRX lacks a
free MVRV). **Top-N ceiling = 37**; a top-50/100 cross-section is impossible on free data because the
newer high-liquidity chains (SOL, SUI, TON, APT, SEI, TIA, …) have their on-chain metrics behind the
Coin Metrics Pro tier — a genuine constraint, not a shortcut.

## 2. Construction — the on-chain signal family

Eight a-priori signals (declared before fitting, all reported, never peak-picked) in
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
DIVERGENCE          divergence            z(address growth) − z(price return): activity outrunning price (orthogonal by build)
```

**Headline = `nvm_val`** — chosen a-priori on the *literature*, not the fit: the one on-chain factor with
hard academic backing is cross-sectional on-chain **value**, market cap per active/new address, LOW = cheap
= long (Liu-Tsyvinski-Wu; Cong et al.). `top-20` is the round mid-cross-section (deliberately **not** the
sweep-maximal N=30), so the headline is not peak-picked; the full top-N × signal surface is in §3.

## 3. Results — a static value tilt that dies out-of-sample

**Top-N × signal sweep** (net Sharpe, monthly rebalance) — the top-10/20/30/37 question:

| signal | N=10 | N=20 | N=30 | N=37 |
|---|---|---|---|---|
| adr_mom30 (adoption mom) | +0.70 | +0.69 | +0.54 | +0.65 |
| tx_mom30 | +0.31 | +0.70 | +0.35 | +0.36 |
| **nvm_val (value, headline)** | −0.10 | **+0.40** | +0.71 | +0.46 |
| nvm_z_val (self-relative) | +0.26 | +0.22 | +0.44 | +0.49 |
| metcalfe_val | −0.09 | +0.24 | +0.51 | +0.47 |
| mvrv_val / mvrv_z_val | −0.25 / −0.07 | +0.09 / −0.55 | +0.03 / −0.31 | +0.10 / −0.16 |
| divergence | −0.01 | −0.06 | −0.27 | −0.10 |
| blend (value+divergence) | −0.27 | −0.39 | −0.28 | −0.23 |

No stable pattern in N — the "value" signals peak at N=30 and fade at N=37; the momentum signals peak at
N≤20. Sign-unstable across the grid is itself a warning. **The headline `nvm_val` top-20 monthly** nets:

| metric | value |
|---|---|
| net Sharpe | **+0.40** |
| MC [P5, P50] | **[−0.28, +0.42]** |
| max drawdown | −23% |
| turnover / bar | **0.03** |
| per-year | 2020 −0.46 · 2021 +0.07 · 2022 **+1.15** · 2023 −0.31 · 2024 +0.22 · 2025 **+1.52** · 2026 +0.77 |

Three out-of-sample gates, all failed:

- **Placebo (200 name-shuffles):** real +0.40 sits at only the **80th percentile** (shuffle mean +0.02,
  p95 **+0.65**). The true cross-section does **not** beat the 95th-percentile bar — a random re-assignment
  of the same signal to different names clears +0.40 one time in five. The structure is not doing the work.
- **Purged/embargoed walk-forward OOS** (2y train / 0.5y test, 365-bar embargo, top-3 config ensemble,
  N=24-trial grid): in-sample best **+0.70 → OOS −0.64** (9 refits). Even *choosing* the construction
  out-of-sample flips it negative. **Deflated Sharpe = 0.22.**
- **MC-P5 −0.28 < 0** — the block-bootstrap 5th percentile is negative; the robust bar (MC-P5 > 0) fails.

**Why it fails — it is a static tilt, not a signal.** Turnover **0.03/bar**, and the median name **never
changes side** (100% side-persistence; 24 of 36 names never flip). The book is permanently:

```
LONG  (cheap per address):  LTC +0.17, ETC +0.16, BCH +0.14, DOGE +0.10, XLM +0.07, ALGO +0.07   ← old PoW coins
SHORT (rich per address):   LINK −0.19, UNI −0.16, SHIB −0.11, XRP −0.10, AAVE −0.07, LDO −0.06   ← newer tokens
```

Market-cap-per-active-address structurally separates legacy PoW coins (many cheap on-chain transfers per
dollar of cap) from smart-contract-era tokens (high cap on a smaller address base) — a **fixed coin
characteristic**, not a time-varying mispricing. So ranking on it is ~ranking on coin generation. The
+0.40 is a bet on old-coin-vs-new-coin relative performance over 2020-2026 (positive in 2022/2025,
negative in 2020/2023) dressed as on-chain value — which is exactly why the placebo can't distinguish it
and the walk-forward is negative. The **self-relative** form (`nvm_z_val`, which removes the coin
fixed-effect) is *weaker* (+0.22), and its z-window is unstable (z180 +0.54, z365 +0.22, z540 +0.08) —
the shorter the window the better, the fingerprint of fitting noise. **Cost is not the binding constraint**
(turnover is tiny; +0.40 at 1×, +0.37 at 3×) — the book is simply below the 0.5 bar and negative OOS.

**Timeframe.** On-chain metrics are **daily aggregates** — there is no free intraday on-chain (and block-
level counts are daily by nature), so the 5m/1h/4h axis the price sleeves sweep does not exist here. The
holding-period axis was swept instead (rebal ∈ {1,7,21,63} bars): daily +0.45, weekly +0.36, monthly
+0.40, quarterly +0.02; a true weekly-resampled book nets +0.27. The edge, such as it is, lives at the
daily-to-monthly horizon and decays by the quarter — consistent with a slow characteristic tilt, not a
timing signal.

## 4. The decisive test — does on-chain add edge *over price*?

The survive/kill bar is "adds decorrelated OOS edge **over price-only sleeves**". Each on-chain book
was regressed on a price-momentum book (`mom 30d`) and a price-reversal book (`−mom 12m`, the price
"value" proxy) built through the *same* engine on the *identical* 37 names — alpha = the intercept:

| on-chain book | α (ann) | **α t-stat** | corr price-mom | corr price-rev | verdict |
|---|---|---|---|---|---|
| nvm_val (value, headline) | +0.065 | **+1.04** | +0.08 | −0.11 | marginal, not significant |
| adr_mom30 (adoption mom) | +0.113 | **+1.91** | **+0.33** | +0.03 | marginal — and it *is* price momentum |
| nvm_z_val | +0.037 | +0.61 | −0.20 | +0.30 | subsumed by price |
| mvrv_val / mvrv_z_val | +0.015 / −0.095 | +0.27 / −1.56 | — | +0.34 / +0.14 | subsumed / negative |
| divergence | −0.012 | −0.18 | −0.22 | −0.03 | subsumed by price |
| blend | −0.072 | −1.01 | −0.14 | +0.06 | subsumed by price |

**No on-chain signal clears t > 2.** The two that come closest are the value level (t=1.04) and adoption
momentum (t=1.91) — and adoption momentum carries the highest price-momentum correlation (+0.33),
confirming it is largely re-labelled momentum. This reproduces the published cross-section exactly: the
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

| model | on-chain features | **price features** | both |
|---|---|---|---|
| ridge | −0.04 | +0.12 | +0.52 |
| random-forest | −0.57 | +0.67 | −0.23 |
| extra-trees | +0.20 | **+1.02** | +0.14 |
| hist-GBM | +0.19 | +0.45 | +0.17 |
| LightGBM | −0.20 | +0.89 | −0.50 |

Three findings, all pointing the same way:

- **ML on on-chain features fails** — best +0.20 (extra-trees), most ≤ 0 (RF −0.57, LightGBM −0.20);
  classifiers no better (logit +0.49, RF −0.34, LGBM −0.13); every non-21d/other-N robustness cell
  ≤ +0.12. No model beats the linear on-chain baseline in any useful way, and none clears the bar OOS.
- **The harness works — on PRICE features it finds a strong edge** (+0.67 to **+1.02**). This is the key
  control: the pipeline is not broken or over-conservative; it recovers the known price cross-sectional
  premium. The on-chain failure is the **data**, not the method.
- **Adding on-chain to price DEGRADES it** — LightGBM +0.89 → −0.50, RF +0.67 → −0.23 when the on-chain
  block is appended. On-chain features are **noise to the model**: they add nothing and cost generalisation.

**Decisive, again:** best on-chain-ML **+0.20** vs best price-ML **+1.02** → on-chain features **do not
beat** price. The best on-chain-ML book's alpha over price momentum+reversal is **t=+0.52** (no edge —
same verdict as the linear §4). And feature importance on the combined model is led entirely by **price**
features (px_vol_60, px_mom_120, px_beta_60, px_radj_120, px_mom_180); the on-chain features that rank at
all are the valuation ones (nvm_z, nvm_log) mid-pack — the model reaches for price whenever allowed. Even
the best *price*-ML book fails the multiple-testing haircut (MC-P5 **−0.30**, deflated Sharpe **0.06** at
N=21 trials) — it is the selection-inflated winner of 21 tries, not a robust sleeve.

**Meta-gate** (an ML model predicts P(the value book is up next month) from
panel-regime state and gates exposure) does what it does everywhere in this repo — **cuts drawdown, not a
Sharpe boost**: the value book's −23% max-DD falls to **−17.7%**, but Sharpe drops **+0.40 → +0.08** (in
market only 26% of the time). The book is too weak to gate profitably. Consistent with the repo's standing
result ("ML buys risk reduction and precision, not out-of-sample alpha", [docs/TREND.md](TREND.md)) — and
here even the risk-reduction is not worth the Sharpe it costs.

**So ML changes nothing: there is no on-chain alpha, linear or non-linear.** Figure:
[reports/figures/onchain_ml.png](../../reports/figures/onchain_ml.png); `scripts/onchain/run_onchain_ml.py`.

## 5. Time-series overlays (BTC/ETH) — regime context, not backtestable alpha

The classic on-chain *timing* indicators (MVRV z-score, NVT, Puell, stablecoin-SSR) are BTC-macro
overlays with only **2-3 non-overlapping cycles** of history — near-unbacktestable by construction, and
reported as context. Long/flat (risk-on when the metric is below its own trailing mean), t+2, vs
vol-targeted buy-and-hold:

| overlay | Sharpe | maxDD | | overlay | Sharpe | maxDD |
|---|---|---|---|---|---|---|
| **BTC buy-and-hold** | **+0.85** | — | | BTC NVT long/flat | +0.18 | −45% |
| BTC MVRV-z long/flat | +0.07 | −32% | | BTC Puell long/flat | −0.49 | −48% |
| BTC MVRV-z long/short | −0.99 | −74% | | BTC stablecoin-SSR growth | +0.55 | −13% |
| ETH MVRV-z long/flat | +0.17 | −36% | | | | |

**Every overlay underperforms simply holding BTC** (+0.85). The best, stablecoin-SSR-growth (+0.55, and a
tidy −13% max-DD from sitting out risk-off), still trails buy-hold and rests on one structural
stablecoin-supply uptrend. MVRV long/short (−0.99) is actively destructive — fading "expensive" MVRV
shorts the biggest bull legs. On-chain top/bottom timing does not beat the hold.

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

- **Correlation to the deliverable book: +0.07** (per stream: trend −0.13, breakout +0.13, carry +0.04,
  x-sect-mom +0.05, volprem +0.07). Genuinely decorrelated, as a new-information source should be — but
  decorrelation only helps a *positive* sleeve:

| on-chain weight | 0% | 15% | 30% | 50% |
|---|---|---|---|---|
| blended Sharpe | 3.77 | 3.73 | 3.53 | 2.85 |

Adding it monotonically lowers the blend. **Marginal contribution is negative → exclude.**

## 8. Honest verdict & ceiling

- **Reachable here:** nothing portfolio-admissible. The free on-chain axis (network-activity + valuation)
  is either a **static coin-type tilt** (value: +0.40 in-sample, −0.64 OOS, fails placebo/MC/DSR) or
  **re-labelled price momentum** (adoption growth: t=1.91 alpha, +0.33 corr to price momentum).
- **Binding constraints:** (1) the high-information flow metrics (exchange net-flows, adjusted transfer
  value, realized cap) are **pay-walled** — the free set is the low-information tail; (2) the documented
  on-chain *value* premium lives in the **illiquid small-cap** universe the tradable funnel excludes;
  (3) what remains does not clear the price-orthogonality bar (no signal t>2 over price); (4) the free
  cross-section caps at **37 names** (top-50/100 impossible — newer chains Pro-walled), too thin for a
  robust cross-sectional book.
- **What did not work (kept, not hidden):** all 8 signals, both value normalisations (level & self-
  relative), the full top-10/20/30/37 × signal surface, the holding-period and z-window sweeps, the
  weekly resample, the placebo, the purged walk-forward OOS, three cost levels, the price-orthogonality
  regressions, the BTC/ETH time-series overlays, **and a 21-trial ML ranker (5 regressors + 3 classifiers ×
  on-chain/price/both features × horizons × top-N) plus an ML meta-gate (§4b)** — the model finds no
  on-chain alpha, recovers a strong edge on price features (proving the harness), and is degraded by adding
  on-chain to price. The value delivered is the **map** (H3 now covered with rigour, and the free-vs-paid
  data boundary documented) plus confirmation that the crypto book's edge is price-based premia, not free
  on-chain information — linear **or** ML.
- **Where an edge might still be** (the honest upgrade path, not free): the pay-walled **exchange
  net-flow** and **entity-adjusted** metrics (Coin Metrics Pro / Glassnode / CryptoQuant paid) are the
  genuinely new-information signals; and the documented **price-to-new-address value** premium needs a
  **wide small-cap** on-chain panel (hundreds of coins), not the liquid-37 cut. Both require paid data —
  a budget decision, deliberately not made here.

## 9. Reproduce

```bash
python -m src.data.onchain      # build the free on-chain cache (Coin Metrics community + blockchain.com)
make onchain                    # scripts/onchain/run_onchain.py    -> reports/onchain/onchain_{summary.json,returns.parquet}
                                # scripts/onchain/run_onchain_ml.py -> reports/onchain/onchain_ml_{summary.json,returns.parquet}
                                #                              + reports/figures/onchain{,_ml}.png
```

Fixed seed (7) throughout; the final walk-forward block runs once. The universe map, the free-metric set,
and the point-in-time alignment are in `src/data/onchain.py`; signals in `src/sleeves/onchain.py`.
Sources: Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency" (JF 2022); Cong, Karolyi, Tang &
Zhao, "Crypto Wash-Trading… Value, Factor Pricing & Market Segmentation" (Mgmt Sci 2024); Kalichkin,
"Rethinking NVT Ratio" (2018); Glassnode "MVRV Z-Score"; Coin Metrics community API.
