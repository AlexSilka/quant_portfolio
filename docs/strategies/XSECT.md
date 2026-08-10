# Cross-Sectional Momentum — Study & Book

**Dashboard:** [xsect_dashboard.html](../../reports/xs/xsect_dashboard.html) · **Reproduce:** `python scripts/xs/build_panels.py && scripts/xs/{sweep,walk_forward,ml,portfolio,audit,make_report}.py`

Cross-sectional (relative-value) momentum: each bar, rank the names of a panel by a momentum
signal, go **long the top quantile and short the bottom, dollar-neutral**. It bets on *relative*
ranking, not each asset's own trend, so it is structurally **market-neutral** — a real
diversifier against the time-series trend book, not a re-labelled copy of it.

---

## Perp funding was never charged — and it is a credit

`scripts/xs/portfolio.py` is the writer the build actually runs (`make xs`) and the one that produced
the committed `reports/xs/xs_book.parquet`; its crypto legs trade **perps**. It passed `cost_bps` and
an ADV panel to `xs_backtest` but never charged funding, although that function's own docstring says
crypto callers must: "crypto perps pay funding, not borrow, so those callers leave
borrow_bps_annual=0 and charge funding separately." Fixed 2026-08-10; funding is now binned onto each
panel's bar grid so no 8h settlement is dropped inside a 1d bar, and coverage is 100% of names on all
three panels.

**It pays the book, it does not cost it** — the opposite of the obvious guess, and the same sign the
breakout family's cross-sectional leg shows ([BREAKOUT.md §12](BREAKOUT.md)). Momentum is long the
winners, where funding is dearest; but measured on the names actually held, the **short** book carries
more, because beaten-down alts keep stubbornly positive funding:

| leg | long book | short book | net | Sharpe | CAGR |
|---|---|---|---|---|---|
| crypto_1d | +3.3%/yr | **+8.6%/yr** | +5.3% | 0.79 → **0.93** | 12.3% → 15.0% |
| crypto_4h | +6.1%/yr | **+10.4%/yr** | +4.3% | 0.41 → **0.58** | 5.9% → 9.0% |
| crypto_1h | +7.3%/yr | **+10.1%/yr** | +2.8% | 0.62 → **0.75** | 9.4% → 12.0% |

Family level: crypto book **+0.67 → +0.83**, cross-asset book **+0.81 → +0.88**, max DD −18.7% →
−17.6%. The family had been *under*-stated, not flattered.

**A second writer exists and is not the one that runs.** `scripts/xs/build_xs_book.py` builds a
different construction (crypto on the **spot** panel, idiosyncratic momentum) into the same
`xs_book.parquet`. It is in no Makefile target and no orchestrator; the committed artifact's
composition (`crypto_1d/4h/1h + stocks_broad`, window 2012-01-03..2026-08-04) matches
`portfolio.py`'s and not its. Treat `portfolio.py` as canonical. If the spot construction is ever
made canonical instead, note that it has the mirror-image defect: a spot short must borrow the coin,
and that leg passes no `borrow_bps_annual` (see `CRYPTO_SPOT_BORROW_BPS_ANNUAL`, ~2.9%/yr live).


## 1. Headline

Cross-sectional momentum is a **real but modest, market-neutral edge** — a decorrelated
diversifier, not a standalone workhorse. On a hand-curated "major coins" list the crypto book scores
Sharpe ~1.2, but that selection is biased; **on the honest survivorship-free, tradable universe it is
~0.6** — the ~1.2 is a selection artifact, and quantifying it is the most important finding here (§2).

> **Crypto x-sect book (survivorship-free, top-100 liquid, no config cherry-pick): net Sharpe
> ~0.67**, max DD −19%, Monte-Carlo P5 ≈ 0 — a *marginal* standalone edge (deflated Sharpe 0.11
> after multiple testing). **Cross-asset book (+ a ~0-correlated US-equity leg): Sharpe 0.67 at −14%
> DD, MC-P5 +0.36.** Its real value is decorrelation: **50/50 with the trend book → Sharpe 0.85 at
> −14% DD** (corr +0.13), lower drawdown than either alone.

Where the edge lives, net and honest: **crypto ~0.4–0.7** across 1d/4h/1h (1d strongest, holding
walk-forward to ~0.65–0.95; intraday deflates); **US equities ~0.4** at 1d only, not helped by
breadth (§6); **everything else dead** — FX at all timeframes, and stocks/FX *intraday* x-sect fail
outright (placebo beats signal). Execution is t+2 bars; costs are liquidity-aware; every number is
net of a $20M/day tradability filter.

## 2. The survivorship correction (the key finding)

Ranking crypto over a hand-curated **50-coin "major perps" list** (CoinGecko mega-cap-ranked, ≥3y
history) scores well, but that list is chosen *with hindsight*: to be on it a coin had to survive and
grow to top-mcap **today** — a textbook survivorship/selection bias for a momentum long-short. On the
full **830-perp universe**, filtered only to a **$20M/day tradability floor** and time-varying
trailing liquidity (so the universe is survivorship-free and fillable at each bar), the number falls
hard. An A/B on the same config and window isolates it:

| Crypto universe (1d) | Sharpe | what it is |
|---|---|---|
| curated 50 "majors" (run_book list) | **+1.06** | selection-biased — only 26/50 are actually top-50 by liquidity |
| broad top-50 by trailing liquidity | +0.87 | mechanical, mildly survivorship-lenient |
| broad universe, all ≥$5M/day (~150) | +0.52 | diluted by micro-cap noise |
| **time-varying tradable (~80 names ≥$20M/day)** | **+0.70** | **the honest, tradable number** |

So ~0.3–0.5 of the original "edge" was the curated list, not the anomaly. The honest a-priori and
walk-forward numbers (§4) *do* hold at 1d (~0.65–0.95) but **deflate at intraday** (4h a-priori
0.77→0.40, 1h 0.83→0.71) where the broad universe is noisier.

**How many names, then?** The right move — trade a *focused* liquid universe, not all 300 (small-cap
microstructure is genuinely different, and excluding it is correct, not cheating). Testing top-N by
**trailing** liquidity (survivorship-free, rotating), a-priori riskadj-30d, crypto 1d:

| universe | top-20 | top-50 | **top-100** | top-150 | all-300 |
|---|---|---|---|---|---|
| a-priori Sharpe | +0.42 | +0.23 | **+0.79** | +0.54 | +0.33 |
| anchored WF | +0.99* | +0.48 | **+1.25** | +0.55 | +0.21 |

*top-20's WF is inflated by config-overfit on a tiny panel. Focusing clearly beats the diluted full
universe (0.79 vs 0.33), and **~top-100 is the robust zone** — but the curve is **noisy** (top-50
dips), so the edge is somewhat fragile to universe size; ~0.7 is the honest point estimate, not the
1.25 peak. Equities are **flat ~0.30 for any N ∈ 20–150** — focusing neither helps nor hurts, and
top-10 is too concentrated (DD −60%). The book (§5) is rebuilt on a **top-100 liquid** universe:
crypto book **~0.67**, down from the inflated ~1.2.

*(Construction still matters — the repo's original `pct_change(120)`, daily-rebalance, crypto-only
build scored 0.74 and was called a null; risk-adjusted signal + monthly rebalance + the right
lookback is what makes even the modest ~0.6 edge appear. But construction was never the whole story:
the universe was.)*

## 3. Correct construction (what the literature and the data agree on)

| Knob | Crypto | US equities | Why |
|---|---|---|---|
| Signal | risk-adjusted / multi-horizon blend | risk-adjusted | return÷vol rewards smooth trends; blends are horizon-robust |
| Lookback | **fast, 20–45d** | **slow, 252d (12-month)** | crypto trends are shorter-lived; equities show the classic 12-1 effect |
| Skip (gap) | 0 | **~1 month** | equities have short-term reversal to skip (Jegadeesh-Titman); crypto does not |
| Breadth | tercile (top/bottom 30%) | **decile** | more names → deeper, cleaner tails on equities |
| Rebalance | **monthly** | monthly | daily rebalancing is turnover suicide — monthly cuts turnover ~10× at ~equal Sharpe |
| Sizing | vol-target 15% | vol-target 15% | equal risk across sleeves/timeframes |

`src/sleeves/xsect.py` implements the whole grid (signal × lookback × skip × breadth × weighting ×
rebalance) behind one vol-targeted, cost-charged, t+2 backtest.

## 4. Edge map — in-sample vs walk-forward vs placebo (the answer)

Every panel was swept over its construction surface, then re-selected out-of-sample by
**walk-forward** (four schemes: rolling/anchored × short/long train, top-10 ensemble per block) so
the number pays the cost of *choosing* parameters. The placebo column is the best Sharpe a random
signal earns on the same panel — the pipeline's false-discovery floor.

All crypto numbers are on the **survivorship-free tradable universe** ($20M/day floor, 300/199/174
names at 1d/4h/1h); equity/FX on their own panels. Walk-forward is the honest number (it pays for
choosing parameters); the a-priori column applies one textbook config (riskadj-30d, tercile,
monthly) with *no* selection; placebo is the best a random signal earns.

| Panel | in-sample best (overfit) | **walk-forward OOS (honest)** | textbook a-priori | placebo max | verdict |
|---|---|---|---|---|---|
| **crypto 1d** | +1.12 | +0.47 … **+0.95** | +0.65 | +0.14 | **modest, holds** |
| **crypto 4h** | +1.33 | +0.51 … +0.84 | +0.40 | +0.01 | modest, deflates |
| **crypto 1h** | +1.11 | +0.36 … +0.44 | +0.71 | +0.02 | modest, noisy |
| **crypto 15m** | — | — | +0.64 (probe) | +0.30 | modest, holds (monthly rebal) |
| **stocks 1d (broad, survivorship-free)** | +0.48 | +0.36 | +0.39 | +0.15 | modest |
| **stocks 4h / 1h** | +0.65 / +0.87 | — | — | +0.36 / **+1.15** | **dead** (placebo ≥ signal) |
| **fx 1d / 4h / 1h** | +0.37 / +0.58 / +0.51 | ≤ +0.24 | +0.20 | ~signal | **dead** (honest null) |

The shape: the edge is **real but modest on crypto and strongest/most-robust at 1d**; it **deflates
as the timeframe speeds up** (intraday adds micro-cap noise and turnover); on **equities it is a
~0.4 large-cap-only effect**; and **intraday cross-section (crypto 15m/5m, all stock/FX intraday)
is dominated by turnover and noise** — for stocks-1h the placebo (+1.15) actually beats the signal,
the signature of a small panel over-fit by chance.

- **Crypto is a plateau at 1d, thinner intraday** — 88–90% of the 1d grid positive and the
  walk-forward ensemble recovers ~0.95; 4h/1h are 63–80% positive and the honest OOS is ~0.4–0.8.
- **Equities are a modest 1d-only effect** — the classic 12-1 shape (long lookback, skip a month)
  but only ~0.4 survivorship-free (§6); breadth does not help and intraday is dead.
- **FX has no edge at any timeframe** — walk-forward at/under zero, placebo-grade.

## 5. The book

Four sleeves, each the **same textbook a-priori config** (riskadj-30d crypto / 12-1 equity, no
per-timeframe cherry-pick), vol-targeted to 15% on the survivorship-free universe, daily P&L:

| Sleeve (top-100 liquid) | net Sharpe | max DD |
|---|---|---|
| crypto 1d | +0.79 | −19% |
| crypto 4h | +0.41 | −22% |
| crypto 1h | +0.62 | −21% |
| equity (broad S&P, survivorship-free) | +0.39 | −29% |

**Correlation is the whole point — and the limit.** The three crypto sleeves co-move **0.66–0.84**
(one edge sampled at three clocks — they barely diversify each other), but the **equity leg
correlates ≈ 0.00 with all of them**. So combining the crypto cluster buys little; the equity leg,
weak as it is, is the only real second source.

- **Crypto x-sect book:** Sharpe **+0.67**, DD −19%, MC-P5 **≈ 0**, deflated Sharpe **0.11** — a
  *marginal* standalone edge after honest controls (the correlated crypto sleeves don't rescue it).
- **Cross-asset x-sect book** (+ survivorship-free equity leg): Sharpe **+0.67**, DD **−13.9%**,
  MC-P5 **+0.36** — the decorrelated equity leg lifts robustness more than raw Sharpe.
- **Diversification is the real deliverable:** a **50/50 with the trend book → +0.85 at −14% DD**
  (corr **+0.13**) — lower drawdown than either alone. As a standalone book x-sect is marginal; as an
  overlay it earns its place.

**Shipped construction — residual momentum (the H5 upgrade).** The master book's crypto x-sect leg does
*not* ship the raw riskadj-30d sleeve tabled above. The H5 deep-dive ([RESIDMOM.md](RESIDMOM.md)) ran
**residual (idiosyncratic) momentum** — the same cross-sectional ranking on the market-beta-neutralised
residual (BHM), monthly rebalance on the 300-name PIT spot panel — through the full funnel and found it a
steadier, lower-turnover crypto momentum (net Sharpe ≈ **+0.6**) that lifts the assembled book's
out-of-sample consistency at no return cost. That construction lives in `build_xs_book.crypto_spot_xsect`, which is **not** the writer the build runs — see the funding section above
ships; the equity leg keeps the risk-adjusted-momentum engine above. The two legs (≈0.00 correlated) are
risk-parity combined into the family series the master book reads.

## 6. Breadth investigation — does a broader equity universe strengthen it? (no)

The narrow equity panel (78 names) mixes mega-caps with sector/bond/commodity ETFs and uses today's
survivors. The literature says single-stock cross-sectional momentum is strongest on a **broad
small/mid-cap** universe, so the honest test is to rebuild it properly — on **Twelve Data Pro**
(deep, complete history; not yfinance, which is full of holes on delisted tickers):

- **Survivorship-free S&P 500** — every name in the index at any point 2012–2026 from the PIT
  membership file (`fja05680/sp500`): **692 names, the 325 that were dropped included** — the honest
  correction to the momentum survivorship bias.
- **S&P 400 mid-cap + S&P 600 small-cap** — **893 names**, where the anomaly is supposed to peak.
- Added **residual (idiosyncratic) momentum** to the signal set and a $10M/day liquidity mask.

| Equity universe | names | in-sample best | walk-forward OOS | a-priori 12-1 | verdict |
|---|---|---|---|---|---|
| narrow mixed (stocks+ETFs) | 78 | +0.90 | +0.55 | +0.85 | strong — but ETF asset-class rotation, low-survivorship mega-caps |
| **broad large-cap** (S&P 500 PIT, survivorship-free) | 692 | +0.48 | **+0.36** | +0.39 | modest — the honest pure-stock number |
| **mid/small-cap** (S&P 400+600) | 893 | +0.39 | **+0.17** | −0.00 | weakest — breadth *hurts*, not helps |

- **Breadth does not strengthen it.** Pure single-stock S&P momentum, survivorship-free and net of
  costs, is **modest (~0.4 large-cap, ~0.2 small-cap)** — and small/mid-caps are **weaker, not
  stronger**, once tradable liquidity is required. The documented "small-cap momentum premium" does
  not survive a realistic, cost-charged, survivorship-controlled build. An honest negative result.
- The narrow panel's higher +0.85 is largely **asset-class rotation across the ETFs** (a legitimate,
  low-survivorship signal since ETFs and mega-caps do not delist) — not S&P stock-picking.
- **Residual momentum** is the best single-stock signal on the broad panel (top sweep config),
  consistent with the literature, but still tops out at +0.48 in-sample.
- **ML learning-to-rank** (LightGBM, 1.9M rows on mid/small) does not rescue it (+0.17) — no feature
  combination recovers an edge that is not there.

The takeaway sharpens the edge map: on equities cross-sectional momentum is a **modest ~0.4
large-cap-only effect that breadth does not improve** — the same lesson §2 found on crypto (a
broad, honest universe is weaker than the curated one, not stronger).

## 7. ML — measured incremental value, and an honest asymmetry

The meta-label confidence gate was built two ways, both leakage-controlled (features stamped at bar t,
labels purged, all prediction walk-forward): a **learning-to-rank** model (predict each name's
cross-sectionally demeaned forward return from a multi-signal feature vector) and a **meta-label
gate** (predict P(the rule book wins next period), trade only high-probability periods). Optimizers
compared: Ridge, RandomForest, HistGradientBoosting, LightGBM.

- **Crypto: ML does *not* beat the rule.** Rule vs best learning-to-rank was measured on the
  curated panel (0.71 vs 0.61); a single risk-adjusted momentum is already a clean signal
  and ML mostly adds estimation noise on daily crypto.
- **Equities: it depends on the panel.** On the narrow mixed panel learning-to-rank beats the rule
  (+0.62 → **+0.89**, LightGBM and even Ridge, drawdown −32%→−25%) — a real lift where several
  factors (reversal, low-vol, beta) combine. But on the **honest survivorship-free broad panel ML
  does not rescue a weak edge** (mid/small +0.17): where the rule is already ~0.4, no feature
  combination manufactures alpha that is not in the cross-section.
- **Meta-gate** reduces Sharpe everywhere — not the win here.

So ML's incremental value is **conditional and measured**, not asserted: it can combine signals on
a rich mixed panel, but it does not conjure an edge on the clean broad universe where none exists.

## 8. Validation & leakage

- **Placebo / FDR:** on the survivorship-free universe, random-signal placebos are near zero for
  crypto (+0.14 at 1d, +0.01–0.02 at 4h/1h) — so the real edge, thin as it is, is clean. But
  **stock/FX intraday placebos are not** (stocks-1h placebo +1.15 ≥ its best real signal): a small
  panel over-fits random rankings, which is exactly why those cells are called dead.
- **Monte Carlo** (stationary block bootstrap): crypto book P5 **≈ 0** (marginal), cross-asset book
  P5 **+0.34** — the decorrelated equity leg is what buys the robustness.
- **Deflated Sharpe:** crypto book DSR **0.11** at N = 2916 grid trials — it barely survives
  multiple testing standalone; the book is defensible only as a decorrelated overlay, not solo.
- **Cost sensitivity** (cross-asset book): Sharpe +0.66 / +0.57 / +0.47 at 1× / 2× / 3× base cost;
  **break-even ≈ 8× base** — monthly rebalancing keeps it cost-cheap even at a modest Sharpe.
- **Look-ahead audit:** shift audit `max|full − truncated| = 0` (computable-at-bar); exec-lag
  sensitivity flat (0.72 / 0.80 / 0.70 at lag 1/2/3 — no same-bar fill dependence).

## 9. Honest limits & ceiling

- **It is a modest edge, marginal standalone.** On the survivorship-free top-100 universe the crypto
  book is ~0.67 with MC-P5 ≈ 0 and DSR 0.11 — it barely clears multiple testing on its own. The three
  crypto sleeves are one edge sampled at three clocks (corr 0.66–0.84), so they do not diversify
  each other; the equity leg (corr ≈ 0) is the only real second source, and it too is only ~0.4.
  The honest case for the sleeve is **diversification** (combo with trend −13% DD), not standalone
  return.
- **Intraday deflates.** With monthly-cadence rebalancing the edge survives to 15m (~0.64) but the
  a-priori Sharpe erodes as the timeframe speeds up and the universe gets noisier; **stock/FX
  intraday cross-section is dead** (placebo ≥ signal on a 50-name panel). Only low-turnover configs
  are trustworthy net; fast-rebalance peaks are turnover artifacts.
- **2026-YTD is weak** (crypto x-sect −0.5 on the partial year) — a low-dispersion regime where
  relative momentum has little to sort; shown, not hidden.
- **Survivorship — the correction that mattered.** Equities use a **point-in-time S&P 500** (692
  names incl. the 325 dropped since 2012); crypto uses the **full 830-perp universe** with a
  time-varying $20M/day tradability floor, not a hand-picked major-coin list. This is what took the
  headline crypto number from an inflated ~1.2 down to the honest ~0.6. Residual bias remains
  (Twelve Data lacks ~15% of fully-purged equity tickers; the crypto floor is a design choice, and
  the sensitivity to it is shown in §2) — quantified, not hidden.
- **Ceiling:** ~0.67 net Sharpe standalone (crypto book), 0.67 cross-asset — a modest,
  market-neutral, cost-robust book, well below the aspirational 2.5–4.0. Its defensible use is as a
  **decorrelated overlay** that takes a trend book's drawdown down while nudging Sharpe up (combo
  0.85 at −14% DD) — real value, honestly small.

## 10. Reproduce

```bash
python scripts/xs/build_panels.py crypto     # crypto 1d/4h/1h/15m/5m, 830-perp universe + $5M filter
python scripts/xs/build_panels.py equity_intraday   # stocks/fx 4h/1h (liquid core)
python scripts/xs/fetch_broad_equity.py      # survivorship-free S&P 500 (Twelve Data, PIT)
python scripts/xs/fetch_broad_equity.py _midsmall_universe.json stocks_midsmall  # S&P 400+600
python scripts/xs/sweep.py crypto_1d crypto_4h crypto_1h stocks_1d fx_1d stocks_4h stocks_1h fx_4h fx_1h
python scripts/xs/walk_forward.py crypto_1d crypto_4h crypto_1h  # honest OOS (with $20M liq mask)
python scripts/xs/broad.py stocks_broad ; python scripts/xs/broad.py stocks_midsmall  # breadth test
python scripts/xs/ml.py crypto_1d stocks_1d  # learning-to-rank + meta-gate
python scripts/xs/portfolio.py               # honest book + diversification + cost/MC/DSR
python scripts/xs/audit.py                   # shift audit, panel placebo, exec-lag
python scripts/xs/make_report.py             # figures + dashboard
```

Fixed seeds throughout; the final block is never tuned against.
