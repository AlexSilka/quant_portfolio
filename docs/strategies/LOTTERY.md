# Cross-sectional skewness / lottery (MAX) — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.52** full / **2.64** OOS). Any master-book Sharpe quoted below is the book *snapshot at the time this family was evaluated*, not the current headline.

**Scope.** H2 from [HYPOTHESES.md](../HYPOTHESES.md): investors overpay for lottery-like assets
(high idiosyncratic skew, high recent maximum daily return) so those assets *underperform* — short the
high-lottery tail, long the low-lottery tail (Bali-Cakici-Whitelaw 2011 "MAX"; Kumar 2009). Run through
the same funnel as every other family (dollar-neutral top/bottom-tercile long/short, vol-target 15%,
t+2-style delay, liquidity-aware costs, shuffled-signal placebo, purged/embargoed walk-forward OOS,
block-bootstrap MC, deflated Sharpe at the true trial count, correlation to the deliverable book + lift
curve, and an orthogonality regression vs a low-vol/BAB proxy). **Crypto is the primary test** (retail memecoin-lottery demand is most
acute there); equity is secondary. All numbers net of costs. Figure:
[reports/figures/lottery.png](../../reports/figures/lottery.png). Reproduce: `make lottery`.

---

## 0. TL;DR

- **The lottery/MAX family is dead as a decorrelated long/short premium in every tradable universe
  here** — it joins overnight, cross-sectional reversal and pairs in the "tested, real-as-something-
  else-or-inverted, not viable" pile. A rigorous edge-map entry, not a hidden gap.
- **In crypto the documented direction is *inverted*: short-high-skew nets Sharpe −0.38, short-high-MAX
  −0.67** (a-priori: skew 30d / MAX(5) 21d, tercile, monthly, top-100-liquid). Every one of the 24
  construction cells (skew/MAX × 4 windows × 3 tails) is **negative** — there is no positive region to
  select. The reason is structural: in crypto the *momentum* premium dominates the monthly horizon, so
  the same recently-exploded names the lottery bet wants to **short** are exactly the names momentum
  **longs** — and momentum wins. The only positive side is the opposite bet (long-high-skew +0.15,
  long-high-MAX +0.50), which is just **re-labelled momentum**, already in the book.
- **In equity it is real-but-weak and below the bar.** Short-high-skew nets **+0.25** (broad 692-name)
  / +0.11 (mid-small), short-high-MAX **−0.95 / −0.52**. The classic MAX effect needs small, low-priced,
  retail-held names; on the top-100-*liquid* mega-cap cross-section the tradable universe forces, those
  names are excluded, so what survives is a faint skew tilt that never clears **0.5**.
- **It is not independent of low-vol.** Regressing the crypto skew book on a low-vol/BAB proxy built
  through the same engine gives corr **+0.42**, beta 0.42, and a **residual Sharpe of 0.00** — strip the
  low-vol component and nothing is left. Against H1's actual crypto BAB book the residual Sharpe is also
  **0.00**. So it is neither an independent lottery effect nor a usable re-labelling of low-vol.
- **Perp funding is a headwind, not a wash.** Charged at every 8h settlement, the dollar-neutral book
  *pays* **−5.5%/yr** (skew) / **−7.5%/yr** (MAX) — the long low-skew leg carries more positive funding
  than the short high-skew leg collects — deepening the verdict to **−0.57 / −0.85**.
- **The sign is not stable across timeframes.** Re-run on the crypto 4h/1h panels (windows × bars/day),
  the skew tilt *flips*: 1d skew-short **−0.38**, 4h **+0.20**, 1h **+0.04** (and the momentum side flips
  with it) — at intraday horizons it becomes a weak short-term *reversal*, not the daily lottery anomaly,
  and it never clears **0.5**. A real premium is sign-stable across horizons; this is not (§4).
- **The extreme returns that drive the signal are real, not data artifacts** — unlike overnight, no
  data bug fakes a number here (§2). The verdict is genuinely economic.
- **Decorrelated to the book (corr −0.17) but negative, so it drags** (1.62 → 1.43 at 30% weight).
  Correctly excluded. Documented, not traded.

---

## 1. Construction — the lottery bet, and its collision with momentum

Two a-priori signals (declared before fitting, both reported, never peak-picked) in
`src/sleeves/lottery.py`, each swapped into `src/sleeves/xsect.py::xs_backtest` — the identical
dollar-neutral / liquidity-aware-cost / vol-target engine the momentum and carry books use, so the
numbers are directly comparable:

```
skew : trailing skewness of daily log-returns over 20-60d  (features/engine.py's skew_60)
MAX  : mean of the top-5 daily returns over the past ~month (Bali et al.'s MAX(5))
```

The lottery bet **shorts the high tail / longs the low tail** — expressed by passing the *negated*
signal to the engine, which longs top-ranked and shorts bottom-ranked names. The un-negated sign (long
the high tail) is reported alongside, because in crypto it is the whole story: high-skew / high-MAX
names are last month's big movers, so the lottery **short** and the momentum **long** are betting on the
same names in opposite directions. Universe = the top-100-liquid names each bar (survivorship-free,
identical to the momentum/carry sibling books); tercile tails; monthly rebalance; delayed so a bar-t
signal never fills at its own close.

Costs are the engine's liquidity-aware model (commission + half-spread + √-impact on ADV, never flat).
Perp **funding** is charged at every 8h settlement (`−Σ wᵢ·fᵢ`, Binance USD-M archive): the headline
Sharpes below use the no-funding convention so they compare apples-to-apples with the momentum/carry
sibling books (which price funding as their own carry sleeve, not as a cost on every book), and the
with-funding number is reported alongside — funding only makes the lottery book worse (§3), never used
to flatter.

## 2. Data integrity — the moment-signal trap, and why it does *not* bite here

skew and MAX are, by construction, dominated by a name's largest daily returns, so a single bad print
would hijack the ranking — the exact failure that faked a +0.18 in the overnight sleeve. Actively
checked (`lottery.return_diagnostics`), on the 392,614 finite crypto name-day returns:

- **Extreme returns are genuine.** 381 name-days |ret| > 50%, 52 > 100%, max +513% — and **zero**
  spike-and-revert pairs (a > 100% move undone ≥ 50% the next bar, the signature of a round-trip tick).
  Every extreme move *persists*: they are real memecoin pumps / crashes, not adjustment artifacts.
  Winsorising log-returns at ±ln 2 (a +100%/−50% day) before building skew moves the headline
  **−0.38 → −0.40** — the number is not riding a handful of extreme prints.
- **Dead names are kept, not survived-past.** 116 names end below 2% of their all-time peak (LUNA,
  which the panel correctly ends at 2022-05-18, and many memecoins). Their presence is *good* — the
  cross-section is not biased toward survivors. The residual hazard is that a position held into a
  delisting stops marking-to-market when the series goes NaN, so terminal-crash losses are
  under-captured. A **delisting-trimmed** variant (NaN the final 5 bars of any name that delists having
  crashed, `lottery.predelist_mask`) nets **−0.38**, identical to untrimmed — the verdict does not hinge
  on delisting mechanics.
- **ffill flats** (3.87% of bars, from `build_panels`' 5-bar gap-fill) inject return-0 rows that merely
  dilute a moment window; they cannot manufacture an edge.

Unlike overnight, there is no data bug doing the work. This is the honest, economic result.

## 3. Results — inverted in crypto, sub-bar in equity, dead everywhere

**Chosen a-priori config** (skew 30d, tercile, monthly, top-100-liquid), crypto primary:

| metric | skew-short (lottery) | MAX-short (lottery) | skew-long (≈ momentum) |
|---|---|---|---|
| **net Sharpe** | **−0.38** | **−0.67** | +0.15 |
| MC [P5, P50, P95] | [−1.06, −0.42, +0.22] | — | — |
| max drawdown | −48.9% | — | — |

- **Surface (§10 sensitivity):** all **24** cells (skew/MAX × window {20,30,45,60}/{14,21,30,45} × tail
  {0.1,0.2,0.3}) are **negative**, −0.01 → −0.71. Not a bad-parameter artifact — there is no positive
  region. 0% of cells clear 0.5, 0% are even positive.
- **Per-year:** negative in **6 of 7** years; the sole positive year is **2022 (+1.6)** — the crypto
  crash, when shorting the pumped high-skew names paid because they fell hardest. It is a crash-tilt,
  not a stable premium.
- **Placebo:** the real book sits at the **31st percentile** of 100 column-shuffled runs (placebo mean
  −0.15) — the true cross-section loses *more* than the median shuffle. Noise clears the +0.5 robust
  bar in **3%** of runs, so the pipeline is not generous; the strategy simply has no positive signal.
- **Purged/embargoed walk-forward OOS:** stitch annual OOS blocks, selecting the best-Sharpe config
  (top-5 ensemble) on each expanding training block with a 60-bar embargo gap — OOS Sharpe **−0.43**,
  in-sample best only **−0.01**. Even *choosing* the construction out-of-sample cannot make it positive.
- **Cost:** −0.38 → −0.46 (2×) → −0.54 (3×); monotonically worse, never a break-even.
- **Funding:** charged at every 8h settlement, the dollar-neutral book *pays* skew −5.5%/yr, MAX
  −7.5%/yr (the long low-skew leg's positive funding exceeds what the short high-skew leg collects),
  taking the a-priori books to **−0.57** (skew) and **−0.85** (MAX). A headwind, not a wash.
- **Deflated Sharpe:** the single best *positive* book found anywhere in the search (equity-broad
  skew-short, +0.25) deflated at the true **N = 31** trials is **0.10** — below any significance bar.
  Nothing survives the multiple-testing haircut.

## 4. Where the premium isn't — momentum eats it, liquidity excludes it (the edge-map answer)

Two structural reasons the lottery bet fails exactly where the lottery anomaly was expected to pay:

| universe | skew-short | MAX-short | why |
|---|---|---|---|
| **crypto** (300 perps) | **−0.38** | **−0.67** | monthly-horizon momentum dominates; the lottery short = the momentum long, inverted |
| equity broad (692) | +0.25 | −0.95 | faint skew tilt, below 0.5; MAX is pure momentum-continuation on mega-caps |
| equity mid/small (893) | +0.11 | −0.52 | same — the top-100-liquid cut removes the low-priced retail names MAX needs |

The classic MAX anomaly lives in small, low-priced, retail-held lottery stocks. The tradable funnel
ranks only the **top-100 most-liquid** names each bar — which are precisely *not* lottery stocks — so
the universe constraint removes the effect at its source, the same way the daily-round-trip constraint
removed the overnight edge. In crypto there is no liquidity escape: the high-lottery names *are* liquid
(memecoin perps), but there the monthly-horizon **momentum** premium (the book's dominant crypto source)
overwhelms any lottery-reversal, so shorting them loses.

**Timeframe robustness — the sign is not stable.** Re-running the a-priori crypto book on the 4h/1h
panels (signal windows × bars/day, monthly-equivalent rebalance — the sibling momentum book's multi-TF
convention) shows the skew tilt does not merely weaken, it *flips*:

| horizon | skew-short (lottery) | skew-long (≈ momentum) | MAX-short |
|---|---|---|---|
| **1d** | **−0.38** | +0.15 | −0.67 |
| **4h** | +0.20 | −0.46 | −0.33 |
| **1h** | +0.04 | −0.29 | — |

At the daily horizon high-skew names carry momentum (they keep running, so long-high wins slightly); at
intraday horizons they mean-revert (short-high wins slightly). This is diagnostic: a genuine lottery
premium would be **sign-stable across horizons**, whereas a horizon-dependent sign flip is the fingerprint
of the underlying momentum/reversal term structure, not a mispricing. And the intraday "positive" side is
sub-bar (best 4h +0.20 < 0.5) and is short-term *reversal* on intraday-return skew — a different signal
from the daily lottery anomaly. (The intraday cells are counted in the deflated-Sharpe trial total, N=31.)

## 5. Portfolio value — decorrelated but negative, and not independent of low-vol

- **Correlation to the deliverable book: −0.17** (to each stream |corr| ≤ 0.45; notably **−0.45** to the
  crypto x-sect *momentum* sleeve — quantifying that skew-short is largely inverse-momentum). Genuinely
  decorrelated, as a mispricing effect should be — but decorrelation only helps a *positive* sleeve:

| skew-short weight | 0% | 15% | 30% |
|---|---|---|---|
| book Sharpe | 1.62 | 1.57 | 1.43 |

- **Not an independent lottery effect.** Regressed on a low-vol/BAB proxy (short high-vol / long low-vol
  through the same engine): corr **+0.42**, beta 0.42, alpha **−0.03/yr**, **residual Sharpe 0.00**.
  Against H1's actual crypto dollar-neutral BAB book (`reports/bab/bab_returns.parquet`): corr +0.14,
  **residual Sharpe 0.00**. Once the low-vol component is stripped there is no residual lottery premium
  — the key worry ("show it is independent, not re-labelled low-vol") resolves against the
  hypothesis: it is *partly* low-vol and its residual is pure noise.

So the correct portfolio decision is to **exclude it**. Documented, not traded.

## 6. Honest verdict & ceiling

- **Reachable here:** nothing market-neutral. The lottery/MAX premium is either inverted (crypto, eaten
  by momentum) or real-but-sub-bar and non-independent (equity, and it needs the illiquid retail names
  the tradable universe excludes).
- **Binding constraints:** (1) in crypto, the momentum premium dominates the monthly horizon and the
  lottery short is the momentum long inverted; (2) in equity, the top-100-liquid cut removes the
  low-priced retail names where the anomaly lives; (3) what skew signal remains overlaps low-vol and
  adds zero independent Sharpe.
- **What did not work (kept, not hidden):** both signals (skew, MAX), both signs (the momentum side is
  the only positive one — and it is not new), the full 24-cell surface, the walk-forward OOS, three cost
  levels, and the low-vol / BAB orthogonalisation. The value delivered is the **map** (H2 now covered
  with rigour) and the confirmation that the crypto book's edge is momentum/carry, not retail-lottery
  mispricing — not a sleeve.
- **Follow-up:** the orthogonality vs H1's BAB book is wired in and ran (residual Sharpe 0.00), checked
  against the returns in `reports/bab/bab_returns.parquet`.

## 7. Reproduce

```bash
make lottery       # scripts/lottery/run_lottery.py -> reports/lottery/lottery_{summary.json,surface.csv,returns.parquet}
                   #                            + reports/figures/lottery.png
```

Fixed seed (7) throughout. Signals, the winsor threshold, the delisting-trim rule and the top-k walk-
forward embargo are in `src/sleeves/lottery.py` / `scripts/lottery/run_lottery.py`.
Sources: Bali, Cakici & Whitelaw, "Maxing Out: Stocks as Lotteries and the Cross-Section of Expected
Returns" (JFE 2011); Kumar, "Who Gambles in the Stock Market?" (JF 2009).
