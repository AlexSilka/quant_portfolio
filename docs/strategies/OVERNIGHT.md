# Calendar / session — deep-dive findings

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the eight-family equal-weight master book in [REPORT.md](../../REPORT.md) (Sharpe **3.52** full / **2.64** OOS). Any master-book Sharpe quoted below is the book *snapshot at the time this family was evaluated*, not the current headline; the canonical assembler is `scripts/run_master_book.py` (not the older `build_book.py`).

**Scope.** Calendar and session effects are a standard feature family. This is the
strategy sleeve that trades it, run through the same funnel as every other family (vol-target 15%,
t+2-style delay, liquidity-aware costs, block-bootstrap MC, shuffled-signal placebo, cost sensitivity,
correlation to the deliverable book). All numbers net of costs, US equities, 2016-01→2026-08, the
692-name point-in-time broad panel, top-100-liquid each bar. Figure:
[reports/figures/overnight.png](../../reports/figures/overnight.png). Reproduce: `make overnight`.

---

## 0. TL;DR

- **The overnight/intraday split is real but is NOT a tradable market-neutral sleeve.** It joins
  mean-reversion, cross-sectional reversal and pairs in the "tested, real-as-beta-or-artifact,
  not viable net of cost" pile — a rigorous edge-map entry, not a hidden gap.
- **The genuine overnight premium is *beta*, not timing.** At the index level it is real and well
  known — SPY overnight Sharpe **0.72** vs intraday **0.45**; IWM (small-cap) overnight **0.99** vs
  intraday **−0.11** (small-cap intraday return is *negative*). But this is long-only, market-directional
  exposure — a *slice* of the trend book, not a decorrelated alpha. It does not diversify the source.
- **The market-neutral cross-sectional book is dead.** Long recent overnight winners / short losers
  (or the reverse), earning only the overnight leg, nets **Sharpe −3.13** (MC P5 −3.61, maxDD −99.7%),
  negative at every lookback × top-fraction and every cost level (break-even at **0.1×** realistic
  cost). The killer is execution: to harvest *only* the overnight leg you must be flat intraday, i.e.
  a **full round-trip every single day** (~2× gross traded daily → 10–30%/yr), which a levered-to-15%
  book cannot survive.
- **The overnight *signal* carries no alpha either.** Held 24h (earning close→close, paying only
  rebalance cost) the overnight-momentum signal nets **−0.57**, essentially the same as a plain
  close-to-close reversal reference (**−0.39**) — the "overnight" framing adds nothing.
- **Two data-quality bugs found and fixed** (each materially moved the number — the cautionary tale):
  a naive first build looked *marginally positive* (+0.18) purely because of them. See §2.
- **Decorrelated (+0.07 to the book) but negative**, so adding it **drags the book down**
  (0.85 → 0.29 at 15% weight). Not deployable. Correctly excluded from the portfolio.

---

## 1. Construction — and why *where* you capture the return is the whole story

Each equity bar splits into two disjoint sessions (`src/sleeves/overnight.py`):

```
overnight[t] = open[t]  / close[t-1] - 1     # the close-to-open gap (held through the night)
intraday[t]  = close[t] / open[t]   - 1      # the open-to-close move (held through the day)
```

The documented anomaly (Cliff-Cooper-Gulen 2008; Lou-Polk-Skouras "A Tug of War" 2019;
Hendershott-Livdan-Rösch 2020) is that the equity premium accrues *overnight* while the intraday leg
is flat/negative, and that the two sessions are traded by different clienteles. The cross-sectional
book ranks names by trailing session return, goes dollar-neutral long-top / short-bottom on the
top-100-liquid names, vol-targets to 15%, and is delayed so a signal never fills at its own bar's
close. Two honest execution models — and the gap between them is the finding:

| model | what it captures | turnover charged |
|---|---|---|
| **overnight-only** | *only* the night leg — flat intraday | enter at close(t−1) **and** exit at open(t) → a **full round-trip every bar** |
| **hold-24h** | close→close, merely *tilted* by the overnight signal | only the rebalance (`w.diff`) |

You cannot have it both ways: capture the isolated overnight leg **or** pay only rebalance cost, not
both. The first probe conflated them (earned overnight PnL, charged only rebalance) and looked ~0.3
better than reality — corrected here.

## 2. Data integrity — two bugs, each of which faked a better number

The honest headline of this sleeve is a *data-integrity* one — exactly what §9 tests.

- **Split-adjustment artifacts.** `open` and `close` are adjusted on their own schedules, so on an
  adjustment day `open / close.shift(1)` prints a spurious ±100%+ (or ∞ when a prior close rounds to
  zero). **543 name-days** carry |overnight| > 50%. Left in, those few hundred rows dominate the
  vol-target and mask the strategy's true economics; the raw panel reports a benign **−0.11** where the
  cleaned panel reports **−3.13**. `session_returns` drops ±∞ and winsorises |session| > 50%; the
  driver reports the raw-vs-clean delta rather than hiding it.
- **Calendar misalignment.** The cached broad panel carries a *union* calendar (forward-filled rows on
  non-NYSE dates). Reindexing the raw open/close onto it scatters NaNs — which silently starved the
  rolling signal so the book was only "live" from 2022 — and makes `close.shift(1)` cross non-sessions
  (a wrong overnight gap). Fixed by anchoring everything to the true NYSE calendar (`_nyse_sessions`,
  from SPY's raw index) *before* building the session panels.

A naive build with both bugs looked *marginally positive* (Sharpe +0.18) and even "survived" a placebo
and a skip-day bounce test — the textbook trap of validating a data artifact. The corrected number is
**−3.13**. This is the sleeve's real contribution: a worked example of how a plausible micro-edge
dissolves under data hygiene.

## 3. Results — dead at every setting

**Chosen a-priori config** (1-month signal, quintile tails, top-100 liquid), momentum sign (the
classic documented direction, long recent overnight winners):

| metric | overnight-only | 24h-hold tilt | plain c2c reference |
|---|---|---|---|
| **net Sharpe** | **−3.13** | −0.57 | −0.39 |
| MC [P5, P50, P95] | [−3.61, −3.14, −2.65] | — | — |
| max drawdown | −99.7% | — | — |
| break-even cost | **0.1×** base | — | — |

- **Surface (§10 sensitivity):** every one of the 12 (lookback × top-fraction) cells is negative,
  −1.45 → −3.85. Not a bad-parameter artifact — there is no positive region.
- **Per-year:** negative in 10 of 11 years (only 2026-YTD +1.0). Consistently, not episodically, dead.
- **Placebo:** the real book sits at the 100th percentile of 100 shuffled-signal runs (placebo mean
  −5.54) — i.e. the true cross-section loses *less* than random, but still loses heavily; noise clears
  the 0.5 robust bar in **0%** of runs (the pipeline's own false-discovery rate here is zero).
- **Cost:** −3.13 → −6.33 (2×) → −9.35 (3×); it only breaks even at **10%** of realistic cost. The
  daily round-trip is decisive.
- **Skip/bounce:** the signal survives a 1–3-bar gap (−3.13 → −2.69), so it is a *real slow signal* —
  it is just a real *losing* one after the execution cost, not bid-ask bounce.

## 4. Where the premium actually is — beta, not timing (the edge-map answer)

At the index/ETF level the overnight premium is unmistakably real, which is why the family is worth
mapping rather than dismissing:

| ETF | overnight Sharpe | intraday Sharpe | buy&hold 24h | overnight %/yr | intraday %/yr |
|---|---|---|---|---|---|
| SPY | 0.72 | 0.45 | 0.80 | +8.2% | +6.1% |
| QQQ | 0.90 | 0.46 | 0.91 | +12.3% | +8.1% |
| **IWM** | **0.99** | **−0.11** | 0.53 | +14.1% | **−2.0%** |
| DIA | 0.64 | 0.40 | 0.71 | +7.3% | +5.1% |

Small-cap IWM is the classic case: essentially **all** of its return accrues overnight and its
intraday leg is *negative*, so overnight-only (0.99) even out-Sharpes buy&hold (0.53). But this is a
**long-only, market-directional** effect — beta earned at night — not a market-neutral source. For a
book already carrying the trend premium it adds exposure to the same factor, not diversification.

## 5. Portfolio value — decorrelated but negative, so it hurts

Correlation of the overnight-only book to the deliverable book is **+0.07** (to each stream: |corr|
≤ 0.14) — genuinely decorrelated, as a session-timing effect should be. But decorrelation only helps a
*positive* sleeve; blending in a −3 Sharpe leg drags the book monotonically:

| overnight weight | 0% | 15% | 30% | 50% |
|---|---|---|---|---|
| book Sharpe | 0.85 | 0.29 | −0.44 | −1.56 |

So the correct portfolio decision is to **exclude it** — which the canonical `run_master_book.py` does (it is not among the eight families). The
sleeve is documented, not traded.

## 6. Honest verdict & ceiling

- **Reachable here:** nothing market-neutral. The overnight premium is real but is beta; there is no
  decorrelated alpha in the session split net of the round-trip cost it demands.
- **Binding constraint:** execution. Isolating the overnight leg forces a daily round-trip whose cost
  (10–30%/yr, levered) exceeds any cross-sectional signal — the same turnover×cost failure mode that
  killed mean-reversion and cross-sectional reversal, in its most acute form.
- **What did not work (kept, not hidden):** both signal directions (momentum and reversal), both
  execution models, four lookbacks × three tail-fractions, and — critically — the naive build whose
  +0.18 was a data artifact. The value delivered is the *map* (§4 family now covered) and the
  *methodology* (a caught, documented data-integrity trap), not a sleeve.

## 7. Reproduce

```bash
make overnight     # scripts/overnight/run_overnight.py -> reports/overnight/overnight_{summary.json,grid.csv,returns.parquet}
                   #                              + reports/figures/overnight.png
```

Fixed seed (7) throughout; the winsor threshold and NYSE-calendar anchor are in `src/sleeves/overnight.py`.
Sources: Cliff, Cooper & Gulen, "Return Differences between Trading and Non-Trading Hours" (2008);
Lou, Polk & Skouras, "A Tug of War: Overnight vs Intraday Expected Returns" (JFE 2019);
Hendershott, Livdan & Rösch, "Asset Pricing: A Tale of Night and Day" (JFE 2020).
