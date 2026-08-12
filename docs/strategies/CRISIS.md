# Crisis-alpha — multi-asset managed-futures trend, and how big a slot a hedge deserves

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the equal-weight master book assembled by `scripts/run_master_book.py`; its composition, scorecard, leverage and target verdict live in [REPORT.md](../../REPORT.md), which is RENDERED from the artifacts and so cannot disagree with the run. Restated here they would go stale the next time the book is re-run, which is exactly what happened to the numbers this line used to carry — so this page quotes none of them. Any master-book figure below is a snapshot from when this family was evaluated, and is labelled as one.

**Scope.** The book's long-gamma leg. Five liquid classes — equity indices/sectors, commodities,
Treasuries and credit, FX, and top-20 crypto — each traded with the same multi-lookback time-series
momentum, per-asset vol-targeted, the three speed tranches averaged, the class vol-targeted to 15%, the
five classes combined at equal risk. Long uptrends and **short** downtrends, so it turns positive in
sustained sell-offs (Hurst-Ooi-Pedersen; Moskowitz-Ooi-Pedersen). Signals lagged one bar, ~2bps per
unit of turnover.

---

## 0. TL;DR

- **The leg is weak on its own terms and that is mostly the strategy class, not this build.** Standalone
  Sharpe **+0.51** over 21 years, **+7.1%** a year, **−11.9%** in 2025. Kaminski-Wen (AlphaSimplex,
  Jun-2025) date the trend industry's drawdown from Apr-2024 at **−21.8%** on the SG Trend Index — the
  second-deepest since 2000, behind only Trade War 1.0 (2015-19, −23%). This leg's −11.9% is half that.
- **Improving the SIGNAL makes it worse at its job.** Baz et al.'s EWMAC with a response function lifts
  standalone Sharpe to **+0.66** and CAGR to **+9.2%** — and takes COVID from **+13.9% to +0.9%** and the
  2011 sell-off from **+11.6% to −4.1%**. The response function damps an already-extended trend, and a
  crash *is* an extended trend. A leg selected on its own ratio would have shipped and stopped hedging.
- **What was mis-set is the SIZE, and fixing it is what ships.** Ramping the slot on market stress — a
  quarter slot when nothing is moving, a slot and a half when the VIX curve inverts or the S&P is 12% off
  its trailing-year high — buys the **same average protection at the times it pays**, and beats both of
  its controls in all five sub-windows. Against the flat slot it is better on every metric that measures
  the book: CAGR **49.5% → 57.0%** full and **41.5% → 47.2%** on the frozen block, worst month **−5.72% →
  −5.10%**, losing streak **3 → 2**, months-in-profit **81.4% → 84.6%**, drawdown **−8.3% → −7.7%**
  (`run_master_book.HEDGE_SLOT`, [`src/risk/stress.py`](../../src/risk/stress.py)).

---

## 1. Where the leg leaks: turnover, not signal

Per class, the shipped sign-blend nets **+0.25** equity, **+0.67** commodity, **+1.28** crypto — and
**−0.27 bonds**, **−0.14 FX**. With the cost switched off the same two books are **+0.05** and **+0.17**:
the negative sign is a turnover artifact, not a dead signal.

| class | Sharpe, costs off | Sharpe, costs on | turnover / yr |
|---|---|---|---|
| equity | +0.29 | +0.21 | 34× |
| commodity | +0.65 | +0.56 | 29× |
| bonds | +0.05 | −0.23 | **91×** |
| FX | +0.17 | −0.12 | **70×** |
| crypto | +1.31 | +1.28 | 9× |

The mechanism is specific: a ±1 sign rule flips a position by its full size on a one-day sign change, and
on a low-vol asset that position sits at the 3× leverage cap. Bonds therefore pay ~28bps of Sharpe a year
to trade a signal worth ~5bps. Crypto, whose vol keeps its leverage near 0.3×, turns over a ninth as much.

## 2. Construction changes tested, and why none of them ships

Each lever is defensible a-priori rather than picked off a result, and none is selected on 2024-26 — that
window is the industry drawdown above, and fitting to it would fit the premium away.

| variant | standalone Sharpe | 2008 GFC | 2011 | 2018 Q4 | COVID | yen 2024 |
|---|---|---|---|---|---|---|
| sign blend (**ships**) | +0.51 | **+33.5%** | **+11.6%** | +9.5% | **+13.9%** | **+3.6%** |
| EWMAC + response function | **+0.66** | +25.6% | −4.1% | +5.4% | +0.9% | −0.2% |
| Donchian channel | +0.51 | **+40.6%** | +10.6% | **+10.2%** | **+24.1%** | **+4.6%** |
| sign + signal smoothing | +0.63 | +36.2% | +10.0% | +9.0% | +11.5% | +2.2% |
| sign + wider ETF panel | +0.39 | +24.3% | +10.3% | +8.0% | +12.8% | +4.0% |

- **EWMAC and its relatives buy Sharpe with convexity.** Every EWMAC variant tested goes *negative*
  through 2011 and gives up most of COVID. This is the single most useful result in the family: the
  ratio and the purpose point in opposite directions, and only a crash-window table shows it.
- **Breadth hurts.** Adding sector, single-country and extra commodity/bond ETFs takes the leg to +0.39.
  Sector ETFs are the S&P again — the panel gets wider without getting more independent.
- **Donchian is the best hedge on the table** and the honest runner-up: same Sharpe, materially better
  crash payoff. It is not shipped because at the sizing that ships it lands slightly behind the sign
  blend on the book's scored window, and switching the signal on that margin is cell-picking. Its full
  numbers are in the lab artifact.

## 3. The size is the lever

A hedge and an earner want opposite sizing rules. Through a calm decade this leg is the weakest earner in
the book (standalone ~0.5 against a book above 3), so a full slot dilutes every calm month; through a
crash it is the only leg paying. Held flat it had also stopped covering the failure it was bought for:
with the vol-premium leg's regime gate in place, the book's losing streak is **2 months without the leg
and 3 with it**, and months-in-profit falls 86% → 81%.

The shipped rule ramps the slot linearly on a 0..1 market-stress reading — the max of the VIX term
structure (VIX/VIX3M, 0.90 calm → 1.05 inverted) and the S&P's drawdown from its trailing-year high
(0 → −12%), both read at t−1. Average slot ≈ **0.70**, so this is not a way of holding more hedge.

**Controls, because holding less of a drag raises return on its own.** At the *same average weight*:

| the slot at the same average weight | full window: CAGR / worst month | frozen block: CAGR |
|---|---|---|
| ramped on market stress | **+57.0%** / **−5.10%** | **+47.2%** |
| the same ramp, rotated to the wrong days | +53.6% / −6.55% | +45.4% |
| held flat at the ramp's own average | +53.2% / −6.20% | +44.2% |

The rotated ramp lands on the flat one — it re-sizes as often and as violently, just not when it matters
— so the **timing** is what pays, not the smaller average. The gain holds in **all five** sub-windows of
2011-2026, and every neighbouring ramp tested (0.0-1.0 through 0.5-2.0) also beats the flat slot on both
return and worst month.

**A crypto-drawdown term was built and rejected.** The book's crypto legs are dollar-neutral, so a BTC
drawdown is not stress for *this* book — correlation +0.09, and the book earns +3.4%/mo with BTC more
than 20% off its high against +4.7% otherwise, same worst month either way. Including it raised the
hedge's average weight by a third and moved neither the worst month nor the drawdown.

## 4. The one thing it "costs"

Book Sharpe rises past the brief's 2.5–4.0 band (3.66 → 4.06). That is a *ceiling*, not a risk, and an
earlier version of this work held the hedge flat to stay under it — which is holding a weak leg to flatter
a ratio, and is not a risk control. Every metric that measures the book rather than a threshold improves.

The prior question, for a book run for return, is whether this leg belongs at all: at **+0.50** standalone
it is the weakest earner in the set, and dropping it along with global-macro takes the portfolio from
49.5% a year to **88.9%** at the same leverage. That book is `scripts/run_live_book.py`, and it does not
hold crisis. This leg earns its slot in a portfolio that wants the crash payoff; it does not earn one in a
portfolio that wants the most money.

## 5. Reproduce

```
python scripts/run_crisis.py        # the family series -> reports/book/crisis_sleeve.parquet
python scripts/run_crisis_lab.py    # the whole evidence base -> reports/lab/crisis_lab.json
```

The lab publishes the per-class cost diagnosis, every construction variant standalone and in-book at
three sizings, the rotated-conditioner and flat-weight controls, the ramp neighbourhood, and the
crash-window table each verdict rests on.
