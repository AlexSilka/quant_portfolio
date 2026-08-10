# Crisis-alpha — multi-asset managed-futures trend, and how big a slot a hedge deserves

> **Canonical-book note.** Single-family deep-dive. The shipped portfolio is the six-family master book
> in [REPORT.md](../../REPORT.md); any book figure quoted below is the snapshot at the time this family
> was re-evaluated, not the current headline. The family's own series is built by
> [`scripts/run_crisis.py`](../../scripts/run_crisis.py); the sizing evidence is
> [`scripts/run_crisis_lab.py`](../../scripts/run_crisis_lab.py) → `reports/lab/crisis_lab.json`.

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
- **What was mis-set is the SIZE.** Held flat at one equal-risk slot the leg costs the book ~8pp of CAGR
  and, since the vol-premium leg gained its regime gate, *lengthens* the losing streak it was bought to
  shorten. Ramping the slot on market stress — a quarter slot when nothing is moving, a slot and a half
  when the VIX curve inverts or the S&P is 12% off its trailing-year high — buys the **same average
  protection at the times it pays**. That is what ships (`run_master_book.HEDGE_SLOT`,
  [`src/risk/stress.py`](../../src/risk/stress.py)).

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
| ramped on market stress | **+58.6%** / **−5.10%** | **+45.4%** |
| the same ramp, rotated to the wrong days | +55.1% / −6.55% | +43.5% |
| held flat at the ramp's own average | +54.6% / −6.20% | +42.2% |

The rotated ramp lands on the flat one — it re-sizes as often and as violently, just not when it matters
— so the **timing** is what pays, not the smaller average. The gain holds in **all five** sub-windows of
2011-2026, and every neighbouring ramp tested (0.0-1.0 through 0.5-2.0) also beats the flat slot on both
return and worst month.

**A crypto-drawdown term was built and rejected.** The book's crypto legs are dollar-neutral, so a BTC
drawdown is not stress for *this* book — correlation +0.09, and the book earns +3.4%/mo with BTC more
than 20% off its high against +4.7% otherwise, same worst month either way. Including it raised the
hedge's average weight by a third and moved neither the worst month nor the drawdown.

## 4. What it costs, stated

The full-window Sharpe rises past the brief's 4.0 ceiling, so that window now fails a target it used to
pass — and it used to pass it *because* a Sharpe-0.5 leg was dragging the ratio down. Holding a weak leg
to flatter a ratio is not a risk control. All four risk targets clear on both windows, and the frozen
block the brief scores keeps all five.

## 5. Reproduce

```
python scripts/run_crisis.py        # the family series -> reports/book/crisis_sleeve.parquet
python scripts/run_crisis_lab.py    # the whole evidence base -> reports/lab/crisis_lab.json
```

The lab publishes the per-class cost diagnosis, every construction variant standalone and in-book at
three sizings, the rotated-conditioner and flat-weight controls, the ramp neighbourhood, and the
crash-window table each verdict rests on.
