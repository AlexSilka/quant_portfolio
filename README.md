<!-- Generated from scripts/report_assets/readme.md by scripts/render_report.py — edit the
     template, not this file. Every figure below is resolved from reports/ at render time. -->
# Cross-Asset Alpha Discovery & Portfolio Assembly (Task A)

Systematic cross-asset (US equities + crypto) alpha research and portfolio assembly, built for honest
validation: leakage control, multiple-testing correction, realistic execution costs. The deliverable is a
portfolio **and** an honest map of where edge exists and where it does not.

**▶ Live dashboard:** https://claude.ai/code/artifact/231e7947-7022-44cd-ac2e-967f799ef48f — equity curves,
drawdown, monthly heatmap, rolling Sharpe, exposure, correlations, edge map. If that link asks for a login the
identical page is committed and self-contained: open [reports/dashboard.html](reports/dashboard.html) (no server,
no network).

---

## The result in one page

An **eight-family, equal-weight cross-asset book** at a constant **1.15× leverage** (~9.5% annualised vol).
The brief scores its five targets on the **final out-of-sample block**; the 15-year window is shown alongside
as supporting evidence, not as a second scorecard.

| §11 target | OOS block (2024-07 →) | full window (2011 → 2026) |
|---|---|---|
| Sharpe, net, 2.5–4.0 | **3.63** ✓ | **3.63** ✓ |
| months in profit ≥ 80% | **80.8%** ✓ | **80.9%** ✓ |
| max drawdown ≤ 15% | **−4.7%** ✓ | **−7.4%** ✓ |
| longest losing streak ≤ 2 mo | **2** ✓ | **3** ✗ |
| worst single month ≥ −6% | **−2.0%** ✓ | **−4.1%** ✓ |
| | **5 / 5** | **4 / 5** |

The one miss is the full window's three-month losing streak — Dec-2021 −2.5%, Jan-2022 −0.8%,
Feb-2022 −1.7%, three months comfortably negative rather than a rounding artifact. It appeared when the
equity panel was corrected to stop carrying foreign companies behind dead US tickers; the substituted
prices had been filling one of those months in. Nothing was re-tuned to win it back, and the leverage
grid shows nothing could be — scaling every return leaves the sign of a month exactly where it was.

On the brief's $500k of sizing capital that is **$2.69M** of P&L, **~$172k/yr**
(+34.5%/yr not reinvested, +40.5%/yr compounded). Positive in **16 of 16 calendar years**.
Mean pairwise correlation between families **≈ 0.06**.

**The eight sources** — each developed in its own deep-dive, combined at genuine equal-weight risk parity
(no per-leg selection), every one on a **survivorship-free / point-in-time** universe:

| family | what it earns on | Sharpe | share of P&L |
|---|---|---|---|
| [short-vol / VRP](docs/strategies/VOLPREM.md) | selling insurance against volatility across 18 Cboe underlyings | +5.71 | **57%** |
| [global-macro](scripts/run_gmacro.py) | trend on EM FX + commodities — asset classes no other family trades | +0.93 | 9% |
| [trend](docs/strategies/TREND.md) | price trend, the only family spanning both asset classes | +0.89 | 7% |
| [BAB / low-vol](docs/strategies/BAB.md) | the leverage-constraint premium: long low-beta, short high-beta | +1.29 | 6% |
| [breakout](docs/strategies/BREAKOUT.md) | channel breakouts held on a trailing stop, ML-gated on fast bars | +1.38 | 6% |
| [x-sect momentum](docs/strategies/XSECT.md) | relative strength, market-neutral | +0.85 | 7% |
| [carry](docs/strategies/CARRY.md) | perpetual funding: being paid to hold the unpopular side | +1.27 | 5% |
| [crisis-alpha](scripts/run_crisis.py) | long-gamma managed futures — it pays when the other seven bleed | +0.46 | 4% |

The short-vol leg carries its own **VIX-term-structure gate** (flat unless both curve segments are in contango),
which is what holds the worst month and the losing streak. Remove that leg entirely and a genuine **Sharpe 1.55**
book still stands.

**Three honest limits, quantified in [REPORT.md](REPORT.md), not buried:**

1. **Concentration.** Short-vol is 56% of P&L and its standalone tail is **−78%** (one day: −76% in the 2010
   flash crash). No term-structure rule reaches that day — the curve was in contango the session before.
2. **Capacity.** That same leg is a variance-swap *replication*, not an executed option book, and the 18-leg
   construction caps out around low tens of $M before the thin legs stop filling.
3. **Crypto-heavy.** Breakout, carry, BAB and x-sect are crypto; only trend spans both classes. US single-name
   and FX breakout did not survive — reported, not hidden.

**Read next:** [REPORT.md](REPORT.md) for the full argument · [docs/APPROACH.md](docs/APPROACH.md) for rationale ·
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the build sequence ·
[scripts/run_master_book.py](scripts/run_master_book.py) is the one script that assembles the portfolio.

---

## Status — complete and reproducible

Full pipeline runs end to end: data (Binance bulk + Twelve Data Pro + FRED) → 82-feature engine
(look-ahead-audited) → per-family deep-dives (discovery, ML, walk-forward, robustness) → **one canonical
portfolio assembly** ([scripts/run_master_book.py](scripts/run_master_book.py)) → edge map + dashboard.

## Verify the headline (~15 min)

Everything under `reports/` is committed, so each command reads the committed series and recomputes —
no key, offline, seconds each:

| command | what it recomputes | expected |
|---|---|---|
| `make master` | the whole portfolio, from scratch | full **Sharpe 3.63** (4/5), OOS **3.63** (5/5), −7.4% max-DD, 8 families |
| `make risk-budget` | how much leverage the book can carry (§4b) | shipped **1.15×**; bootstrap-P5 max-DD is what binds first, at 1.45× |
| `make cscv` | the overfit / multiple-testing control | **PBO 13%**, in-sample-best +0.088 → OOS +0.004 /bar |
| `python scripts/smoke_features.py` | the look-ahead audit | `max\|full − truncated\| = 0` |
| `python scripts/smoke_math.py` | the metric / cost / overlay math (known-answer) | every invariant ✓ |

Re-running `make master` then `git diff reports/master_book_summary.json` shows **no change** —
byte-for-byte reproducibility. The Sharpe is high because the book **selects no single sleeve** (the
best sleeve's deflated Sharpe ≈ 0.00 at N = 2,129): it stacks eight decorrelated premia (mean ρ ≈ 0.06).
Every Sharpe is annualised by actual obs/yr (not a flat 365), and the short-vol leg is net of
per-underlying vega spreads (`reports/volprem/volprem_cost_robustness.csv`).

**Where the edge is _not_** (kept, not hidden): cross-sectional reversal, stat-arb pairs, calendar/session,
lottery/skew and free-data on-chain were tested and rejected (REPORT §7).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .        # pinned deps (from requirements.txt) + the src package, editable
# macOS only: brew install libomp   # OpenMP runtime for lightgbm
```

No API key is needed to read the results or to reproduce the headline offline — the committed `reports/`
folder already holds every output. A Twelve Data key is required only to rebuild equities/FX from raw data.

## Data sources

- **Crypto:** `data.binance.vision` bulk dumps (spot + USD-M perp klines, funding).
  Full 5m/15m depth (BTC since 2017). No key required.
- **Equities / FX:** Twelve Data Pro (professional feed) — split-adjusted daily from 2006 and
  intraday (5m/15m/1h) from ~2020, one bar contract. Yahoo/yfinance was dropped as unreliable for an
  unattended `make reproduce` (see [src/data/equity.py](src/data/equity.py)).
- **Macro:** FRED `fredgraph.csv` (rates/DXY/VIX), applied with a **1-month release lag** as the
  vintage proxy — no ALFRED first-release vintage feed is wired (no FRED key); the lag is defended for the
  near-non-revised 3-month interbank series it uses.

## Reproduce

**Nothing needs to run to read the results** — every output is committed under `reports/` and `docs/`:
[REPORT.md](REPORT.md), the [dashboard](reports/dashboard.html), the §13 charts ([reports/figures/](reports/figures/)),
the per-year / per-quarter tables and the out-of-sample trade log ([reports/](reports/)), and the eight
per-family write-ups ([docs/](docs/)).

```bash
# 1. Reproduce the headline OFFLINE — no key, no download, ~seconds. Works on a fresh clone as-is:
#    because reports/ is committed, run_master_book.py simply reads the eight family series already
#    there and re-assembles the risk-parity portfolio (Sharpe 3.63 full / 3.63 OOS).
make master

# 2. Rebuild the pipeline from raw data — discovery, the crisis/gmacro diversifier legs, validation,
#    master-book assembly, CSCV, charts, dashboard. Budget ~1 hour: it mines the FULL 2,129-candidate
#    grid including 5m/15m, because the trial count is what sets the deflation haircut quoted in the
#    report — reproducing on the cheap 1h/4h/1d grid would report a smaller N and a weaker penalty.
#    (The other six family deep-dives are heavy one-offs; rebuild any from its own target, e.g.
#    `make trend carry volprem xs breakout bab`.)
cp .env.example .env            # paste a Twelve Data key (equities/FX); crypto (Binance) + macro (FRED) need none
python scripts/smoke_test.py    # optional: proves the data layer end to end (needs the key)
make reproduce                  # crypto auto-downloads keyless (~10 GB, cached to data/); without a key the
                                # equities step stops immediately with "TWELVEDATA_API_KEY not set"
```

Fixed seeds throughout; the final out-of-sample block is run exactly once.
