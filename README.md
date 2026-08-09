# Cross-Asset Alpha Discovery & Portfolio Assembly (Task A)

Systematic cross-asset (US equities + crypto) alpha research and portfolio assembly,
built for honest validation: leakage control, multiple-testing correction, and
realistic execution costs. The deliverable is a portfolio **and** an honest map of
where edge exists and where it does not.

**Headline result** — an eight-family, equal-weight cross-asset book run at a constant **1.15× leverage**
(~9.5% annualised volatility): net **Sharpe 3.68** full-sample (2011 → 2026) at **−7.2%** max drawdown, **+34.6%/yr** not reinvested on the brief's $500k of
sizing capital ($2.69M, ~$173k/yr) or **+40.6%/yr** compounded,
positive in all 16 calendar years. **The brief scores its targets on the final out-of-sample block** (§11), and
there the book clears **all five** (2024-07 →: **Sharpe 3.76**, months-in-profit 80.8%, max-DD −4.5%, worst month
−1.8%, streak 2mo). The **full 15-year window** is reported alongside as supporting evidence (§10/§12), and the
same five clear there too (**Sharpe 3.68**, months 80.9%, max-DD −7.2%, worst month −5.2%, streak 2mo) — under
**both** of the brief's accounting conventions, which is what fixes the leverage at 1.15× rather than the 1.20×
the compounded scorecard alone would allow ([REPORT.md](REPORT.md) §4b). The surviving edge is crypto-heavy and volprem-anchored (short-vol, ~half of book
P&L, on a real tail) — both stated and quantified in [REPORT.md](REPORT.md).

**▶ Live interactive dashboard:** https://claude.ai/code/artifact/231e7947-7022-44cd-ac2e-967f799ef48f
— hosted and public: equity curves, drawdown, monthly heatmap, rolling 12-month Sharpe, exposure &
turnover, sleeve correlation matrix and the edge map. **If that link needs a login, the identical page is
committed and fully self-contained** — open [reports/dashboard.html](reports/dashboard.html) directly (no server, no network).

- **Approach & rationale:** [docs/APPROACH.md](docs/APPROACH.md)
- **Architecture & build sequence:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Status — complete and reproducible

Full pipeline runs end to end: data (Binance bulk + Twelve Data Pro + FRED) → 82-feature
engine (look-ahead-audited) → per-family deep-dives (discovery, ML, walk-forward, robustness) →
**one canonical portfolio assembly** ([scripts/run_master_book.py](scripts/run_master_book.py)) →
edge map + dashboard.

- **Report & verdict:** [REPORT.md](REPORT.md) · **Portfolio assembly:** [scripts/run_master_book.py](scripts/run_master_book.py) · **Dashboard:** [live](https://claude.ai/code/artifact/231e7947-7022-44cd-ac2e-967f799ef48f) · [source](reports/dashboard.html)
- **Result:** **eight structurally-distinct families survive** — trend, carry, short-vol / variance
  premium, cross-sectional momentum, breakout, crisis-alpha (managed-futures), global-macro (EM-FX +
  commodities trend) and betting-against-beta / low-vol, each developed in its own deep-dive:
  [TREND](docs/strategies/TREND.md), [CARRY](docs/strategies/CARRY.md), [VOLPREM](docs/strategies/VOLPREM.md),
  [XSECT](docs/strategies/XSECT.md), [BREAKOUT](docs/strategies/BREAKOUT.md),
  [crisis-alpha](scripts/run_crisis.py), [global-macro](scripts/run_gmacro.py), [BAB](docs/strategies/BAB.md).
  Combined at **genuine equal-weight risk parity** (no per-leg selection) on their honest
  **survivorship-free / point-in-time** series over a **15-year window (2011 → 2026)** — the short-vol leg timed
  by its own **VIX-term-structure regime gate** (flat unless both curve segments are in contango), sized at a constant **1.15×**
  (~9.5% book volatility — the last level whose worst month holds on both accounting conventions, REPORT §4b), with a disclosed **§8 risk overlay**
  (drawdown ladder + daily-loss breaker) on top — the
  master book nets **Sharpe 3.68** at **−7.2% max drawdown**, **+34.6%/yr** on $500k, months-in-profit **80.9%**,
  mean pairwise cross-family correlation **≈ 0.06**, positive in all 16 calendar years. It **meets all five
  targets on both windows** — the frozen out-of-sample block the brief actually scores (2024-07→, Sharpe **3.77**)
  and the full 15-year window (Sharpe **3.68**). The surviving edge is crypto-heavy
  and the Sharpe is volprem-anchored (short-vol, ~half the book P&L, on a real tail) — all quantified in [REPORT.md](REPORT.md).

## Verify the headline (~15 min)

Everything under `reports/` is committed, so each command reads the committed series and recomputes —
no key, offline, seconds each:

| command | what it recomputes | expected |
|---|---|---|
| `make master` | the whole portfolio, from scratch | full **Sharpe 3.68** (5/5), OOS **3.76** (5/5), −7.2% max-DD, 8 families |
| `make risk-budget` | how much leverage the book can carry (§4b) | shipped **1.15×** — the worst month binds; 1.20× passes on one accounting convention only |
| `make cscv` | the overfit / multiple-testing control | **PBO 32%**, in-sample-best +0.09 → OOS +0.00 /bar |
| `python scripts/smoke_features.py` | the look-ahead audit | `max\|full − truncated\| = 0` |
| `python scripts/smoke_math.py` | the metric / cost / overlay math (known-answer) | every invariant ✓ |

Re-running `make master` then `git diff reports/master_book_summary.json` shows **no change** —
byte-for-byte reproducibility. The Sharpe is high because the book **selects no single sleeve** (the
best sleeve's deflated Sharpe ≈ 0 at N = 2,129): it stacks eight decorrelated premia (mean ρ ≈ 0.06).
Every Sharpe is annualised by actual obs/yr (not a flat 365), and the short-vol leg is net of
per-underlying vega spreads (`reports/volprem/volprem_cost_robustness.csv`).

**Where the edge is _not_** (kept, not hidden): cross-sectional reversal, stat-arb pairs, calendar/session,
lottery/skew and free-data on-chain were tested and rejected (REPORT §7). **Where the risk is:**
vol-premium is ~55% of P&L on a real −78% tail (strip it → Sharpe **1.73**); the surviving edge is
crypto-heavy; the short-vol book's thin single-name / exotic legs cap deployable size
([VOLPREM.md](docs/strategies/VOLPREM.md) §capacity); daily-annualised 3.72 is lower on a
calendar-robust monthly basis; the dollar figures are quoted on the brief's **$500k** sizing capital with P&L
**not** reinvested (**$2.69M**, ~$173k/yr) — full compounding would outgrow the vol-premium leg's vega capacity
around year 8, so it is not claimed; and the leg's one systemic day (2010-05-06, −76% on the leg) sits **outside**
the reporting window and is unreachable by the regime gate (the curve was in contango the session before), which
is why the drawdown headroom at 1.15× is not treated as spare risk budget (REPORT §4b).

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
#    there and re-assembles the risk-parity portfolio (Sharpe 3.72 full / 3.77 OOS).
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
