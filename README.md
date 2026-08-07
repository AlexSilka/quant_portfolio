# Cross-Asset Alpha Discovery & Portfolio Assembly (Task A)

Systematic cross-asset (US equities + crypto) alpha research and portfolio assembly,
built for honest validation: leakage control, multiple-testing correction, and
realistic execution costs. The deliverable is a portfolio **and** an honest map of
where edge exists and where it does not.

**Headline result** — an eight-family, equal-weight cross-asset book: net **Sharpe 3.77** full-sample
(2011 → 2026) at **−8.0%** max drawdown, positive in all 16 calendar years. On the run-once **out-of-sample
block** (2024-07 →, the window the brief scores) it **meets all five targets** (**Sharpe 3.61**, months-in-profit
81%, worst-month −2.1%, streak 2mo); on the **full 15-year window** it meets **4 of 5** — the one miss a single
3-month losing streak (vs ≤2). The surviving edge is crypto-heavy and volprem-anchored (short-vol, ~half of book
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
  **survivorship-free / point-in-time** series over a **15-year window (2011 → 2026)**, with a disclosed
  **§8 risk overlay** on top (drawdown ladder **+ a VIX-term-structure regime gate on the short-vol leg**), the
  master book nets **Sharpe 3.77** at **−8.0% max drawdown**, months-in-profit **80%**,
  mean pairwise cross-family correlation **≈ 0.06**, positive in all 16 calendar years. **On the frozen
  out-of-sample block the brief actually scores (2024-07→) it meets all five targets (Sharpe 3.61)**; on the full
  15-year window it meets **4 of 5** (the one miss a 3-month losing streak vs ≤2). The surviving edge is crypto-heavy
  and the Sharpe is volprem-anchored (short-vol, ~half the book P&L, on a real tail) — all quantified in [REPORT.md](REPORT.md).

## Verify the headline (~15 min)

Everything under `reports/` is committed, so each command reads the committed series and recomputes —
no key, offline, seconds each:

| command | what it recomputes | expected |
|---|---|---|
| `make master` | the whole portfolio, from scratch | full **Sharpe 3.77** (4/5), OOS **3.61** (5/5), −8.0% max-DD, 8 families |
| `make cscv` | the overfit / multiple-testing control | **PBO 32%**, in-sample-best +0.09 → OOS +0.00 /bar |
| `python scripts/smoke_features.py` | the look-ahead audit | `max\|full − truncated\| = 0` |
| `python scripts/smoke_math.py` | the metric / cost / overlay math (known-answer) | every invariant ✓ |

Re-running `make master` then `git diff reports/master_book_summary.json` shows **no change** —
byte-for-byte reproducibility. The Sharpe is high because the book **selects no single sleeve** (the
best sleeve's deflated Sharpe ≈ 0 at N = 1,279): it stacks eight decorrelated premia (mean ρ ≈ 0.06).
Every Sharpe is annualised by actual obs/yr (not a flat 365), and the short-vol leg is net of
per-underlying vega spreads (`reports/volprem/volprem_cost_robustness.csv`).

**Where the edge is _not_** (kept, not hidden): cross-sectional reversal, stat-arb pairs, calendar/session,
lottery/skew and free-data on-chain were tested and rejected (REPORT §7). **Where the risk is:**
vol-premium is ~52% of P&L on a real −78% tail (strip it → Sharpe **1.75**); the surviving edge is
crypto-heavy; the short-vol book's thin single-name / exotic legs cap deployable size
([VOLPREM.md](docs/strategies/VOLPREM.md) §capacity); and daily-annualised 3.77 is ~2.9 on a
calendar-robust monthly basis.

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
#    there and re-assembles the risk-parity portfolio (Sharpe 3.77 full / 3.61 OOS).
make master

# 2. Rebuild the pipeline from raw data — discovery, the crisis/gmacro diversifier legs, validation,
#    master-book assembly, CSCV, charts, dashboard. (The other six family deep-dives are heavy one-offs;
#    rebuild any from its own target, e.g. `make trend carry volprem xs breakout bab`.)
cp .env.example .env            # paste a Twelve Data key (equities/FX); crypto (Binance) + macro (FRED) need none
python scripts/smoke_test.py    # optional: proves the data layer end to end (needs the key)
make reproduce                  # crypto auto-downloads keyless (~10 GB, cached to data/); without a key the
                                # equities step stops immediately with "TWELVEDATA_API_KEY not set"
```

Fixed seeds throughout; the final out-of-sample block is run exactly once.
