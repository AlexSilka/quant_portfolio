# Cross-Asset Alpha Discovery & Portfolio Assembly (Task A)

Systematic cross-asset (US equities + crypto) alpha research and portfolio assembly,
built for honest validation: leakage control, multiple-testing correction, and
realistic execution costs. The deliverable is a portfolio **and** an honest map of
where edge exists and where it does not.

**Headline result** — an eight-family, equal-weight cross-asset book: net **Sharpe 3.52** full-sample
(2011 → 2026) at **−8.1%** max drawdown, positive in all 16 calendar years; on the run-once
out-of-sample block (2024-07 →) **Sharpe 2.64**, meeting **4 of 5** brief targets. The surviving edge is
crypto-heavy and volprem-anchored (short-vol, ~half of book P&L, on a real tail) — both stated and
quantified in [REPORT.md](REPORT.md).

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
  **§8 drawdown-ladder risk overlay** on top, the master book nets **Sharpe 3.52** at **−8.1% max drawdown**,
  mean pairwise cross-family correlation **≈ 0.06**, positive in all 16 calendar years. **On the frozen
  out-of-sample block the brief actually scores (2024-07→), Sharpe is 2.64** — clearing the 2.5 floor on
  genuinely unseen data, not the full-sample figure. The surviving edge is crypto-heavy and the Sharpe is
  volprem-anchored (short-vol, ~half the book P&L, on a real tail) — all quantified in [REPORT.md](REPORT.md).

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
#    there and re-assembles the risk-parity portfolio (Sharpe 3.52 full / 2.64 OOS).
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
