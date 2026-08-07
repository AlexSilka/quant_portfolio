# Architecture and build sequence

The technical skeleton of the project. Strategy and rationale live in
[APPROACH.md](APPROACH.md); this document covers how it is wired in code, in what order it
is built, and where the report, UI and deployment attach later.

## Ordering principle

Research rigour is the substance; the UI and deployment are peripheral work that feeds on
the core's outputs. So the build order is:

```
core pipeline  →  report (from validation)  →  results dashboard  →  deploy
  (the substance)    (§13)                       (review artifact)    (light, last)
```

Nothing on the right is built until the left produces real numbers.

## Repository layout

```
src/config.py            frozen periods / costs / universe params + filesystem layout (single source)
src/
  data/                  Binance bulk (klines+funding), equity daily + intraday (Twelve Data Pro), macro (FRED)
  features/              82 features, computable-at-bar, PIT normalisation
  labels/                triple-barrier + meta-label, sample weights
  sleeves/               per-family primary rules + cross-sectional engine (xsect.py, incl. short-borrow)
  backtest/              liquidity-aware costs (√-impact + funding + borrow) + engine (t+2 execution)
  metrics.py             deflated / probabilistic Sharpe (Bailey–López de Prado)
  risk/                  book vol-target + drawdown-ladder overlay (overlay.py)
  validation/            purged+embargo CV, CSCV/PBO (cscv.py), walk-forward, single OOS, Monte Carlo (4 schemes)
scripts/                 screening (discovery zoo, run_book), portfolio assembly (run_master_book),
                         reporting (make_report/make_figures), CSCV (run_cscv), feature report, OOS ledger, edge map
scripts/
  smoke_test.py          DONE: crypto+equity-daily data layer end-to-end
  smoke_twelvedata.py    DONE: equity intraday loader
  smoke_features.py      DONE: feature engine + look-ahead audit
  run_all.py             `make reproduce` — one command to the headline results
```

## Data contracts (interfaces between layers)

- **Bars:** `DataFrame`, UTC `open_time` index, columns `open/high/low/close/volume/
  quote_volume/count/taker_buy_volume/…`. The source sits behind the interface: crypto
  (Binance) and equities (Twelve Data for daily + intraday) — a professional feed
  plugs into the same contract via config.
- **Funding:** UTC settlement-time index, `last_funding_rate`, `funding_interval_hours`.
  Charged at each settlement on notional.
- **Features:** a wide `DataFrame`, each column stamped to a bar, values from that bar's past
  only.
- **Labels:** `t0`/`t1` (window start/end) + side + meta-label; the CV embargo = the label
  window length.
- **Sleeve P&L:** a per-bar net-return series → the input to the portfolio and validation.

## Where the UI and deploy ideas fit

- **UI = a results/validation dashboard** (not a trading app): edge map, survival funnel,
  portfolio and per-sleeve equity/drawdown, rolling-12m Sharpe, cost sensitivity, trade-off
  frontier. It reads `report/` artifacts — it has no logic of its own — and is the single
  self-contained page a reviewer opens to see the results.
- **Deploy:** a self-contained HTML page (a clickable link, no server) as the primary path;
  Streamlit if a "live" re-run is wanted. Done last and minimal.

## Target dependency stack

The data layer (locked and tested) is in `requirements.txt`. As modules land, dependencies move
here, pinned (versions verified on Python 3.12):

- features/TA: in-house vectorised engine (82 features); selection — in-house IC/stability/redundancy
  report (`scripts/feature_report.py`), no `shap`/`boruta` dependency.
- models: `lightgbm` 4.7, `scikit-learn` 1.5 (macOS: `brew install libomp`).
- validation/overfit: **in-house** deflated Sharpe (`src/metrics.py`) + **in-house CSCV/PBO**
  (`src/validation/cscv.py`); no `purgedcv` dependency. Monte Carlo — `arch` 8.0 block bootstrap +
  in-house trade-order/jitter/random-start schemes.
- labelling: in-house triple-barrier/meta-label (not `mlfinlab` — commercial; not `mlfinpy` —
  immature).
- portfolio: **in-house** equal-weight risk parity + §8 drawdown-ladder overlay (`run_master_book.py`,
  `src/risk/overlay.py`); no `skfolio`/`riskfolio-lib` dependency.
- reporting: in-house `matplotlib` PNGs + self-contained SVG dashboard (no `quantstats`).

## Reproducibility

Python 3.12, fixed seeds (`config.seed`), locked versions. `make reproduce` (→
`scripts/run_all.py`) runs the whole path to the headline results. The final OOS block is run
exactly once and never tuned against.
