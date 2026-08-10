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

A **{{n_families_word}}-family book** at a constant **{{leverage}} leverage** (~{{book_vol}} annualised vol): the
{{n_families}} earners at equal risk, and the one long-gamma hedge sized by market stress instead of held flat
(a quarter slot when nothing is moving, a slot and a half when the VIX curve inverts — REPORT §6c-ter).
§11 scores the five targets on the **final out-of-sample block**, so that is the scorecard. The 15-year
column is the same book measured over the longer window — supporting evidence, reported because a book that
works only on the block it is scored on is not a book, but not a second scorecard and not counted as one.

| §11 target | OOS block (2024-07 →) | full window (2011 → 2026), not scored |
|---|---|---|
| Sharpe, net, 2.5–4.0 | **{{oos_sharpe}}** ✓ | {{book_sharpe}} |
| months in profit ≥ 80% | **{{oos_months}}** ✓ | {{book_months}} |
| max drawdown ≤ 15% | **{{oos_dd}}** ✓ | {{book_dd}} |
| longest losing streak ≤ 2 mo | **{{oos_streak}}** ✓ | {{book_streak}} |
| worst single month ≥ −6% | **{{oos_worst_month}}** ✓ | {{book_worst_month_2dp}} |
| | **5 / 5** | — |

Over fifteen years the book has one **{{book_streak}}-month** losing run, which is longer than the block's
target allows; it is stated here rather than dropped, and §6d-quater gives the window.

On the brief's {{capital}} of sizing capital that is **{{pnl_usd}}** of P&L, **~{{pnl_usd_per_year}}/yr**
({{return_not_reinvested}}/yr not reinvested, {{return_compounded}}/yr compounded). Positive in **{{n_years_positive}} of {{n_years}} calendar years**.
Mean pairwise correlation between families **≈ {{mean_corr_abs}}**.

**The composition was fixed before the sleeve-level gate below, and has not been re-picked since.** Trend
and carry were dropped under the earlier rule — the one pair, of the {{comp_n_configs}} single- and
double-removal configurations, that then cleared all five targets on both windows. With the gate in place
{{comp_n_passing_word}} configurations clear both; six clear the scored block, and the shipped book is one of
them while the eight-family book is not ({{comp_base_targets_oos}} on the block — {{comp_base_miss_oos}}).
Re-running the search now would mean choosing a composition against the block §10 says to run exactly once,
so the search is published as the denominator (§6d-ter) and the composition is left where it was. Neither
dropped leg is weak on its own terms (carry's standalone Sharpe is {{comp_carry_solo}}, the
{{comp_carry_rank}} of the eight). **Return went up, not down**: {{comp_d_cagr_full}} of CAGR on the full
window and {{comp_d_cagr_oos}} on the block, since six legs at equal risk run hotter than eight. What it costs
is **concentration and breadth**: the short-vol leg's share of P&L up from {{comp_share_before}} to
{{comp_share_after}}, and no family left that spans both asset classes. The eight-family alternative is one
line away in `scripts/run_master_book.py`.

**The {{n_families_word}} sources** — each developed in its own deep-dive, combined at genuine equal-weight
risk parity (no per-leg weighting *fitted*: the hedge slot follows market state, never anyone's P&L), every
one on a **survivorship-free / point-in-time** universe:

{{family_source_table}}

The short-vol leg carries **two regime gates**, ANDed, and they are what hold the worst month and the losing
streak. They cover different failures. The shared one is the **VIX term structure** (flat unless both curve
segments are in contango), applied to all eighteen sleeves — not as a forecast of what gold's volatility will
do, but as a read on *systemic* stress, when the sleeves fall together whatever they sell. The second is per
sleeve: the same contango test on the sleeve's **own** implied vol, which is what catches a vol event one
market has on its own and the VIX never sees. Remove the leg entirely and a genuine
**Sharpe {{top_removed_sharpe}}** book still stands.

**One disclosure §14 asks for.** That second gate was added after a stall *inside* the scored block was
diagnosed, so it is a change made with the block visible. What defends it: the defect is structural and
checkable without looking at a single return — thirteen sleeves were gated on a market they do not trade;
neither constant is fitted (63 trading days is the span a 3M vol index covers, and the 1.0 threshold is the
contango boundary the VIX gate already used); and it pays **more in the in-sample years than in the scored
one** — +13.4pp of book return in 2012 and +13.0pp in 2013 against +6.9pp in 2026, better in 14 of 16 calendar
years. A rule fitted to the block would show that the other way round. §6d-quater carries the audit: a random
gate at each sleeve's own duty cycle, added execution lag, and the whole threshold surface.

**Three honest limits, quantified in [REPORT.md](REPORT.md), not buried:**

1. **Concentration.** Short-vol is {{volprem_pnl_share}} of P&L. Ungated, its standalone tail is **−78%** (one day:
   −76% in the 2010 flash crash), and no *VIX* rule reaches that day — that curve was in contango the session
   before. The sleeve-level gate does reach it, on the sleeves' own curves: the deployed leg loses **0.6%**
   that session and draws down **−15.8%** at worst. That is the tail timed, not removed — a dislocation out of
   a state that is calm in every sleeve at once would still land in full.
2. **Capacity.** That same leg is a variance-swap *replication*, not an executed option book, and the 18-leg
   construction caps out around low tens of $M before the thin legs stop filling.
3. **Crypto-heavy.** Breakout, BAB and x-sect are crypto; short-vol is US index options, global-macro is
   EM FX + commodities, crisis-alpha is multi-asset futures — and since trend was dropped, **no single
   family spans both asset classes**. US single-name and FX breakout did not survive — reported, not hidden.

**Read next:** [REPORT.md](REPORT.md) for the full argument · [docs/APPROACH.md](docs/APPROACH.md) for rationale ·
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the build sequence ·
[docs/AUDIT.md](docs/AUDIT.md) for what running every script found ·
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
| `make master` | the whole portfolio, from scratch | full **Sharpe {{book_sharpe}}** ({{book_targets}}), OOS **{{oos_sharpe}}** ({{oos_targets}}), {{book_dd}} max-DD, {{n_families}} families |
| `make risk-budget` | how much leverage the book can carry (§4b) | shipped **{{leverage}}**; {{binding_constraint}} is what binds first, at {{binding_leverage}} |
| `make cscv` | the overfit / multiple-testing control | **PBO {{cscv_pbo}}**, in-sample-best {{cscv_is_bar}} → OOS {{cscv_oos_bar}} /bar |
| `python scripts/smoke_features.py` | the look-ahead audit | `max\|full − truncated\| = 0` |
| `python scripts/smoke_math.py` | the metric / cost / overlay math (known-answer) | every invariant ✓ |

Re-running `make master` then `git diff reports/master_book_summary.json` shows **no change** —
byte-for-byte reproducibility. The Sharpe is high because the book **selects no single sleeve** (the
best sleeve's deflated Sharpe ≈ {{zoo_dsr}} at N = {{zoo_trials}}): it stacks {{n_families_word}} decorrelated premia (mean ρ ≈ {{mean_corr_abs}}).
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
the per-year / per-quarter tables and the out-of-sample trade log ([reports/](reports/)), and the twelve
per-family write-ups ([docs/strategies/](docs/strategies/)) — {{n_families_word}} that ship and the rest that did not.

```bash
# 1. Reproduce the headline OFFLINE — no key, no download, ~seconds. Works on a fresh clone as-is:
#    because reports/ is committed, run_master_book.py simply reads the {{n_families_word}} family series already
#    there and re-assembles the risk-parity portfolio (Sharpe {{book_sharpe}} full / {{oos_sharpe}} OOS).
make master

# 2. Rebuild the pipeline from raw data — discovery, the crisis/gmacro diversifier legs, validation,
#    master-book assembly, CSCV, charts, dashboard. Budget ~1 hour: it mines the FULL {{zoo_trials}}-candidate
#    grid including 5m/15m, because the trial count is what sets the deflation haircut quoted in the
#    report — reproducing on the cheap 1h/4h/1d grid would report a smaller N and a weaker penalty.
#    (The family deep-dives are heavy one-offs; rebuild any from its own target, e.g.
#    `make volprem xs breakout bab trend carry`.)
cp .env.example .env            # paste a Twelve Data key (equities/FX); crypto (Binance) + macro (FRED) need none
python scripts/smoke_test.py    # optional: proves the data layer end to end (needs the key)
make reproduce                  # crypto auto-downloads keyless (~10 GB, cached to data/); without a key the
                                # equities step stops immediately with "TWELVEDATA_API_KEY not set"
```

Fixed seeds throughout; the final out-of-sample block is run exactly once.
