.PHONY: attribution gate-ablation gate-coverage integrity setup reproduce clean smoke smoke-math carry carry-wide volprem trend xs breakout overnight lottery bab crisis seasonal onchain residmom master composition risk-budget ml-contribution ml-portfolio longgamma discovery figures cscv selection-bias wf features sessions ledger lint

PY := .venv/bin/python

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"
	@echo "macOS: if lightgbm fails to load, run: brew install libomp"

# One-command headline: discovery -> ML overlay -> MASTER portfolio (risk-parity over the traded
# families' committed honest series) -> dashboard. Rebuild a family's series with its target below.
reproduce:
	$(PY) scripts/run_all.py

# Assemble the master book from the traded families' published series (the canonical portfolio).
master:
	$(PY) scripts/run_master_book.py

# §6d-ter why the book trades six families and not eight: every single- and double-removal of the eight
# validated families, all five targets on both windows. Publishes the denominator, not just the winner.
composition:
	$(PY) scripts/run_composition_search.py

# §8 how much leverage the book can carry: constant-leverage grid 1.0-2.0x through the canonical
# assembler, both §8 limit conventions, bootstrap tail + the 2010 systemic event the window excludes.
risk-budget:
	$(PY) scripts/run_risk_budget.py

# §5d book-level ML: does ML lift the ASSEMBLED book? Leg-swap each family for its ML variant through
# the risk-parity assembly (breakout/carry/trend) + uniform gate + magnitude sizing. ~several minutes.
ml-contribution:
	$(PY) scripts/run_ml_book_contribution.py
	$(PY) scripts/trend/run_trend_sleeve_ml.py

# §5d portfolio-level ML: does ML on top of the WHOLE book lift it? (measured — it does not: whole-book
# gate / soft-exposure / ML-allocation, six engines, honest controls, all five targets.) ~3-4 min.
ml-portfolio:
	$(PY) scripts/run_ml_portfolio_overlay.py

# §6c search for a 9th family: a second long-gamma source, incl. the size sweep a hedge needs.
longgamma:
	$(PY) scripts/run_longgamma_search.py

# §6 probability of backtest overfitting (CSCV) on the full 1,279-sleeve trial set.
cscv:
	$(PY) scripts/run_cscv.py

# §6 dose-response behind CSCV: what a search budget buys in-sample vs. out of sample, and the same
# winner deflated at a range of declared trial counts. Reads the committed zoo matrix; seconds.
selection-bias:
	$(PY) scripts/run_selection_bias.py

# §10 book-level walk-forward: rolling & anchored, periodic allocation re-fit -> accumulated OOS track.
wf:
	$(PY) scripts/run_wf_book.py

# §4 per-feature IC / stability-over-time / redundancy-cluster analysis + the stated reduction.
features:
	$(PY) scripts/feature_report.py

# Panel integrity: is each cached series still being published, or merely still being returned?
# Reports stale series, interior gaps, frozen tapes and scale outliers — the four shapes behind the
# 2026-08-10 data defects. Reports only; screens live next to the results they change.
integrity:
	$(PY) -m src.data.integrity --verbose

# Per-family attribution of any window of the book — weighted contribution (what a leg did to the BOOK,
# not standalone), plus realised crypto beta and the alpha left after it. "The book stalled, which leg?"
# Optional args: `make attribution ARGS=2026` or `ARGS="2026-01 2026-08"`.
attribution:
	$(PY) scripts/run_window_attribution.py $(ARGS)

# Does the short-vol book's regime gate cover the risk it gates? The shipped VIX gate reads equity vol,
# but 13 of 18 sleeves are not equity. Tests three per-sleeve rules + placebo/lag/threshold audits.
# Reports only — run_master_book.py is untouched (the winner costs a brief target; see the printout).
gate-coverage:
	$(PY) -m scripts.volprem.run_gate_coverage

# The deployed short-vol leg ANDs a shared VIX gate with a per-sleeve own-curve one. This ablates each
# half on the full metric set (return, vol, drawdown depth AND duration, tails, monthly distribution)
# plus the five brief targets at book level, and checks the two-gate arm reproduces the shipped series.
gate-ablation:
	$(PY) -m scripts.volprem.run_gate_ablation

# §3 equity NYSE session/half-day integrity check via pandas_market_calendars.
sessions:
	$(PY) scripts/validate_sessions.py

# §13 portfolio-level out-of-sample trade/position ledger + combined per-trade log.
ledger:
	$(PY) scripts/make_oos_ledger.py

# §13 regenerate the required charts (equity, per-sleeve equity, drawdown, monthly heatmap, rolling
# Sharpe, exposure, turnover, correlation, edge map, survival funnel) from the current artifacts.
figures:
	$(PY) scripts/make_figures.py

# Full discovery grid (incl. 5m/15m intraday) — the honest N for the multiple-testing haircut. Slow
# (~45 min). `reproduce` runs this same grid, so the published N cannot drift from the reproduced one.
discovery:
	$(PY) scripts/run_book.py --intraday
	$(PY) scripts/walk_forward.py

# Carry deep-dive (cross-sectional funding carry + basis + ML + walk-forward). See CARRY.md.
carry:
	$(PY) scripts/carry/run_carry.py
	$(PY) scripts/carry/audit_carry.py
	$(PY) scripts/carry/run_carry_ml.py
	$(PY) scripts/carry/run_carry_wfo.py
	$(PY) scripts/carry/run_carry_refine.py
	$(PY) scripts/carry/run_carry_portfolio.py
	$(PY) scripts/carry/run_carry_basis.py
	$(PY) scripts/carry/run_carry_fx.py
	$(PY) scripts/carry/run_carry_equity.py
	$(PY) scripts/carry/make_carry_figures.py
	$(PY) scripts/carry/make_carry_xasset_fig.py

# Wide-universe carry: breadth curve on the full point-in-time perp set.
# Downloads the whole Binance USD-M archive first (~830 perps incl. delisted) — slow, one-off. See CARRY.md §2.6/§6.
carry-wide:
	$(PY) scripts/data/_dl_universe.py
	$(PY) scripts/carry/run_carry_breadth.py

# Short-vol / variance-risk-premium deep-dive (Deribit DVOL + Cboe VIX/VXN/EVZ vs realised). See VOLPREM.md.
volprem:
	$(PY) scripts/volprem/run_vol_premium.py
	$(PY) scripts/volprem/run_vol_premium_xasset.py
	$(PY) scripts/volprem/run_vol_premium_book.py
	$(PY) scripts/volprem/run_vol_premium_tf.py
	$(PY) scripts/volprem/run_vol_premium_deploy.py
	$(PY) scripts/volprem/run_vol_premium_gates.py
	$(PY) scripts/volprem/run_wing_cost.py

# Trend deep-dive -> publishes reports/trend/trend_block_returns.parquet. See docs/TREND.md.
trend:
	$(PY) scripts/trend/run_trend_book.py
	$(PY) scripts/trend/run_trend_in_portfolio.py

# Cross-sectional momentum deep-dive -> publishes reports/xs/xs_book.parquet. See docs/XSECT.md.
xs:
	$(PY) scripts/xs/portfolio.py

# Calendar/session deep-dive: overnight-vs-intraday decomposition (a tested-and-mapped family). See OVERNIGHT.md.
overnight:
	$(PY) scripts/overnight/run_overnight.py

# Skewness/lottery (MAX) deep-dive: short-high-skew cross-sectional book (a tested-and-mapped family). See LOTTERY.md.
lottery:
	$(PY) scripts/lottery/run_lottery.py

# Betting-against-beta / low-vol deep-dive: crypto + equity, dollar- vs beta-neutral (a tested-and-mapped family). See BAB.md.
bab:
	$(PY) scripts/bab/run_bab.py
	$(PY) scripts/bab/run_bab_robust.py
	$(PY) scripts/bab/run_bab_deep.py
	$(PY) scripts/bab/run_bab_ml.py
	$(PY) scripts/bab/run_bab_portfolio.py

# Residual / idiosyncratic momentum deep-dive (H5): residualise the momentum signal, raw-vs-residual head-to-head
# across crypto/equity/FX × timeframe × universe size (a tested in-family refinement). See RESIDMOM.md.
residmom:
	$(PY) scripts/residmom/run_residmom.py
	$(PY) scripts/residmom/run_residmom_robust.py
	$(PY) scripts/residmom/run_residmom_ml.py

# Crisis-alpha deep-dive: the book's long-gamma leg, and how big a slot a hedge deserves. Rebuilds the
# family series, then the evidence behind its sizing — per-class cost diagnosis, every construction
# variant with its crash-window payoff, the slot sweep and the rotated/flat controls. See CRISIS.md.
crisis:
	$(PY) scripts/run_crisis.py
	$(PY) scripts/run_crisis_lab.py

# Calendar-seasonality deep-dive: pre-FOMC drift + turn-of-month (a tested-and-mapped family). See SEASONAL.md.
seasonal:
	$(PY) scripts/seasonal/run_seasonal.py
	$(PY) scripts/seasonal/run_seasonal_variants.py
	$(PY) scripts/seasonal/run_seasonal_xasset_ml.py

# On-chain / network-signal deep-dive (H3): free Coin Metrics community + blockchain.com + DefiLlama;
# activity/valuation cross-section, BTC/ETH exchange-flow overlays, and a chain-fundamentals (fees /
# revenue / TVL) cross-section over the chains CM cannot see (tested and mapped, dead). See ONCHAIN.md.
onchain:
	$(PY) -m src.data.onchain
	$(PY) scripts/onchain/run_onchain.py
	$(PY) scripts/onchain/run_onchain_ml.py
	$(PY) -m src.data.defillama
	$(PY) scripts/onchain/run_fundamentals.py
	$(PY) scripts/onchain/run_onchain_blend.py

# Breakout deep-dive -> publishes reports/breakout/bo_combined_portfolio.parquet. See docs/BREAKOUT.md.
breakout:
	$(PY) scripts/breakout/run_bo_exits.py
	$(PY) scripts/breakout/run_bo_sweep.py
	$(PY) scripts/breakout/run_bo_book.py d55_atr3
	$(PY) scripts/breakout/run_bo_frozen.py
	$(PY) scripts/breakout/run_bo_walkforward.py
	$(PY) scripts/breakout/run_bo_ml.py
	$(PY) scripts/breakout/run_bo_fast_tf.py
	$(PY) scripts/breakout/run_bo_final.py
	$(PY) scripts/breakout/run_bo_xs.py
	$(PY) scripts/breakout/run_bo_xs_tf.py
	$(PY) scripts/breakout/run_bo_xs_big.py
	$(PY) scripts/breakout/run_bo_xs_liq.py
	$(PY) scripts/breakout/run_bo_xs_pit.py
	$(PY) scripts/breakout/run_bo_xs_costs.py
	$(PY) scripts/breakout/run_bo_combined.py
	$(PY) scripts/breakout/run_bo_contribution.py
	$(PY) scripts/breakout/run_bo_spot.py
	$(PY) scripts/breakout/run_bo_ml_wf.py
	$(PY) scripts/breakout/run_bo_improve.py
	$(PY) scripts/breakout/run_bo_ts_book.py
	$(PY) scripts/breakout/run_bo_pre_perp.py
	$(PY) scripts/breakout/run_bo_deep_history.py
	$(PY) scripts/breakout/make_bo_figures.py

smoke:
	$(PY) scripts/smoke_test.py
	$(PY) scripts/smoke_features.py
	$(PY) scripts/smoke_math.py

# Known-answer invariants for the headline-integrity math (CSCV/PBO, 4x Monte-Carlo, DD-ladder, borrow).
smoke-math:
	$(PY) scripts/smoke_math.py

# Lint gate: the curated ruff ruleset (pyflakes + syntax errors) from pyproject.toml [tool.ruff], plus
# the report-freshness check — REPORT.md is generated, so a book re-run that leaves it behind fails here.
lint:
	.venv/bin/ruff check .
	$(PY) scripts/render_report.py --check
	$(PY) scripts/make_report.py --check

# Rebuild REPORT.md and README.md from scripts/report_assets/ + the measured numbers.
report:
	$(PY) scripts/render_report.py

# §9 per family: cost as a share of gross P&L, by re-running each family with its cost model off. ~5 min.
family-costs:
	$(PY) scripts/measure_family_costs.py

clean:
	find reports -type f \( -name '*.parquet' -o -name '*.png' \) -delete
