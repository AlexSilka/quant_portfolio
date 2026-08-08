"""Project constants in one auditable place — filesystem layout, frozen backtest periods,
trading costs (split, per venue) and the cross-sectional-momentum strategy parameters.
No magic numbers or hard-coded paths scattered across scripts.

Fees are the Binance PUBLISHED SCHEDULE, verified live 2026-08 (not from memory):
  - Spot:  taker/maker 0.10% / 0.10%  (0.075% each with the 25% BNB discount)
  - USD-M perpetual futures: taker/maker 0.05% / 0.02%
Spot commission is 2x futures — a real difference that matters for a high-turnover book, so it is
split by venue below. The total per-trade cost is always commission + half-spread + √-impact
(Almgren), never a flat constant (`src/backtest/costs.py::trade_cost_bps` and
`src/sleeves/xsect.py::xs_backtest` implement it).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── filesystem layout: every input/output path is anchored at the repo root, so a script runs
#    identically from any working directory. Each script previously hard-coded a CWD-relative
#    "reports/..."/"data/..." string that silently broke if launched from elsewhere. ───────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"            # market data: raw vendor pulls + derived caches (git-ignored)
RAW_DIR = DATA_DIR / "raw"              # raw pulls: Binance klines/funding, TwelveData equities
CACHE_DIR = DATA_DIR / "cache"          # derived per-strategy caches
REPORTS_DIR = ROOT_DIR / "reports"      # strategy artifacts: *.parquet/*.csv/*.json + dashboard.html
FIGURES_DIR = REPORTS_DIR / "figures"   # generated plots (git-ignored)
LOGS_DIR = ROOT_DIR / "logs"            # run logs
# per-family artifact sub-books — one folder per strategy family so reports/ stays navigable
# (the `bo_`/`carry_`/… filename prefixes map to these; trend & xs already followed this layout).
TREND_DIR = REPORTS_DIR / "trend"       # trend deep-dive          (trend_*)
XS_DIR = REPORTS_DIR / "xs"             # cross-sectional momentum  (xs_*)
BREAKOUT_DIR = REPORTS_DIR / "breakout" # breakout deep-dive        (bo_*)
CARRY_DIR = REPORTS_DIR / "carry"       # carry / funding           (carry_*)
BAB_DIR = REPORTS_DIR / "bab"           # betting-against-beta       (bab_*)
VOLPREM_DIR = REPORTS_DIR / "volprem"   # short-vol / variance risk premium (volprem_*)
RESIDMOM_DIR = REPORTS_DIR / "residmom" # residual momentum         (residmom_*)
SEASONAL_DIR = REPORTS_DIR / "seasonal" # calendar / seasonal       (seasonal_*)
ONCHAIN_DIR = REPORTS_DIR / "onchain"   # on-chain / network        (onchain_*)
OVERNIGHT_DIR = REPORTS_DIR / "overnight"  # overnight / session    (overnight_*)
LOTTERY_DIR = REPORTS_DIR / "lottery"   # lottery / skewness        (lottery_*)
VOLSPIKE_DIR = REPORTS_DIR / "volspike" # volume-spike              (volspike_*)
BOOK_DIR = REPORTS_DIR / "book"         # cross-family discovery "zoo" + combined returns (zoo_*, all_returns, walk_forward, …)
LAB_DIR = REPORTS_DIR / "lab"           # experimental / book-candidate / overlay / stress-study outputs

REPORT_SUBDIRS = (FIGURES_DIR, TREND_DIR, XS_DIR, BREAKOUT_DIR, CARRY_DIR, BAB_DIR, VOLPREM_DIR,
                  RESIDMOM_DIR, SEASONAL_DIR, ONCHAIN_DIR, OVERNIGHT_DIR, LOTTERY_DIR, VOLSPIKE_DIR, BOOK_DIR, LAB_DIR)
for _sub in REPORT_SUBDIRS:                 # ensure the per-family sub-books exist so any writer can drop into them
    _sub.mkdir(parents=True, exist_ok=True)  # (idempotent; survives a fresh clone / `make clean`)

# ── Binance commission (exchange fee), bps of traded notional, split by venue ──────────────
BINANCE_SPOT_TAKER_BPS = 10.0     # 0.10%  spot taker  (7.5 with 25% BNB discount)
BINANCE_SPOT_MAKER_BPS = 10.0     # 0.10%  spot maker  (7.5 with BNB)
BINANCE_FUT_TAKER_BPS = 5.0       # 0.05%  USD-M perp taker
BINANCE_FUT_MAKER_BPS = 2.0       # 0.02%  USD-M perp maker

# ── half bid/ask spread (bps): a floor cost beyond the fee. Liquid majors ~0.5-1bps; illiquid
#    mid-caps are wider, but that extra cost is added by the √-impact term (size/ADV-dependent),
#    NOT by inflating this flat spread. ──────────────────────────────────────────────────────
CRYPTO_HALF_SPREAD_BPS = 1.0
EQUITY_COMMISSION_BPS = 1.0       # US equities ~1bp/side all-in via a prime broker
EQUITY_HALF_SPREAD_BPS = 2.0      # penny-ish spread on liquid large caps
EQUITY_BORROW_BPS_ANNUAL = 50.0   # stock-borrow on the SHORT leg (§9): ~0.5%/yr general-collateral for
                                  # liquid large caps (the equity x-sect shorts top-100 names; hard-to-borrow
                                  # small caps run far higher and are excluded). Charged per bar on short gross.

# ── Almgren √-impact: impact_bps = IMPACT_K * sigma_bar * sqrt(order_notional / ADV_bar) * 1e4.
#    Scales the cost up for a large order in a thin name — this is what makes the mid-cap tail of
#    the top-50 pay its true, wider effective spread instead of a flat 1bps. ──────────────────
IMPACT_K = 0.1
CAPITAL_USD = 500_000             # book notional, drives the √-impact order-size term

# ── cross-sectional momentum (crypto): SPOT price signal, executed on FUTURES (shorts need perps).
#    So commission is the SPOT taker before perps exist (pre-2020, spot-only execution) and the
#    FUTURES taker after (2020+, the tradable venue). ─────────────────────────────────────────
XS_LOOKBACK_D = 30                # momentum formation window (days) — the survivorship-free winner
XS_TOP_N_LIQUID = 50              # rank only the top-N by trailing liquidity (universe sweep optimum)
XS_TOP_FRAC = 0.3                 # long / short tercile of the eligible set
XS_WEIGHTING = "volinv"           # inverse-vol legs — robust on the thin pre-2020 cross-section
XS_NO_TRADE_BUFFER = 0.02         # skip weight moves below this fraction — controls turnover
XS_VOL_TARGET = 0.15              # annualised vol target per stream
XS_EQUITY_LOOKBACK_D = 252        # equity leg: ~12-month formation
XS_EQUITY_SKIP_D = 7              # skip the last week (reversal gap); a plateau — skip 5-15 all ≈+0.5
XS_EQUITY_TOP_FRAC = 0.1          # equity leg: decile (deeper, cleaner tails on a broad panel)

STABLECOINS = frozenset({"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT", "USDPUSDT",
                         "EURUSDT", "AEURUSDT", "USD1USDT"})   # no momentum — excluded from ranking

# ── general ────────────────────────────────────────────────────────────────────────────────
SEED = 7
VOL_TARGET_ANNUAL = 0.15          # annualised vol target per return stream — the per-sleeve sizing knob
                                  # (also the master book's per-leg risk-parity target: run_master_book
                                  # rescale() reads THIS, it has no target of its own)
BOOK_LEVERAGE = 1.20              # the assembled book's constant leverage — the only dial that sets book
                                  # risk (the equal-weight stack runs at ~8.4% annualised vol on its own,
                                  # so this is ~10.3%). The stated risk budget is the -15% drawdown
                                  # mandate, and the level is what that mandate allows, measured in
                                  # run_risk_budget.py (reports/book/risk_budget.json), not read off a
                                  # scorecard: 1.35x is the hard ceiling — above it a repeat of the
                                  # vol-premium leg's documented one-day systemic tail (2010-05-06,
                                  # -50.9% at book weight, OUTSIDE the 2011+ report window) breaches -15%
                                  # — and 1.40x is a cliff, not a slope (the ladder's deepest rung flattens
                                  # the book and the drawdown jumps -14.5% -> -16.3%), so the level sits
                                  # below the ceiling rather than on it.
CRYPTO_PPY = 365                  # crypto trades 24/7
EQUITY_PPY = 252                  # US trading days
PERP_HISTORY_START = "2020-01-01"  # USD-M perps + funding begin here; spot reaches back to 2017-08

# ── frozen out-of-sample boundary — this constant is the single source, set BEFORE any result
#    was scored and run exactly once. Moving it after seeing results is a documented bias source,
#    not a free parameter. tz-aware so it compares directly against the UTC-indexed return frames. ─
OOS_START = pd.Timestamp("2024-07-01", tz="UTC")


def crypto_cost_bps(venue: str) -> tuple[float, float]:
    """(commission_bps, half_spread_bps) for a crypto trade on the given venue ('spot' | 'futures').

    Use this so the cost is explicit and venue-correct: the deep pre-2020 track is spot-only
    (spot taker), the tradable 2020+ book executes on perps (futures taker). The √-impact term is
    added on top by the backtest, per name, from ADV.
    """
    taker = BINANCE_SPOT_TAKER_BPS if venue == "spot" else BINANCE_FUT_TAKER_BPS
    return taker, CRYPTO_HALF_SPREAD_BPS
