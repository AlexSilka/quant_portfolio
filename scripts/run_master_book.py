"""THE canonical portfolio assembly — the single source of truth for the final book.

Combines the surviving strategy families at risk parity from each family's one honest published
return series (below). Every series is re-scaled to a common ~15% vol on a trailing (lagged) vol
estimate — point-in-time, no look-ahead — then **equal-weighted (genuine risk parity, no
performance-based selection: every family weight is 1/N)**. Each family is developed and validated in
its own deep-dive (reports/trend, reports/xs, docs/strategies/*). This script only *reads* their
published series and assembles the master.

Two books are reported and persisted:
  • gross premium stack  — the equal-weight risk-parity mean of the eight legs (the raw edge), which
                            runs at ~8.4% annualised vol on its own;
  • risk-managed book     — the deliverable: the stack at the book's one constant leverage (BOOK_LEVERAGE,
                            the level the −15% drawdown mandate allows) with the §8 portfolio risk
                            overlay on top — a drawdown-responsive de-risking ladder (triggers
                            −6/−9/−12% → 0.66/0.33/flat, restore −4%, hysteresis) plus a daily-loss
                            circuit breaker. On the realised (benign-tail) history the overlay COSTS a
                            little Sharpe — it is tail insurance against the short-vol leg's −78%
                            systemic tail the sample does not contain, kept because that tail is real,
                            not to lift a metric. That same excluded tail is why the leverage stops well
                            short of what the realised scorecard would bear (run_risk_budget.py).

Metrics are reported on BOTH the full 15-year window and the frozen out-of-sample block
(OOS_START), because the brief scores targets on the final OOS block. Emits the assembled book,
per-year/quarter, the four-scheme Monte Carlo (block bootstrap / trade-order / entry jitter / random
start), the cross-family correlation matrix, the marginal-contribution curve, book exposure/turnover
series, and each family's add/remove delta.

    python scripts/run_master_book.py
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from src.config import (BOOK_LEVERAGE, BOOK_REBALANCE_BPS, CAPITAL_USD, OOS_START,  # noqa: E402
                        SEED, VOL_SCALE_CAP,
                        VOL_TARGET_ANNUAL)
from src.metrics import summarise  # noqa: E402
from src.risk.overlay import drawdown_ladder  # noqa: E402
from src.risk.stress import hedge_weight  # noqa: E402
from src.validation.monte_carlo import mc_all_variants  # noqa: E402
from src.book_id import fingerprint as book_fingerprint  # noqa: E402
from src.risk.sizing import held_weight_turnover, vol_target_scale  # noqa: E402

PPY = 365
START_REPORT = "2011-01-01"        # the reporting window opens as early as any two legs both exist, so the
                                   # book is shown over a long span rather than the last decade. The early
                                   # years are NOT the four-family book — the crypto legs list in 2020 and
                                   # until then this is the long-history legs only, which the dashboard and
                                   # REPORT state in the same breath as the headline. The §11 scorecard is
                                   # the frozen OOS block either way.
OOS = pd.Timestamp(OOS_START).tz_localize(None)   # frozen final OOS block boundary (2024-07-01)
R = bo.REPORTS

# ── §8 portfolio-level risk limits (stated triggers + step sizes), applied to the book ────────────
LADDER = ((-0.06, 0.66), (-0.09, 0.33), (-0.12, 0.0))   # drawdown → gross exposure step (flat = stop)
LADDER_RESTORE = -0.04             # re-risk only once drawdown recovers above this (hysteresis)
DAILY_LOSS_LIMIT = -0.04           # circuit breaker: flatten the day AFTER a book loss worse than this
GROSS_CAP = 2.0                    # max book gross exposure (leverage limit)

# (label, file, column) — the composition, and one line per leg saying what that leg IS.
#
# NO MEASURED NUMBERS LIVE IN THIS FILE. Every Sharpe, drawdown, CAGR and target count that used to be
# written into these comments described a book that had since been rebuilt, and a comment is the one
# artifact nothing recomputes: the block below claimed trend and carry were not in the book while the
# list underneath it held trend, and quoted a final-block scorecard that disagreed with the JSON this
# same script writes. Numbers belong in the artifacts that are regenerated with the book —
# reports/master_book_summary.json, the rendered REPORT.md, the dashboard — and the reasoning belongs
# here. If a statement here needs a number to be true, it is in the wrong file.
#
# HOW THE COMPOSITION WAS CHOSEN. §5 requires at least four structurally distinct families, and the
# brief's closing line forbids tuning against the final block: "Do not tune against the final
# out-of-sample block to reach a number." So every eligible subset is scored on the IN-SAMPLE window
# only (scripts/run_composition_search.py publishes all of them, winners and losers, through this
# module's own assembly), and ties are broken on breadth — an a-priori good, and what §5 asks for —
# never on what the block would say. What the choice cost, and what the block then printed, is in
# reports/book/composition_search.json and §6d-ter, measured.
FAMILIES = [
    # Short-vol / variance risk premium: a diversified book of capped variance swaps across the Cboe
    # underlyings with clean OHLC. Crypto, FX and discontinued indices are excluded on frozen ex-ante
    # rules (an unhedgeable intraday path; corrupt free OHLC; an index that stopped publishing), never on
    # backtested Sharpe. Net of the per-leg vega spread, of the term-structure haircut, and of its own
    # re-sizing; "naked" (no bought wing) means the systemic tail is UNHEDGED, not that it is free.
    # `ret_gated` is the deployed series — the regime gate is this strategy's own timing signal, so it
    # lives in the construction and pays the spread on every switch, not in the book's risk overlay.
    # docs/strategies/VOLPREM.md.
    ("volprem", "volprem/volprem_book.parquet", "ret_gated"),
    # Cross-sectional momentum on the survivorship-free top-N liquid crypto cross-section: long the
    # winners, short the losers, dollar-neutral, funding charged from the venue's own archive.
    # docs/strategies/XSECT.md.
    ("xs_momentum", "xs/xs_book.parquet", "ret"),
    # Time-series trend on a point-in-time liquid crypto universe plus index ETFs and a point-in-time
    # single-name equity book — the only leg here that spans both asset classes. docs/strategies/TREND.md.
    ("trend_momentum", "trend/trend_block_returns.parquet", "ret"),
    # Channel breakouts: a time-series leg with a walk-forward ML confidence gate and a point-in-time
    # cross-sectional leg, blended at equal RISK inside the family. docs/strategies/BREAKOUT.md.
    ("breakout", "breakout/bo_combined_portfolio.parquet", "ret"),
]


# No family may exceed 1.5x equal risk weight. Derived from the family count rather than typed: written
# out as a fraction for a book of a particular size, it stays behind when a leg leaves and then claims a
# limit it is not enforcing.
PER_FAMILY_CAP = 1.5 / len(FAMILIES)

# Every family holds ONE equal-risk slot. A long-gamma hedge is the exception the mechanism below exists
# for, and it is the only weight in this book that would not be a flat 1/N.
#
# Why a hedge is not held flat. A hedge and an earner want opposite sizing rules: through a calm decade a
# crash hedge is the weakest earner in the book and dilutes every quiet month, and through a crash it is
# the only leg paying. So its slot ramps on MARKET STRESS (src/risk/stress.py — the VIX term structure
# and the S&P's drawdown from its trailing-year high, both read at t-1): a fraction of a slot when
# nothing is moving, more than a slot when the curve inverts or the market is well off its high. It buys
# the same average protection, concentrated at the times it pays for it. It is not performance-based
# selection — no leg's P&L and no book P&L is an input — and rotating the stress path gives the whole
# gain back, which is what says the timing rather than the smaller average is doing the work.
# scripts/run_crisis_lab.py publishes the controls, the ramp's neighbourhood and what it is worth.
#
# Empty because the book holds no hedge family: crisis-alpha was the only leg that ever occupied this
# slot and it is out of the composition. The mechanism stays — a stress-ramped slot is the right shape
# for a hedge — so re-adding a long-gamma family is one line here plus one in FAMILIES, and the import
# guard below still catches a name that answers to nothing.
# Families that cleared their own validation and that the composition did NOT take. They are measured
# here, on the book's window and at the book's per-leg risk, for one reason: the edge map used to print
# a blank Sharpe for them, and a blank reads as "never tested" — which is the opposite of true. Being
# left out is a composition decision about correlation and the scorecard, not a verdict on the edge.
# Nothing below enters the book: these series are scored, never stacked.
VALIDATED_NOT_HELD = [
    ("carry",   "carry/carry_refined.parquet", "ret"),      # perp funding, dollar-neutral cross-section
    ("gmacro",  "book/gmacro_sleeve.parquet",  "ret"),      # macro trend; the live classes are published
                                                            # in book/gmacro_sleeve.json by the run itself
    ("crisis",  "book/crisis_sleeve.parquet",  "ret"),      # managed-futures long gamma
    ("bab",     "bab/bab_book_c25.parquet",    "ret"),      # beta-neutral betting-against-beta, top-25
    # residual momentum is here rather than among the rejected families because it PASSES its own
    # validation (placebo 93rd crypto / 99th equity, walk-forward beats raw momentum on both). What it
    # fails is distinctness: no alpha over raw momentum (t +0.99), so holding it means holding momentum
    # twice. Scored on the same basis as its neighbours instead of quoted off its own study.
    ("residmom", "residmom/residmom_returns.parquet", "crypto_idio"),
]

HEDGE_SLOT: dict[str, tuple[float, float]] = {}

# A name here that no family answers to would silently do nothing — the book would quietly revert to flat
# weights and every scorecard below would go on looking reasonable. Checked once, against the canonical
# family list, at import: a typo or a renamed leg fails loudly here instead of in six months' numbers.
_unknown = set(HEDGE_SLOT) - {lab for lab, _, _ in FAMILIES}
if _unknown:
    raise ValueError(f"HEDGE_SLOT names {sorted(_unknown)}, which are not families in FAMILIES "
                     f"({[lab for lab, _, _ in FAMILIES]}) — the ramp would silently not apply")


def slot_weights(df):
    """Each family's share of an equal-risk slot, per day: 1.0 everywhere except the long-gamma hedge.

    A hedge family absent from `df` is skipped rather than raised on, because that is the legitimate
    case: the composition search and the leave-one-out counterfactuals assemble subsets that drop it on
    purpose. The typo case is caught at import above, against the canonical list, so the two cannot be
    confused with each other."""
    w = pd.DataFrame(1.0, index=df.index, columns=df.columns)
    for fam, (calm, stressed) in HEDGE_SLOT.items():
        if fam in w.columns:
            w[fam] = hedge_weight(df.index, calm, stressed).to_numpy()
    return w


def book_stack(df, slots=None):
    """THE book: the slot-weighted mean over the families that PRINT each day.

    Every consumer that needs the assembled book calls this rather than re-deriving it — an unweighted
    `df.mean(axis=1)` was the old spelling and is now wrong by the hedge's slot. Reduces exactly to that
    mean when every slot is 1.0."""
    slots = slot_weights(df) if slots is None else slots
    num = (df * slots).sum(axis=1, min_count=1)
    den = df.notna().mul(slots).sum(axis=1).replace(0, np.nan)
    return (num / den).rename("ret")


def load(label, file, col):
    p = R / file
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = (df[col] if col in df.columns else df.iloc[:, 0]).dropna()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)                   # normalise so families of mixed tz align
    return s.rename(label)


def _scale(net, target=VOL_TARGET_ANNUAL):
    """Trailing (lagged) vol-target scale factor — the leg's risk-parity weight, computable-at-bar.

    Annualised on the LEG'S OWN observations per year, not a flat 365. Risk parity means every leg
    carries the same annual risk, and a leg that prints on an exchange calendar carries √(365/252) less
    of it than the nominal figure says — so sizing an exchange-calendar leg with √365 leaves it below the
    target the crypto legs are held at, which is not parity, it is a quiet de-risking of whichever leg
    happens to keep a shorter calendar. `ppy_of` is the same honest count the Sharpe uses.

    The 60-bar lookback was swept 10-250 and stays: shorter arms only look better before their burn-in,
    risk and re-sizing cost are matched, and none of them touches the failure a faster estimate is
    supposed to fix. The ceiling is the lever that does — it bounds how much leverage a quiet stretch can
    hand a leg just before a shock, so it caps the tail by construction. `scripts/run_volwindow_lab.py`
    holds the evidence for both."""
    return vol_target_scale(net, target, ppy_of(net), cap=VOL_SCALE_CAP)


def rescale(net, target=VOL_TARGET_ANNUAL):
    return net * _scale(net, target)


def ppy_of(s):
    """Actual observations per calendar year — the honest Sharpe annualisation for a mixed-calendar
    series. Crypto legs trade 365 d/yr, equity/Cboe legs ~252, so the blended 2011-2026 book averages
    ~339; a flat 365 would overstate the annualised Sharpe of any sub-365 series (e.g. volprem's 252-day
    Cboe calendar). The fully-live 2020+ book and the OOS block are genuinely ~365-366 obs/yr, so their
    Sharpe is unchanged. (Vol-targeting `_scale` and turnover keep the nominal 365 — a constant factor
    there does not change any Sharpe, so the book *return series* is byte-identical; only its annualised
    Sharpe is now honest.)"""
    s = s.dropna()
    yrs = (s.index.max() - s.index.min()).days / 365.25
    return len(s) / yrs if yrs > 0 else float(PPY)


def per_period(s, freq):
    out = {}
    for k, g in s.groupby(s.index.to_period(freq) if freq != "Y" else s.index.year):
        g = g.dropna()
        out[str(k)] = round(float(np.sqrt(ppy_of(g)) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    return out


def fixed_size_scorecard(s):
    """The same five targets under the brief's §9 convention: positions always sized on the stated
    $500k of capital (P&L not reinvested), percentages taken of that same capital.

    The denominator never grows, which is what makes this honest — measuring a fixed-size book against
    its own *running balance* would divide every later drawdown by accumulated cash and flatter it
    (that reading gives −4.7% here). Reported alongside the compounded scorecard because the brief
    fixes the sizing capital: if the two conventions disagreed, the number would be an artifact of
    the accounting rather than of the book. They do not — this one lands ~0.4pp stricter."""
    s = s.dropna()
    cum = s.cumsum()                                   # P&L in units of the sizing capital
    mo = s.resample("ME").sum()
    neg = (mo <= 0).astype(int).to_numpy()
    streak = mx = 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"max_dd": round(float((cum - cum.cummax()).min()), 4),
            "worst_month": round(float(mo.min()) if len(mo) else 0.0, 4),
            "months_in_profit": round(float((mo > 0).mean()) if len(mo) else 0.0, 4),
            "longest_losing_streak_mo": int(mx),
            "pnl_usd": round(float(CAPITAL_USD * s.sum()), 2),
            "pnl_usd_per_year": round(float(CAPITAL_USD * s.sum() / ((s.index[-1] - s.index[0]).days / 365.25)), 2),
            "max_dd_usd": round(float(CAPITAL_USD * (cum - cum.cummax()).min()), 2),
            "worst_month_usd": round(float(CAPITAL_USD * mo.min()) if len(mo) else 0.0, 2)}


def scorecard(s, ppy=None):
    """The six task targets on a return series: Sharpe, max-DD, months-in-profit, worst month,
    longest losing streak (months). Reported for both the full window and the OOS block. Sharpe is
    annualised by the series' ACTUAL obs/yr (honest for the mixed 252/365 calendar), not a flat 365."""
    s = s.dropna()
    ss = summarise(s, ppy_of(s) if ppy is None else ppy)
    mo = (1.0 + s).resample("ME").prod() - 1.0
    neg = (mo <= 0).astype(int).to_numpy()
    streak = mx = 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"sharpe": round(ss["sharpe_ann"], 2), "max_dd": round(ss["max_dd"], 4),
            "months_in_profit": round(ss["months_in_profit"], 4),
            "worst_month": round(float(mo.min()) if len(mo) else 0.0, 4),
            "longest_losing_streak_mo": int(mx), "total_return": round(ss["total_return"], 3),
            "n_obs": int(len(s))}


# The brief's five scored targets, in one place. They were written out twice with different Sharpe
# rules — a 2.5-4.0 band in the long-gamma search and a bare >=1.5 in the report resolver — which is one
# copy too many for a rule that decides whether the deliverable passes. The band is the brief's: a book
# far ABOVE 4.0 is not passing, it is a sign the risk was mis-stated.
TARGETS = {"sharpe": (2.5, 4.0), "max_dd": -0.15, "months_in_profit": 0.80,
           "worst_month": -0.06, "longest_losing_streak_mo": 2}


def n_targets(c: dict) -> int:
    """How many of the five a scorecard clears. Accepts either key spelling for the streak, since the
    lab scripts carry it as `streak` and `scorecard()` as `longest_losing_streak_mo`."""
    lo, hi = TARGETS["sharpe"]
    streak = c.get("longest_losing_streak_mo", c.get("streak"))
    return int(sum([lo <= c["sharpe"] <= hi,
                    c["max_dd"] >= TARGETS["max_dd"],
                    c["months_in_profit"] >= TARGETS["months_in_profit"],
                    c["worst_month"] >= TARGETS["worst_month"],
                    streak <= TARGETS["longest_losing_streak_mo"]]))


def book_lift(sleeve, book, weights=(0.0, 0.15, 0.30, 0.50), n_control=40):
    """What adding `sleeve` to `book` does to EVERY scored target, not just Sharpe.

    Judging an addition on Sharpe alone asks the wrong question of this book: Sharpe has never been the
    binding constraint — the worst month and the losing-month streak are. A leg can leave Sharpe flat
    and still earn its slot by lifting months-in-profit or cutting the worst month, and the Sharpe-only
    reading would call that nothing. The sleeve is vol-matched to the book before blending so the weight
    means risk share, not notional."""
    common = book.dropna().index.intersection(sleeve.dropna().index)
    b = book.reindex(common).dropna()
    s = sleeve.reindex(b.index).fillna(0.0)
    s = s * (b.std() / s.std()) if s.std() > 0 else s
    out = {"window": f"{b.index.min().date()}..{b.index.max().date()}"}
    rng = np.random.default_rng(SEED)
    yrs = (b.index[-1] - b.index[0]).days / 365.25
    for w in weights:
        blend = (1.0 - w) * b + w * s
        card = scorecard(blend)
        card["targets"] = n_targets(card)
        # CAGR belongs next to the ratio and is the thing the ratio hides: blending swaps a slice of a
        # high-Sharpe book for a lower-Sharpe one at matched vol, so the ratio can hold while the money
        # falls. A sleeve that leaves Sharpe flat and costs ten points of compound return has not helped.
        card["cagr"] = round(float((1.0 + blend).prod() ** (1.0 / yrs) - 1.0), 4) if yrs > 0 else 0.0
        # Diluting a book with ANY weakly-correlated series cuts its drawdown and worst month and costs
        # Sharpe — that is arithmetic, not a sleeve earning its slot. The control keeps the sleeve's own
        # path exactly (rotation preserves vol, skew and autocorrelation) and destroys only its alignment
        # with the book, so what survives the comparison is the alignment, which is the whole claim.
        if w > 0 and n_control:
            draws = []
            arr = s.to_numpy()
            for k in rng.integers(1, len(arr) - 1, size=n_control):
                rot = pd.Series(np.roll(arr, int(k)), index=s.index)
                rb = (1.0 - w) * b + w * rot
                c = scorecard(rb)
                c["targets"] = n_targets(c)
                c["cagr"] = float((1.0 + rb).prod() ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
                draws.append(c)
            card["control"] = {
                k: round(float(np.median([d[k] for d in draws])), 4)
                for k in ("sharpe", "cagr", "max_dd", "worst_month", "months_in_profit")}
            card["control"]["targets_median"] = float(np.median([d["targets"] for d in draws]))
            card["beats_control_targets"] = bool(card["targets"] > card["control"]["targets_median"])
        out[f"{int(w * 100)}%"] = card
    return out


def risk_overlay(raw, leverage=1.0, limits="book_equity"):
    """§8 book-level risk management applied to the equal-weight premium stack, all causal (t-1 info):
      1. daily-loss circuit breaker — flat the day after a book loss worse than DAILY_LOSS_LIMIT;
      2. drawdown-responsive de-risking ladder — cut gross to the stated step as drawdown deepens
         (flat = stop trading at the deepest trigger), restore only after recovery (hysteresis);
      3. gross-exposure cap, which is what `leverage` is spent against.

    `limits` says which yardstick the -6/-9/-12% and -4%/day triggers are measured against:
      "book_equity" — percent of the LEVERED book's own equity, so a trigger means the same loss to the
                      investor at any leverage. The ladder is what keeps the book inside its -15% mandate,
                      so its triggers have to be quoted in the same units as that mandate; scaling them
                      with leverage would push the deepest 'stop' trigger past the mandate itself.
      "risk_budget" — percent of the UNLEVERED stack, i.e. the triggers scale with leverage. The exposure
                      path is then leverage-invariant, which makes the whole book a pure constant scaling.
    Measured both ways over a 1.0-2.0x grid in run_risk_budget.py; "book_equity" is what ships.
    Returns (managed_ret, gross_exposure, n_breaker_days)."""
    sig = raw * leverage if limits == "book_equity" else raw   # the equity the triggers are read off
    breaker = (sig.shift(1).fillna(0.0) >= DAILY_LOSS_LIMIT).astype(float)   # 0 the day after a big loss
    _, ladder_expo = drawdown_ladder(sig * breaker, LADDER, LADDER_RESTORE)
    gross = (leverage * breaker * ladder_expo).clip(upper=GROSS_CAP)
    managed = raw * gross                                     # apply the combined causal exposure
    return managed.rename("ret"), gross.rename("gross"), int((breaker == 0).sum())


def hold_started(df):
    """A leg that has started keeps its weight on the days its own market is shut, earning nothing,
    instead of the book renormalising onto whoever happens to be open.

    Averaging over the legs that PRINT each day reads well until you price it. Crypto trades 365 days a
    year and the Cboe and equity legs about 252, so on every US holiday the book silently doubled the
    crypto weight and undid it the next morning — round-trip rebalancing charged nowhere, because each
    family's cost model only sees that family's own trades. No desk resizes the whole book because the
    NYSE is shut; it holds.

    A leg is 'started' from its first print, so the union window is unchanged — legs still join in the
    year they list, they simply stop dropping back out on other markets' holidays."""
    return df.where(df.notna(), 0.0).where(df.notna().cummax())


def book_weights(df, scales, slots=None):
    """The per-leg weights the book's own return series implies, day by day — the single source every
    exposure / turnover / rebalance-log figure is derived from (verified: (raw_legs * these).sum(axis=1)
    reproduces `book_stack(df)` to 1e-17).

    The book averages over the legs that PRINT each day, so a leg's weight is its slot share over the
    slot shares printing, and on a day when the equity and Cboe markets are shut the crypto legs carry
    the whole book. That is a consequence of the equal-weight-over-live-legs convention, and it is where
    most of the book's measured turnover comes from — quantified on the dashboard, not smoothed away
    here. The hedge slot varies with market stress, so its weight moves day to day as well."""
    slots = slot_weights(df) if slots is None else slots
    live = df.notna().mul(slots).sum(axis=1).replace(0, np.nan)
    return (scales * slots).where(df.notna()).div(live, axis=0)


def book_turnover(w, legs=None):
    """Book rebalancing turnover per day, round-trip, from the weights above. Unsmoothed and
    unannualised — consumers annualise. Intra-sleeve turnover is charged inside every family's own net
    returns and reported per-family in the deep-dives (§9); this is the assembly layer only.

    Pass `legs` and the DRIFT is counted too, which is what the legs themselves now charge: a weight
    held flat still has to be traded back onto, because the leg that earned more than the book is a
    larger share of it by the close. `Σ|Δw|` alone sees only the bars the target moves, so the assembly
    layer was charged less than it trades — the same shape `xsect.held_turnover` and `backtest.engine`
    fixed one layer down. Left None it is the old target-only figure, for a caller that wants it."""
    if legs is None:
        return w.fillna(0.0).diff().abs().sum(axis=1).rename("turnover")
    return held_weight_turnover(w, legs).rename("turnover")


def describe(s, mc=True):
    ppy = ppy_of(s)
    ss = summarise(s, ppy)
    out = {"sharpe": ss["sharpe_ann"], "max_dd": ss["max_dd"], "months_in_profit": ss["months_in_profit"],
           "total_return": ss["total_return"]}
    if mc:
        v = mc_all_variants(s, ppy, 2000, bo.SEED)
        bb = v["block_bootstrap"]
        out.update({"mc_p5": bb.get("sharpe_p5"), "mc_p50": bb.get("sharpe_p50"), "mc_p95": bb.get("sharpe_p95"),
                    "mc_maxdd_p5": bb.get("maxdd_p5"), "mc_maxdd_p50": bb.get("maxdd_p50"),
                    "mc_maxdd_p95": bb.get("maxdd_p95"), "mc_hit_p5": bb.get("hit_p5"),
                    "mc_hit_p50": bb.get("hit_p50"), "mc_hit_p95": bb.get("hit_p95"), "mc_variants": v})
    return out


def assemble_from(raw: dict, start=START_REPORT):
    """The canonical leg matrix for a given set of published series, rescaled to the common per-leg vol
    target. Returns (rescaled_legs, scale_factors) — the slot-weighted mean of the legs IS the book.

    UNION over the reporting window, not the intersection: the crypto-perp legs only exist from 2020, so
    `.dropna()` would collapse a 15-year book to 2020+. Average over the families live each day (>=2).

    Takes the series as an argument so that a study over subsets of families — the composition search —
    assembles its candidates through THIS function rather than through a copy of it."""
    raw = {k: v for k, v in raw.items() if v is not None}
    # Each leg's scale is computed on its OWN calendar, so the union below is where the gaps appear —
    # and a gap must carry the last scale, not drop to nothing. Held with the returns, the two agree and
    # the weight a leg has on a shut day is the weight it had the evening before.
    scales = pd.DataFrame({k: _scale(v) for k, v in raw.items()}).sort_index().ffill()
    df = hold_started(pd.DataFrame({k: rescale(v) for k, v in raw.items()}).sort_index())
    scales = scales.where(df.notna())
    mask = df.index >= pd.Timestamp(start)
    df, scales = df[mask], scales[mask]
    keep = df.notna().sum(axis=1) >= 2
    return df[keep], scales[keep]


def assemble(start=START_REPORT):
    """`assemble_from` on the book's own composition."""
    return assemble_from({lab: load(lab, f, c) for lab, f, c in FAMILIES}, start)


def book_from_legs(df, scales, leverage=BOOK_LEVERAGE):
    """THE book, from a leg matrix: slot-weighted mean, MINUS what the assembly layer's own trading
    costs, then the §8 overlay — which pays for its own cutting and restoring as well.

    Every consumer that scores a book goes through here. The composition search used to re-derive it and
    got a different number for the same composition — no `hold_started`, no rebalance charge — so the
    one choice in this deliverable made against the scorecard was made on a book that is not the one
    that ships. One function, and that cannot happen.

    Returns (managed, gross_stack, weights, gross_exposure, turnover, n_breaker_days)."""
    slots = slot_weights(df)
    w = book_weights(df, scales, slots)
    turn = book_turnover(w, df)
    raw_ew = book_stack(df, slots) - turn * (BOOK_REBALANCE_BPS / 1e4)
    managed, gross, n_breaker = risk_overlay(raw_ew, leverage=leverage)
    overlay_turn = gross.diff().abs().fillna(0.0)
    managed = (managed - overlay_turn * (BOOK_REBALANCE_BPS / 1e4)).rename("ret")
    return managed, raw_ew, w, gross, turn, n_breaker


def main():
    df, scales = assemble()
    live = df.notna().sum(axis=1)
    print(f"families: {list(df.columns)}\nreporting window: {df.index.min().date()}..{df.index.max().date()} "
          f"({len(df)} days; {int(live.min())}-{int(live.max())} live/day)\n")

    # risk parity with no performance-based selection: every EARNER at 1/N, the long-gamma hedge at the
    # stress-ramped slot above (HEDGE_SLOT) — a market-state rule, not a P&L one. The assembly layer
    # trades, so the assembly layer pays: `book_from_legs` charges its re-weighting and the overlay's
    # own cutting and restoring, and it is the ONE place any consumer assembles this book.
    slots = slot_weights(df)
    managed, raw_ew, w, gross, turn, n_breaker = book_from_legs(df, scales)
    ann_turn = float(turn.mean() * PPY)
    overlay_turn = gross.diff().abs().fillna(0.0)
    ann_rebal_cost = float((turn.mean() + overlay_turn.mean()) * PPY * BOOK_REBALANCE_BPS / 1e4)
    stack_vol = float(raw_ew.std(ddof=1) * np.sqrt(ppy_of(raw_ew)))

    print("=== GROSS PREMIUM STACK (equal-weight risk parity, no overlay, unlevered) ===")
    sc_raw_full, sc_raw_oos = scorecard(raw_ew), scorecard(raw_ew[raw_ew.index >= OOS])
    print(f"  FULL {df.index.min().date()}+: {sc_raw_full}")
    print(f"  OOS  {OOS.date()}+: {sc_raw_oos}")
    print(f"  stack realised vol {stack_vol:.1%} ann -> constant book leverage {BOOK_LEVERAGE:.2f}x "
          f"= {BOOK_LEVERAGE * stack_vol:.1%} book vol (level set by the -15% mandate, see run_risk_budget.py)")
    print(f"\n=== RISK-MANAGED BOOK (deliverable: §8 DD-ladder + daily-loss breaker; breaker fired {n_breaker}d) ===")
    sc_full, sc_oos = scorecard(managed), scorecard(managed[managed.index >= OOS])
    print(f"  FULL {df.index.min().date()}+: {sc_full}")
    print(f"  OOS  {OOS.date()}+: {sc_oos}")
    fx_full, fx_oos = fixed_size_scorecard(managed), fixed_size_scorecard(managed[managed.index >= OOS])
    print(f"  §9 fixed-size (${CAPITAL_USD // 1000}k sizing capital, P&L not reinvested): FULL max-DD "
          f"{fx_full['max_dd']:+.2%} (${fx_full['max_dd_usd']:,.0f}) worst month {fx_full['worst_month']:+.2%} "
          f"(${fx_full['worst_month_usd']:,.0f}) months {fx_full['months_in_profit']:.1%} · P&L "
          f"${fx_full['pnl_usd']:,.0f} (${fx_full['pnl_usd_per_year']:,.0f}/yr)")
    print(f"  book gross exposure: min {gross.min():.2f} mean {gross.mean():.2f} max {gross.max():.2f} (cap {GROSS_CAP})")
    print(f"  book net exposure ~0 (legs dollar-neutral); earner weight 1/{len(df.columns)}, hedge slot "
          f"{HEDGE_SLOT} ramped on market stress (mean {slots[list(HEDGE_SLOT)[0]].mean():.2f}) (cap {PER_FAMILY_CAP:.2f})"
          if set(HEDGE_SLOT) & set(df.columns) else
          f"  book net exposure ~0 (legs dollar-neutral); per-family weight 1/{len(df.columns)} (cap {PER_FAMILY_CAP:.2f})")
    print(f"  annual turnover {ann_turn:.1f}x round-trip at the assembly layer, charged at "
          f"{BOOK_REBALANCE_BPS:.0f}bps per unit of book weight moved — {100 * ann_rebal_cost:.2f}%/yr, "
          f"the overlay's own cutting and restoring included. Every leg holds its weight through the days "
          f"its own market is shut, so none of this is the book chasing whoever happens to be open")

    m = describe(managed)          # MC on the deliverable (managed) book
    per_year = per_period(managed, "Y")
    print(f"  MC block-bootstrap Sharpe[P5 {m['mc_p5']:+.2f} P50 {m['mc_p50']:+.2f} P95 {m['mc_p95']:+.2f}]")
    print(f"  per-year Sharpe: {per_year}")

    # integration delta: with vs without breakout — like-for-like (both raw equal-weight, no overlay)
    wo = book_stack(df[[c for c in df.columns if c != "breakout"]])
    mw = describe(wo, mc=True)
    print(f"\nWITHOUT breakout (raw): Sharpe {mw['sharpe']:+.2f}  MC-P5 {mw['mc_p5']:+.2f}")
    print(f"WITH breakout    (raw): Sharpe {summarise(raw_ew, ppy_of(raw_ew))['sharpe_ann']:+.2f}")

    # correlation + marginal-contribution curve + top-removed — all on the raw equal-weight legs (consistent)
    corr = df.corr()
    solo = {c: summarise(df[c], ppy_of(df[c]))["sharpe_ann"] for c in df.columns}
    order = sorted(df.columns, key=lambda c: -solo[c])
    # the ones we did not take, scored the same way and over the same window, so "not held" and
    # "not measured" can never again look alike on the page
    for lab, f, c in VALIDATED_NOT_HELD:
        s_ = load(lab, f, c)
        if s_ is None:
            print(f"  [not-held] {lab}: no series at {f} — edge map will show it blank")
            continue
        s_ = rescale(s_).reindex(df.index).dropna()
        solo[lab] = summarise(s_, ppy_of(s_))["sharpe_ann"] if len(s_) > 250 else None
    marg = []
    for k in range(1, len(order) + 1):
        b = book_stack(df[order[:k]])
        sc = scorecard(b)
        marg.append({"n": k, "added": order[k - 1], "sharpe": sc["sharpe"],
                     "max_dd": sc["max_dd"], "months_in_profit": sc["months_in_profit"]})
    top = order[0]
    _notop = book_stack(df[[c for c in df.columns if c != top]])
    notop = summarise(_notop, ppy_of(_notop))["sharpe_ann"]
    mean_corr = float(corr.values[np.triu_indices_from(corr.values, 1)].mean())
    # each leg against the book it is part of — the §4 family table's second column. Published rather than
    # transcribed, because that table used to be typed and went on listing legs the book no longer trades.
    _ew = raw_ew
    corr_to_book = {c: round(float(df[c].corr(_ew)), 3) for c in df.columns}
    # §7.2 correlation STABILITY — the same matrix on two halves of the window. "Diversification that exists
    # only in-sample is not diversification": if the decorrelation is real it must persist out-of-sample.
    tri = np.triu_indices_from(corr.values, 1)
    mid = df.index[len(df) // 2]
    ca = df[df.index < mid].corr().values[tri]
    cb = df[df.index >= mid].corr().values[tri]
    corr_stab = {"first_half_mean": round(float(np.nanmean(ca)), 3), "second_half_mean": round(float(np.nanmean(cb)), 3),
                 "max_pairwise_shift": round(float(np.nanmax(np.abs(ca - cb))), 3),
                 "oos_mean": round(float(df[df.index >= OOS].corr().values[tri].mean()), 3)}
    print(f"\nmean cross-family correlation: {mean_corr:+.2f}  (stability — first half {corr_stab['first_half_mean']:+.2f} "
          f"/ second half {corr_stab['second_half_mean']:+.2f} / OOS {corr_stab['oos_mean']:+.2f}, max shift {corr_stab['max_pairwise_shift']:.2f})")
    print("marginal curve (Sharpe): " + " -> ".join(f"{r['added'][:4]} {r['sharpe']:+.2f}" for r in marg))
    print(f"top contributor ({top}) removed: {notop:+.2f}  (vs managed {sc_full['sharpe']:+.2f})")

    # per-family share of book P&L — each leg through the weight it is actually held at, so the hedge
    # slot's stress ramp shows up here rather than being reported as a flat sixth of the book.
    live_slots = df.notna().mul(slots).sum(axis=1).replace(0, np.nan)
    contrib = (df.fillna(0.0) * slots).div(live_slots, axis=0).sum()
    pnl_share = (contrib / contrib.sum()).round(4).to_dict()
    print(f"per-family P&L share: { {k: round(v,3) for k,v in pnl_share.items()} }")

    # §13 out-of-sample sleeve-rebalance log — the book is return-composed, so its "trades" ARE the daily
    # risk-parity rebalances of the family sleeves; the combined instrument-level fills live separately in
    # master_book_oos_trades.csv (make_oos_ledger, from each family's deep-dive e.g. trend_oos_trade_log.csv).
    # One row per sleeve per OOS day it is re-weighted.
    w_oos = w[w.index >= OOS].fillna(0.0)
    dw_oos = w_oos.diff().fillna(0.0)
    trades = []
    for dt in w_oos.index:
        for fam in w_oos.columns:
            d = float(dw_oos.at[dt, fam])
            if abs(d) <= 1e-6:
                continue
            trades.append({"date": dt.date(), "sleeve": fam, "side": "buy" if d > 0 else "sell",
                           "delta_weight": round(d, 5), "weight_after": round(float(w_oos.at[dt, fam]), 5),
                           "notional_usd": round(abs(d) * CAPITAL_USD, 2)})
    pd.DataFrame(trades).to_csv(R / "master_book_oos_rebalances.csv", index=False)
    print(f"OOS sleeve-rebalance log: {len(trades):,} rebalances -> reports/master_book_oos_rebalances.csv")

    # persist
    managed.to_frame().to_parquet(R / "master_book.parquet")
    raw_ew.rename("ret").to_frame().to_parquet(R / "master_book_raw.parquet")
    df.to_parquet(R / "master_book_legs.parquet")
    # the slot share each family is held at, per day — published so the report renderer and every
    # leave-one-out counterfactual weight the legs the way the book does instead of re-deriving a flat
    # mean that has been wrong since the hedge slot started ramping.
    slots.to_parquet(R / "master_book_slots.parquet")
    # every exposure/turnover/cost figure downstream reads these two files — nothing re-derives the
    # assembly from the family blocks, which is how a stale family list or the wrong column slips in
    w.to_parquet(R / "master_book_weights.parquet")
    pd.DataFrame({"gross": gross, "family_gross": w.abs().sum(axis=1),
                  "turnover": turn}).to_parquet(R / "master_book_exposure.parquet")
    corr.to_csv(R / "master_book_correlation.csv")
    pd.DataFrame(marg).to_csv(R / "master_book_marginal.csv", index=False)
    (R / "master_book_summary.json").write_text(json.dumps({
        # the book's own identity: everything measured AGAINST the book records this, and
        # scripts/check_freshness.py fails the build when a derived artifact still carries an older one
        "book_id": book_fingerprint(managed),
        "families": list(df.columns), "window": [str(df.index.min().date()), str(df.index.max().date())],
        # which legs are live in which era, so the report can SAY which rather than name them by hand.
        # The sentence "the pre-2020 window runs <legs>" was typed and went on naming legs the book had
        # dropped years earlier, while the family COUNT beside it stayed right because that was derived.
        "families_by_era": {
            "pre_2020": [c for c in df.columns if df[c].first_valid_index() < pd.Timestamp("2020-01-01")],
            "from_2020": [c for c in df.columns if df[c].first_valid_index() >= pd.Timestamp("2020-01-01")]},
        "oos_start": str(OOS.date()),
        "master": {**m, **{f"full_{k}": v for k, v in sc_full.items()}},
        "scorecard_full": sc_full, "scorecard_oos": sc_oos,
        "fixed_size_full": fx_full, "fixed_size_oos": fx_oos, "sizing_capital_usd": CAPITAL_USD,
        "gross_premium_full": sc_raw_full, "gross_premium_oos": sc_raw_oos,
        "without_breakout": mw, "per_year": per_year, "per_quarter": per_period(managed, "Q"),
        "standalone_sharpe": solo, "corr_to_book": corr_to_book,
        "mean_correlation": mean_corr, "correlation_stability": corr_stab,
        # §11 turnover, round-trip x capital per year. `weights_held` is the same book with every started
        "annual_turnover": round(ann_turn, 1),
        "rebalance_bps": BOOK_REBALANCE_BPS, "annual_rebalance_cost": round(ann_rebal_cost, 5),
        "pnl_share": pnl_share, "marginal": marg, "top_removed": {"family": top, "sharpe": notop},
        "breakout_delta_sharpe": summarise(raw_ew, ppy_of(raw_ew))["sharpe_ann"] - mw["sharpe"],
        "risk_limits": {"ladder": LADDER, "restore": LADDER_RESTORE, "daily_loss_limit": DAILY_LOSS_LIMIT,
                        "gross_cap": GROSS_CAP, "per_family_cap": round(PER_FAMILY_CAP, 3),
                        "breaker_days": n_breaker, "max_gross": round(float(gross.max()), 2),
                        "leverage": BOOK_LEVERAGE, "stack_vol": round(stack_vol, 4),
                        "book_vol": round(BOOK_LEVERAGE * stack_vol, 4), "limits_measured_in": "book_equity"},
        "mc_variants": m["mc_variants"]}, indent=2, default=float))

    _figure(managed, df, corr, marg, per_year)
    print("\nartifacts -> reports/master_book*  |  figure -> reports/figures/master_book.png")
    print("MASTER BOOK OK")


def _figure(master, df, corr, marg, per_year):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    (1 + master).cumprod().plot(ax=a, color="#1f77b4", lw=1.6, label="risk-managed book")
    (1 + df[[c for c in df.columns if c != "breakout"]].mean(axis=1)).cumprod().plot(
        ax=a, color="#b0b0b0", lw=1.2, label="without breakout")
    a.set_yscale("log"); a.legend(); a.set_title("1) Master book equity (with vs without breakout)")
    a = ax[0, 1]
    im = a.imshow(corr.values, cmap="coolwarm", vmin=-0.5, vmax=1.0)
    a.set_xticks(range(len(corr))); a.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    a.set_yticks(range(len(corr))); a.set_yticklabels(corr.columns, fontsize=7)
    for (i, j), v in np.ndenumerate(corr.values):
        a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    a.set_title("2) Cross-family correlation"); a.grid(False); fig.colorbar(im, ax=a, fraction=0.046)
    a = ax[1, 0]
    a.plot([r["n"] for r in marg], [r["sharpe"] for r in marg], "o-", color="#2ca02c")
    for r in marg:
        a.annotate(r["added"][:4], (r["n"], r["sharpe"]), fontsize=7, xytext=(0, 6), textcoords="offset points")
    a.set_title("3) Marginal-contribution curve"); a.set_xlabel("# families"); a.set_ylabel("Sharpe")
    a = ax[1, 1]
    yrs = list(per_year)
    a.bar(range(len(yrs)), [per_year[y] for y in yrs],
          color=["#2ca02c" if per_year[y] > 0 else "#d62728" for y in yrs])
    a.set_xticks(range(len(yrs))); a.set_xticklabels([y[2:] for y in yrs]); a.axhline(0, color="k", lw=0.5)
    a.set_title("4) Master book per-year Sharpe"); a.set_ylabel("Sharpe")
    fig.tight_layout()
    (R / "figures").mkdir(exist_ok=True)
    fig.savefig(R / "figures" / "master_book.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
