"""Known-answer invariant checks for the headline-integrity math.

The numbers the deliverable rests on — PBO/CSCV, the four Monte-Carlo schemes, the §8 drawdown
ladder and the equity short-borrow charge — are produced by pure functions with analytic
invariants, yet nothing asserted them (the only prior invariant test was the feature look-ahead
audit). This pins each to a *known answer* exercised through the real function boundary, so a sign
flip / off-by-one / look-ahead in the code that computes the headline is caught, not trusted:

  - CSCV PBO → 0.5 on an iid null (no skill), → 0 when one strategy genuinely dominates.
  - trade-order MC leaves Sharpe permutation-invariant but spreads max-drawdown (the module's own claim).
  - the drawdown ladder steps 1.0→0.66→0.33→0.0 at −6/−9/−12%, restores above −4%, and is causal
    (exposure on bar t uses only the drawdown through t-1 — proven by truncation, like the feature audit).
  - equity borrow is charged only on the short leg, subtracted from net, and matches k·notional/ppy.

Run: python scripts/smoke_math.py   (or `make smoke-math`).
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


from src.backtest.costs import panel_impact_cost, trade_cost_bps  # noqa: E402
from src.backtest.engine import backtest  # noqa: E402
from src.config import EQUITY_BORROW_BPS_ANNUAL, SEED  # noqa: E402
from src.metrics import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, sharpe  # noqa: E402
from src.risk.overlay import drawdown_ladder, vol_managed  # noqa: E402
from src.sleeves.xsect import xs_backtest  # noqa: E402
from src.validation.cscv import pbo_cscv  # noqa: E402
from src.validation.monte_carlo import mc_all_variants, mc_metrics, trade_order_mc  # noqa: E402
from src.validation.purged_cv import purged_kfold  # noqa: E402


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


# ── CSCV / PBO ────────────────────────────────────────────────────────────────────────────
def test_cscv_pbo() -> None:
    rng = np.random.default_rng(SEED)
    T, N, S = 384, 24, 16

    # null: N iid-noise strategies, no skill → IS-best is random w.r.t. OOS rank → PBO ≈ 0.5
    noise = pd.DataFrame(rng.standard_normal((T, N)) * 0.01, index=_days(T),
                         columns=[f"s{i}" for i in range(N)])
    null = pbo_cscv(noise, n_blocks=S)
    assert null["n_splits"] == comb(S, S // 2), "CSCV must evaluate C(S,S/2) splits"
    assert 0.35 < null["pbo"] < 0.65, f"iid-null PBO must be ~0.5, got {null['pbo']:.3f}"

    # signal: one strategy genuinely dominates every block → always IS-best AND OOS-best → PBO ≈ 0
    strong = noise.copy()
    strong["s0"] = 0.02 + rng.standard_normal(T) * 0.001            # high, near-constant Sharpe
    sig = pbo_cscv(strong, n_blocks=S)
    assert sig["pbo"] < 0.10, f"a genuinely dominant strategy must give low PBO, got {sig['pbo']:.3f}"
    assert sig["pbo"] < null["pbo"], "PBO must discriminate real skill from the noise null"
    assert sig["oos_sharpe_mean"] > null["oos_sharpe_mean"], "selected OOS Sharpe should be higher with real skill"

    # guards: odd block count rejected; too-few strategies flagged not crashed
    try:
        pbo_cscv(noise, n_blocks=15)
    except ValueError:
        pass
    else:
        raise AssertionError("n_blocks must be even → ValueError expected")
    assert "error" in pbo_cscv(noise.iloc[:, :3], n_blocks=S), "N<4 must return an error dict, not crash"
    print(f"  CSCV/PBO      ✓  null={null['pbo']:.3f}  signal={sig['pbo']:.3f}  splits={null['n_splits']}")


# ── Monte-Carlo (4 schemes) ──────────────────────────────────────────────────────────────
def test_monte_carlo() -> None:
    rng = np.random.default_rng(SEED)
    r = pd.Series(rng.standard_normal(300) * 0.01 + 0.0004, index=_days(300))

    mc = mc_metrics(r, 365, n_reps=500, seed=SEED)
    for k in ("sharpe", "maxdd", "hit"):
        assert mc[f"{k}_p5"] <= mc[f"{k}_p50"] <= mc[f"{k}_p95"], f"{k} percentiles must be ordered"
    assert mc == mc_metrics(r, 365, n_reps=500, seed=SEED), "same seed must reproduce exactly"

    # trade-order: Sharpe is permutation-invariant (mean/std unchanged) → degenerate band;
    # only path-dependent max-drawdown spreads. This pins the module's own honesty claim.
    to = trade_order_mc(r, 365, n_reps=500, seed=SEED)
    assert to["sharpe_p5"] == to["sharpe_p50"] == to["sharpe_p95"], "trade-order Sharpe must be permutation-invariant"
    assert to["maxdd_p5"] < to["maxdd_p95"], "trade-order must spread max-drawdown via path re-ordering"

    allv = mc_all_variants(r, 365, n_reps=300, seed=SEED)
    assert set(allv) == {"block_bootstrap", "trade_order", "entry_jitter", "random_start"}, "all 4 schemes required"
    assert all(v.get("sharpe_p50") is not None for v in allv.values()), "every scheme must percentile Sharpe"
    assert allv == mc_all_variants(r, 365, n_reps=300, seed=SEED), "all-variants must be deterministic under a fixed seed"
    print(f"  MonteCarlo    ✓  block P5={mc['sharpe_p5']:+.2f}  trade-order dd-spread="
          f"{to['maxdd_p95'] - to['maxdd_p5']:.3f}  4 schemes")


# ── §8 drawdown ladder + vol targeting (causal) ──────────────────────────────────────────
def _ladder_expo(returns):
    return drawdown_ladder(pd.Series(returns, index=_days(len(returns))))[1]

def test_drawdown_ladder() -> None:
    # causality: the bar that CAUSES the loss is still fully on (drawdown is only known at t-1)
    managed, expo = drawdown_ladder(pd.Series([-0.07] + [0.0] * 5, index=_days(6)))
    assert expo.iloc[0] == 1.0 and managed.iloc[0] == -0.07, "loss-causing bar must run at full exposure (causal)"

    # the three trigger steps: −7%→0.66, −10%→0.33 (deepest wins over −6%), −13%→0.0 (stop)
    assert _ladder_expo([-0.07] + [0.0] * 5).iloc[1] == 0.66, "−6% trigger → 0.66"
    assert _ladder_expo([-0.10] + [0.0] * 5).iloc[1] == 0.33, "−9% trigger → 0.33 (deepest match wins)"
    assert _ladder_expo([-0.13] + [0.0] * 5).iloc[1] == 0.00, "−12% trigger → flat (stop)"

    # hysteresis: de-risk on the drop, re-risk only once drawdown recovers above −4%
    assert _ladder_expo([-0.07, 0.10] + [0.0] * 3).iloc[2] == 1.0, "recovery above −4% must restore full exposure"

    # exposure never takes a value outside the stated ladder steps
    rng = np.random.default_rng(SEED)
    walk = rng.standard_normal(200) * 0.03
    _, ex = drawdown_ladder(pd.Series(walk, index=_days(200)))
    steps = np.array([1.0, 0.66, 0.33, 0.0])
    assert all(min(abs(v - steps)) < 1e-9 for v in ex), "exposure must only ever be a stated ladder step"

    # no look-ahead: exposure on a truncated series equals the full series on the shared prefix
    _, ex_full = drawdown_ladder(pd.Series(walk, index=_days(200)))
    _, ex_trunc = drawdown_ladder(pd.Series(walk[:120], index=_days(120)))
    assert np.array_equal(ex_full.iloc[:120].to_numpy(), ex_trunc.to_numpy()), "ladder must not use future returns"
    print("  DD-ladder     ✓  steps 1.0/0.66/0.33/0.0 at −6/−9/−12%, restore −4%, causal")

def test_vol_managed() -> None:
    rng = np.random.default_rng(SEED)
    _, ex = vol_managed(pd.Series(rng.standard_normal(300) * 0.01, index=_days(300)))
    assert ex.iloc[0] == 0.0, "vol exposure is lagged → first bar must be flat (no look-ahead)"
    lo = vol_managed(pd.Series(rng.standard_normal(300) * 0.005, index=_days(300)))[1].iloc[50:].mean()
    hi = vol_managed(pd.Series(rng.standard_normal(300) * 0.020, index=_days(300)))[1].iloc[50:].mean()
    assert lo > hi, "higher realised vol must earn lower exposure (vol targeting)"
    print(f"  vol-managed   ✓  lagged; expo(low-vol)={lo:.2f} > expo(high-vol)={hi:.2f}")


# ── equity short-borrow charge ───────────────────────────────────────────────────────────
def test_borrow_cost() -> None:
    idx = _days(40)
    rng = np.random.default_rng(SEED)
    cols = list("ABCDEF")
    px = pd.DataFrame(100 * np.cumprod(1 + rng.standard_normal((40, 6)) * 0.01, axis=0), index=idx, columns=cols)
    signal = pd.DataFrame(np.tile(np.arange(6.0), (40, 1)), index=idx, columns=cols)   # stable cross-section

    kw = dict(top_frac=0.34, weighting="equal", min_names=6, ppy=252)
    off = xs_backtest(px, signal, borrow_bps_annual=0.0, **kw)
    on = xs_backtest(px, signal, borrow_bps_annual=EQUITY_BORROW_BPS_ANNUAL, **kw)

    assert (off["carry"] == 0.0).all(), "no borrow charge when the rate is zero"
    assert on["carry"].sum() > 0.0, "a dollar-neutral book has a short leg → borrow must be charged"
    # only borrow changed → the whole net delta IS the borrow charge (subtracted, not double-counted)
    assert np.allclose((off["net"] - on["net"]).to_numpy(), on["carry"].to_numpy()), "net must drop by exactly the borrow"
    assert np.allclose(off["gross"].to_numpy(), on["gross"].to_numpy()), "borrow must not touch gross return"
    # magnitude matches k·(short notional)/ppy on the executed (shifted) weights
    expected = on["weights"].clip(upper=0.0).abs().sum(axis=1) * (EQUITY_BORROW_BPS_ANNUAL / 1e4) / 252
    assert np.allclose(expected.to_numpy(), on["carry"].to_numpy()), "borrow must equal rate × short-notional / ppy"
    print(f"  short-borrow  ✓  {EQUITY_BORROW_BPS_ANNUAL:.0f}bps/yr on shorts only; net drag={on['carry'].sum():.2e}")


# ── carry is charged because the INSTRUMENT says so, not because the caller remembered ─────
def test_carry_is_not_opt_in() -> None:
    """The regression this exists to prevent: a book of perps backtested with no funding charge.

    It happened three times — the x-sect leg, the lottery sleeve, BAB — because `xs_backtest` is
    handed a price panel and cannot see a venue, so carry was the caller's job and callers forgot.
    The test crosses the real boundary rather than asserting against a literal: it names symbols the
    Binance funding archive actually covers, and requires the charge to equal that archive's rates on
    the executed weights, with NOBODY having asked for it.
    """
    from src.backtest.carry import NoCarry, funding_panel, perp_symbols

    perps = sorted(perp_symbols() & {"BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT"})
    assert len(perps) >= 6, f"funding archive is missing majors — got {perps}"
    idx = pd.date_range("2023-01-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(SEED)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.standard_normal((60, len(perps))) * 0.01, axis=0),
                      index=idx, columns=perps)
    signal = pd.DataFrame(np.tile(np.arange(float(len(perps))), (60, 1)), index=idx, columns=perps)
    kw = dict(top_frac=0.34, weighting="equal", min_names=6, ppy=365)

    auto = xs_backtest(px, signal, **kw)                       # caller says NOTHING about carry
    off = xs_backtest(px, signal, carry=NoCarry(), **kw)       # opting out is explicit
    assert abs(auto["carry"]).sum() > 0.0, "a perp panel must be charged funding with no caller action"
    assert (off["carry"] == 0.0).all(), "NoCarry() must charge nothing"
    # the charge IS the archive: −Σ wᵢ·fᵢ on the executed weights, every 8h settlement binned per bar
    f = funding_panel(perps, idx)
    expected = (auto["weights"] * f.reindex_like(auto["weights"])).sum(axis=1)
    assert np.allclose(expected.to_numpy(), auto["carry"].to_numpy()), "carry must equal Σ w·funding"
    assert np.allclose((off["net"] - auto["net"]).to_numpy(), auto["carry"].to_numpy()), \
        "net must move by exactly the carry — charged once, not twice"
    # A panel of names the venue never settled funding on is CASH, and a dollar-neutral cash book
    # borrows every share it shorts — so its unasked default is the config's borrow rate, not zero.
    # That was the other half of the same hole: the shipped broad-equity sleeve shorts a decile of
    # 692 names and passed no rate at all.
    ren = dict(zip(perps, list("ABCDEF")[:len(perps)]))
    cash = xs_backtest(px.rename(columns=ren), signal.rename(columns=ren), **kw)
    assert cash["carry"].sum() > 0.0, "a cash panel with a short leg must be charged borrow unasked"
    expect = (cash["weights"].clip(upper=0.0).abs().sum(axis=1)
              * (EQUITY_BORROW_BPS_ANNUAL / 1e4) / kw["ppy"])
    assert np.allclose(expect.to_numpy(), cash["carry"].to_numpy()), "borrow must be the config rate"
    assert (xs_backtest(px.rename(columns=ren), signal.rename(columns=ren),
                        borrow_bps_annual=0.0, **kw)["carry"] == 0.0).all(), \
        "an explicit 0.0 is a statement that this book pays none, and must be honoured"
    yr = float(auto["carry"].sum()) / ((idx[-1] - idx[0]).days / 365.25)
    print(f"  carry-default ✓  perp panel charged {yr:+.2%}/yr unasked; cash panel charged "
          f"{EQUITY_BORROW_BPS_ANNUAL:.0f}bps borrow unasked; NoCarry()/0.0 opt out")


# ── core: execution lag (no look-ahead), cost model, deflated Sharpe, purged CV ────────────
def test_execution_lag() -> None:
    close = pd.Series(100 * 1.01 ** np.arange(20), index=_days(20))
    target = pd.Series([0.0] * 10 + [1.0] * 10, index=_days(20))      # go long from bar 10, held
    bt = backtest(close, target, capital=500_000, commission_bps=0.0, half_spread_bps=0.0, impact_k=0.0)
    assert bt["position"].iloc[10] == 0.0 and bt["position"].iloc[11] == 0.0, "no same-bar or t+1 fill"
    assert bt["position"].iloc[12] == 1.0, "a signal at bar t must fill exactly exec_lag=2 bars later"
    assert bt["gross_ret"].iloc[10] == 0.0, "the signal bar itself must earn nothing (t+2 execution)"
    print("  exec-lag      ✓  t+2 execution — a signal never fills at its own bar")


def test_cost_model() -> None:
    # infinite ADV (a $500k order in an infinitely deep book) → impact 0, cost = commission+half-spread
    base = float(trade_cost_bps(1e5, np.inf, 0.02, 5.0, 1.0, 0.1))
    assert abs(base - 6.0) < 1e-9, "with infinite ADV cost must be commission+half-spread only"
    # √-impact: quadrupling the order quadruples notional/ADV → doubles the impact term (Almgren)
    c1 = float(trade_cost_bps(1e6, 1e8, 0.02, 0.0, 0.0, 0.1))
    c4 = float(trade_cost_bps(4e6, 1e8, 0.02, 0.0, 0.0, 0.1))
    assert c1 > 0.0 and abs(c4 - 2.0 * c1) < 1e-9, "impact must scale as √(notional/ADV)"
    print(f"  cost-model    ✓  commission+spread floor={base:.0f}bps; impact √-scales, never flat")


def test_deflated_sharpe() -> None:
    # PSR rises with the observed Sharpe; the max-under-null benchmark rises with the trial count;
    # deflating against it can only LOWER the probability (DSR ≤ PSR-vs-0).
    assert probabilistic_sharpe(0.15, 1000, 0.0, 3.0) > probabilistic_sharpe(0.05, 1000, 0.0, 3.0), "PSR ↑ in Sharpe"
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01), "E[max Sharpe] grows with trial count"
    psr0 = probabilistic_sharpe(0.15, 1000, 0.0, 3.0)
    dsr = deflated_sharpe(0.15, 1000, 0.0, 3.0, n_trials=100, var_across_trials=0.01)
    assert dsr <= psr0, "deflating against the multiple-testing benchmark cannot raise the probability"
    assert sharpe(pd.Series([0.0] * 10), 365) == 0.0, "a zero-variance series must return Sharpe 0, not NaN"
    print(f"  deflated-SR   ✓  PSR↑ in Sharpe; DSR({dsr:.2f}) ≤ PSR({psr0:.2f}); E[max]↑ in N")


def test_panel_impact_cost() -> None:
    # hand-computed: 1 name, turnover 1.0, $1M order into $100M ADV → q=0.01, √q=0.1, σ=0.02,
    # impact_k=0.1 → impact_bps = 0.1·0.02·0.1 = 2e-4, cost = 2e-4·(dw=1) = 2e-4 per bar.
    idx = _days(3)
    dw = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    sig = pd.DataFrame({"A": [0.02, 0.02, 0.02]}, index=idx)
    adv = pd.DataFrame({"A": [1e8, 1e8, 1e8]}, index=idx)
    c = panel_impact_cost(dw, sig, adv, capital=1e6, impact_k=0.1)
    assert np.allclose(c.to_numpy(), 2e-4), f"√-impact known answer 2e-4, got {c.iloc[0]:.2e}"
    assert (panel_impact_cost(dw, sig, adv, 1e6, 0.0) == 0.0).all(), "impact_k=0 must cost nothing"
    # √(notional): a 4× larger order raises per-name impact_bps 2× (q 4× → √q 2×)
    c4 = panel_impact_cost(dw, sig, adv, capital=4e6, impact_k=0.1)
    assert np.allclose(c4.to_numpy(), 4e-4), "impact_bps must scale as √(notional/ADV)"
    print("  panel-impact  ✓  shared √-impact = 2e-4 known answer; √-scales; one source for 4 sleeves")


def test_purged_cv() -> None:
    n = 120
    t0 = _days(n)
    t1 = t0 + pd.Timedelta(days=3)                                    # each label spans 3 days
    embargo = pd.Timedelta(days=1)
    folds = list(purged_kfold(t0, t1, n_splits=4, embargo=embargo))
    assert len(folds) == 4, "must yield the requested number of folds"
    for tr, te in folds:
        ts, te_end = t0[te].min(), t1[te].max()
        # every retained training label lies ENTIRELY before the test span or after test-end+embargo
        for j in tr:
            assert t1[j] < ts or t0[j] > te_end + embargo, "purge/embargo must drop every overlapping train label"
    print(f"  purged-CV     ✓  {len(folds)} folds, no train label overlaps test span + embargo")


def main() -> None:
    print("known-answer invariants for the correctness-critical math:\n")
    # core engine / cost / metrics / CV — the invariants every reported number depends on
    test_execution_lag()
    test_cost_model()
    test_panel_impact_cost()
    test_deflated_sharpe()
    test_purged_cv()
    # the sprint modules that produce the headline figures
    test_cscv_pbo()
    test_monte_carlo()
    test_drawdown_ladder()
    test_vol_managed()
    test_borrow_cost()
    test_carry_is_not_opt_in()
    print("\nSMOKE MATH OK — every correctness-critical invariant holds")


if __name__ == "__main__":
    main()
