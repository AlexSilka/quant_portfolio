"""BAB robustness — does the crypto beta-neutral (FP) premium survive off the a-priori config?

The main deliverable (scripts/bab/run_bab.py) fixes top-100-liquid, 1d, 90-day beta, monthly rebalance. This
sweeps every axis around that on the honest beta-neutral construction (t+2, liquidity-aware cost):

  1. UNIVERSE SIZE  (top-10..200)              — is +0.77 a top-100 artifact?  (no: broad plateau)
  2. TIMEFRAME      (5m..1d)                    — a daily-sampling artifact?    (no: bar-invariant)
  3. REBALANCE × BETA-LOOKBACK (1h grid)        — is there a faster/intraday BAB? (no: slow is optimal)
  4. FX             (currencies, FP-2014 asset) — does it work off crypto/equity? (no: dead)

For the size/timeframe cells: full plus first/second-half Sharpe (the second half exposes the 2025-26
decay). The rebalance grid reports net Sharpe + a gross→net cost decomposition at the fastest cell.

    python scripts/bab/run_bab_robust.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from src.config import BAB_DIR, CACHE_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE = CACHE_DIR / "xs"
COST, WINSOR, TVOL = 6.0, 1.0, VOL_TARGET_ANNUAL
BETA_LB_D, TOPFRAC, REBAL_D, EXEC_LAG, IMPACT_K = 90, 0.2, 21, 2, 0.1
BPD = {"1d": 1, "4h": 6, "1h": 24, "15m": 96, "5m": 288}
PPY = {tf: 365 * b for tf, b in BPD.items()}
PPY_FX = 252   # FX trades ~5d/week


def _beta_neutral_net(C, A, top_n, bpd, ppy, cost=COST):
    """Vol-targeted net series of the a-priori beta-neutral (FP) book (params scaled by bars/day)."""
    beta = top_n_liquid(bab.panel_beta(C, BETA_LB_D * bpd), A, top_n)
    w = bab.bab_weights(beta, top_frac=TOPFRAC, neutral="beta", rebal=REBAL_D * bpd)
    bt = bab.bab_backtest(C, w, exec_lag=EXEC_LAG, cost_bps=cost, adv=A, impact_k=IMPACT_K)
    return vol_target(bt["net"], ppy, TVOL)


def _orient_usd(C):
    """FX: keep the 12 USD-pairs oriented to *foreign-vs-USD* (USDXXX → reciprocal), dropping the 13
    crosses — so the equal-weight panel mean is the (anti-)dollar risk factor and beta to it is the
    clean dollar-factor beta the Frazzini-Pedersen FX-BAB ranks on. XXXUSD pairs are already
    foreign-per-USD; USDXXX are USD-per-foreign, so 1/price flips them to the same convention."""
    out = pd.DataFrame(index=C.index)
    for c in [c for c in C.columns if "USD" in c]:
        out[c] = 1.0 / C[c] if c.startswith("USD") else C[c]
    return out


def _halves(net, ppy):
    """Full / first-half / second-half Sharpe (temporal-stability OOS)."""
    n = net.dropna()
    mid = n.index[len(n) // 2]
    return (round(summarise(n, ppy)["sharpe_ann"], 3),
            round(summarise(n.loc[:mid], ppy)["sharpe_ann"], 3),
            round(summarise(n.loc[mid:], ppy)["sharpe_ann"], 3))


def _load(tf):
    C = pd.read_parquet(CACHE / f"crypto_{tf}_close.parquet")
    A = pd.read_parquet(CACHE / f"crypto_{tf}_adv.parquet").reindex_like(C)
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC"); A.index = A.index.tz_localize("UTC")
    return bab.winsorize_panel(C, WINSOR), A


def main():
    rows = []

    # ── universe-size sweep (crypto 1d, top-N liquid) ──────────────────────────────────────────
    C, A = _load("1d")
    print(f"\n=== UNIVERSE SIZE sweep (crypto 1d, beta-neutral FP, a-priori) — {C.shape[1]} names ===")
    print(f"{'top_N':>6} {'names/leg':>10} {'full':>7} {'1st-half':>9} {'2nd-half':>9}")
    for n in (10, 25, 50, 100, 200):
        net = _beta_neutral_net(C, A, n, 1, PPY["1d"])
        full, h1, h2 = _halves(net, PPY["1d"])
        per_leg = int(round(min(n, C.shape[1]) * TOPFRAC))
        rows.append({"sweep": "universe", "cell": f"top{n}", "full": full, "first_half": h1, "second_half": h2})
        print(f"{n:>6} {per_leg:>10} {full:>+7.2f} {h1:>+9.2f} {h2:>+9.2f}")

    # ── timeframe sweep (crypto, top-100 liquid) ───────────────────────────────────────────────
    # NOTE: BAB is a slow signal (90d beta, monthly rebalance), so finer bars re-sample the SAME
    # monthly-turnover book — expect ~flat Sharpe. vol_target's window is the repo default (60 bars),
    # which is short in calendar time at 5m/15m; the near-flat result confirms it is not distorting.
    print("\n=== TIMEFRAME sweep (crypto, top-100 liquid, beta-neutral FP, a-priori) ===")
    print(f"{'tf':>6} {'bars':>8} {'names':>6} {'full':>7} {'1st-half':>9} {'2nd-half':>9}")
    for tf in ("1d", "4h", "1h", "15m", "5m"):
        Ct, At = _load(tf)
        net = _beta_neutral_net(Ct, At, 100, BPD[tf], PPY[tf])
        full, h1, h2 = _halves(net, PPY[tf])
        rows.append({"sweep": "timeframe", "cell": tf, "full": full, "first_half": h1, "second_half": h2})
        print(f"{tf:>6} {Ct.shape[0]:>8} {Ct.shape[1]:>6} {full:>+7.2f} {h1:>+9.2f} {h2:>+9.2f}")

    # ── rebalance-frequency × beta-lookback sweep (crypto 1h) — is there a FAST/intraday BAB? ────
    # The only axis a slow monthly BAB leaves open: shorten the beta window AND rebalance faster, to
    # capture intraday beta dynamics. If faster only lowers net Sharpe, the edge is genuinely slow and
    # turnover×cost kills the fast version (the cost-check line quantifies gross→net at the fastest cell).
    Ch, Ah = _load("1h")
    ppy_h = PPY["1h"]
    LBS = [("1d", 24), ("3d", 72), ("7d", 168), ("30d", 720), ("90d", 2160)]
    REBALS = [("1h", 1), ("4h", 4), ("1d", 24), ("1w", 168), ("1M", 504)]
    print("\n=== REBALANCE × BETA-LOOKBACK sweep (crypto 1h, top-100, beta-neutral FP net Sharpe) ===")
    print(f"{'lb / rebal':>11} " + " ".join(f"{r[0]:>6}" for r in REBALS))
    for lbn, lbb in LBS:
        beta = top_n_liquid(bab.panel_beta(Ch, lbb), Ah, 100)
        cells = []
        for rn, rb in REBALS:
            w = bab.bab_weights(beta, top_frac=TOPFRAC, neutral="beta", rebal=rb)
            net = vol_target(bab.bab_backtest(Ch, w, exec_lag=EXEC_LAG, cost_bps=COST, adv=Ah,
                                              impact_k=IMPACT_K)["net"], ppy_h, TVOL)
            s = round(summarise(net.dropna(), ppy_h)["sharpe_ann"], 2)
            cells.append(s)
            rows.append({"sweep": "rebalance", "cell": f"1h_lb{lbn}_rb{rn}", "full": s,
                         "first_half": None, "second_half": None})
        print(f"{lbn:>11} " + " ".join(f"{c:>+6.2f}" for c in cells))
    # cost decomposition at the fastest cell (90d beta, hourly rebalance): gross vs net
    wf = bab.bab_weights(top_n_liquid(bab.panel_beta(Ch, 2160), Ah, 100), top_frac=TOPFRAC, neutral="beta", rebal=1)
    btf = bab.bab_backtest(Ch, wf, exec_lag=EXEC_LAG, cost_bps=COST, adv=Ah, impact_k=IMPACT_K)
    sg = summarise(vol_target(btf["gross"], ppy_h, TVOL).dropna(), ppy_h)["sharpe_ann"]
    sn = summarise(vol_target(btf["net"], ppy_h, TVOL).dropna(), ppy_h)["sharpe_ann"]
    print(f"  cost check (90d beta, hourly rebal): gross {sg:+.2f} → net {sn:+.2f}  (turnover×cost eats {sg - sn:+.2f})")

    # ── FX as a third asset class (BAB is documented on currencies; FP 2014) ────────────────────
    # FX has no ADV panel → flat ~1bp cost, no √-impact; the universe is small (12 clean USD-pairs).
    FX = pd.read_parquet(CACHE / "fx_1d_close.parquet")
    print(f"\n=== FX BAB (crypto/equity's third asset, 1bp, no impact, {FX.index.min().date()}..) ===")
    print(f"{'panel':>34} {'pairs':>6} {'dollar−β':>9} {'beta-neut':>10} {'MC-P5':>7} {'1st':>6} {'2nd':>6}")
    for panel, tag in [(_orient_usd(FX), "12 USD-pairs oriented (clean)"), (FX, "all 25 pairs (muddled market)")]:
        Cw = bab.winsorize_panel(panel, 0.25)                 # FX daily moves are small; 25% clips only peg breaks
        beta = bab.panel_beta(Cw, BETA_LB_D)                  # no top_n mask — FX universe is already liquid/small
        dn = vol_target(xs_backtest(Cw, -beta, top_frac=TOPFRAC, rebal=REBAL_D, exec_lag=EXEC_LAG,
                                    cost_bps=1.0, adv=None, impact_k=0.0)["net"], PPY_FX, TVOL)
        bn = _beta_neutral_net(Cw, None, 0, 1, PPY_FX, cost=1.0)
        f, h1, h2 = _halves(bn, PPY_FX)
        mc = bootstrap_sharpe(bn.dropna(), PPY_FX, 1000, SEED).get("sharpe_p5", float("nan"))
        rows.append({"sweep": "fx", "cell": tag.split(" (")[0], "full": f, "first_half": h1, "second_half": h2})
        print(f"{tag:>34} {panel.shape[1]:>6} {summarise(dn.dropna(), PPY_FX)['sharpe_ann']:>+9.2f} "
              f"{f:>+10.2f} {mc:>+7.2f} {h1:>+6.2f} {h2:>+6.2f}")

    pd.DataFrame(rows).to_csv(BAB_DIR / "bab_robust.csv", index=False)
    print("\nRUN BAB ROBUST OK  -> reports/bab_robust.csv")


if __name__ == "__main__":
    main()
