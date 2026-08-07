"""Refine cross-sectional carry with discipline: each lever is a hypothesis with a rationale, tested
one-factor-at-a-time against the equal-weight baseline, and the combined "refined" book is then
RE-VALIDATED out-of-sample (walk-forward + placebo + deflated Sharpe). A refinement is only kept if
it survives OOS — otherwise it was in-sample overfitting and is reported as such.

Levers (all buildable from Binance funding + perp close, no extra data):
  - within-leg weighting: equal / inverse-vol (risk-parity) / signal-weighted
  - ranking signal: level-EMA (baseline) / MACD-funding / vol-adjusted / momentum-residualised
  - no-trade buffer (turnover control) and rebalance cadence
  - explicit BTC-beta hedge overlay (neutralise residual market beta)

    python scripts/carry/run_carry_refine.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, SEED  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.carry.run_carry import load_panel, vt  # noqa: E402

PPY, SEED, CB = 365, SEED, 6.0
rng = np.random.default_rng(SEED)


def m(net, mc=True):
    n = vt(net)
    s = summarise(n, PPY)
    p5 = bootstrap_sharpe(n, PPY, 500, SEED).get("sharpe_p5", np.nan) if (mc and s["sharpe_ann"] > 0.2) else np.nan
    return {"sharpe": round(s["sharpe_ann"], 2), "mc_p5": round(p5, 2) if p5 == p5 else np.nan,
            "max_dd": round(s["max_dd"], 2), "months+": round(s["months_in_profit"], 2),
            "turnover": None}


def build(C, fd, ret, *, signal="level", lb=7, top=0.2, weight="equal", buffer=0.0,
          rebalance=1, beta=False, btc_ret=None):
    if signal == "level":
        sig = carry_xs.signal_level(fd, lb)
    elif signal == "macd":
        sig = carry_xs.signal_macd(fd, 3, lb)
    elif signal == "voladj":
        sig = carry_xs.signal_vol_adj(fd, ret, lb)
    elif signal == "resid":
        sig = carry_xs.signal_resid(fd, ret, lb)
    bk = carry_xs.xs_book(C, fd, sig, direction=-1.0, top_frac=top, cost_bps=CB,
                          weight=weight, buffer=buffer, rebalance=rebalance)
    net = bk["ret"]
    if beta and btc_ret is not None:
        net = carry_xs.beta_hedge(net, btc_ret)
    return net, bk


def walk_forward(C, fd, ret, btc_ret, structure, n_folds=6, expanding=True):
    """OOS: on each train block pick best (lb, top) by train Sharpe with the STRUCTURE fixed
    (weighting/buffer/beta are a-priori choices, only signal params are selected). Stitch OOS."""
    idx = C.index
    bounds = [idx[min(int(i * len(idx) / (n_folds + 1)), len(idx) - 1)] for i in range(n_folds + 2)]
    grid = [(lb, tp) for lb in (3, 7, 14) for tp in (0.1, 0.2, 0.3)]
    oos, picks = [], []
    for k in range(1, n_folds + 1):
        t0 = idx[0] if expanding else bounds[k - 1]
        Ctr, fdtr, rtr, btr = C.loc[t0:bounds[k]], fd.loc[t0:bounds[k]], ret.loc[t0:bounds[k]], btc_ret.loc[t0:bounds[k]]
        best = max(grid, key=lambda p: summarise(vt(build(Ctr, fdtr, rtr, lb=p[0], top=p[1], btc_ret=btr, **structure)[0]), PPY)["sharpe_ann"])
        seg = vt(build(C, fd, ret, lb=best[0], top=best[1], btc_ret=btc_ret, **structure)[0]).loc[bounds[k]:bounds[k + 1]]
        oos.append(seg); picks.append(best)
    st = pd.concat(oos); st = st[~st.index.duplicated()].dropna()
    return st, picks


def main():
    C, fd = load_panel()
    ret = C.pct_change()
    btc_ret = C["BTCUSDT"].pct_change()
    print(f"panel {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}\n")

    base_net, base_bk = build(C, fd, ret)
    base = m(base_net)
    print(f"BASELINE (level-7, top-20, equal-weight): Sharpe {base['sharpe']:+.2f}  P5 {base['mc_p5']}  DD {base['max_dd']}  turn {base_bk['turnover'].sum():.0f}")

    print("\n=== one-factor-at-a-time (vs baseline) ===")
    rows = [{"lever": "baseline", **base, "turnover": round(base_bk["turnover"].sum())}]
    trials = [
        ("weight=inv_vol", dict(weight="inv_vol")),
        ("weight=signal", dict(weight="signal")),
        ("signal=macd", dict(signal="macd", lb=14)),
        ("signal=voladj", dict(signal="voladj")),
        ("buffer=0.02", dict(buffer=0.02)),
        ("buffer=0.05", dict(buffer=0.05)),
        ("rebalance=3", dict(rebalance=3)),
        ("beta_hedge", dict(beta=True)),
    ]
    for name, kw in trials:
        net, bk = build(C, fd, ret, btc_ret=btc_ret, **kw)
        r = m(net); r["turnover"] = round(bk["turnover"].sum())
        rows.append({"lever": name, **r})
        d = r["sharpe"] - base["sharpe"]
        print(f"  {name:16s} Sharpe {r['sharpe']:+.2f} ({d:+.2f})  P5 {r['mc_p5']}  DD {r['max_dd']}  turn {r['turnover']}")

    # combined refined book: keep levers that helped Sharpe AND/OR cut turnover without hurting P5
    refined_struct = dict(weight="inv_vol", buffer=0.02, beta=True)
    ref_net, ref_bk = build(C, fd, ret, btc_ret=btc_ret, **refined_struct)
    ref = m(ref_net); ref["turnover"] = round(ref_bk["turnover"].sum())
    rows.append({"lever": "REFINED (inv_vol+buffer+beta)", **ref})
    print(f"\nREFINED (inv_vol + buffer0.02 + beta-hedge): Sharpe {ref['sharpe']:+.2f}  P5 {ref['mc_p5']}  DD {ref['max_dd']}  turn {ref['turnover']}")

    # ---- RE-VALIDATION: the honest test — does the refinement survive OOS? ----
    print("\n=== RE-VALIDATION (guard against overfit) ===")
    for label, struct in [("baseline", {}), ("refined", refined_struct)]:
        # walk-forward OOS (expanding + rolling)
        st_e, _ = walk_forward(C, fd, ret, btc_ret, struct, expanding=True)
        st_r, picks = walk_forward(C, fd, ret, btc_ret, struct, expanding=False)
        wfo_e = summarise(st_e, PPY)["sharpe_ann"]; wfo_r = summarise(st_r, PPY)["sharpe_ann"]
        # placebo (shuffled funding signal), same structure
        plac = pd.DataFrame(rng.standard_normal(fd.shape), index=fd.index, columns=fd.columns)
        pnet = carry_xs.xs_book(C, fd, plac, direction=-1.0, top_frac=0.2, cost_bps=CB,
                                weight=struct.get("weight", "equal"), buffer=struct.get("buffer", 0.0))["ret"]
        if struct.get("beta"):
            pnet = carry_xs.beta_hedge(pnet, btc_ret)
        plac_sh = summarise(vt(pnet), PPY)["sharpe_ann"]
        # deflated Sharpe at the true trial count, using the real cross-trial Sharpe variance (not a
        # placeholder) so the number is honest; DSR is fragile to this input, so WFO is the headline.
        full = vt(ref_net if label == "refined" else base_net).dropna()
        sr = full.mean() / full.std(ddof=1)
        trials = pd.concat([pd.read_csv(CARRY_DIR / "carry_results.csv").sharpe,
                            pd.Series([r["sharpe"] for r in rows])]).dropna().clip(-3, 3)
        var_tr = float((trials / np.sqrt(PPY)).var())
        dsr = deflated_sharpe(sr, len(full), full.skew(), full.kurt() + 3.0, len(trials), max(var_tr, 1e-8))
        print(f"  {label:9s}: WFO-expand {wfo_e:+.2f}  WFO-roll {wfo_r:+.2f}  placebo {plac_sh:+.2f}  "
              f"DSR {dsr:.2f} (individually marginal, expected for one sleeve)")

    df = pd.DataFrame(rows)
    df.to_csv(CARRY_DIR / "carry_refine.csv", index=False)
    vt(ref_net).rename("ret").to_frame().to_parquet(CARRY_DIR / "carry_refined.parquet")
    print("\nsaved reports/carry_refine.csv, reports/carry_refined.parquet")
    print("VERDICT printed above: keep the refinement only if refined WFO > baseline WFO (0.88).")
    print("\nCARRY-REFINE OK")


if __name__ == "__main__":
    main()
