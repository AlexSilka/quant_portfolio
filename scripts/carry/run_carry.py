"""Carry deep-dive: is there a real, harvestable perp funding-carry premium, and under what
construction? Runs the directional baseline (single-asset funding timer), the cross-sectional
dollar-neutral carry factor across a signal/param grid, price-only reversal & momentum controls
(is carry just reversal?), a funding-residualised carry (incremental info beyond momentum), and a
shuffled-funding placebo — all vol-targeted to 15% so they compare on equal risk.

    python scripts/carry/run_carry.py [quick]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.config import CACHE_DIR, CAPITAL_USD, CARRY_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry, carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

SEED, CAP, TVOL, PPY = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL, 365
CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
START, END = "2020-01", "2026-07"
CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "TRXUSDT", "DOGEUSDT", "ZECUSDT",
          "ADAUSDT", "XMRUSDT", "LINKUSDT", "XLMUSDT", "BCHUSDT", "LTCUSDT", "HBARUSDT", "1000SHIBUSDT",
          "AVAXUSDT", "SUIUSDT", "UNIUSDT", "NEARUSDT", "DOTUSDT", "AAVEUSDT", "1000PEPEUSDT", "ICPUSDT",
          "ETCUSDT", "QNTUSDT", "ALGOUSDT", "ATOMUSDT", "FILUSDT", "ARBUSDT", "APTUSDT", "INJUSDT",
          "DASHUSDT", "VETUSDT", "FETUSDT", "CRVUSDT", "1000LUNCUSDT", "STXUSDT", "LDOUSDT", "IMXUSDT",
          "XTZUSDT", "JASMYUSDT", "CFXUSDT", "OPUSDT", "1000FLOKIUSDT", "ENSUSDT", "COMPUSDT", "GRTUSDT",
          "IOTAUSDT", "RUNEUSDT"]
rng = np.random.default_rng(SEED)


def load_panel():
    close, fund = {}, {}
    for s in CRYPTO:
        px = load_klines(s, "1d", START, END, market="um")
        if len(px):
            close[s] = px["close"]
        f = load_funding(s, START, END)["last_funding_rate"]
        if len(f):
            fund[s] = f
    C = pd.DataFrame(close).sort_index()
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(C.index)
    return C, fd


def vt(net: pd.Series) -> pd.Series:
    """Vol-target a daily net-return series to 15% annualised (lagged, look-ahead-free)."""
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def row(name, bk, mc=True, extra=None):
    net = vt(bk["ret"])
    s = summarise(net, PPY)
    p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan) if (mc and s["sharpe_ann"] > 0.3) else np.nan
    yrs = (net.index[-1] - net.index[0]).days / 365.25
    # attribution: annualised return contribution of each leg (pre-vol-target, on gross-1 book)
    fund_ann = float(bk["funding"].mean() * PPY) if "funding" in bk else np.nan
    px_ann = float((bk.get("price", bk.get("basis")).mean()) * PPY) if len(bk) else np.nan
    r = {"sleeve": name, "sharpe": s["sharpe_ann"], "sortino": s["sortino_ann"],
         "mc_p5": p5, "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
         "psr_gt0": s["psr_gt0"], "ann_funding_%": fund_ann * 100, "ann_price_%": px_ann * 100,
         "turnover": float(bk["turnover"].sum()), "n_obs": s["n_obs"]}
    if extra:
        r.update(extra)
    RET[name] = net
    return r


RET = {}


def main():
    quick = "quick" in sys.argv
    (CACHE_DIR / "carry").mkdir(parents=True, exist_ok=True)
    C, fd = load_panel()
    ret = C.pct_change()
    print(f"panel: {C.shape[1]} names, {C.index.min().date()}..{C.index.max().date()}, {len(C)} days\n")
    rows = []

    # ---- 1) directional single-asset baseline (confirm it is ~0 / negative) ----
    print("=== directional single-asset carry (baseline) ===")
    dvals = []
    for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]:
        px = load_klines(s, "1d", START, END, market="um")
        f = load_funding(s, START, END)["last_funding_rate"]
        pos = vol_target(carry.primary_side(f, px["close"]), px["close"], TVOL, PPY)
        bt = backtest(px["close"], pos, capital=CAP, funding=f, **CC)
        d = (1 + bt["net_ret"]).resample("D").prod() - 1
        sh = summarise(d, PPY)["sharpe_ann"]
        dvals.append(sh)
        print(f"  {s:14s} directional Sharpe {sh:+.2f}")
    print(f"  -> mean directional Sharpe {np.mean(dvals):+.2f}  (price risk swamps funding)\n")

    # ---- 2) cross-sectional carry: signal x top_frac x rebalance x lookback grid ----
    print("=== cross-sectional dollar-neutral carry (grid) ===")
    lookbacks = [3, 7, 14] if quick else [1, 3, 7, 14, 30]
    tops = [0.2, 0.3] if quick else [0.1, 0.2, 0.3, 0.4]
    rebals = [1] if quick else [1, 3, 5]
    for lb in lookbacks:
        sig = carry_xs.signal_level(fd, lb)
        for tf in tops:
            for rb in rebals:
                bk = carry_xs.xs_book(C, fd, sig, direction=-1.0, top_frac=tf, rebalance=rb,
                                      cost_bps=CC["commission_bps"] + CC["half_spread_bps"])
                r = row(f"XScarry_lvl{lb}_top{int(tf*100)}_rb{rb}", bk,
                        extra={"family": "carry_xs", "signal": "level", "lb": lb, "top": tf, "rb": rb})
                rows.append(r)
    # vol-adjusted and residualised signals at a central config
    for signame, sig in [("voladj", carry_xs.signal_vol_adj(fd, ret, 7)),
                         ("resid", carry_xs.signal_resid(fd, ret, 7))]:
        bk = carry_xs.xs_book(C, fd, sig, direction=-1.0, top_frac=0.3,
                              cost_bps=CC["commission_bps"] + CC["half_spread_bps"])
        rows.append(row(f"XScarry_{signame}_top30", bk,
                        extra={"family": "carry_xs", "signal": signame, "lb": 7, "top": 0.3, "rb": 1}))

    # ---- 3) controls: price-only reversal & momentum, same engine (is carry just reversal?) ----
    print("=== controls: price reversal / momentum (same book engine) ===")
    for cname, csig, cdir in [("XSreversal_14", ret.rolling(14).sum(), 1.0),   # long losers -> dir maps to low
                              ("XSmomentum_14", ret.rolling(14).sum(), -1.0)]:
        # reversal: long low trailing-return -> direction=-1 on the return signal? keep explicit:
        # use direction so that reversal LONGS losers (low return) and momentum LONGS winners.
        bk = carry_xs.xs_book(C, fd, csig, direction=(-1.0 if "reversal" in cname else 1.0),
                              top_frac=0.3, cost_bps=CC["commission_bps"] + CC["half_spread_bps"])
        rows.append(row(cname, bk, extra={"family": "control", "signal": cname, "lb": 14, "top": 0.3, "rb": 1}))

    # ---- 4) placebo: shuffled funding signal (destroys the carry information) ----
    print("=== placebo: shuffled-funding carry ===")
    sig = carry_xs.signal_level(fd, 7)
    plac = pd.DataFrame(rng.standard_normal(sig.shape), index=sig.index, columns=sig.columns)
    bk = carry_xs.xs_book(C, fd, plac, direction=-1.0, top_frac=0.3,
                          cost_bps=CC["commission_bps"] + CC["half_spread_bps"])
    rows.append(row("PLACEBO_shuffled", bk, mc=False,
                    extra={"family": "placebo", "signal": "shuffled", "lb": 7, "top": 0.3, "rb": 1}))

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(CARRY_DIR / "carry_results.csv", index=False)
    pd.DataFrame(RET).to_parquet(CARRY_DIR / "carry_returns.parquet")

    # correlation of the headline carry vs the controls (incremental-info check)
    best = df[df.family == "carry_xs"].iloc[0]["sleeve"]
    corr_block = pd.DataFrame({k: RET[k] for k in [best, "XSreversal_14", "XSmomentum_14"]
                               if k in RET}).corr()

    pd.set_option("display.width", 200, "display.max_columns", 20)
    show = ["sleeve", "sharpe", "mc_p5", "max_dd", "months_in_profit", "ann_funding_%", "ann_price_%", "turnover"]
    print("\n=== RESULTS (vol-targeted 15%, net of costs, sorted by Sharpe) ===")
    print(df[show].round(2).to_string(index=False))
    print(f"\nbest carry sleeve: {best}")
    print("correlation (best carry vs price controls):")
    print(corr_block.round(2).to_string())
    print("\nCARRY OK")


if __name__ == "__main__":
    main()
