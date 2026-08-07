"""Residual-momentum (H5) robustness — does residualising the momentum signal help off the a-priori
config, and across the axes the brief asks for (universe size, timeframe, parameter variations)?

The main deliverable (scripts/residmom/run_residmom.py) fixes the a-priori config per asset and runs the full
funnel. This sweeps every axis around it, always **raw risk-adjusted momentum vs residual (idio)
side by side**, so the question is not "is residual good" but "does residualising *beat raw* here":

  1. UNIVERSE SIZE  (top-10..all)     — crypto & equity: does the residual edge hold across breadth?
  2. TIMEFRAME      (1d/4h/1h[/15m])  — crypto & FX: is the signal a daily artifact?
  3. PARAMETERS     (skip/weight/rebal/formation) — incl. the literature's skip-t-1 (esp. crypto)
  4. MARKET FACTOR  (EW panel vs BTC) — crypto: how much does the residualisation factor matter?

Each cell reports net Sharpe (vol-targeted 15%, t+2, liquidity-aware cost) and the realised book beta
(the residual's headline claim is lower beta). Split-half where a temporal-stability read helps.

    python scripts/residmom/run_residmom_robust.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CACHE_DIR, REPORTS_DIR, RESIDMOM_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import (idio_mom, risk_adj_mom, top_n_liquid,  # noqa: E402
                               vol_target, xs_backtest)

CACHE = CACHE_DIR / "xs"
SEED, TVOL = SEED, VOL_TARGET_ANNUAL
BPD = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}

CFG = {
    "crypto": dict(cost=6.0, imp=0.1, winsor=1.0, form_d=30, beta_d=90, sk_d=0, tf=0.3, rebal_d=21,
                   topn=100, ppy_d=365, mkt_col="BTCUSDT"),
    "equity": dict(cost=3.0, imp=0.1, winsor=0.5, form_d=252, beta_d=756, sk_d=7, tf=0.1, rebal_d=21,
                   topn=100, ppy_d=252, mkt_col=None),
    "fx": dict(cost=1.0, imp=0.0, winsor=0.5, form_d=90, beta_d=250, sk_d=0, tf=0.3, rebal_d=21,
               topn=0, ppy_d=252, mkt_col=None),
}


def _load(tag):
    C = pd.read_parquet(CACHE / f"{tag}_close.parquet")
    ap = CACHE / f"{tag}_adv.parquet"
    A = pd.read_parquet(ap).reindex_like(C) if ap.exists() else None
    if C.index.tz is None:
        C.index = C.index.tz_localize("UTC")
        if A is not None:
            A.index = A.index.tz_localize("UTC")
    return C, A


def _sh(net, ppy):
    n = net.dropna()
    return round(summarise(n, ppy)["sharpe_ann"], 2) if len(n) > 2 else float("nan")


def _beta(net, mkt):
    df = pd.concat([net.rename("n"), mkt.rename("m")], axis=1).dropna()
    if len(df) < 30 or df["m"].var() == 0:
        return float("nan")
    return round(float(np.cov(df["m"], df["n"])[0, 1] / df["m"].var()), 3)


def _halves(net, ppy):
    n = net.dropna()
    if len(n) < 60:
        return _sh(n, ppy), float("nan"), float("nan")
    mid = n.index[len(n) // 2]
    return _sh(n, ppy), _sh(n.loc[:mid], ppy), _sh(n.loc[mid:], ppy)


def _book(C, sig, A, kind, *, tf=None, weighting="equal", rebal=None, cost_mult=1.0, ppy=None):
    c = CFG[kind]
    s = top_n_liquid(sig, A, c["topn"]) if c["topn"] else sig
    vol = C.pct_change().rolling(20).std() if weighting == "volinv" else None
    bt = xs_backtest(C, s, top_frac=tf or c["tf"], weighting=weighting, rebal=rebal or c["rebal_d"],
                     exec_lag=2, cost_bps=c["cost"] * cost_mult, adv=A, impact_k=c["imp"])
    return vol_target(bt["net"], ppy or c["ppy_d"], TVOL)


def universe_sweep(rows):
    print(f"\n{'='*84}\n1. UNIVERSE SIZE — raw riskadj vs residual (idio), a-priori config, 1d\n{'='*84}")
    for kind in ("crypto", "equity"):
        c = CFG[kind]
        tag = "crypto_1d" if kind == "crypto" else "stocks_broad_1d"
        C, A = _load(tag)
        C = bab.winsorize_panel(C, c["winsor"])
        mkt = C[c["mkt_col"]].pct_change() if c["mkt_col"] else C.pct_change().mean(axis=1)
        raw = risk_adj_mom(C, c["form_d"], c["sk_d"])
        idio = idio_mom(C, c["form_d"], c["beta_d"], c["sk_d"], market=None)
        print(f"\n  {kind.upper()} ({C.shape[1]} names)   "
              f"{'topN':>6} {'raw Sh':>7} {'res Sh':>7} {'Δ':>6} {'raw β':>7} {'res β':>7}")
        for n in (10, 25, 50, 100, 200, 0):
            rs = top_n_liquid(raw, A, n) if n else raw          # mask to the swept N (0 = all)
            is_ = top_n_liquid(idio, A, n) if n else idio
            def bk(sig):                                         # backtest the already-masked signal
                bt = xs_backtest(C, sig, top_frac=c["tf"], weighting="equal", rebal=c["rebal_d"],
                                 exec_lag=2, cost_bps=c["cost"], adv=A, impact_k=c["imp"])
                return vol_target(bt["net"], c["ppy_d"], TVOL)
            nr, ni = bk(rs), bk(is_)
            sr, si = _sh(nr, c["ppy_d"]), _sh(ni, c["ppy_d"])
            br, bi = _beta(nr, mkt), _beta(ni, mkt)
            lbl = "all" if n == 0 else str(n)
            rows.append({"sweep": "universe", "asset": kind, "cell": f"top{lbl}",
                         "raw_sharpe": sr, "resid_sharpe": si, "raw_beta": br, "resid_beta": bi})
            print(f"  {'':>16}{lbl:>6} {sr:>+7.2f} {si:>+7.2f} {si-sr:>+6.2f} {br:>+7.3f} {bi:>+7.3f}")


def timeframe_sweep(rows):
    print(f"\n{'='*84}\n2. TIMEFRAME — raw vs residual (idio), a-priori windows scaled by bars/day, top-100\n{'='*84}")
    for kind, tfs in (("crypto", ("1d", "4h", "1h", "15m")), ("fx", ("1d", "4h", "1h"))):
        c = CFG[kind]
        print(f"\n  {kind.upper()}   {'tf':>5} {'names':>6} {'raw Sh':>7} {'res Sh':>7} {'Δ':>6} "
              f"{'raw β':>7} {'res β':>7} {'res 1st':>8} {'res 2nd':>8}")
        for tf in tfs:
            bpd = BPD[tf]
            C, A = _load(f"{kind}_{tf}")
            C = bab.winsorize_panel(C, c["winsor"])
            ppy = c["ppy_d"] * bpd
            mkt = C[c["mkt_col"]].pct_change() if c["mkt_col"] else C.pct_change().mean(axis=1)
            raw = risk_adj_mom(C, c["form_d"] * bpd, c["sk_d"] * bpd)
            idio = idio_mom(C, c["form_d"] * bpd, c["beta_d"] * bpd, c["sk_d"] * bpd, market=None)
            nr = _book(C, raw, A, kind, rebal=c["rebal_d"] * bpd, ppy=ppy)
            ni = _book(C, idio, A, kind, rebal=c["rebal_d"] * bpd, ppy=ppy)
            sr, si = _sh(nr, ppy), _sh(ni, ppy)
            f, h1, h2 = _halves(ni, ppy)
            rows.append({"sweep": "timeframe", "asset": kind, "cell": tf, "raw_sharpe": sr,
                         "resid_sharpe": si, "raw_beta": _beta(nr, mkt), "resid_beta": _beta(ni, mkt),
                         "resid_1st": h1, "resid_2nd": h2})
            print(f"  {'':>6}{tf:>5} {C.shape[1]:>6} {sr:>+7.2f} {si:>+7.2f} {si-sr:>+6.2f} "
                  f"{_beta(nr, mkt):>+7.3f} {_beta(ni, mkt):>+7.3f} {h1:>+8.2f} {h2:>+8.2f}")


def parameter_sweep(rows):
    print(f"\n{'='*84}\n3. PARAMETERS — residual (idio) net Sharpe off the a-priori (crypto 1d, equity 1d)\n{'='*84}")
    for kind in ("crypto", "equity"):
        c = CFG[kind]
        tag = "crypto_1d" if kind == "crypto" else "stocks_broad_1d"
        C, A = _load(tag)
        C = bab.winsorize_panel(C, c["winsor"])
        base = idio_mom(C, c["form_d"], c["beta_d"], c["sk_d"], market=None)
        b0 = _sh(_book(C, base, A, kind), c["ppy_d"])
        print(f"\n  {kind.upper()}  a-priori idio = {b0:+.2f}")
        # skip (the literature's skip-t-1; crypto a-priori is 0)
        skips = (0, 1, 2, 5) if kind == "crypto" else (0, 5, 21, 42)
        cells = []
        for sk in skips:
            s = _sh(_book(C, idio_mom(C, c["form_d"], c["beta_d"], sk, market=None), A, kind), c["ppy_d"])
            cells.append((f"sk{sk}", s)); rows.append({"sweep": "skip", "asset": kind, "cell": f"sk{sk}", "resid_sharpe": s})
        print("    skip:    " + "  ".join(f"{n}={v:+.2f}" for n, v in cells))
        # weighting
        cells = []
        for wt in ("equal", "rank", "volinv"):
            s = _sh(_book(C, base, A, kind, weighting=wt), c["ppy_d"])
            cells.append((wt, s)); rows.append({"sweep": "weighting", "asset": kind, "cell": wt, "resid_sharpe": s})
        print("    weight:  " + "  ".join(f"{n}={v:+.2f}" for n, v in cells))
        # rebalance
        rbs = (5, 10, 21, 42, 63)
        cells = []
        for rb in rbs:
            s = _sh(_book(C, base, A, kind, rebal=rb), c["ppy_d"])
            cells.append((f"rb{rb}", s)); rows.append({"sweep": "rebal", "asset": kind, "cell": f"rb{rb}", "resid_sharpe": s})
        print("    rebal:   " + "  ".join(f"{n}={v:+.2f}" for n, v in cells))
        # formation lookback
        forms = (10, 20, 30, 45, 90) if kind == "crypto" else (63, 126, 252, 378, 504)
        cells = []
        for fl in forms:
            s = _sh(_book(C, idio_mom(C, fl, c["beta_d"], c["sk_d"], market=None), A, kind), c["ppy_d"])
            cells.append((f"f{fl}", s)); rows.append({"sweep": "formation", "asset": kind, "cell": f"f{fl}", "resid_sharpe": s})
        print("    formation:" + "  ".join(f"{n}={v:+.2f}" for n, v in cells))


def market_factor_sweep(rows):
    print(f"\n{'='*84}\n4. MARKET FACTOR — residualise crypto on EW panel vs BTC (does the factor matter?)\n{'='*84}")
    c = CFG["crypto"]
    C, A = _load("crypto_1d")
    C = bab.winsorize_panel(C, c["winsor"])
    btc = C["BTCUSDT"].pct_change()
    ew = C.pct_change().mean(axis=1)
    print(f"  {'factor':>10} {'idio Sh':>8} {'β(BTC)':>8} {'β(EW)':>7}")
    for name, mkt in (("EW panel", None), ("BTC", btc)):
        sig = idio_mom(C, c["form_d"], c["beta_d"], c["sk_d"], market=mkt)
        net = _book(C, sig, A, "crypto")
        s, bb, be = _sh(net, c["ppy_d"]), _beta(net, btc), _beta(net, ew)
        rows.append({"sweep": "market_factor", "asset": "crypto", "cell": name, "resid_sharpe": s,
                     "resid_beta": bb})
        print(f"  {name:>10} {s:>+8.2f} {bb:>+8.3f} {be:>+7.3f}")


def main():
    rows = []
    universe_sweep(rows)
    timeframe_sweep(rows)
    parameter_sweep(rows)
    market_factor_sweep(rows)
    pd.DataFrame(rows).to_csv(RESIDMOM_DIR / "residmom_robust.csv", index=False)
    print("\nRUN RESIDMOM ROBUST OK  -> reports/residmom_robust.csv")


if __name__ == "__main__":
    main()
