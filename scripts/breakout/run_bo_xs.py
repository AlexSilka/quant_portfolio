"""Cross-sectional breakout — the honest second look at equities/FX, where single-name breakout
died. Instead of betting each name's own breakout, rank the panel each bar and go long the most
broken-out / short the least, dollar-neutral (market-neutral). The evidenced signal is George &
Hwang 52-week-high nearness (long names nearest their high, short farthest), which dominates
trailing-return momentum and does not reverse.

Tested on the stock panel (where time-series breakout was dead), plus crypto and FX panels for
completeness. Real vs shuffled-signal placebo, vol-targeted, net of costs, with the 2024-07 OOS split.

CAVEAT (stated, per Task A §2): the stock panel is today's large caps — survivorship-biased (no
delisted names), so the equity cross-sectional number is optimistic; the crypto panel likewise
omits dead coins. Reported as a diagnostic, not a clean tradeable Sharpe.

    python scripts/breakout/run_bo_xs.py
"""

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from src.config import OOS_START, VOL_TARGET_ANNUAL  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import breakout_signal  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

TVOL = VOL_TARGET_ANNUAL
SIGNALS = [("nearness", 252), ("nearness", 126), ("donchian", 55), ("donchian", 120)]


def vt(net, ppy):
    scale = vol_target_scale(net, TVOL, ppy)
    return (net * scale).dropna()


def panel(kind):
    if kind == "crypto":
        cols = {s: bo.load_crypto(s, "1d")["close"] for s in bo.CRYPTO if bo.load_crypto(s, "1d") is not None}
    else:
        syms = bo.STOCKS if kind == "equity" else bo.FX
        cols = {}
        for s in syms:
            px = bo.load_eqfx(s, "1d")
            if px is not None:
                cols[s] = px["close"]
    raw = pd.DataFrame(cols).sort_index()
    # ffill only small gaps: a name idle >5 bars drops out of the cross-section (handles
    # delistings/halts without propagating a stale price forever), then require >=300 obs.
    px = raw.ffill(limit=5)
    px = px.loc[:, px.notna().sum() >= 300]
    return px


def xs_ls(pnl, sig, top_frac, cost_bps, ppy, rebal=1):
    """Dollar-neutral long-top/short-bottom on a large, sparse panel — robust to stale-price
    artifacts (daily returns winsorised to +-50%; NaN returns contribute zero, not NaN).

    rebal>1 holds the target weights for `rebal` bars between rebalances (the evidenced 52-week-high
    strategy rebalances monthly, not daily — daily churn on a slow signal dies to turnover)."""
    rets = pnl.pct_change().clip(-0.5, 0.5)
    ranks = sig.rank(axis=1, pct=True)
    longs = (ranks >= 1.0 - top_frac).astype(float)
    shorts = (ranks <= top_frac).astype(float)
    wl = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0)
    ws = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws)
    if rebal > 1:
        hold = pd.Series(False, index=w.index)
        hold.iloc[::rebal] = True
        w = w.where(hold, np.nan).ffill()          # refresh weights only on rebalance bars
    w = w.shift(2).fillna(0.0)
    gross = (w * rets.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1)
    return vt(gross - turn * cost_bps / 1e4, ppy)


def evaluate(pnl, sig, cost_bps, ppy, rebal=1):
    net = xs_ls(pnl, sig, 0.3, cost_bps, ppy, rebal)
    if len(net) < 252:
        return None
    s = summarise(net, ppy)
    mc = bootstrap_sharpe(net, ppy, 500, bo.SEED) if s["sharpe_ann"] > 0.3 else {}
    is_, oos = net[net.index < OOS_START], net[net.index >= OOS_START]
    return {"sharpe": s["sharpe_ann"], "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
            "mc_p5": mc.get("sharpe_p5", np.nan), "sharpe_is": summarise(is_, ppy)["sharpe_ann"],
            "sharpe_oos": summarise(oos, ppy)["sharpe_ann"], "net": net}


def main():
    costs = {"equity": 3.0, "crypto": 6.0, "fx": 1.0}   # round-trip commission+half-spread, bps/side
    ppy = 365
    # daily vs monthly rebalance — the 52-week-high edge is slow (GH hold 6mo); daily churn dies to cost
    CADENCE = [("daily", 1), ("monthly", 21)]
    rows, best_net = [], {}
    for kind in ["equity", "crypto", "fx"]:
        pnl = panel(kind)
        print(f"\n=== {kind} panel: {pnl.shape[1]} names, {pnl.index.min().date()}..{pnl.index.max().date()} ===")
        for clab, rb in CADENCE:
            for sk, lb in SIGNALS:
                sig = breakout_signal(pnl, sk, lb)
                r = evaluate(pnl, sig, costs[kind], ppy, rb)
                if r is None:
                    continue
                plac = pd.DataFrame(bo.rng.standard_normal(pnl.shape), index=pnl.index, columns=pnl.columns)
                rp = evaluate(pnl, plac, costs[kind], ppy, rb)
                rows.append({"kind": kind, "cadence": clab, "signal": f"{sk}_{lb}", "sharpe": r["sharpe"],
                             "mc_p5": r["mc_p5"], "max_dd": r["max_dd"], "sharpe_is": r["sharpe_is"],
                             "sharpe_oos": r["sharpe_oos"], "placebo": rp["sharpe"] if rp else np.nan})
                best_net[f"{kind}_{clab}_{sk}_{lb}"] = r["net"]
                print(f"    {clab:7s} {sk+'_'+str(lb):13s}: Sharpe {r['sharpe']:+.2f} (IS {r['sharpe_is']:+.2f}/OOS "
                      f"{r['sharpe_oos']:+.2f})  MC-P5 {r['mc_p5']:+.2f}  DD {r['max_dd']:+.1%}  "
                      f"placebo {rp['sharpe'] if rp else float('nan'):+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(bo.BREAKOUT / "bo_xs.csv", index=False)
    if best_net:
        pd.DataFrame(best_net).to_parquet(bo.BREAKOUT / "bo_xs_returns.parquet")
    # per-year for the single best (kind,cadence,signal) overall
    best = df.sort_values("sharpe", ascending=False).iloc[0]
    net = best_net[f"{best['kind']}_{best['cadence']}_{best['signal']}"]
    py = {int(y): round(float(np.sqrt(ppy) * g.dropna().mean() / g.dropna().std(ddof=1)), 2)
          for y, g in net.groupby(net.index.year) if g.dropna().std(ddof=1) > 0}
    print(f"\nbest XS overall: {best['kind']} {best['cadence']} {best['signal']} "
          f"Sharpe {best['sharpe']:+.2f}  per-year: {py}")
    print("\nBO XS OK")


if __name__ == "__main__":
    main()
