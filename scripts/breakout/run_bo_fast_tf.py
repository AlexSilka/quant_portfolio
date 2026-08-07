"""Did the deep-intraday timeframes get a fair shot? The sweep tested 15m/5m only with the
baseline + reversal exits (both catastrophically negative: base_d55_tb 15m -3.31, 5m -8.83).
Here 15m and 5m get the FULL treatment the slow TFs got — the trend-riding chandelier exit AND
the LightGBM meta-label gate — to answer definitively whether either rescues them (the gate lifted
1h from +0.05 to +0.37, so this is a fair test, not a foregone conclusion).

Core-10 only, processed coin-by-coin so the 700k-bar 5m feature matrices never co-reside in memory.

    python scripts/breakout/run_bo_fast_tf.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_ml import CORE10, OOS_START, models, oos_proba, sleeve_data, uniqueness_weights  # noqa: E402
from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.metrics import summarise  # noqa: E402

THR = 0.55


def daily(px, side, t1, events, tf, fund, adv):
    pos = positions_from_events(px.index, side, t1, events)
    pos = vol_target(pos, px["close"], bo.TVOL, bo.CRYPTO_TF[tf])
    bt = backtest(px["close"], pos, capital=bo.CAP, funding=fund, adv=adv, **bo.CC)
    return (1 + bt["net_ret"]).resample("D").prod() - 1


def book_split(rets):
    port = rets.fillna(0.0).mean(axis=1)
    is_, oos = port[port.index < OOS_START], port[port.index >= OOS_START]
    return (summarise(port, 365)["sharpe_ann"], summarise(port, 365)["max_dd"],
            summarise(is_, 365)["sharpe_ann"], summarise(oos, 365)["sharpe_ann"])


def main():
    fac = models()["lightgbm"]
    print("=== Deep-intraday breakout: full treatment (chandelier exit + ML gate), core-10 ===\n")
    for tf in ["15m", "5m"]:
        ung, gat, precs = {}, {}, []
        for sym in CORE10:
            d = sleeve_data(sym, tf)                       # features computed+cached, then subset
            if d is None:
                print(f"    {sym} {tf}: too few trades"); continue
            px, trades, X, y = d
            adv, fund = px["quote_volume"].rolling(20).median().shift(1), bo.safe_funding(sym)
            w = uniqueness_weights(pd.DatetimeIndex(X.index), trades["t1"], px.index)
            p = oos_proba(X, y, trades["t1"], fac, tf, w)
            kept = p.index[p.values >= THR]
            precs.append((float(y.reindex(p.index).mean()),
                          float(y.reindex(kept).mean()) if len(kept) else np.nan))
            ung[f"{sym}_{tf}"] = daily(px, trades["side"], trades["t1"], X.index, tf, fund, adv)
            gat[f"{sym}_{tf}"] = daily(px, trades["side"], trades["t1"], kept, tf, fund, adv)
            del X, px, trades       # free the big frames before the next coin
            print(f"    {sym} {tf}: {len(p)} trades, kept {len(kept)}", flush=True)
        su = book_split(pd.DataFrame(ung))
        sg = book_split(pd.DataFrame(gat))
        pb, pg = np.nanmean([a for a, _ in precs]), np.nanmean([b for _, b in precs])
        print(f"\n  {tf} chandelier UNGATED : Sharpe {su[0]:+.2f}  maxDD {su[1]:+.1%}  (IS {su[2]:+.2f}/OOS {su[3]:+.2f})")
        print(f"  {tf} chandelier ML-GATED: Sharpe {sg[0]:+.2f}  maxDD {sg[1]:+.1%}  (IS {sg[2]:+.2f}/OOS {sg[3]:+.2f})  "
              f"precision {pb:.0%}->{pg:.0%}\n")
    print("BO FAST-TF OK")


if __name__ == "__main__":
    main()
