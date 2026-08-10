"""Where does the cross-sectional crypto breakout edge cut off by LIQUIDITY? The expanded ~800-name
panel (mostly micro-cap alts) scores ~0.3 vs the liquid core-50's ~1.0 — so more names hurt. This
ranks every cached perp by median daily dollar-volume and sweeps top-N tiers (50/100/150/200/300),
so the edge-vs-liquidity-breadth curve is explicit rather than an all-or-core-50 dichotomy.

Liquidity is ranked once on 1d quote-volume (a coin property) and the same top-N symbol lists are
applied at each timeframe. Tokenized stock/commodity perps are excluded (pure-crypto answer).

    python scripts/breakout/run_bo_xs_liq.py
"""

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_xs_big import NONCRYPTO, build_panel, symbols_with_tf  # noqa: E402
from scripts.breakout.run_bo_xs_tf import BPD, PPY, adv_panel, funding_panel, xs_daily  # noqa: E402
from src.config import OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import breakout_signal  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

TIERS = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]   # inverted-U: peak at ~top-20/30


def liquidity_rank():
    """All crypto perps ranked by median 1d dollar-volume (quote_volume), most liquid first."""
    liq = {}
    for s in symbols_with_tf("1d"):
        if s[:-4] in NONCRYPTO:
            continue
        px = bo.load_crypto(s, "1d")
        if px is None or "quote_volume" not in px:
            continue
        dv = px["quote_volume"].median()
        if np.isfinite(dv) and dv > 0:
            liq[s] = dv
    return [s for s, _ in sorted(liq.items(), key=lambda kv: kv[1], reverse=True)]


def score(pnl, tf):
    if pnl.shape[1] < 8:
        return None
    adv, fund = adv_panel(pnl.columns, tf), funding_panel(pnl.columns, pnl.index)
    net = xs_daily(pnl, breakout_signal(pnl, "nearness", 126 * BPD[tf]), PPY[tf],
                   rebal=BPD[tf], adv=adv, funding=fund)
    s = summarise(net, 365)
    mc = bootstrap_sharpe(net, 365, 500, bo.SEED) if s["sharpe_ann"] > 0.3 else {}
    oos = net[net.index >= OOS_START]
    return {"n": pnl.shape[1], "sharpe": s["sharpe_ann"], "oos": summarise(oos, 365)["sharpe_ann"],
            "mc_p5": mc.get("sharpe_p5", np.nan), "max_dd": s["max_dd"]}


def main():
    ranked = liquidity_rank()
    print("=== XS breakout by liquidity tier (top-N most-liquid perps), across timeframes ===")
    print(f"ranked {len(ranked)} crypto perps by median 1d $-volume; most liquid: "
          f"{', '.join(s[:-4] for s in ranked[:8])}\n")
    rows = []
    for tf in ["1d", "4h", "1h"]:
        avail = set(symbols_with_tf(tf))
        print(f"--- {tf} ---", flush=True)
        for n in TIERS:
            syms = [s for s in ranked if s in avail][:n]
            pnl = build_panel(syms, tf)
            r = score(pnl, tf)
            if r is None:
                print(f"    top-{n:3d}: panel too small"); continue
            rows.append({"tf": tf, "tier": n, **r})
            print(f"    top-{n:3d} ({r['n']:3d} usable): Sharpe {r['sharpe']:+.2f}  OOS {r['oos']:+.2f}  "
                  f"MC-P5 {r['mc_p5']:+.2f}  DD {r['max_dd']:+.1%}", flush=True)
        print()
    pd.DataFrame(rows).to_csv(bo.BREAKOUT / "bo_xs_liq.csv", index=False)
    print("BO XS-LIQ OK")


if __name__ == "__main__":
    main()
