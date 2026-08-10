"""Is the equity cross-sectional NULL breakout-specific, or a broken construction? Orthogonal check:
run cross-sectional MOMENTUM (trailing return) and cross-sectional BREAKOUT (52w-high nearness /
Donchian rank) through the SAME long-short harness on the SAME panels. If momentum works on equities
where breakout does not, the null is breakout-specific (honest, and consistent with breakout being a
coarse momentum encoding that single-stock short-term reversal punishes). If both fail, the harness is
the problem.

Equity = broad panel, monthly rebalance (slow anomaly); crypto = liquid, daily. Net of costs, vs a
shuffled-signal placebo.

    python scripts/breakout/run_bo_xs_signals.py
"""

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_xs import evaluate, panel  # noqa: E402
from src.sleeves.cross_sectional import breakout_signal, momentum_signal  # noqa: E402

SIGNALS = [("momentum", 120), ("momentum", 252), ("breakout_near", 126),
           ("breakout_near", 252), ("breakout_donch", 120)]


def make_signal(pnl, kind, lb):
    if kind == "momentum":
        return momentum_signal(pnl, lb)
    if kind == "breakout_near":
        return breakout_signal(pnl, "nearness", lb)
    return breakout_signal(pnl, "donchian", lb)


def main():
    print("=== Cross-sectional MOMENTUM vs BREAKOUT, same harness (is the equity null signal-specific?) ===\n")
    setups = [("equity", 3.0, 21), ("crypto", 6.0, 1)]   # (panel, cost bps/side, rebalance bars)
    rows = []
    for kind, cost, rb in setups:
        pnl = panel(kind)
        cad = "monthly" if rb > 1 else "daily"
        print(f"--- {kind} panel ({pnl.shape[1]} names, {cad} rebalance) ---")
        for sk, lb in SIGNALS:
            r = evaluate(pnl, make_signal(pnl, sk, lb), cost, 365, rb)
            if r is None:
                continue
            plac = pd.DataFrame(bo.rng.standard_normal(pnl.shape), index=pnl.index, columns=pnl.columns)
            rp = evaluate(pnl, plac, cost, 365, rb)
            rows.append({"kind": kind, "signal": f"{sk}_{lb}", "sharpe": r["sharpe"],
                         "oos": r["sharpe_oos"], "mc_p5": r["mc_p5"],
                         "placebo": rp["sharpe"] if rp else np.nan})
            print(f"    {sk+'_'+str(lb):18s}: Sharpe {r['sharpe']:+.2f}  OOS {r['sharpe_oos']:+.2f}  "
                  f"MC-P5 {r['mc_p5']:+.2f}  placebo {rp['sharpe'] if rp else float('nan'):+.2f}", flush=True)
        print()
    pd.DataFrame(rows).to_csv(bo.BREAKOUT / "bo_xs_signals.csv", index=False)
    print("BO XS-SIGNALS OK")


if __name__ == "__main__":
    main()
