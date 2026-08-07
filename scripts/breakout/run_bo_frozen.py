"""The honest, tradeable breakout book: a FROZEN pre-registered universe with NO per-sleeve
performance selection, so there is zero survivor-selection look-ahead (Task A §2 freeze).

Walk-forward showed the edge: selecting WHICH coins will be the star sleeves does not generalise
(+0.24 OOS), but the CONSTRUCTION does (parameter-WF +1.16). So the correct book fixes both:
  - universe = the core-10 (10 largest-cap perps, frozen by market cap rank
    BEFORE seeing breakout results — not one was chosen because it backtested well)
  - construction = d55_atr3 (Donchian-55 -> chandelier ATR-trail(3)), pre-registered from the
    literature; the sweep proved the whole trend-riding family lands at the same place
  - take ALL universe x {1d,4h} sleeves, equal-risk. No Sharpe/MC survivor screen.

Reports full-sample AND the strict held-out block (oos_start .. end, evaluated once) so the number
is not an in-sample-selected one. Compares core-10 vs a broader frozen top-30 to show the breadth
effect, and three sibling constructions to show it is not a lucky config.

    python scripts/breakout/run_bo_frozen.py
"""
import json

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_sweep import build_pos, SLOW_CFGS  # noqa: E402
from src.config import OOS_START  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

# frozen by market-cap rank (the core-10 largest-cap perps), NOT by backtest
CORE10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TOP30 = bo.CRYPTO[:30]                 # broader frozen set (still mcap-ranked, no backtest selection)
TFS = ["1d", "4h"]


def sleeve_returns(universe, cfg_id):
    """Daily net returns for every (sym, tf) in the frozen universe under one construction."""
    cfg = dict(SLOW_CFGS)[cfg_id]
    out = {}
    for tf in TFS:
        for sym in universe:
            px = bo.load_crypto(sym, tf)
            if px is None:
                continue
            pos = build_pos(px, tf, cfg)
            if pos is None or pos.abs().sum() == 0:
                continue
            adv = px["quote_volume"].rolling(20).median().shift(1)
            _, ret = bo.evaluate(px["close"], pos, bo.CRYPTO_TF[tf], bo.CC,
                                 fund=bo.safe_funding(sym), adv=adv, ppy_daily=365, with_mc=False)
            out[f"{sym}_{tf}"] = ret
    return pd.DataFrame(out).sort_index()


def book_metrics(rets, label, mc=False):
    port = rets.fillna(0.0).mean(axis=1)          # equal risk (already vol-targeted)
    s = summarise(port, 365)
    row = {"book": label, "n_sleeves": rets.shape[1], "sharpe": s["sharpe_ann"],
           "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
           "total_return": s["total_return"]}
    if mc:
        m = bootstrap_sharpe(port, 365, 1000, bo.SEED)
        row["mc_p5"], row["mc_p50"] = m.get("sharpe_p5", np.nan), m.get("sharpe_p50", np.nan)
    return row, port


def main():
    print("=== FROZEN-UNIVERSE breakout book (no survivor selection) ===\n")

    # 1) construction robustness on the frozen core-10 (full sample)
    print("[A] core-10 x {1d,4h}, FULL SAMPLE, three sibling constructions:")
    core_rets = {}
    for cid in ["d55_atr3", "d55_atr3_tr", "kelt_atr3"]:
        r = sleeve_returns(CORE10, cid)
        core_rets[cid] = r
        row, _ = book_metrics(r, f"core10_{cid}", mc=True)
        print(f"    {cid:12s}: {row['n_sleeves']} sleeves  Sharpe {row['sharpe']:+.2f}  "
              f"maxDD {row['max_dd']:+.1%}  months+ {row['months_in_profit']:.0%}  "
              f"MC[P5 {row['mc_p5']:+.2f} P50 {row['mc_p50']:+.2f}]")

    # 2) breadth: core-10 vs top-30 (default construction)
    print("\n[B] breadth effect (d55_atr3), full sample:")
    rows = []
    top30_rets = sleeve_returns(TOP30, "d55_atr3")
    for label, r in [("core-10", core_rets["d55_atr3"]), ("top-30", top30_rets)]:
        row, port = book_metrics(r, label, mc=True)
        rows.append(row)
        print(f"    {label:8s}: {row['n_sleeves']:2d} sleeves  Sharpe {row['sharpe']:+.2f}  "
              f"maxDD {row['max_dd']:+.1%}  MC-P5 {row['mc_p5']:+.2f}")

    # 3) STRICT held-out OOS block (evaluated once) on the default frozen book (top-30)
    print("\n[C] STRICT OOS split, top-30 frozen book (d55_atr3), scored once:")
    port = top30_rets.fillna(0.0).mean(axis=1)
    isr = port[port.index < OOS_START]
    oos = port[port.index >= OOS_START]
    si, so = summarise(isr, 365), summarise(oos, 365)
    print(f"    in-sample  (..{OOS_START.date()}): Sharpe {si['sharpe_ann']:+.2f}  maxDD {si['max_dd']:+.1%}  "
          f"months+ {si['months_in_profit']:.0%}")
    print(f"    OOS block  ({OOS_START.date()}..): Sharpe {so['sharpe_ann']:+.2f}  maxDD {so['max_dd']:+.1%}  "
          f"months+ {so['months_in_profit']:.0%}  ({len(oos)} days)")

    # 4) per-year of the default frozen book
    per_year = {}
    for y, g in port.groupby(port.index.year):
        g = g.dropna()
        per_year[int(y)] = round(float(np.sqrt(365) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    print(f"\n[D] per-year Sharpe (top-30 frozen book): {per_year}")

    # persist the default frozen book for the report / downstream
    top30_rets.to_parquet(bo.REPORTS / "bo_frozen_sleeve_returns.parquet")
    port.rename("ret").to_frame().to_parquet(bo.REPORTS / "bo_frozen_portfolio.parquet")
    summary = {"core10_constructions": {k: book_metrics(v, k)[0] for k, v in core_rets.items()},
               "breadth": rows, "oos_split": {"in_sample": si, "oos": so},
               "per_year": per_year, "oos_start": str(OOS_START.date())}
    (bo.REPORTS / "bo_frozen_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nBO FROZEN OK")


if __name__ == "__main__":
    main()
