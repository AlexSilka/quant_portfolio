"""Assemble the breakout book: one principled construction across the universe, screen each sleeve
(Sharpe>0.5 & Monte-Carlo 5th-pct>0 & shuffle-placebo control), combine survivors equal-risk.

Headline construction `d55_atr3_tr` is pre-registered from the literature, NOT picked from the
sweep results: Donchian-55 entry (canonical breakout) -> chandelier ATR-trail(3) exit (captures the
fat right tail the triple-barrier discards) -> long-trend(100) alignment filter (best-evidenced
false-breakout filter). The sweep shows the whole trend-riding family (kelt_atr3, d55_atr3, chan)
performs similarly, so this is a robustness property, not a lucky config.

Universe = crypto 1d/4h/1h + US equities + FX (1d/4h/1h) — the same discover-everywhere principle
as run_book; equities/FX are expected to die (breakout is a crypto trend-premium sleeve), and that
non-result is the edge map. Outputs mirror run_book so the two are directly comparable.

    python scripts/breakout/run_bo_book.py [config_id]
"""
import json
import sys

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_sweep import build_pos, SLOW_CFGS  # noqa: E402
from src.config import CACHE_DIR  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CFG_ID = sys.argv[1] if len(sys.argv) > 1 else "d55_atr3_tr"
CFG = dict(SLOW_CFGS)[CFG_ID]
TFS = ["1d", "4h", "1h"]
CACHE = CACHE_DIR / "book_bo"


def eval_sleeve(kind, sym, tf, px, fund, adv):
    tfmap, costs = bo.cfg_for(kind, sym)
    ppy_bar, ppy_d = tfmap[tf], (365 if kind == "crypto" else 252)
    pos = build_pos(px, tf, CFG)
    if pos is None or pos.abs().sum() == 0:
        return None, None
    s, ret = bo.evaluate(px["close"], pos, ppy_bar, costs, fund=fund, adv=adv, ppy_daily=ppy_d)
    plac = pos.abs() * pd.Series(bo.rng.choice([-1.0, 1.0], len(pos)), index=pos.index)
    sp, _ = bo.evaluate(px["close"], plac, ppy_bar, costs, fund=fund, adv=adv,
                        ppy_daily=ppy_d, with_mc=False)
    # cost drag / turnover for the report come from the daily frame; recompute the pieces
    robust = bool(s["sharpe_ann"] > 0.5 and s.get("mc_p5", -9) > 0.0 and s["n_obs"] > 100)
    row = {"sleeve": f"{sym}_{tf}_breakout", "kind": kind, "tf": tf, "sym": sym,
           "sharpe": s["sharpe_ann"], "mc_p5": s.get("mc_p5", np.nan),
           "mc_p50": s.get("mc_p50", np.nan), "max_dd": s["max_dd"],
           "months_in_profit": s["months_in_profit"], "ann_turnover": s["ann_turnover"],
           "placebo_sharpe": sp["sharpe_ann"], "n_obs": s["n_obs"], "robust": robust}
    return row, ret


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, all_ret = [], {}
    for tf in TFS:
        for sym in bo.CRYPTO:
            px = bo.load_crypto(sym, tf)
            if px is None:
                continue
            adv = px["quote_volume"].rolling(20).median().shift(1)
            row, ret = eval_sleeve("crypto", sym, tf, px, bo.safe_funding(sym), adv)
            if row:
                rows.append(row); all_ret[row["sleeve"]] = ret
                if row["robust"]:
                    ret.to_frame().to_parquet(CACHE / f"{row['sleeve']}.parquet")
        for kind, syms in [("equity", bo.STOCKS), ("fx", bo.FX)]:
            for sym in syms:
                px = bo.load_eqfx(sym, tf)
                if px is None:
                    continue
                adv = ((px["close"] * px["volume"]).rolling(20).median().shift(1)
                       if "volume" in px and px["volume"].abs().sum() > 0 else None)
                row, ret = eval_sleeve(kind, sym, tf, px, None, adv)
                if row:
                    rows.append(row); all_ret[row["sleeve"]] = ret
                    if row["robust"]:
                        ret.to_frame().to_parquet(CACHE / f"{row['sleeve']}.parquet")
        print(f"  [{tf}] tested {len(rows)} sleeves, {sum(r['robust'] for r in rows)} robust so far",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(bo.BREAKOUT / f"bo_book_results_{CFG_ID}.csv", index=False)
    pd.DataFrame(all_ret).sort_index().to_parquet(bo.BREAKOUT / f"bo_all_returns_{CFG_ID}.parquet")
    surv = df[df.robust].sort_values("sharpe", ascending=False)
    print(f"\n=== {CFG_ID}: {len(surv)}/{len(df)} sleeves robust "
          f"(Sharpe>0.5 & MC-P5>0); placebo-robust {int((df.placebo_sharpe>0.5).sum())}/{len(df)} ===")
    print(surv.groupby(["kind", "tf"]).size().to_string())
    _portfolio(surv, df, all_ret)


def _portfolio(surv, df, all_ret):
    dfs = {s: pd.read_parquet(CACHE / f"{s}.parquet")["ret"] for s in surv.sleeve
           if (CACHE / f"{s}.parquet").exists()}
    if not dfs:
        print("no survivors -> no portfolio"); return
    rets = pd.DataFrame(dfs).sort_index()
    port = rets.fillna(0.0).mean(axis=1)          # equal risk (already vol-targeted)
    s = summarise(port, 365)
    mc = bootstrap_sharpe(port, 365, 1000, bo.SEED)
    per_year = {}
    for y, g in port.groupby(port.index.year):
        g = g.dropna()
        per_year[int(y)] = round(float(np.sqrt(365) * g.mean() / g.std(ddof=1)), 2) if g.std(ddof=1) > 0 else 0.0
    n_trials = int(len(df))
    var_tr = float((df["sharpe"].clip(-3, 3).dropna() / np.sqrt(365)).var())
    best = surv.iloc[0]; b = dfs[best.sleeve].dropna()
    best_dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0,
                               n_trials, max(var_tr, 1e-8))

    out = {"config": CFG_ID, "portfolio": s, "mc": mc, "per_year": per_year,
           "n_trials": n_trials, "n_survivors": len(dfs),
           "best_sleeve": best.sleeve, "best_sleeve_dsr": best_dsr,
           "placebo_fdr": float((df.placebo_sharpe > 0.5).mean()),
           "survivors": list(dfs), "mean_turnover": float(surv.ann_turnover.mean())}
    (bo.BREAKOUT / f"bo_book_summary_{CFG_ID}.json").write_text(json.dumps(out, indent=2, default=float))
    rets.to_parquet(bo.BREAKOUT / f"bo_book_sleeve_returns_{CFG_ID}.parquet")
    port.rename("ret").to_frame().to_parquet(bo.BREAKOUT / f"bo_book_portfolio_{CFG_ID}.parquet")
    print(f"\n=== BREAKOUT BOOK ({CFG_ID}, equal-risk over {len(dfs)} survivors, net of costs) ===")
    print(f"Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]")
    print(f"per-year Sharpe: {per_year}")
    print(f"best single sleeve ({best.sleeve}) deflated Sharpe (N={n_trials}): {best_dsr:.2f}")
    print(f"mean annual turnover across sleeves: {surv.ann_turnover.mean():.1f}x  |  placebo FDR "
          f"{out['placebo_fdr']:.0%}")
    print("BO BOOK OK")


if __name__ == "__main__":
    main()
