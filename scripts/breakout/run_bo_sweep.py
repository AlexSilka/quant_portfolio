"""Full breakout edge map: every instrument x timeframe x construction, one harness, net of costs.

Constructions (entry x exit x filter), chosen from the exit experiment + the literature:
  base_d55_tb : Donchian-55 -> triple-barrier            (the current book baseline)
  d55_rev     : Donchian-55 -> held-to-reversal          (pure trend system, cheap; shows fast-TF death)
  d55_atr3    : Donchian-55 -> chandelier ATR-trail(3)    (best exit from the exit experiment)
  d55_atr3_tr : + long-trend(200) alignment filter        (best-evidenced false-breakout filter)
  d55_atr3_vol: + volume-expansion filter                 (equities/crypto; FX has no volume)
  d55_chan20  : Donchian-55 -> opposite-20 channel exit   (Turtle System 2)
  d20_chan10  : Donchian-20 -> opposite-10 channel exit   (Turtle System 1)
  boll_atr3   : Bollinger(20,2) -> ATR-trail(3)           (alt entry)
  kelt_atr3   : Keltner(20,2)  -> ATR-trail(3)            (alt entry)

Fast TFs (5m/15m) run only the two cheap configs (triple-barrier events + vectorised reversal) —
the trend-riding exits loop over every bar and the point at 5m/15m is only to confirm cost death.
No Monte Carlo here (that is applied to the assembled book); a shuffle-sign placebo runs per config
so the pipeline's own false-discovery rate on this construction is measurable.

    python scripts/breakout/run_bo_sweep.py
"""

import pandas as pd

from src import bo_common as bo  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402

SLOW_CFGS = [
    ("base_d55_tb", dict(entry="donchian", lookback=55, exit="triple_barrier")),
    ("d55_rev", dict(entry="donchian", lookback=55, exit="reversal")),
    ("d55_atr3", dict(entry="donchian", lookback=55, exit="atr_trailing", k=3.0)),
    ("d55_atr3_tr", dict(entry="donchian", lookback=55, exit="atr_trailing", k=3.0, filt="trend")),
    ("d55_atr3_vol", dict(entry="donchian", lookback=55, exit="atr_trailing", k=3.0, filt="vol")),
    ("d55_chan20", dict(entry="donchian", lookback=55, exit="channel", exit_lookback=20)),
    ("d20_chan10", dict(entry="donchian", lookback=20, exit="channel", exit_lookback=10)),
    ("boll_atr3", dict(entry="bollinger", lookback=20, k=2.0, exit="atr_trailing")),
    ("kelt_atr3", dict(entry="keltner", lookback=20, k=2.0, exit="atr_trailing")),
]
FAST_CFGS = [SLOW_CFGS[0], SLOW_CFGS[1]]   # 5m/15m: baseline + cheap reversal only


def build_pos(px, tf, cfg):
    """Entry -> optional filter -> exit -> held +1/-1/0 position for one config."""
    side = bo.entry_side(px, cfg["entry"], cfg["lookback"], k=cfg.get("k", 2.0))
    filt = cfg.get("filt")
    if filt == "trend":
        side = bl.apply_filters(side, align=bl.trend_filter(px["close"], 100))
    elif filt == "vol":
        if "volume" in px and px["volume"].abs().sum() > 0:
            side = bl.apply_filters(side, bl.volume_filter(px["volume"], 20, 0.5))
        else:
            return None   # no usable volume (FX) — skip this config for this instrument
    return bo.held_position(cfg["exit"], px, side, tf, k=cfg.get("k_exit", 3.0),
                            exit_lookback=cfg.get("exit_lookback", 20))


def run_instrument(rows, kind, sym, tf, px, fund, adv, cfgs):
    tfmap, costs = bo.cfg_for(kind, sym)
    ppy_bar, ppy_d = tfmap[tf], (365 if kind == "crypto" else 252)
    for cid, cfg in cfgs:
        pos = build_pos(px, tf, cfg)
        if pos is None or pos.abs().sum() == 0:
            continue
        s, _ = bo.evaluate(px["close"], pos, ppy_bar, costs, fund=fund, adv=adv,
                           ppy_daily=ppy_d, with_mc=False)
        plac = pos.abs() * pd.Series(bo.rng.choice([-1.0, 1.0], len(pos)), index=pos.index)
        sp, _ = bo.evaluate(px["close"], plac, ppy_bar, costs, fund=fund, adv=adv,
                            ppy_daily=ppy_d, with_mc=False)
        rows.append({"kind": kind, "sym": sym, "tf": tf, "config": cid,
                     "sharpe": s["sharpe_ann"], "max_dd": s["max_dd"],
                     "months_in_profit": s["months_in_profit"], "ann_turnover": s["ann_turnover"],
                     "n_obs": s["n_obs"], "placebo_sharpe": sp["sharpe_ann"]})


def main():
    rows = []
    for tf in ["1d", "4h", "1h", "15m", "5m"]:
        cfgs = SLOW_CFGS if tf in ("1d", "4h", "1h") else FAST_CFGS
        for sym in bo.CRYPTO:
            px = bo.load_crypto(sym, tf)
            if px is None:
                continue
            fund = bo.safe_funding(sym)
            adv = px["quote_volume"].rolling(20).median().shift(1)
            run_instrument(rows, "crypto", sym, tf, px, fund, adv, cfgs)
        for kind, syms in [("equity", bo.STOCKS), ("fx", bo.FX)]:
            for sym in syms:
                px = bo.load_eqfx(sym, tf)
                if px is None:
                    continue
                adv = ((px["close"] * px["volume"]).rolling(20).median().shift(1)
                       if "volume" in px and px["volume"].abs().sum() > 0 else None)
                run_instrument(rows, kind, sym, tf, px, None, adv, cfgs)
        pd.DataFrame(rows).to_csv(bo.REPORTS / "bo_sweep.csv", index=False)  # partial, per-TF
        print(f"  [{tf}] cumulative rows: {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(bo.REPORTS / "bo_sweep.csv", index=False)
    print("\n=== EDGE MAP: mean Sharpe by config x timeframe (all instruments) ===")
    print(df.pivot_table(index="config", columns="tf", values="sharpe", aggfunc="mean")
          .reindex(columns=["1d", "4h", "1h", "15m", "5m"]).round(3).to_string())
    print("\n=== mean Sharpe by config x asset class (1d+4h+1h only) ===")
    slow = df[df.tf.isin(["1d", "4h", "1h"])]
    print(slow.pivot_table(index="config", columns="kind", values="sharpe", aggfunc="mean").round(3).to_string())
    print("\n=== survivors (Sharpe>0.5) per config, on 1d/4h/1h ===")
    surv = slow[slow.sharpe > 0.5].groupby("config").size().sort_values(ascending=False)
    tot = slow.groupby("config").size()
    for c in surv.index:
        print(f"  {c:14s} {surv[c]:3d}/{tot[c]:3d}  mean_plac {slow[slow.config==c].placebo_sharpe.mean():+.2f}")
    print("\nBO SWEEP OK")


if __name__ == "__main__":
    main()
