"""Meta-label 'confidence factor' overlay — measures the ML layer's incremental value over the
non-ML trend baseline (Task A §5). A LightGBM model predicts P(this trend segment wins) from the
feature library; the sleeve trades a segment only when confidence clears a threshold. Applied to
fast timeframes (15m/1h) where there are enough segments to train on.

    python scripts/run_meta_overlay.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.config import BOOK_DIR, CACHE_DIR, CAPITAL_USD, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.features.engine import compute_features, pit_normalize  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.pipeline import model_factory, signal_events  # noqa: E402
from src.sleeves import momentum  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.validation.purged_cv import cv_oos_predictions  # noqa: E402

CAP, TVOL, THRESHOLD = CAPITAL_USD, VOL_TARGET_ANNUAL, 0.55
CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
SLEEVES = [("BTCUSDT", "15m", 96 * 365), ("ETHUSDT", "15m", 96 * 365),
           ("SOLUSDT", "15m", 96 * 365), ("BTCUSDT", "1h", 24 * 365),
           ("ETHUSDT", "1h", 24 * 365), ("SOLUSDT", "1h", 24 * 365)]


def features_fast(sym, tf, btc):
    cache = CACHE_DIR / f"features_{sym}_{tf}_fast.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px = load_klines(sym, tf, "2020-01", market="um")
    feats = pit_normalize(compute_features(px, benchmark=btc, fast=True))
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


def segments(close, side):
    ev = list(signal_events(side))
    rows = []
    for k, t0 in enumerate(ev):
        t1 = ev[k + 1] if k + 1 < len(ev) else close.index[-1]
        seg = float(side.loc[t0]) * (close.loc[t1] / close.loc[t0] - 1.0)
        rows.append((t0, t1, int(seg > 0)))
    return pd.DataFrame(rows, columns=["t0", "t1", "win"]).set_index("t0")


def gated_side(side, labels, keep):
    g = pd.Series(0.0, index=side.index)
    for t0 in keep:
        g.loc[t0:labels.loc[t0, "t1"]] = side.loc[t0]
    return g


def main():
    btc = {tf: load_klines("BTCUSDT", tf, "2020-01", market="um")["close"]
           for tf in ("15m", "1h")}
    base_rets, gate_rets, table = {}, {}, []
    for sym, tf, ppy in SLEEVES:
        px = load_klines(sym, tf, "2020-01", market="um")
        close = px["close"]
        fund = load_funding(sym, "2020-01")["last_funding_rate"]
        feats = features_fast(sym, tf, btc[tf])
        side = momentum.primary_side(close, 50, 200)

        lab = segments(close, side)
        X = feats.reindex(lab.index).dropna()
        y = lab["win"].reindex(X.index)
        emb = pd.Timedelta(days=2)
        oos, _ = cv_oos_predictions(X, y, lab["t1"], model_factory, n_splits=5, embargo=emb)
        oos = oos.dropna()
        keep = oos.index[oos.values >= THRESHOLD]
        prec_base, prec_gate = float(y.reindex(oos.index).mean()), float(y.reindex(keep).mean())

        adv = px["quote_volume"].rolling(20).median().shift(1)  # liquidity-aware impact, lagged
        base_pos = vol_target(side, close, TVOL, ppy)
        gate_pos = vol_target(gated_side(side, lab, keep), close, TVOL, ppy)
        bt_b = backtest(close, base_pos, capital=CAP, funding=fund, adv=adv, **CC)
        bt_g = backtest(close, gate_pos, capital=CAP, funding=fund, adv=adv, **CC)
        db = ((1 + bt_b["net_ret"]).resample("D").prod() - 1).dropna()
        dg = ((1 + bt_g["net_ret"]).resample("D").prod() - 1).dropna()
        sb, sg = summarise(db, 365), summarise(dg, 365)
        base_rets[f"{sym}_{tf}"], gate_rets[f"{sym}_{tf}"] = db, dg
        table.append((f"{sym}_{tf}", len(y), prec_base, prec_gate,
                      sb["sharpe_ann"], sg["sharpe_ann"], sb["max_dd"], sg["max_dd"]))
        print(f"{sym}_{tf:3s}  segments {len(y):4d}  precision {prec_base:.0%}->{prec_gate:.0%}  "
              f"Sharpe {sb['sharpe_ann']:+.2f}->{sg['sharpe_ann']:+.2f}  "
              f"DD {sb['max_dd']:+.0%}->{sg['max_dd']:+.0%}")

    pb = pd.DataFrame(base_rets).mean(axis=1)
    pg = pd.DataFrame(gate_rets).mean(axis=1)
    sb, sg = summarise(pb, 365), summarise(pg, 365)
    mcg = bootstrap_sharpe(pg, 365, 1000, SEED)
    print("\n=== fast-TF sub-portfolio: ML incremental value ===")
    print(f"baseline (non-ML rules): Sharpe {sb['sharpe_ann']:+.2f}  DD {sb['max_dd']:+.1%}  months+ {sb['months_in_profit']:.0%}")
    print(f"meta-gated (confidence): Sharpe {sg['sharpe_ann']:+.2f}  DD {sg['max_dd']:+.1%}  months+ {sg['months_in_profit']:.0%}  "
          f"MC-P5 {mcg.get('sharpe_p5', float('nan')):+.2f}")
    print(f"ML incremental value: {sg['sharpe_ann'] - sb['sharpe_ann']:+.2f} Sharpe")
    pd.DataFrame(table, columns=["sleeve", "segments", "prec_base", "prec_gate",
                                 "sharpe_base", "sharpe_gate", "dd_base", "dd_gate"]
                 ).to_csv(BOOK_DIR / "meta_overlay.csv", index=False)
    print("META OVERLAY OK")


if __name__ == "__main__":
    main()
