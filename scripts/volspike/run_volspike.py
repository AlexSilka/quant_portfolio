"""Volume-spike sleeve — honest evaluation on 15m altcoin perps (Task A §10).

Ports the author's live Rust bot (spike_bot / VolumeSpike) into the framework and tests it the SAME
way every other sleeve was tested, so the number is comparable to the book:
  entry  = src/sleeves/volume_spike.primary_side  (vol-spike + non-falling price; simplest form)
  exit   = triple barrier (vol-scaled TP/SL/vertical) OR fixed time-stop                [bo_common/bl]
  exec   = t+2 bars (never the signal bar's own close)                                  [engine]
  costs  = liquidity-aware crypto perp taker (5 + 1 bps + sqrt-impact) + funding @ 8h   [bo_common.CC]
  size   = vol-target 15% annualised, equal-risk book across the alt universe

Honesty stance vs the live bot: the bot per-coin-optimises and the operator hand-launches a few
"pump" coins (FHE, PIPPIN, ...). That is survivorship and cannot be walk-forward validated. Here ONE
global parameter set is applied uniformly to a FROZEN alt universe and results are pooled — this
answers "does the volume-spike edge exist systematically, net of cost", the portfolio question.

Stages (argv):  smoke | wf | robust | all
  .venv/bin/python scripts/volspike/run_volspike.py all
"""
import sys

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from src.config import RAW_DIR  # noqa: E402
from src.sleeves import volume_spike as vs  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.backtest.engine import positions_from_events  # noqa: E402
from src.labels.triple_barrier import triple_barrier_labels, trailing_vol  # noqa: E402
from src.metrics import summarise, deflated_sharpe  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

TF = "15m"
PPY_BAR = bo.CRYPTO_TF[TF]        # 96*365 — the 15m annualisation factor
HORIZON = bo.HORIZON[TF]          # 32-bar vertical default
PPY = 365


def _liq_alts(n=30, min_months=36):
    """Frozen universe: the n most liquid USD-M alt perps (ex BTC/ETH) with >= min_months history.
    Ranked by full-sample median $-volume/bar so the pick does not depend on a single recent month."""
    import glob
    import os
    base = str(RAW_DIR / "futures/um/klines")
    rows = []
    for p in glob.glob(base + "/*/15m"):
        s = os.path.basename(os.path.dirname(p))
        if s in ("BTCUSDT", "ETHUSDT"):
            continue
        fs = sorted(glob.glob(p + "/*.parquet"))
        if len(fs) < min_months:
            continue
        qv = pd.concat([pd.read_parquet(f, columns=["quote_volume"]) for f in fs[::6]])
        rows.append((s, float(qv["quote_volume"].median())))
    df = pd.DataFrame(rows, columns=["sym", "adv"]).sort_values("adv", ascending=False)
    return df.sym.head(n).tolist()


ALTS = _liq_alts(30)

# --- a-priori grid (10 constructions). Kept small to bound the deflated-Sharpe trial penalty. Spans
#     the bot's toy default, its thesis region (rare strong spikes) and the drift-capturing long holds.
GRID = [
    dict(k_vol=2, vol_win=10, pcp=0.5, exit="tb", pt=1.5, sl=1.5, horizon=32),    # bot toy default
    dict(k_vol=3, vol_win=20, pcp=0.5, exit="tb", pt=3.0, sl=2.0, horizon=48),
    dict(k_vol=4, vol_win=48, pcp=0.5, exit="tb", pt=4.0, sl=3.0, horizon=96),
    dict(k_vol=6, vol_win=96, pcp=0.5, exit="tb", pt=4.0, sl=3.0, horizon=96),
    dict(k_vol=3, vol_win=48, pcp=0.5, exit="time", horizon=64),
    dict(k_vol=4, vol_win=48, pcp=0.5, exit="time", horizon=96),
    dict(k_vol=6, vol_win=96, pcp=0.5, exit="time", horizon=96),
    dict(k_vol=6, vol_win=96, pcp=0.5, exit="time", horizon=192),               # best in-sample so far
    dict(k_vol=8, vol_win=96, pcp=1.0, exit="time", horizon=192),
    dict(k_vol=4, vol_win=48, pcp=0.0, exit="time", horizon=96),
]

_CACHE = {}


def _load(sym):
    if sym not in _CACHE:
        px = bo.load_crypto(sym, TF)
        _CACHE[sym] = None if px is None else (px, bo.safe_funding(sym),
                                               px["quote_volume"].rolling(20).median().shift(1))
    return _CACHE[sym]


def _pos(cfg, px, placebo_rng=None):
    c = px["close"]
    side = vs.primary_side(c, px["quote_volume"], k_vol=cfg["k_vol"],
                           vol_win=cfg["vol_win"], min_price_chg_pct=cfg["pcp"])
    events = bl.entry_events(side)
    if len(events) == 0:
        return None
    if placebo_rng is not None:
        valid = c.index[100:-cfg["horizon"] - 2]
        if len(valid) < len(events):
            return None
        events = valid[np.sort(placebo_rng.choice(len(valid), size=len(events), replace=False))]
        side = pd.Series(0.0, index=c.index)
        side.loc[events] = 1.0
    if cfg["exit"] == "time":
        return bl.hold_time_stop(side, cfg["horizon"])
    lab = triple_barrier_labels(c, events, trailing_vol(c, 100), cfg["pt"], cfg["sl"], cfg["horizon"])
    return positions_from_events(c.index, side, lab["t1"], events)


def sym_daily(sym, cfg, placebo_seed=None, cost_mult=1.0):
    got = _load(sym)
    if got is None:
        return None
    px, fund, adv = got
    rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
    pos = _pos(cfg, px, placebo_rng=rng)
    if pos is None or pos.abs().sum() == 0:
        return None
    costs = dict(bo.CC)
    if cost_mult != 1.0:
        for kk in ("commission_bps", "half_spread_bps", "impact_k"):
            costs[kk] = bo.CC[kk] * cost_mult
    s, ret = bo.evaluate(px["close"], pos, PPY_BAR, costs, fund=fund, adv=adv, ppy_daily=PPY, with_mc=False)
    return ret.rename(sym), s


def book_daily(cfg, syms=None, placebo_seed=None, cost_mult=1.0):
    syms = syms or ALTS
    cols, stats = [], []
    for i, sym in enumerate(syms):
        out = sym_daily(sym, cfg, placebo_seed=(placebo_seed + i if placebo_seed is not None else None),
                        cost_mult=cost_mult)
        if out is None:
            continue
        cols.append(out[0])
        stats.append(out[1])
    if not cols:
        return pd.Series(dtype=float), []
    return pd.concat(cols, axis=1).mean(axis=1).dropna(), stats


# =================================================================================================

def smoke(n_syms=None):
    syms = ALTS if n_syms is None else ALTS[:n_syms]
    cfg = GRID[0]
    print(f"[SMOKE] bot toy-default cfg {cfg}\n        universe: {len(syms)} liquid alt perps, {TF}, "
          f"t+2 exec, liquidity-aware costs, vol-target 15%")
    book, stats = book_daily(cfg, syms)
    if not len(book):
        print("  no trades"); return
    s = summarise(book, PPY)
    print(f"  symbols traded {len(stats)}  span {book.index.min().date()}..{book.index.max().date()}")
    print(f"  POOLED  Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}"
          f"  total {s['total_return']:+.0%}  ann.turnover {np.mean([x['ann_turnover'] for x in stats]):,.0f}x")


def _precompute(placebo_seed=None):
    """Per-config book daily returns over the whole timeline (the expensive step, done once)."""
    books = {}
    for i, cfg in enumerate(GRID):
        bk, _ = book_daily(cfg, placebo_seed=(placebo_seed * 100 + i * 7 if placebo_seed is not None else None))
        if len(bk):
            books[i] = bk
    return books


def _wf(books, dates, window_years):
    cfgs = list(books)
    oos, picks = [], []
    for i in range(len(dates) - 1):
        T, Tn = dates[i], dates[i + 1]
        lo = (T - pd.DateOffset(years=window_years)) if window_years else None

        def train_sh(c):
            w = books[c].loc[:T]
            w = w[w.index < T]
            if lo is not None:
                w = w[w.index >= lo]
            return summarise(w, PPY)["sharpe_ann"] if len(w) > 60 else -9.0
        best = max(cfgs, key=train_sh)
        seg = books[best].loc[T:Tn]
        seg = seg[seg.index < Tn]
        oos.append(seg); picks.append(best)
    return pd.concat(oos).sort_index().dropna(), picks


def wf():
    print("\n[WF] parameter walk-forward — best-of-grid on train, applied OOS on the next block")
    print(f"     universe {len(ALTS)} liquid alt perps; grid {len(GRID)} configs; net of liquidity-aware costs")
    books = _precompute()
    tz = next(iter(books.values())).index.tz
    # full-sample in-sample peak (overfit reference) + per-config spread for deflated Sharpe
    full = {i: summarise(books[i], PPY)["sharpe_ann"] for i in books}
    best_is = max(full, key=full.get)
    print(f"     in-sample full-sample Sharpe by config: min {min(full.values()):+.2f}  "
          f"max {max(full.values()):+.2f} (cfg {best_is})  frac>0 {np.mean([v > 0 for v in full.values()]):.0%}")

    schemes = [("anchored", "annual", None, "YS"), ("anchored", "semiann", None, "6MS"),
               ("rolling2y", "annual", 2, "YS"), ("rolling2y", "semiann", 2, "6MS")]
    shs = []
    prim = None
    for wlab, clab, wy, freq in schemes:
        dates = pd.date_range("2021-07-01", "2026-07-01", freq=freq, tz=tz)
        oos, picks = _wf(books, dates, wy)
        s = summarise(oos, PPY)
        shs.append(s["sharpe_ann"])
        if prim is None:
            prim = oos
        print(f"     {wlab:9s} {clab:8s}: OOS Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  "
              f"months+ {s['months_in_profit']:.0%}  total {s['total_return']:+.0%}")
    mc = bootstrap_sharpe(prim, PPY, 1000, bo.SEED)
    print(f"     OOS Sharpe across schemes: {min(shs):+.2f}..{max(shs):+.2f}   "
          f"primary MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]")

    # deflated Sharpe at the true trial count (grid size)
    r = books[best_is]
    sr_bar = r.mean() / r.std(ddof=1)
    var_tr = np.var([v / np.sqrt(PPY) for v in full.values()], ddof=1)  # per-bar SR variance across trials
    dsr = deflated_sharpe(sr_bar, len(r), r.skew(), r.kurt() + 3.0, len(GRID), max(var_tr, 1e-8))
    print(f"     deflated Sharpe of best config (N={len(GRID)} trials): P(SR>E[maxSR|null]) = {dsr:.2f}")

    # placebo: same pipeline, entries moved to random bars -> false-discovery reference
    pl_books = _precompute(placebo_seed=1)
    pl_full = [summarise(pl_books[i], PPY)["sharpe_ann"] for i in pl_books]
    print(f"     PLACEBO (random-entry) full-sample Sharpe: min {min(pl_full):+.2f} max {max(pl_full):+.2f} "
          f"median {np.median(pl_full):+.2f}  -> real max {max(full.values()):+.2f} vs placebo max {max(pl_full):+.2f}")
    pd.concat([prim.rename("volspike_wf_oos")], axis=1).to_parquet(bo.REPORTS / "volspike_wf_oos.parquet")
    return prim, best_is, books


def robust(best_cfg_idx=7, books=None):
    print("\n[ROBUST] cost sensitivity, break-even, per-year, correlation to the book")
    cfg = GRID[best_cfg_idx]
    print(f"     primary cfg = {cfg}")
    # cost sensitivity: base / 2x / 3x
    for m in (1.0, 2.0, 3.0):
        bk, _ = book_daily(cfg, cost_mult=m)
        s = summarise(bk, PPY)
        print(f"     cost x{m:.0f}: Sharpe {s['sharpe_ann']:+.2f}  total {s['total_return']:+.0%}")
    # break-even cost multiple (bisect total-return -> 0); if base already <=0, report <1
    lo, hi = 0.0, 3.0
    base = summarise(book_daily(cfg)[0], PPY)["total_return"]
    if base <= 0:
        print(f"     break-even cost: base total {base:+.0%} <= 0 -> already below break-even at 1x cost")
    else:
        for _ in range(12):
            mid = (lo + hi) / 2
            if summarise(book_daily(cfg, cost_mult=mid)[0], PPY)["total_return"] > 0:
                lo = mid
            else:
                hi = mid
        print(f"     break-even cost multiple ~ {lo:.2f}x base")
    # per-year Sharpe
    bk, _ = book_daily(cfg)
    yr = bk.groupby(bk.index.year).apply(lambda x: summarise(x, PPY)["sharpe_ann"])
    print("     per-year Sharpe: " + "  ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
    # correlation to existing book sleeves
    try:
        book = pd.read_parquet(bo.REPORTS / "master_book_legs.parquet")
        j = pd.concat([bk.rename("volspike"), book], axis=1).dropna()
        cor = j.corr()["volspike"].drop("volspike")
        print("     corr to book sleeves: " + "  ".join(f"{k.split('·')[-1] if '·' in k else k}:{v:+.2f}"
              for k, v in cor.items()))
    except Exception as e:
        print(f"     corr step skipped: {e}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if stage == "smoke":
        smoke(n)
    elif stage == "wf":
        wf()
    elif stage == "robust":
        robust()
    else:
        smoke()
        prim, best, books = wf()
        robust(best, books)
    print("\nVOLSPIKE OK")


if __name__ == "__main__":
    main()
