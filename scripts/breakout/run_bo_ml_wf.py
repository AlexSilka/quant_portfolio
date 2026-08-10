"""The breakout ML gate, re-measured under a walk-forward that a live desk could have run.

Two defects in the shipped gate are corrected here and each is priced separately, so the honest
incremental value of meta-labelling is separable from the value of having seen the future.

  1. **The CV trains on the future.** `run_bo_ml.py` gates every trade with `purged_kfold`, whose
     folds are contiguous in time but whose *training set is the whole complement* — so a trade in
     2021 is filtered by a model fit on 2022-2026. Purging removes label-overlap leakage, not
     future information; it estimates generalisation, it does not simulate a track record. The
     replacement is a purged, embargoed, expanding-window walk-forward: block k is predicted only
     by trades whose label had already resolved before block k opened.
  2. **The labels are gross and mis-timed.** The label is `close(t1)/close(t0)`, priced at the
     signal bars, while the book fills at t+2 and pays commission, spread and funding. A +3bps
     trade is labelled a win and loses money. The corrected label prices the trade the way the
     book actually experiences it: execution-lagged prices, round-trip cost, funding over the hold.

Both venues are run, because the answer differs: on spot the funding term leaves the label, and the
question "does the gate still add value once it can no longer see the future" is asked separately
for each. Reported with return and volatility beside the ratio.

    python scripts/breakout/run_bo_ml_wf.py
"""
from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import bo_common as bo  # noqa: E402
from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.config import BREAKOUT_DIR, CACHE_DIR, CRYPTO_PPY, OOS_START  # noqa: E402
from src.features.engine import compute_features, pit_normalize  # noqa: E402
from src.sleeves import breakout_lab as bl  # noqa: E402
from src.validation.purged_cv import purged_kfold  # noqa: E402
from scripts.breakout.run_bo_spot import (CORE10, VENUE, _borrow, load, stats)  # noqa: E402

TFS = ["4h", "1h"]              # 1d has ~30 Donchian-55 trades/sleeve — too few to meta-label
THR = 0.55
EMB = {"1d": pd.Timedelta(days=10), "4h": pd.Timedelta(days=5), "1h": pd.Timedelta(days=2)}
WINDOW = {"perp": ("2020-01-01", "2026-07-31"), "spot": ("2018-01-01", "2026-07-31")}
N_BLOCKS = 8                    # per-sleeve walk-forward test blocks; the first is train-only
MIN_TRAIN = 60                  # trades required before the gate is allowed to speak
RETRAIN = "QS"                  # pooled walk-forward retrains quarterly on everything resolved


def model():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=bo.SEED, n_jobs=-1, verbose=-1)


# --- features (venue-tagged cache: spot and perp prices give different features) --------

def features(venue: str, sym: str, tf: str, lo: str, hi: str) -> pd.DataFrame | None:
    cache = CACHE_DIR / "book_bo" / f"feat_{venue}_{sym}_{tf}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    px = load(venue, sym, tf, lo, hi)
    btc = load(venue, "BTCUSDT", tf, lo, hi)
    if px is None or btc is None:
        return None
    feats = pit_normalize(compute_features(px, benchmark=btc["close"], fast=True))
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


# --- labels ------------------------------------------------------------------------------

def trade_labels(venue: str, px: pd.DataFrame, trades: pd.DataFrame, tf: str,
                 fund: pd.Series | None) -> tuple[pd.Series, pd.Series]:
    """(gross-at-signal label, net-at-execution label) for each chandelier trade.

    The first is what the shipped gate trains on. The second prices the same trade the way the book
    fills it: entry and exit both moved forward by the engine's execution lag, a round-trip
    commission+spread charge, and the funding actually accrued over the hold (a long pays positive
    funding; a spot short pays coin-borrow instead).
    """
    close = px["close"]
    lag = VENUE[venue]["costs"]["exec_lag"]
    exec_px = close.shift(-lag)                     # price the fill actually gets, stamped at t0/t1
    r_gross = trades["side"] * (close.reindex(trades["t1"]).to_numpy()
                                / close.reindex(trades.index).to_numpy() - 1.0)
    r_exec = trades["side"] * (exec_px.reindex(trades["t1"]).to_numpy()
                               / exec_px.reindex(trades.index).to_numpy() - 1.0)
    c = VENUE[venue]["costs"]
    roundtrip = 2.0 * (c["commission_bps"] + c["half_spread_bps"]) / 1e4
    carry = np.zeros(len(trades))
    if fund is not None and len(fund):
        cum = fund.reindex(close.index.union(fund.index)).fillna(0.0).cumsum()
        a = cum.reindex(trades.index, method="ffill").to_numpy()
        b = cum.reindex(trades["t1"], method="ffill").to_numpy()
        carry = -trades["side"].to_numpy() * (b - a)          # long pays positive funding
    elif VENUE[venue]["borrow_bps"]:
        held = (pd.DatetimeIndex(trades["t1"]) - pd.DatetimeIndex(trades.index)).days / 365.0
        carry = np.where(trades["side"] < 0, -held * VENUE[venue]["borrow_bps"] / 1e4, 0.0)
    r_net = pd.Series(r_exec.to_numpy() + carry - roundtrip, index=trades.index)
    return (r_gross > 0).astype(int), (r_net > 0).astype(int)


# --- one sleeve ----------------------------------------------------------------------------

def sleeve_data(venue: str, sym: str, tf: str):
    lo, hi = WINDOW[venue]
    px = load(venue, sym, tf, lo, hi)
    if px is None:
        return None
    close, high, low = px["close"], px["high"], px["low"]
    trades = bl.chandelier_trades(close, high, low, bl.donchian_side(close, high, low, 55), 3.0, 14)
    if len(trades) < 60:
        return None
    feats = features(venue, sym, tf, lo, hi)
    if feats is None:
        return None
    X = feats.reindex(trades.index).dropna()
    if len(X) < 60:
        return None
    trades = trades.reindex(X.index)
    fund = bo.safe_funding(sym) if VENUE[venue]["funding"] else None
    y_gross, y_net = trade_labels(venue, px, trades, tf, fund)
    y_gross, y_net = y_gross.reindex(X.index), y_net.reindex(X.index)
    if y_gross.nunique() < 2 or y_net.nunique() < 2:
        return None
    return dict(venue=venue, sym=sym, tf=tf, px=px, trades=trades, X=X,
                y_gross=y_gross, y_net=y_net, fund=fund,
                adv=px["quote_volume"].rolling(20).median().shift(1))


# --- the two gates ----------------------------------------------------------------------------

def proba_kfold(s: dict, y: pd.Series) -> pd.Series:
    """The shipped estimator: purged k-fold — every fold trains on the past AND the future."""
    X = s["X"]
    t0 = pd.DatetimeIndex(X.index)
    t1 = pd.DatetimeIndex(s["trades"]["t1"].reindex(X.index).to_numpy())
    out = pd.Series(np.nan, index=X.index)
    for tr, te in purged_kfold(t0, t1, n_splits=5, embargo=EMB[s["tf"]]):
        m = model().fit(X.iloc[tr], y.iloc[tr])
        out.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out.dropna()


def proba_walkforward(s: dict, y: pd.Series) -> pd.Series:
    """Expanding-window walk-forward: a block is scored only by labels that had already resolved.

    Training is restricted to trades whose exit (t1) precedes the test block's first entry by the
    embargo — no future price, and no label still open when the block starts.
    """
    X = s["X"]
    t0 = pd.DatetimeIndex(X.index)
    t1 = pd.DatetimeIndex(s["trades"]["t1"].reindex(X.index).to_numpy())
    order = np.argsort(t0.values)
    out = pd.Series(np.nan, index=X.index)
    for block in np.array_split(order, N_BLOCKS)[1:]:
        te = np.sort(block)
        cutoff = t0[te].min() - EMB[s["tf"]]
        tr = np.flatnonzero(t1.values < np.datetime64(cutoff))
        if len(tr) < MIN_TRAIN or y.iloc[tr].nunique() < 2:
            continue
        m = model().fit(X.iloc[tr], y.iloc[tr])
        out.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out.dropna()


# --- book ---------------------------------------------------------------------------------------

def proba_walkforward_pooled(sleeves: list[dict], y_key: str) -> dict[str, pd.Series]:
    """One model per timeframe, retrained quarterly on every sleeve's resolved trades.

    A per-sleeve model sees ~200 trades over the whole sample, which is too few for an 80-feature
    gate to learn anything but noise. Pooling the ten symbols is both the data-efficient choice and
    the realistic one — a desk runs one breakout model, not ten. Still strictly causal: each quarter
    trains only on trades whose label had resolved before the quarter opened.
    """
    out = {}
    for tf in sorted({s["tf"] for s in sleeves}):
        grp = [s for s in sleeves if s["tf"] == tf]
        X = pd.concat([s["X"] for s in grp], keys=[s["sym"] for s in grp], names=["sym", "t0"])
        y = pd.concat([s[y_key] for s in grp], keys=[s["sym"] for s in grp], names=["sym", "t0"])
        t0 = pd.DatetimeIndex(X.index.get_level_values("t0"))
        t1 = pd.DatetimeIndex(np.concatenate(
            [s["trades"]["t1"].reindex(s["X"].index).to_numpy() for s in grp]))
        pred = pd.Series(np.nan, index=range(len(X)))
        for qs in pd.date_range(t0.min().ceil("D"), t0.max(), freq=RETRAIN, tz="UTC"):
            te = np.flatnonzero((t0 >= qs) & (t0 < qs + pd.tseries.frequencies.to_offset(RETRAIN)))
            tr = np.flatnonzero(t1 < qs - EMB[tf])
            if not len(te) or len(tr) < MIN_TRAIN or y.iloc[tr].nunique() < 2:
                continue
            m = model().fit(X.iloc[tr], y.iloc[tr])
            pred.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
        pred.index = X.index
        for s in grp:
            p = pred.xs(s["sym"], level="sym").dropna()
            out[f"{s['sym']}_{tf}"] = p
    return out


def daily(s: dict, keep: pd.Index, size: pd.Series | None = None) -> pd.Series:
    pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], keep)
    if size is not None:                       # continuous bet size instead of a binary gate
        scale = pd.Series(0.0, index=s["px"].index)
        for t0 in keep:
            scale.loc[t0:s["trades"]["t1"].get(t0, t0)] = size.get(t0, 0.0)
        pos = pos * scale
    posv = vol_target(pos, s["px"]["close"], bo.TVOL, bo.CRYPTO_TF[s["tf"]])
    v = VENUE[s["venue"]]
    bt = backtest(s["px"]["close"], posv, capital=bo.CAP, funding=s["fund"], adv=s["adv"], **v["costs"])
    net = bt["net_ret"] - _borrow(bt["position"], v["borrow_bps"], bo.CRYPTO_TF[s["tf"]])
    return ((1 + net).resample("D").prod() - 1).rename(f"{s['sym']}_{s['tf']}")


def book(series: dict[str, pd.Series], starts: dict[str, pd.Timestamp] | None = None) -> pd.Series:
    """Equal-weight over the sleeves live that day; `starts` blanks a sleeve before its model exists."""
    cols = {}
    for k, v in series.items():
        s = v.copy()
        if starts is not None:
            if k not in starts:
                continue
            s = s.loc[starts[k]:]
        cols[k] = s
    return pd.DataFrame(cols).sort_index().mean(axis=1) if cols else pd.Series(dtype=float)


def vol_matched(leg: pd.Series, ref_vol: float) -> pd.Series:
    """Rescale a stream to a reference volatility — the only way to read return off two gates that
    hold wildly different amounts of risk. A gate that rejects 80% of trades wins on ratio while
    making less money; matching vol puts both on the same capital."""
    v = leg.std(ddof=1) * np.sqrt(CRYPTO_PPY)
    return leg * (ref_vol / v) if v > 0 else leg


def main():
    print("=== BREAKOUT ML GATE — purged k-fold (shipped) vs purged WALK-FORWARD (honest) ===")
    print(f"primary Donchian-55 + chandelier(3); threshold {THR}; LightGBM; TFs {'+'.join(TFS)}\n")

    out, all_rows = {}, []
    for venue in ("perp", "spot"):
        sleeves = [d for tf in TFS for sym in CORE10
                   if (d := sleeve_data(venue, sym, tf)) is not None]
        if not sleeves:
            continue
        flip = np.mean([(s["y_gross"] != s["y_net"]).mean() for s in sleeves])
        wr_g = np.mean([s["y_gross"].mean() for s in sleeves])
        wr_n = np.mean([s["y_net"].mean() for s in sleeves])
        print(f"[{venue}] {len(sleeves)} sleeves | trade win-rate: gross-at-signal {wr_g:.1%} -> "
              f"net-at-execution {wr_n:.1%}  ({flip:.1%} of labels flip once costs are priced)")

        probas = {"kfold/gross": {}, "kfold/net": {}, "walkfwd/net": {}}
        ung = {}
        for s in sleeves:
            key = f"{s['sym']}_{s['tf']}"
            ung[key] = daily(s, s["X"].index)
            probas["kfold/gross"][key] = proba_kfold(s, s["y_gross"])
            probas["kfold/net"][key] = proba_kfold(s, s["y_net"])
            probas["walkfwd/net"][key] = proba_walkforward(s, s["y_net"])
        probas["walkfwd-pooled/net"] = proba_walkforward_pooled(sleeves, "y_net")

        by_key = {f"{s['sym']}_{s['tf']}": s for s in sleeves}

        # Each gate is compared against the SAME ungated book restricted to exactly the sleeves and
        # dates that gate covers — a walk-forward is silent until it has a model, and crediting it
        # with (or blaming it for) the period it never spoke in would compare windows, not gates.
        # AFML bet sizing: instead of a binary pass/reject, size the trade by how far the model's
        # confidence sits from a coin flip. A hard threshold throws away 80%+ of the primary's
        # trades and with them most of the exposure; a continuous size keeps the weak ones small.
        probas["walkfwd/net (bet-size)"] = probas["walkfwd/net"]
        probas["walkfwd-pooled/net (bet-size)"] = probas["walkfwd-pooled/net"]

        rows, oos = [], []
        for tag, d in probas.items():
            starts = {k: p.index.min() for k, p in d.items() if len(p)}
            if "bet-size" in tag:
                gated = {}
                for k, p in d.items():
                    if not len(p):
                        continue
                    z = (p - 0.5) / np.sqrt((p * (1 - p)).clip(lower=1e-9))
                    m = pd.Series(np.clip(2 * st.norm.cdf(z) - 1, 0.0, None), index=p.index)
                    gated[k] = daily(by_key[k], p.index, size=m)
            else:
                gated = {k: daily(by_key[k], p.index[p.values >= THR])
                         for k, p in d.items() if len(p)}
            base = book(ung, starts)
            g = book(gated, starts)
            kept = (1.0 if "bet-size" in tag else
                    np.mean([float((p.values >= THR).sum()) / len(by_key[k]["X"])
                             for k, p in d.items() if len(p)]))
            ref = base.std(ddof=1) * np.sqrt(CRYPTO_PPY)
            for lab, ser, extra in ((f"{venue} ungated (matched to {tag})", base, {}),
                                    (f"{venue} gate {tag}", g,
                                     {"kept": kept, "cagr_volmatched":
                                      stats(vol_matched(g, ref))["cagr"]})):
                rows.append({**stats(ser, lab), **extra, "start": str(ser.dropna().index[0].date())})
                o = ser[ser.index >= OOS_START]
                oos.append({**stats(o, lab + " | OOS"), **extra})

        print(f"{'variant':<40}{'Sharpe':>8}{'CAGR':>9}{'=vol':>9}{'vol':>8}{'maxDD':>9}{'kept':>7}")
        for r in rows + oos:
            vm = f"{r['cagr_volmatched']:+.1%}" if "cagr_volmatched" in r else "  —"
            k = f"{r['kept']:.0%}" if "kept" in r else "100%"
            print(f"{r['label']:<40}{r['sharpe']:+8.2f}{r['cagr']:+9.1%}{vm:>9}{r['vol']:8.1%}"
                  f"{r['max_dd']:+9.1%}{k:>7}")
        print("   ('=vol' = CAGR after rescaling the gated book to the ungated book's volatility; "
              "'kept' = share of primary trades passing the gate)\n")
        all_rows += rows + oos
        out[venue] = {"n_sleeves": len(sleeves), "label_flip_frac": float(flip),
                      "winrate_gross": float(wr_g), "winrate_net": float(wr_n)}

    (BREAKOUT_DIR / "bo_ml_wf.json").write_text(
        json.dumps({"meta": out, "rows": all_rows}, indent=2, default=float))
    print("BO ML WF OK")


if __name__ == "__main__":
    main()
