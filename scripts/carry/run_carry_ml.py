"""ML on cross-sectional carry — two distinct uses, each measured against the non-ML baseline
(the linear funding-rank book), each leakage-controlled with purged time-series CV (expanding
train, embargoed gap, predict the next block — no training on the future).

  A) ML RANKER  — replace the linear funding rank with a model that predicts each name's forward
     return from a richer feature set (funding level/change/z + price/vol). Long/short by prediction.
     Variants: feature sets x models x horizon. Question: does ML improve the carry SIGNAL?

  B) ML TIMING OVERLAY — a model predicts P(the carry book is up next week) from market-wide state
     (mean funding, funding dispersion, BTC vol/trend, the book's own recent P&L) and scales
     exposure. Question: does an ML regime gate cut drawdown (the "confidence factor" story)?

    python scripts/carry/run_carry_ml.py
"""
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from scripts.carry.run_carry import START, END, pit_symbols  # noqa: E402
# Resolved LAZILY, inside the function that needs it. Binding it at module scope made every
# importer pay a 578-symbol funding load — including network probes for unpublished months —
# before its own first line ran, which is how `run_ml_book_contribution` came to spend minutes
# doing nothing it asked for. Import-time work is work every caller pays whether it wants it.

PPY, TVOL, SEED, CB = 365, VOL_TARGET_ANNUAL, SEED, 6.0
rng = np.random.default_rng(SEED)


def vt(net):
    scale = vol_target_scale(net, TVOL, PPY)
    return (net * scale).dropna()


def metr(net, mc=True):
    n = vt(net)
    s = summarise(n, PPY)
    p5 = bootstrap_sharpe(n, PPY, 500, SEED).get("sharpe_p5", np.nan) if (mc and s["sharpe_ann"] > 0.2) else np.nan
    return {"sharpe": round(s["sharpe_ann"], 2), "mc_p5": round(p5, 2) if p5 == p5 else np.nan,
            "max_dd": round(s["max_dd"], 2), "months_in_profit": round(s["months_in_profit"], 2)}


def load_panel():
    close, vol_, fund = {}, {}, {}
    for s in pit_symbols():
        px = load_klines(s, "1d", START, END, market="um")
        if len(px):
            close[s] = px["close"]
            vol_[s] = px["quote_volume"]
        f = load_funding(s, START, END)["last_funding_rate"]
        if len(f):
            fund[s] = f
    C = pd.DataFrame(close).sort_index()
    V = pd.DataFrame(vol_).reindex(C.index)
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(C.index)
    return C, V, fd


# ---------- per-name feature panel (each computable-at-bar) ----------

def name_features(C, V, fd):
    ret = C.pct_change()
    feats = {
        "fund_ema3": fd.ewm(span=3).mean(), "fund_ema7": fd.ewm(span=7).mean(),
        "fund_ema14": fd.ewm(span=14).mean(), "fund_chg": fd.ewm(span=3).mean().diff(3),
        "fund_z": (fd - fd.rolling(30).mean()) / (fd.rolling(30).std() + 1e-9),
        "fund_xs_rank": fd.ewm(span=7).mean().rank(axis=1, pct=True),
        "ret3": ret.rolling(3).sum(), "ret7": ret.rolling(7).sum(), "ret14": ret.rolling(14).sum(),
        "ret30": ret.rolling(30).sum(), "vol14": ret.rolling(14).std(), "vol30": ret.rolling(30).std(),
        "dvol_rank": V.rolling(20).median().rank(axis=1, pct=True),
    }
    return feats, ret


FEATURE_SETS = {
    "fund_only": ["fund_ema3", "fund_ema7", "fund_ema14", "fund_chg", "fund_z", "fund_xs_rank"],
    "fund_price": ["fund_ema3", "fund_ema7", "fund_ema14", "fund_chg", "fund_z", "fund_xs_rank",
                   "ret3", "ret7", "ret14", "ret30"],
    "full": ["fund_ema3", "fund_ema7", "fund_ema14", "fund_chg", "fund_z", "fund_xs_rank",
             "ret3", "ret7", "ret14", "ret30", "vol14", "vol30", "dvol_rank"],
}


def stack(feats, cols, fwd):
    """Long panel: rows = (date, name), X = features at t, y = forward return over (t, t+h]."""
    parts = {c: feats[c].stack() for c in cols}
    X = pd.DataFrame(parts)
    y = fwd.stack().reindex(X.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    m = X.notna().all(axis=1) & y.notna()
    return X[m], y[m]


def purged_oos(X, y, model_fn, n_blocks=6, embargo_days=10, classify=False):
    """Expanding-window purged prediction: for each of n_blocks test spans, train on all rows whose
    date is at least `embargo_days` before the test span begins, predict the span. No future leak."""
    dates = X.index.get_level_values(0)
    ud = np.array(sorted(dates.unique()))
    blocks = np.array_split(ud, n_blocks + 1)          # block 0 = initial train warmup only
    oos = pd.Series(np.nan, index=X.index)
    emb = pd.Timedelta(days=embargo_days)
    for k in range(1, n_blocks + 1):
        tspan = blocks[k]
        t0 = pd.Timestamp(tspan.min())
        tr = dates < (t0 - emb)
        te = np.isin(dates, tspan)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        model = model_fn()
        yt = (y[tr] > 0).astype(int) if classify else y[tr]
        model.fit(X[tr], yt)
        pred = (model.predict_proba(X[te])[:, 1] if classify else model.predict(X[te]))
        oos[te] = pred
    return oos


def book_from_scores(scores_long, C, fd, top_frac=0.2):
    """Dollar-neutral book: long the top-predicted names, short the bottom, per date."""
    S = scores_long.unstack()          # date x name
    ranks = S.rank(axis=1, pct=True)
    hi = (ranks >= 1 - top_frac).astype(float); lo = (ranks <= top_frac).astype(float)
    wl = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    ws = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).reindex(C.index).fillna(0.0).shift(2).fillna(0.0)
    ret = C.pct_change()
    price = (w * ret).sum(axis=1)
    funding = -(w * fd.fillna(0.0)).sum(axis=1)
    cost = w.diff().abs().sum(axis=1) * CB / 1e4
    return (price + funding - cost).rename("ret")


def main():
    C, V, fd = load_panel()
    feats, ret = name_features(C, V, fd)
    rows = []

    # ---- non-ML baseline: linear funding-rank book (level-7, top-20) ----
    base = carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, 7), direction=-1.0, top_frac=0.2, cost_bps=CB)
    base_net = base["ret"]
    rows.append({"model": "LINEAR funding-rank (baseline)", "featset": "-", "horizon": "-", **metr(base_net)})
    print(f"baseline linear carry: {rows[-1]}")

    # ---- A) ML RANKER: feature sets x models x horizon ----
    models = {
        "ridge": lambda: Ridge(alpha=5.0),
        "lgbm": lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                          subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                          n_jobs=-1, verbose=-1),
    }
    print("\n=== A) ML RANKER (purged OOS, dollar-neutral top-20) ===")
    for h in [5, 10]:
        fwd = C.pct_change(h).shift(-h)                 # forward h-day return (target)
        for fs_name, cols in FEATURE_SETS.items():
            X, y = stack(feats, cols, fwd)
            for mname, mfn in models.items():
                if mname == "ridge" and fs_name != "full":
                    continue                            # ridge only on the full set (linear reference)
                oos = purged_oos(X, y, mfn, n_blocks=6, embargo_days=h + 3).dropna()
                net = book_from_scores(oos, C, fd, top_frac=0.2)
                m = metr(net)
                rows.append({"model": f"ML-{mname}", "featset": fs_name, "horizon": h, **m})
                print(f"  h{h} {mname:5s} {fs_name:10s} Sharpe {m['sharpe']:+.2f}  P5 {m['mc_p5']}  DD {m['max_dd']}")

    # ---- B) ML TIMING OVERLAY on the linear baseline book ----
    print("\n=== B) ML TIMING OVERLAY (predict up-week, scale exposure) ===")
    btc = C["BTCUSDT"]
    mkt = pd.DataFrame({
        "mean_fund": fd.mean(axis=1), "disp_fund": fd.std(axis=1),
        "btc_vol": btc.pct_change().rolling(14).std(), "btc_mom": btc.pct_change(30),
        "book_mom": base_net.rolling(7).mean(), "book_vol": base_net.rolling(14).std(),
    })
    horizon = 5
    ytime = base_net.rolling(horizon).sum().shift(-horizon)     # forward book return
    Xt = mkt.replace([np.inf, -np.inf], np.nan)
    mm = Xt.notna().all(axis=1) & ytime.notna()
    Xt, yt = Xt[mm], ytime[mm]
    Xt.index = pd.MultiIndex.from_arrays([Xt.index, ["_"] * len(Xt)])   # reuse purged_oos (date-level)
    yt.index = Xt.index
    proba = purged_oos(Xt, yt, lambda: LogisticRegression(max_iter=500, C=1.0),
                       n_blocks=6, embargo_days=horizon + 3, classify=True)
    proba.index = proba.index.get_level_values(0)
    proba = proba.reindex(base_net.index).shift(1)              # decision uses yesterday's model output
    for thr_name, scaler in [("gate>0.5", (proba > 0.5).astype(float)),
                             ("soft", (2 * proba.clip(0, 1)).clip(0, 1.5).fillna(1.0)),
                             ("raw(no overlay)", pd.Series(1.0, index=base_net.index))]:
        net = (base_net * scaler.reindex(base_net.index).fillna(0.0)).dropna()
        m = metr(net)
        rows.append({"model": f"TIMING {thr_name}", "featset": "market", "horizon": horizon, **m})
        print(f"  {thr_name:16s} Sharpe {m['sharpe']:+.2f}  P5 {m['mc_p5']}  DD {m['max_dd']}  mip {m['months_in_profit']}")

    df = pd.DataFrame(rows)
    df.to_csv(CARRY_DIR / "carry_ml.csv", index=False)
    print("\n=== SUMMARY (vs linear baseline Sharpe %.2f, DD %.2f) ===" % (rows[0]["sharpe"], rows[0]["max_dd"]))
    print(df.to_string(index=False))
    print("\nCARRY-ML OK")


if __name__ == "__main__":
    main()
