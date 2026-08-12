"""Does the ML meta-label ('confidence gate') rescue the sector-pairs sleeve? Emit every pair trade
with entry features, label by realized win, train LightGBM with purged cross-validation, gate the
low-confidence trades out of the daily stream, and compare gated vs ungated. The base is weak after
walk-forward selection (+0.25), so this tests whether ML can lift a thin signal over the 0.5 bar.

    python scripts/pairs/run_sector_pairs_ml.py
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.data.equity import load_equity_daily  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.pipeline import model_factory  # noqa: E402
from src.sleeves.sector_pairs import SECTOR_ETFS, _cointegration, _positions_from_z  # noqa: E402
from src.validation.purged_cv import cv_oos_predictions  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402

PPY, COST = 252, 2.0


def pair_daily_and_trades(y, x, lookback, entry, mkt_vol, exit_=0.5):
    """Daily vol-targeted return AND per-trade records (entry feats + realised P&L) for one pair."""
    y, x = y.align(x, join="inner")
    ry, rx = y.pct_change(), x.pct_change()
    beta = (ry.rolling(lookback).cov(rx) / (rx.rolling(lookback).var() + 1e-12)).shift(1)
    spread = np.log(y) - beta * np.log(x)
    sd = spread.rolling(lookback).std()
    z = (spread - spread.rolling(lookback).mean()) / (sd + 1e-12)
    pair_ret = ry - beta * rx
    scale = vol_target_scale(pair_ret, 0.15, PPY)
    executed = (pd.Series(_positions_from_z(z.to_numpy(), entry, exit_), index=y.index) * scale).shift(2).fillna(0.0)
    daily = executed * pair_ret - executed.diff().abs().fillna(0.0) * 2 * COST / 1e4
    trades, inpos, start = [], False, 0
    ep = executed.to_numpy()
    for i in range(len(ep)):
        if not inpos and ep[i] != 0:
            inpos, start = True, i
            feat = {"z": float(abs(z.iloc[i]) if np.isfinite(z.iloc[i]) else 0.0),
                    "spread_sd": float(sd.iloc[i] if np.isfinite(sd.iloc[i]) else 0.0),
                    "dir": float(np.sign(ep[i])), "mkt_vol": float(mkt_vol.iloc[i])}
        elif inpos and ep[i] == 0:
            inpos = False
            trades.append((y.index[start], y.index[i], feat, float(daily.iloc[start:i].sum()), start, i))
    return daily, trades


def main():
    panel = pd.DataFrame({s: load_equity_daily(s, start="2016-01-01")["close"]
                          for s in SECTOR_ETFS}).dropna(how="all").ffill().dropna()
    mkt_vol = panel.pct_change().abs().mean(axis=1).rolling(20).mean()
    names = list(panel.columns)
    anchors = list(pd.date_range(panel.index[0] + pd.DateOffset(years=2), panel.index[-1], freq="6MS"))

    daily_total = pd.Series(0.0, index=panel.index)     # ungated basket (sum of pair dailies, /n later)
    rows, seg_map = [], []                              # trade features/labels + (series-slot) for gating
    pair_series = []                                    # per-pair daily series for the basket
    for i, t0 in enumerate(anchors):
        t1 = anchors[i + 1] if i + 1 < len(anchors) else panel.index[-1] + pd.Timedelta(days=1)
        form = panel.loc[t0 - pd.DateOffset(years=2):t0]
        for a, b in itertools.combinations(names, 2):
            p, hl = _cointegration(form[a], form[b])
            if not (p < 0.05 and 3 < hl < 250):
                continue
            lb, ez = 60, 2.0
            daily, trades = pair_daily_and_trades(panel[a].loc[:t1], panel[b].loc[:t1], lb, ez, mkt_vol)
            seg = daily.loc[t0:t1]
            s = pd.Series(0.0, index=panel.index)
            s.loc[seg.index] = seg.values
            slot = len(pair_series)
            pair_series.append(s)
            for (te, tx, feat, pnl, si, xi) in trades:
                if t0 <= te <= t1:
                    rows.append({**feat, "adf_p": p, "half_life": hl, "entry": te, "exit": tx,
                                 "pnl": pnl, "win": int(pnl > 0), "slot": slot, "si": si, "xi": xi})
    trades_df = pd.DataFrame(rows).dropna(subset=["entry"]).reset_index(drop=True)
    basket = pd.concat(pair_series, axis=1).sum(axis=1) / max(len(pair_series), 1)
    base = summarise(basket.loc[anchors[0]:], PPY)
    print(f"ungated sleeve: Sharpe {base['sharpe_ann']:+.2f}  ({len(trades_df)} trades, "
          f"hit-rate {trades_df['win'].mean():.0%})", flush=True)

    feats = ["z", "spread_sd", "dir", "mkt_vol", "adf_p", "half_life"]
    X = trades_df[feats].astype(float)
    X.index = pd.DatetimeIndex(trades_df["entry"])
    y = pd.Series(trades_df["win"].values, index=X.index)
    t1s = pd.Series(pd.DatetimeIndex(trades_df["exit"]).values, index=X.index)
    oos_p, _ = cv_oos_predictions(X, y, t1s, model_factory, n_splits=5, embargo=pd.Timedelta(days=2))
    trades_df["p_win"] = oos_p.values

    print(f"meta-model OOS AUC-ish: mean P(win|took) {trades_df.loc[trades_df.win == 1, 'p_win'].mean():.3f} "
          f"vs P(win|lost) {trades_df.loc[trades_df.win == 0, 'p_win'].mean():.3f}", flush=True)
    for thr in (0.50, 0.55, 0.60):
        gated = pd.concat(pair_series, axis=1)
        keep = trades_df[trades_df["p_win"] >= thr]
        g = pd.Series(0.0, index=panel.index)
        for _, r in keep.iterrows():
            s = pair_series[int(r["slot"])]
            g.loc[s.index[int(r["si"]):int(r["xi"])]] += s.iloc[int(r["si"]):int(r["xi"])].values
        gb = (g / max(len(pair_series), 1)).loc[anchors[0]:]
        m = summarise(gb, PPY)
        print(f"  gate P(win)>={thr}: kept {len(keep)}/{len(trades_df)} trades -> Sharpe {m['sharpe_ann']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
