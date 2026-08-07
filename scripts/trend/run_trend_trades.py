"""OOS trade log + performance-target verification (task §10-13).

Extracts a per-trade log of the headline trend book over the held-out block (2024-07-01+): every
position run (entry→reversal) per instrument×timeframe, with side, hold length, and net return. Then
verifies the portfolio-level targets — max drawdown ≤ 15%, worst calendar month ≥ −6% — on the full
sample and OOS, for the three headline operating points.

    python scripts/trend/run_trend_trades.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.trend.trend_common as T  # noqa: E402
from src.backtest.engine import backtest, vol_target  # noqa: E402
from src.metrics import max_drawdown, summarise  # noqa: E402

ENTRY, DIRECTION, TFS = "ema", "asym", ["1d", "4h", "1h"]
WARMUP = pd.Timestamp("2024-04-01", tz="UTC")   # load from before OOS so positions are warmed up


def segment_trades(pos: pd.Series, close: pd.Series, net_ret: pd.Series, sym: str, tf: str) -> list[dict]:
    """One row per constant-sign position run: t0, t1, side, bars, entry/exit px, net return."""
    p = pos.to_numpy()
    idx = pos.index
    rows, i, n = [], 0, len(p)
    while i < n:
        if p[i] == 0.0:
            i += 1
            continue
        s = np.sign(p[i])
        j = i
        while j + 1 < n and np.sign(p[j + 1]) == s and p[j + 1] != 0.0:
            j += 1
        seg = net_ret.iloc[i:j + 1]
        rows.append({"symbol": sym, "tf": tf, "t0": idx[i], "t1": idx[j], "side": "long" if s > 0 else "short",
                     "bars": int(j - i + 1), "entry_px": round(float(close.iloc[i]), 6),
                     "exit_px": round(float(close.iloc[j]), 6),
                     "net_return": round(float((1 + seg).prod() - 1), 5)})
        i = j + 1
    return rows


def worst_month(ret: pd.Series) -> float:
    m = (1 + ret.dropna()).resample("ME").prod() - 1
    return float(m.min()) if len(m) else float("nan")


def main():
    spec = {"entry": ENTRY, "direction": DIRECTION, "exit": "reversal"}
    print(f"=== OOS trade log — headline book ({ENTRY} {DIRECTION}, {TFS}), block ≥ {T.OOS_START.date()} ===\n")
    trades = []
    for sym in T.CRYPTO:
        for tf in TFS:
            px = T.load_crypto_long(sym, tf)
            if px is None:
                continue
            pos = T.trend_position(px, spec, tf)
            posv = vol_target(pos, px["close"], T.TVOL, T.CRYPTO_TF[tf])
            bt = backtest(px["close"], posv, capital=T.CAP, funding=T.bo.safe_funding(sym),
                          adv=T.crypto_adv(px), **T.CC)
            m = px.index >= WARMUP
            trades += segment_trades(bt["position"][m], px["close"][m], bt["net_ret"][m], sym, tf)
    for sym in T.EQ_CORE:
        px = T.load_equity(sym)
        if px is None:
            continue
        pos = T.trend_position(px, spec, "1d")
        posv = vol_target(pos, px["close"], T.TVOL, T.EQUITY_TF["1d"])
        adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
        bt = backtest(px["close"], posv, capital=T.CAP, adv=adv, **T.EC)
        m = px.index >= WARMUP
        trades += segment_trades(bt["position"][m], px["close"][m], bt["net_ret"][m], f"{sym}_eq", "1d")

    log = pd.DataFrame(trades)
    log = log[log["t1"] >= T.OOS_START].sort_values("t0").reset_index(drop=True)   # closed in OOS
    log.to_csv(T.REPORTS / "trend_oos_trade_log.csv", index=False)
    wins = (log["net_return"] > 0).mean()
    print(f"OOS trades: {len(log)}  win rate {wins:.0%}  avg net/trade {log['net_return'].mean():+.2%}  "
          f"median hold {log['bars'].median():.0f} bars")
    print(f"  long {int((log['side']=='long').sum())} / short {int((log['side']=='short').sum())}  "
          f"best {log['net_return'].max():+.1%}  worst {log['net_return'].min():+.1%}")
    print(f"  wrote reports/trend/trend_oos_trade_log.csv ({len(log)} rows)")

    # ---- target verification on the three operating points ----
    print(f"\n=== performance-target check (max DD ≤ 15%, worst month ≥ −6%) ===")
    books = {}
    for name, fn in [("EMA asym", "trend_book_asym.parquet"),
                     ("Blend long-only", "trend_book_blend_long_only.parquet")]:
        p = T.REPORTS / fn
        if p.exists():
            books[name] = pd.read_parquet(p)["ret"]
    e = T.CACHE / "sleeves_ema_reversal_long_only_lag2_1d4h1h.parquet"
    b = T.CACHE / "sleeves_blend_reversal_long_only_lag2_1d4h1h.parquet"
    if e.exists() and b.exists():
        books["EMA+Blend LO"] = pd.concat([pd.read_parquet(e), pd.read_parquet(b)], axis=1).mean(axis=1)

    print(f"  {'book':16s} {'DD full':>8s} {'DD OOS':>8s} {'worstMo full':>13s} {'worstMo OOS':>12s}  targets")
    for name, ret in books.items():
        eq = (1 + ret.dropna()).cumprod()
        dd_full = max_drawdown(eq)
        oos = ret[ret.index >= T.OOS_START]
        dd_oos = max_drawdown((1 + oos.dropna()).cumprod())
        wm_full, wm_oos = worst_month(ret), worst_month(oos)
        ok = "OK" if (dd_full >= -0.15 and wm_full >= -0.06) else "OVER"
        print(f"  {name:16s} {dd_full:>+8.1%} {dd_oos:>+8.1%} {wm_full:>+13.1%} {wm_oos:>+12.1%}  [{ok}]")
    print("\nTREND TRADES OK")


if __name__ == "__main__":
    main()
