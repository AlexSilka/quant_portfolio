"""Exit-style experiment: is breakout edge in the fat tail (like trend), so a bounded exit throws
it away? Same Donchian-55 entries, five exits, net of costs+funding, vol-targeted, MC-screened.

    triple_barrier  : current book exit (pt/sl = 1 bar of vol, vertical HORIZON) — short bounded hold
    time            : fixed HORIZON hold, no barrier
    channel         : Turtle — ride until the opposite 20-bar channel breaks (trend-riding)
    atr_trailing    : chandelier — ride until close falls 3*ATR from the peak (trend-riding)
    reversal        : stop-and-reverse Donchian, always in market (pure trend system)

    python scripts/breakout/run_bo_exits.py
"""

import pandas as pd

from src import bo_common as bo  # noqa: E402

EXITS = ["triple_barrier", "time", "channel", "atr_trailing", "reversal"]
CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
          "NEARUSDT", "LINKUSDT", "DOGEUSDT", "LTCUSDT", "ETCUSDT"]
EQUITY = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM"]


def run(rows, kind, sym, tf, px, fund):
    tfmap, costs = bo.cfg_for(kind, sym)
    ppy_bar = tfmap[tf]
    adv = (px["quote_volume"].rolling(20).median().shift(1) if kind == "crypto"
           else (px["close"] * px["volume"]).rolling(20).median().shift(1) if "volume" in px else None)
    side = bo.entry_side(px, "donchian", 55)
    ppy_d = 365 if kind == "crypto" else 252
    for ex in EXITS:
        pos = bo.held_position(ex, px, side, tf)
        s, _ = bo.evaluate(px["close"], pos, ppy_bar, costs, fund=fund, adv=adv, ppy_daily=ppy_d)
        rows.append({"kind": kind, "sym": sym, "tf": tf, "exit": ex, "sharpe": s["sharpe_ann"],
                     "mc_p5": s["mc_p5"], "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
                     "ann_turnover": s["ann_turnover"]})


def main():
    rows = []
    for tf in ["1d", "4h"]:
        for sym in CRYPTO:
            print(f"  load crypto {sym} {tf} ...", flush=True)
            px = bo.load_crypto(sym, tf)
            if px is None:
                continue
            fund = bo.safe_funding(sym)
            run(rows, "crypto", sym, tf, px, fund)
        for sym in EQUITY:
            print(f"  load equity {sym} {tf} ...", flush=True)
            px = bo.load_eqfx(sym, tf)
            if px is None:
                print(f"  skip equity {sym} {tf} (not cached)", flush=True)
                continue
            run(rows, "equity", sym, tf, px, None)

    df = pd.DataFrame(rows)
    df.to_csv(bo.REPORTS / "bo_exits.csv", index=False)

    print("\n=== MEAN Sharpe by exit style (across all symbols x {1d,4h}) ===")
    piv = df.pivot_table(index="exit", values="sharpe", aggfunc="mean").reindex(EXITS)
    piv["median"] = df.groupby("exit")["sharpe"].median().reindex(EXITS)
    piv["mean_dd"] = df.groupby("exit")["max_dd"].mean().reindex(EXITS)
    piv["mean_turn"] = df.groupby("exit")["ann_turnover"].mean().reindex(EXITS)
    piv["frac>0.5"] = df.groupby("exit")["sharpe"].apply(lambda x: (x > 0.5).mean()).reindex(EXITS)
    print(piv.round(3).to_string())

    print("\n=== by timeframe (mean Sharpe) ===")
    print(df.pivot_table(index="exit", columns="tf", values="sharpe", aggfunc="mean")
          .reindex(EXITS).round(3).to_string())

    print("\n=== per-symbol on 1d (Sharpe by exit) ===")
    d1 = df[df.tf == "1d"].pivot_table(index="sym", columns="exit", values="sharpe").reindex(columns=EXITS)
    print(d1.round(2).sort_values("triple_barrier", ascending=False).to_string())

    # sanity: the triple_barrier arm must reproduce the book (AVAX_1d ~1.09)
    chk = df[(df.sym == "AVAXUSDT") & (df.tf == "1d") & (df.exit == "triple_barrier")]
    if len(chk):
        print(f"\n[sanity] AVAXUSDT_1d triple_barrier Sharpe {chk.iloc[0].sharpe:+.2f} (book: +1.09)")
    print("\nBO EXITS OK")


if __name__ == "__main__":
    main()
