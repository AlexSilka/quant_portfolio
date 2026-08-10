"""Re-run cross-sectional breakout on the EXPANDED cached universe (a parallel session grew it from
68 to ~830 USDT perps), across timeframes. Two honest caveats drive the design:

  1. The expanded set is NOT pure crypto — it includes tokenized-stock perps (AAPLUSDT, TSLAUSDT,
     SPYUSDT, COINUSDT, ...) and commodity perps (XAUUSDT, XAGUSDT, NATGASUSDT). We therefore report
     THREE universes per timeframe: `all` (everything cached), `crypto` (all minus a real-world-ticker
     blocklist), and `core50` (the known-clean liquid majors, unchanged) — so the reader sees whether a
     bigger panel helps and whether the edge is crypto or contaminated by stock/commodity dispersion.
  2. Many new listings have tiny history; a name needs >~150 days at the timeframe to earn a nearness
     signal, so short-history coins are filtered out (not silently long-biased).

Same construction as run_bo_xs_tf: 52-week-high nearness, long top / short bottom 30%, dollar-neutral,
~daily rebalance, returns resampled to daily, net of 6bps/side, shuffled-signal placebo.

    python scripts/breakout/run_bo_xs_big.py
"""

import numpy as np
import pandas as pd

from src import bo_common as bo  # noqa: E402
from scripts.breakout.run_bo_xs_tf import BPD, PPY, adv_panel, funding_panel, xs_daily  # noqa: E402
from src.config import OOS_START, RAW_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.cross_sectional import breakout_signal  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

KLINES = RAW_DIR / "futures/um/klines"

# tokenized real-world assets on Binance perps — excluded from the `crypto` universe (approximate but
# covers the liquid names); matched as exact symbol roots (ROOT + 'USDT').
NONCRYPTO = {
    # US mega/large-cap stocks & ETFs
    "AAPL", "AMZN", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "NFLX", "COIN", "MSTR", "HOOD", "PLTR",
    "AMD", "INTC", "IBM", "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "TXN", "AVGO", "MRVL", "SMCI", "ASML",
    "TSM", "DELL", "HPE", "WDC", "SNDK", "KLAC", "LRCX", "AMAT", "COHR", "CIEN", "AAOI", "NBIS", "CRWV",
    "CRCL", "CRWD", "PANW", "SNOW", "UBER", "DKNG", "RKLB", "RIVN", "SOFI", "GME", "BMNR", "IREN", "HIMS",
    "NVO", "LLY", "WMT", "COST", "HD", "DIS", "EBAY", "PYPL", "BX", "BRKB", "JPM", "MU", "GLW", "GEV",
    "ASTS", "APP", "BIDU", "QCOM", "NOK", "SONY", "SAMSUNG", "SKHYNIX", "SKHY", "HYUNDAI", "TENCENT",
    "POPMART", "BABA", "HK0700", "HK1810", "DATAIP", "TTWO", "PYPL", "PANW", "FLNC", "NFP",
    # ETFs / indices / leveraged
    "SPY", "QQQ", "TQQQ", "SQQQ", "SOXL", "SOXS", "UVXY", "TZA", "TMF", "XBI", "SMH", "XLE", "IWM",
    "EWJ", "EWY", "EWZ", "EWT", "BITO", "SPX", "INX",
    # commodities / metals
    "XAU", "XAG", "XPT", "XPD", "NATGAS", "COPPER",
}


def symbols_with_tf(tf):
    out = []
    for d in sorted(KLINES.iterdir()):
        if d.is_dir() and (d / tf).exists() and any((d / tf).glob("[0-9]*.parquet")):
            out.append(d.name)
    return out


def build_panel(symbols, tf, min_days=150):
    min_obs = min_days * BPD[tf]
    cols = {}
    for s in symbols:
        px = bo.load_crypto(s, tf)
        if px is None:
            continue
        c = px["close"]
        if c.notna().sum() >= min_obs:
            cols[s] = c
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index().ffill(limit=5)


def score(pnl, tf):
    if pnl.shape[1] < 15:
        return None
    lb = 126 * BPD[tf]
    adv, fund = adv_panel(pnl.columns, tf), funding_panel(pnl.columns, pnl.index)
    net = xs_daily(pnl, breakout_signal(pnl, "nearness", lb), PPY[tf], rebal=BPD[tf], adv=adv, funding=fund)
    plac = pd.DataFrame(bo.rng.standard_normal(pnl.shape), index=pnl.index, columns=pnl.columns)
    netp = xs_daily(pnl, plac, PPY[tf], rebal=BPD[tf], adv=adv, funding=fund)
    s = summarise(net, 365)
    mc = bootstrap_sharpe(net, 365, 500, bo.SEED) if s["sharpe_ann"] > 0.3 else {}
    oos = net[net.index >= OOS_START]
    return {"n": pnl.shape[1], "sharpe": s["sharpe_ann"], "oos": summarise(oos, 365)["sharpe_ann"],
            "mc_p5": mc.get("sharpe_p5", np.nan), "max_dd": s["max_dd"],
            "placebo": summarise(netp, 365)["sharpe_ann"]}


def main():
    print("=== Cross-sectional breakout on the EXPANDED universe, across timeframes ===")
    print("(52w-high nearness, ~daily rebalance, daily-resampled, net 6bps/side; 3 universes)\n")
    rows = []
    for tf in ["1d", "4h", "1h", "15m"]:
        syms = symbols_with_tf(tf)
        allp = build_panel(syms, tf)
        if allp.empty:
            continue
        crypto = allp[[c for c in allp.columns if c[:-4] not in NONCRYPTO]]
        core = allp[[c for c in allp.columns if c in bo.CRYPTO]]
        n_noncrypto = allp.shape[1] - crypto.shape[1]
        print(f"--- {tf}: {len(syms)} symbols with data, {allp.shape[1]} pass >=150d history "
              f"({n_noncrypto} tokenized stock/commodity) ---", flush=True)
        for uni, pnl in [("all", allp), ("crypto", crypto), ("core50", core)]:
            r = score(pnl, tf)
            if r is None:
                print(f"    {uni:7s}: panel too small ({pnl.shape[1]})"); continue
            rows.append({"tf": tf, "universe": uni, **r})
            print(f"    {uni:7s} ({r['n']:3d}): Sharpe {r['sharpe']:+.2f}  OOS {r['oos']:+.2f}  "
                  f"MC-P5 {r['mc_p5']:+.2f}  DD {r['max_dd']:+.1%}  placebo {r['placebo']:+.2f}", flush=True)
        print()
    pd.DataFrame(rows).to_csv(bo.BREAKOUT / "bo_xs_big.csv", index=False)
    print("BO XS-BIG OK")


if __name__ == "__main__":
    main()
