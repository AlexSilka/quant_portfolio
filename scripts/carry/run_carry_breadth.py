"""Does cross-sectional carry get better with a wider universe, and how much did survivorship inflate
the 50-name result? Loads every cached USD-M perp (incl. delisted names in the Binance archive),
selects a POINT-IN-TIME top-N by trailing dollar volume on each date (so a coin is only traded while it
was actually liquid — no survivorship), and runs carry across a breadth curve N = 30..150.

Also: (a) survivorship test — PIT (delisted included) vs current-listed-only; (b) cost stress on the
illiquid tail (smaller coins have wider spreads); (c) per-year, so we see where the extra breadth pays.

    python scripts/carry/run_carry_breadth.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import CARRY_DIR, RAW_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

PPY, TVOL, SEED, CB = 365, VOL_TARGET_ANNUAL, SEED, 6.0
START, END = "2020-01", "2026-07"
rng = np.random.default_rng(SEED)


def vt(net):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def per_year(net):
    return {int(y): round(float(np.sqrt(PPY) * g.dropna().mean() / g.dropna().std(ddof=1)), 2)
            for y, g in net.groupby(net.index.year) if g.dropna().std(ddof=1) > 0}


def load_all():
    """Every symbol with both 1d klines and funding cached -> close, dollar-volume, daily funding."""
    kdir = RAW_DIR / "futures/um/klines"
    fdir = RAW_DIR / "futures/um/fundingRate"
    # crypto-native only: drop stablecoins, tokenized gold and synthetic index perps (different asset class)
    syms = sorted(s.name for s in kdir.iterdir() if (s / "1d").exists() and (fdir / s.name).exists()
                  and s.name not in carry_xs.NON_CRYPTO_NATIVE)
    close, dvol, fund = {}, {}, {}
    for s in syms:
        try:
            px = load_klines(s, "1d", START, END, market="um")
            f = load_funding(s, START, END)["last_funding_rate"]
        except Exception:
            continue
        if len(px) < 120 or not len(f):
            continue
        close[s], dvol[s], fund[s] = px["close"], px["quote_volume"], f
    C = pd.DataFrame(close).sort_index()
    V = pd.DataFrame(dvol).reindex(C.index)
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(C.index)
    return C, V, fd


def run_carry(C, fd, sig_full, elig, *, cost_bps=CB, weight="inv_vol", buffer=0.02, beta_ret=None):
    """Carry on the PIT-eligible universe (signal masked to eligible names), refined construction."""
    sig = sig_full.where(elig)
    bk = carry_xs.xs_book(C, fd, sig, direction=-1.0, top_frac=0.2, cost_bps=cost_bps,
                          weight=weight, buffer=buffer)
    net = bk["ret"]
    if beta_ret is not None:
        net = carry_xs.beta_hedge(net, beta_ret)
    return net, bk


def main():
    C, V, fd = load_all()
    btc = C["BTCUSDT"].pct_change() if "BTCUSDT" in C else C.iloc[:, 0].pct_change()
    n_names = C.notna().sum(axis=1)
    print(f"loaded {C.shape[1]} symbols (incl. delisted), {C.index.min().date()}..{C.index.max().date()}")
    print(f"names with data over time: 2021 {int(n_names.loc['2021'].median())}, 2023 {int(n_names.loc['2023'].median())}, "
          f"2025 {int(n_names.loc['2025'].median())}\n")
    sig_full = carry_xs.signal_level(fd, 7)

    print("=== BREADTH CURVE (point-in-time top-N by $-volume, refined carry) ===")
    rows = []
    for n in [30, 50, 75, 100, 150, 200]:
        if n > C.shape[1]:
            continue
        elig = carry_xs.pit_eligible(V, n)
        net = vt(run_carry(C, fd, sig_full, elig, beta_ret=btc)[0])
        s = summarise(net, PPY)
        p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan) if s["sharpe_ann"] > 0.2 else np.nan
        rows.append({"N": n, "sharpe": round(s["sharpe_ann"], 2), "mc_p5": round(p5, 2) if p5 == p5 else np.nan,
                     "max_dd": round(s["max_dd"], 2), "months+": round(s["months_in_profit"], 2)})
        print(f"  top-{n:<3d} Sharpe {s['sharpe_ann']:+.2f}  MC-P5 {rows[-1]['mc_p5']}  DD {s['max_dd']:+.0%}  months+ {s['months_in_profit']:.0%}")

    # ---- survivorship test: PIT (delisted included) vs current-listed-only, same top-100 ----
    print("\n=== survivorship test (top-100) ===")
    elig100 = carry_xs.pit_eligible(V, 100)
    pit = vt(run_carry(C, fd, sig_full, elig100, beta_ret=btc)[0])
    survivors = C.columns[C.iloc[-1].notna()]          # names still trading at the end
    C_surv = C[survivors]
    surv_only = vt(run_carry(C_surv, fd[survivors], carry_xs.signal_level(fd[survivors], 7),
                             carry_xs.pit_eligible(V[survivors], 100), beta_ret=btc)[0])
    print(f"  PIT (incl. delisted):     Sharpe {summarise(pit, PPY)['sharpe_ann']:+.2f}  DD {summarise(pit, PPY)['max_dd']:+.0%}")
    print(f"  current-listed only:      Sharpe {summarise(surv_only, PPY)['sharpe_ann']:+.2f}  DD {summarise(surv_only, PPY)['max_dd']:+.0%}")
    print(f"  -> survivorship inflation: {summarise(surv_only, PPY)['sharpe_ann'] - summarise(pit, PPY)['sharpe_ann']:+.2f} Sharpe")

    # ---- cost stress on the wide universe (illiquid tail = wider spreads) ----
    print("\n=== cost stress (top-100 PIT) ===")
    for cb in [6, 12, 20, 30]:
        net = vt(run_carry(C, fd, sig_full, elig100, cost_bps=cb, beta_ret=btc)[0])
        print(f"  {cb:2d} bps/side: Sharpe {summarise(net, PPY)['sharpe_ann']:+.2f}")

    best = max(rows, key=lambda r: r["sharpe"]) if rows else None
    print(f"\nbest breadth: top-{best['N']} Sharpe {best['sharpe']:+.2f}  (vs 50-name refined +1.47)")
    print(f"per-year (top-100 PIT): {per_year(pit)}")
    pd.DataFrame(rows).to_csv(CARRY_DIR / "carry_breadth.csv", index=False)
    pit.rename("ret").to_frame().to_parquet(CARRY_DIR / "carry_breadth_headline.parquet")
    print("\nCARRY-BREADTH OK")


if __name__ == "__main__":
    main()
