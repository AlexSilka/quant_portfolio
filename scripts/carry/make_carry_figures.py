"""Carry story in six panels: the directional baseline dies, cross-sectional carry is a real
decorrelated edge, and the delta-neutral basis trade harvests funding at low vol. Writes
reports/figures/carry.png. Recomputes the needed series inline (fast).

    python scripts/carry/make_carry_figures.py
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, vol_target  # noqa: E402
import scripts.run_master_book as mb  # noqa: E402  the assembler weights the book, not a flat mean
from src.config import CAPITAL_USD, FIGURES_DIR, REPORTS_DIR, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry, carry_xs  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from scripts.carry.run_carry import START, END, load_panel  # noqa: E402

PPY, TVOL, CB = 365, VOL_TARGET_ANNUAL, 6.0
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def vt(net):
    scale = vol_target_scale(net, TVOL, PPY)
    return (net * scale).dropna()


def main():
    C, fd = load_panel()
    ret = C.pct_change()
    bk = carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, 7), direction=-1.0, top_frac=0.2, cost_bps=CB)
    carry_net = vt(bk["ret"])
    # refined book: BTC-beta hedge + inverse-vol weighting + no-trade buffer (validated OOS)
    ref_bk = carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, 7), direction=-1.0, top_frac=0.2,
                              cost_bps=CB, weight="inv_vol", buffer=0.02)
    refined_net = vt(carry_xs.beta_hedge(ref_bk["ret"], C["BTCUSDT"].pct_change()))

    # directional single-asset (BTC) for the "dead baseline" curve
    pxb = load_klines("BTCUSDT", "1d", START, END, market="um"); fb = load_funding("BTCUSDT", START, END)["last_funding_rate"]
    dpos = vol_target(carry.primary_side(fb, pxb["close"]), pxb["close"], TVOL, PPY)
    dbt = backtest(pxb["close"], dpos, capital=CAPITAL_USD, funding=fb, commission_bps=5, half_spread_bps=1, impact_k=0.1)
    dir_net = ((1 + dbt["net_ret"]).resample("D").prod() - 1).dropna()

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # (1) equity curves: directional vs cross-sectional vs refined
    a = ax[0, 0]
    (1 + dir_net).cumprod().plot(ax=a, label=f"directional BTC (Sh {summarise(dir_net,PPY)['sharpe_ann']:+.2f})", color="#b0b0b0")
    (1 + carry_net).cumprod().plot(ax=a, label=f"cross-sectional (Sh {summarise(carry_net,PPY)['sharpe_ann']:+.2f})", color="#1f77b4", lw=2)
    (1 + refined_net).cumprod().plot(ax=a, label=f"+ beta-hedge refined (Sh {summarise(refined_net,PPY)['sharpe_ann']:+.2f})", color="#2ca02c", lw=2)
    a.set_title("1) Directional carry is dead; cross-sectional (refined) is the edge"); a.legend(); a.set_yscale("log"); a.set_ylabel("equity (log)")

    # (2) leg decomposition: funding vs price vs total
    a = ax[0, 1]
    for lab, s, c in [("funding-only", bk["funding"], "#2ca02c"), ("price-only", bk["price"], "#ff7f0e"), ("total", bk["ret"], "#1f77b4")]:
        (1 + vt(s)).cumprod().plot(ax=a, label=f"{lab} (Sh {summarise(vt(s),PPY)['sharpe_ann']:+.2f})", color=c)
    a.set_title("2) Leg decomposition (funding is a smooth accrual)"); a.legend(); a.set_yscale("log")

    # (3) execution-lag leak test
    a = ax[0, 2]
    lags = [2, 3, 4, 6, 8, 12]
    ptot = [summarise(vt(carry_xs.xs_book(C, fd, carry_xs.signal_level(fd, 7), direction=-1, top_frac=0.2, exec_lag=l, cost_bps=CB)["price"]), PPY)["sharpe_ann"] for l in lags]
    a.plot(lags, ptot, "o-", color="#d62728")
    a.axhline(0, color="k", lw=0.5); a.set_title("3) Leak test: price-leg Sharpe vs execution lag\n(decays gracefully = real signal, not leak)")
    a.set_xlabel("execution lag (days)"); a.set_ylabel("price-leg Sharpe")

    # (4) per-year Sharpe
    a = ax[1, 0]
    py = {int(y): (np.sqrt(PPY) * g.dropna().mean() / g.dropna().std(ddof=1)) for y, g in carry_net.groupby(carry_net.index.year) if g.dropna().std(ddof=1) > 0}
    a.bar([str(y) for y in py], list(py.values()), color=["#2ca02c" if v > 0 else "#d62728" for v in py.values()])
    a.axhline(0, color="k", lw=0.5); a.set_title("4) Cross-sectional carry: per-year Sharpe"); a.set_ylabel("Sharpe")

    # (5) portfolio blend: master book (ex-carry) vs +carry
    a = ax[1, 1]
    # The book this is compared against is whatever the book trades — carry is no longer in it (§6d-ter),
    # so dropping it by name raised KeyError. And the blend has to be the assembler's weighting, not a
    # flat mean, or the baseline prices the hedge slot at a share the book does not hold it at.
    _legs = pd.read_parquet(REPORTS_DIR / "master_book_legs.parquet")
    book = mb.book_stack(_legs.drop(columns=[c for c in ("carry",) if c in _legs.columns]))
    if carry_net.index.tz is not None:
        book.index = book.index.tz_localize(carry_net.index.tz)
    idx = carry_net.index.intersection(book.index)
    cc, bb = carry_net.reindex(idx).fillna(0), vt(book.reindex(idx)).reindex(idx).fillna(0)
    alphas = np.linspace(0, 0.5, 11)
    shs = [summarise(a_ * cc + (1 - a_) * bb, PPY)["sharpe_ann"] for a_ in alphas]
    dds = [-summarise(a_ * cc + (1 - a_) * bb, PPY)["max_dd"] for a_ in alphas]
    a.plot(alphas, shs, "o-", color="#1f77b4", label="portfolio Sharpe")
    a.set_xlabel("carry risk-budget weight"); a.set_ylabel("Sharpe", color="#1f77b4")
    a2 = a.twinx(); a2.plot(alphas, dds, "s--", color="#d62728", label="max drawdown"); a2.set_ylabel("max DD", color="#d62728"); a2.grid(False)
    a.set_title(f"5) Adding carry to the master book ex-carry (corr {np.corrcoef(cc,bb)[0,1]:+.2f})")

    # (6) cost sensitivity
    a = ax[1, 2]
    mults = np.linspace(1, 8, 15)
    def at(m): return vt(bk["price"] + bk["funding"] - m * bk["cost"])
    a.plot(mults, [summarise(at(m), PPY)["sharpe_ann"] for m in mults], "o-", color="#9467bd")
    a.axhline(0, color="k", lw=0.5); a.set_title("6) Cost sensitivity (break-even ~5x base)"); a.set_xlabel("cost multiple x base"); a.set_ylabel("Sharpe")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "carry.png", bbox_inches="tight")
    print("wrote reports/figures/carry.png")


if __name__ == "__main__":
    main()
