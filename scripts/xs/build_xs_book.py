"""A CANDIDATE x-sect construction: idiosyncratic momentum, published for comparison only.

The leg the master book actually reads is written by `scripts/xs/portfolio.py` (`make xs`).

crypto·x-sect is IDIOSYNCRATIC (residual) momentum — the RESIDMOM.md deep-dive construction (Blitz-
Huij-Martens): the crypto momentum signal on the market-beta-neutralised residual, not the raw price.
On the 300-name PIT spot panel (crypto_1d, 2020+) ranked to the top-100 most-liquid, beta stripped
over 90d (as BAB), residual momentum ranked over ~30d, monthly rebalance, honest 6bps + √-impact
(Almgren k·σ·√(order/ADV) per name), vol-target 15%. Residualising and rebalancing monthly gives a
steadier, lower-turnover crypto momentum than a daily raw-price sleeve — net Sharpe ≈ +0.6, and it
lifts the assembled book's out-of-sample consistency. The deep-dive owns and validated the
construction (scripts/residmom/run_residmom.py); this reuses its exact builder so the two never drift.

Equity·x-sect keeps the broad-panel risk-adjusted-momentum engine (survivorship-free S&P panel,
top-100 liquid, classic 12-1, decile legs, monthly, honest commission + half-spread + √-impact +
borrow). The two legs (≈0.00 correlated) are risk-parity combined. Feeds scripts/run_master_book.py.

    python scripts/xs/build_xs_book.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import config as cfg  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves.xsect import idio_mom, risk_adj_mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
# The crypto x-sect leg IS the RESIDMOM.md deep-dive's idiosyncratic-momentum book — reuse its exact
# builder (panel / winsor / signal / backtest) so the leg can never drift from the validated deep-dive.
from scripts.residmom.run_residmom import ASSETS as RM_ASSETS, _book as rm_idio_book, _load as rm_load  # noqa: E402

CACHE, OUT = cfg.CACHE_DIR / "xs", cfg.XS_DIR


def _norm(s):
    s = s.copy(); s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna()


def crypto_spot_xsect(report=False):
    """Crypto x-sect = idiosyncratic (residual) momentum — the RESIDMOM.md deep-dive construction
    (Blitz-Huij-Martens): the momentum signal on the market-beta-neutralised residual, not the raw
    price. On the 300-name PIT spot panel (crypto_1d, 2020+) ranked to the top-100 most-liquid, beta
    stripped over a 90d window (as BAB), residual momentum ranked over ~30d, monthly rebalance, honest
    6bps + √-impact, vol-target 15%. Residualising and rebalancing monthly gives a steadier, lower-
    turnover crypto momentum than a daily raw-price sleeve (net Sharpe ≈ +0.6, decorrelated from the
    equity leg). Reuses the deep-dive's own builder so this leg can never drift from the validated run."""
    c = RM_ASSETS["crypto"]; b = c["base"]
    Craw, adv = rm_load(c["tag"])                                       # crypto_1d: 300-name PIT spot panel
    C = bab.winsorize_panel(Craw, c["winsor"])                          # clip artifact name-days |ret|>100%
    sig = idio_mom(C, b["lb"], c["beta_lb"], b["sk"], market=None)      # BHM residual momentum vs EW market
    net, bt = rm_idio_book(C, sig, adv, c)                              # top-100 liquid, q0.3, monthly, 6bps+√-impact, vt15%
    if report:
        yrs = (bt["net"].index[-1] - bt["net"].index[0]).days / 365.25
        print(f"  crypto·x-sect (idio) cost/yr: total {bt['cost'].sum()/yrs:.1%}"
              f"  √-impact {bt['impact'].sum()/yrs:.1%}  turnover {bt['turnover'].sum()/yrs:.0f}x/yr"
              f"  (monthly rebalance — far below a daily sleeve)")
    return _norm(net)


def equity_xsect(report=False):
    """Cross-sectional momentum on the survivorship-free S&P panel (book engine, top-100 liquid,
    classic 12-1, decile legs, monthly) — broad and liquid, so turnover and impact are small and the
    equal-weight engine is fine. Cost split into equity commission + half-spread + √-impact."""
    px = pd.read_parquet(CACHE / "stocks_broad_1d_close.parquet")
    px = px[px.notna().sum(axis=1) >= 100]
    adv = pd.read_parquet(CACHE / "stocks_broad_1d_adv.parquet").reindex(px.index)
    sig = top_n_liquid(risk_adj_mom(px, cfg.XS_EQUITY_LOOKBACK_D, cfg.XS_EQUITY_SKIP_D), adv, 100, px=px)
    bt = xs_backtest(px, sig, top_frac=cfg.XS_EQUITY_TOP_FRAC, weighting="equal", rebal=21,
                     commission_bps=cfg.EQUITY_COMMISSION_BPS, half_spread_bps=cfg.EQUITY_HALF_SPREAD_BPS,
                     adv=adv, impact_k=cfg.IMPACT_K, capital=cfg.CAPITAL_USD, min_names=6,
                     borrow_bps_annual=cfg.EQUITY_BORROW_BPS_ANNUAL, ppy=cfg.EQUITY_PPY)
    if report:
        yrs = (bt["net"].index[-1] - bt["net"].index[0]).days / 365.25
        print(f"  equity·x-sect cost/yr: commission {bt['commission'].sum()/yrs:.1%}"
              f"  half-spread {bt['spread'].sum()/yrs:.1%}  √-impact {bt['impact'].sum()/yrs:.1%}"
              f"  borrow {bt['carry'].sum()/yrs:.2%}"
              f"  (total {bt['cost'].sum()/yrs:.1%}, turnover {bt['turnover'].sum()/yrs:.0f}x/yr)")
    return _norm(vol_target(bt["net"], cfg.EQUITY_PPY, cfg.XS_VOL_TARGET).dropna())


def main():
    cr, eq = crypto_spot_xsect(report=True), equity_xsect(report=True)
    for nm, s, ppy in [("crypto·x-sect (idio)", cr, cfg.CRYPTO_PPY), ("equity·x-sect", eq, cfg.EQUITY_PPY)]:
        x = summarise(s, ppy)
        p5 = bootstrap_sharpe(s, ppy, 800, cfg.SEED).get("sharpe_p5", np.nan)
        print(f"  {nm:22s} Sharpe {x['sharpe_ann']:+.2f}  DD {x['max_dd']:+.0%}  MC-P5 {p5:+.2f}  "
              f"{s.index.min().date()}→{s.index.max().date()}")

    # risk-parity (inverse-vol) combine of the two ≈decorrelated legs
    R = pd.concat([cr.rename("crypto"), eq.rename("equity")], axis=1)
    print(f"  corr(crypto, equity) = {R.corr().iloc[0, 1]:+.2f}")
    w = (1.0 / R.std()).replace([np.inf, -np.inf], np.nan)
    book = (R * w).sum(axis=1, min_count=1).div(w.sum()).where(R.notna().any(axis=1)).dropna()
    x = summarise(book, cfg.CRYPTO_PPY)
    mc = bootstrap_sharpe(book, cfg.CRYPTO_PPY, 1000, cfg.SEED)
    print(f"\nx-sect BLOCK (crypto SPOT + equity, risk-parity): Sharpe {x['sharpe_ann']:+.2f}  "
          f"DD {x['max_dd']:+.1%}  MC-P5 {mc.get('sharpe_p5', float('nan')):+.2f}  "
          f"{book.index.min().date()}→{book.index.max().date()}")

    # NOT `xs_book.parquet`. That path is written by `scripts/xs/portfolio.py`, which is what
    # `make xs` runs and what the master book reads; this file wrote the SAME path with a different
    # construction (idiosyncratic momentum, two legs) and whichever ran last won — the two series
    # correlate 0.73. A candidate study does not get to overwrite the shipped leg by being run
    # second, so it publishes under its own name and the assembler is unaffected.
    book.rename("ret").to_frame().to_parquet(OUT / "xs_book_idio_candidate.parquet")
    print(f"\nwrote {OUT/'xs_book_idio_candidate.parquet'}  (a CANDIDATE construction — the leg the "
          f"master book reads is written by scripts/xs/portfolio.py)")


if __name__ == "__main__":
    main()
