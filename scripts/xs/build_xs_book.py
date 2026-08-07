"""Build the x-sect family block (reports/xs/xs_book.parquet) that feeds the master book.

crypto·x-sect is on SPOT (2017-08+), not perp (2020+): a pure price signal needs no funding, so
spot's deeper history is usable. It runs through the book's own engine (`xsect.xs_backtest`) with the
HONEST, liquidity-aware cost, split into three visible pieces (never one opaque bps number):
  - commission : venue-correct Binance taker — SPOT (10bps) while spot is the only venue (pre-2020),
                 FUTURES (5bps) once perps exist and carry the shorts (2020+); spliced at 2020,
  - half-spread: 1bp bid/ask floor,
  - √-impact   : Almgren k·σ·√(order/ADV) per name, so the illiquid mid-cap tail of the top-50 pays
                 its true wider cost instead of a flat spread.
inverse-vol legs + a no-trade buffer keep the thin 2017-18 cross-section (4-18 names) from blowing up
(the equal-weight build does). Honest net Sharpe ≈ +0.66 (venue-correct) / +0.70 (futures-only) — the
√-impact term (~7-8%/yr at this book's turnover) is what a flat-cost build silently omits. All
constants live in `src/config.py`. Equity·x-sect keeps the broad-panel engine. The two legs
(≈0.00 correlated) are risk-parity combined. Feeds scripts/run_master_book.py.

    python scripts/xs/build_xs_book.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config as cfg  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.xsect import mom, risk_adj_mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE, OUT = cfg.CACHE_DIR / "xs", cfg.XS_DIR


def _norm(s):
    s = s.copy(); s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna()


def _tznaive(*frames):
    for D in frames:
        D.index = pd.to_datetime(D.index)
        if D.index.tz is not None:
            D.index = D.index.tz_localize(None)
    return frames


def crypto_spot_xsect(report=False):
    """Robust, survivorship-free cross-sectional momentum on crypto SPOT with venue-correct costs.

    The BROAD spot universe (226 names, 2017-08+) ranked to the top-50 most-liquid each bar (a sweep
    found top-50 optimal even under honest costs — a smaller universe concentrates each order and pays
    *more* impact, not less), plain mom-30d, inverse-vol legs, no-trade buffer. Commission is spliced
    at 2020: spot taker before perps exist, futures taker after (the shorts execute on perps). The
    √-impact term is per name from ADV. Honest full-sample net ≈ +0.66; the 2023-26 tail is weak —
    crypto x-sect momentum has decayed as the market matured (the honest caveat)."""
    close = pd.read_parquet(CACHE / "crypto_spotwide_1d_close.parquet")
    adv = pd.read_parquet(CACHE / "crypto_spotwide_1d_adv.parquet")
    close, adv = _tznaive(close, adv)
    close = close[close.index >= pd.Timestamp("2017-08-01")]           # drop a 1970 glitch bar
    adv = adv.reindex(close.index)
    keep = [c for c in close.columns if c not in cfg.STABLECOINS]       # stablecoins have no momentum
    close, adv = close[keep], adv[keep]

    sig = top_n_liquid(mom(close, cfg.XS_LOOKBACK_D), adv, cfg.XS_TOP_N_LIQUID)

    def leg(venue):
        commission_bps, half_spread_bps = cfg.crypto_cost_bps(venue)
        return xs_backtest(close, sig, top_frac=cfg.XS_TOP_FRAC, weighting=cfg.XS_WEIGHTING, rebal=1,
                           buffer=cfg.XS_NO_TRADE_BUFFER, commission_bps=commission_bps,
                           half_spread_bps=half_spread_bps, adv=adv, impact_k=cfg.IMPACT_K,
                           capital=cfg.CAPITAL_USD, min_names=4)

    bt_spot, bt_fut = leg("spot"), leg("futures")                       # identical but for the taker fee
    split = pd.Timestamp(cfg.PERP_HISTORY_START)
    raw = pd.concat([bt_spot["net"].loc[:split - pd.Timedelta(days=1)],
                     bt_fut["net"].loc[split:]]).sort_index()           # venue-correct: spot<2020, fut>=2020
    if report:
        yrs = (bt_fut["net"].index[-1] - bt_fut["net"].index[0]).days / 365.25
        print(f"  crypto·x-sect cost/yr (futures leg): commission {bt_fut['commission'].sum()/yrs:.1%}"
              f"  half-spread {bt_fut['spread'].sum()/yrs:.1%}  √-impact {bt_fut['impact'].sum()/yrs:.1%}"
              f"  (total {bt_fut['cost'].sum()/yrs:.1%}, turnover {bt_fut['turnover'].sum()/yrs:.0f}x/yr)")
    return _norm(vol_target(raw, cfg.CRYPTO_PPY, cfg.XS_VOL_TARGET).dropna())


def equity_xsect(report=False):
    """Cross-sectional momentum on the survivorship-free S&P panel (book engine, top-100 liquid,
    classic 12-1, decile legs, monthly) — broad and liquid, so turnover and impact are small and the
    equal-weight engine is fine. Cost split into equity commission + half-spread + √-impact."""
    px = pd.read_parquet(CACHE / "stocks_broad_1d_close.parquet")
    px = px[px.notna().sum(axis=1) >= 100]
    adv = pd.read_parquet(CACHE / "stocks_broad_1d_adv.parquet").reindex(px.index)
    sig = top_n_liquid(risk_adj_mom(px, cfg.XS_EQUITY_LOOKBACK_D, cfg.XS_EQUITY_SKIP_D), adv, 100)
    bt = xs_backtest(px, sig, top_frac=cfg.XS_EQUITY_TOP_FRAC, weighting="equal", rebal=21,
                     commission_bps=cfg.EQUITY_COMMISSION_BPS, half_spread_bps=cfg.EQUITY_HALF_SPREAD_BPS,
                     adv=adv, impact_k=cfg.IMPACT_K, capital=cfg.CAPITAL_USD, min_names=6,
                     borrow_bps_annual=cfg.EQUITY_BORROW_BPS_ANNUAL, ppy=cfg.EQUITY_PPY)
    if report:
        yrs = (bt["net"].index[-1] - bt["net"].index[0]).days / 365.25
        print(f"  equity·x-sect cost/yr: commission {bt['commission'].sum()/yrs:.1%}"
              f"  half-spread {bt['spread'].sum()/yrs:.1%}  √-impact {bt['impact'].sum()/yrs:.1%}"
              f"  borrow {bt['borrow'].sum()/yrs:.2%}"
              f"  (total {bt['cost'].sum()/yrs:.1%}, turnover {bt['turnover'].sum()/yrs:.0f}x/yr)")
    return _norm(vol_target(bt["net"], cfg.EQUITY_PPY, cfg.XS_VOL_TARGET).dropna())


def main():
    cr, eq = crypto_spot_xsect(report=True), equity_xsect(report=True)
    for nm, s, ppy in [("crypto·x-sect (SPOT)", cr, cfg.CRYPTO_PPY), ("equity·x-sect", eq, cfg.EQUITY_PPY)]:
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

    book.rename("ret").to_frame().to_parquet(OUT / "xs_book.parquet")
    print(f"\nwrote {OUT/'xs_book.parquet'}  (master book reads this as the x-sect family)")


if __name__ == "__main__":
    main()
