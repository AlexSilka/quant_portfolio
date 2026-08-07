"""Full acceptance bar for the raw-price crypto·x-sect spot construction, under the honest liquidity-aware cost.

Validates the raw-price crypto cross-sectional sleeve — broad spot universe (226 names,
2017-08+), top-50 most-liquid each bar, mom-30d, inverse-vol legs, no-trade buffer, and the split
commission + half-spread + √-impact cost from `src/config.py` — against the shared bar
(docs/HYPOTHESES.md): shuffled-signal placebo (beat the 95th pct), purged/embargoed walk-forward OOS,
block-bootstrap MC-P5, deflated Sharpe at the trial count. This is the construction the crypto x-sect
sleeve shipped before the idiosyncratic-momentum upgrade; the shipped leg now runs on residual momentum
(docs/strategies/RESIDMOM.md, built in `build_xs_book.crypto_spot_xsect`). Costs use the FUTURES taker
(the tradable 2020+ venue where the shorts live).

    python scripts/xs/spot.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src import config as cfg  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.sleeves.xsect import mom, top_n_liquid, vol_target, xs_backtest  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

CACHE, OUT = cfg.CACHE_DIR / "xs", cfg.XS_DIR
COMMISSION, HALF_SPREAD = cfg.BINANCE_FUT_TAKER_BPS, cfg.CRYPTO_HALF_SPREAD_BPS  # tradable (futures) venue


def load_panel():
    """The broad survivorship-free spot panel the book ships on: tz-naive, glitch bar and
    stablecoins dropped."""
    close = pd.read_parquet(CACHE / "crypto_spotwide_1d_close.parquet")
    adv = pd.read_parquet(CACHE / "crypto_spotwide_1d_adv.parquet")
    for D in (close, adv):
        D.index = pd.to_datetime(D.index)
        if D.index.tz is not None:
            D.index = D.index.tz_localize(None)
    close = close[close.index >= pd.Timestamp("2017-08-01")]
    adv = adv.reindex(close.index)
    keep = [c for c in close.columns if c not in cfg.STABLECOINS]
    return close[keep], adv[keep]


def stream(close, adv, sig):
    """The shipped x-sect stream on a panel: top-50 liquid, inverse-vol legs, daily + no-trade buffer,
    honest split cost (commission + half-spread + √-impact), vol-targeted."""
    sig = top_n_liquid(sig, adv, cfg.XS_TOP_N_LIQUID)
    bt = xs_backtest(close, sig, top_frac=cfg.XS_TOP_FRAC, weighting=cfg.XS_WEIGHTING, rebal=1,
                     buffer=cfg.XS_NO_TRADE_BUFFER, commission_bps=COMMISSION, half_spread_bps=HALF_SPREAD,
                     adv=adv, impact_k=cfg.IMPACT_K, capital=cfg.CAPITAL_USD, min_names=4)
    return vol_target(bt["net"], cfg.CRYPTO_PPY, cfg.XS_VOL_TARGET).dropna()


def placebo_beat(close, adv, real_sharpe, n=40):
    """Shuffled-signal placebo: random signals through the same pipeline; real must beat the 95th pct."""
    sh = []
    for i in range(n):
        rng = np.random.default_rng(cfg.SEED + i)
        noise = pd.DataFrame(rng.standard_normal(close.shape), index=close.index, columns=close.columns)
        s = stream(close, adv, noise)
        sh.append(summarise(s, cfg.CRYPTO_PPY)["sharpe_ann"] if len(s) > 50 else 0.0)
    p95 = float(np.percentile(sh, 95))
    return p95, real_sharpe > p95, float(np.mean(sh))


def walk_forward(close, adv):
    """Purged/embargoed anchored WF: best-of-grid on train, apply next block, stitch OOS. Embargo = one
    holding month. Grid over lookback/skip/breadth — the OOS Sharpe pays for that selection."""
    grid = [(lb, sk, tf) for lb in (20, 30, 45, 90) for sk in (0, 7) for tf in (0.2, 0.3)]
    cols = {}
    for lb, sk, tf in grid:
        sig = top_n_liquid(mom(close, lb, sk), adv, cfg.XS_TOP_N_LIQUID)
        bt = xs_backtest(close, sig, top_frac=tf, weighting=cfg.XS_WEIGHTING, rebal=1,
                         buffer=cfg.XS_NO_TRADE_BUFFER, commission_bps=COMMISSION,
                         half_spread_bps=HALF_SPREAD, adv=adv, impact_k=cfg.IMPACT_K,
                         capital=cfg.CAPITAL_USD, min_names=4)
        cols[(lb, sk, tf)] = vol_target(bt["net"], cfg.CRYPTO_PPY, cfg.XS_VOL_TARGET)
    M = pd.DataFrame(cols).dropna(how="all")
    idx = M.index
    tr_b, te_b, emb = int(3 * cfg.CRYPTO_PPY), int(1 * cfg.CRYPTO_PPY), 21
    segs, start = [], tr_b
    while start + te_b <= len(idx):
        train = M.iloc[:max(0, start - emb)]                       # embargo one holding period
        test = M.iloc[start:start + te_b]
        sr = (train.mean() / train.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        best = list(sr.nlargest(5).index)                          # top-5 ensemble (plateau-robust)
        segs.append(test[best].mean(axis=1))
        start += te_b
    return (pd.concat(segs) if segs else pd.Series(dtype=float)), len(grid)


def main():
    close, adv = load_panel()
    live = int(top_n_liquid(mom(close, cfg.XS_LOOKBACK_D), adv, cfg.XS_TOP_N_LIQUID).notna().sum(axis=1).mean())
    s = stream(close, adv, mom(close, cfg.XS_LOOKBACK_D))
    full = summarise(s, cfg.CRYPTO_PPY)
    p5 = bootstrap_sharpe(s, cfg.CRYPTO_PPY, 1000, cfg.SEED).get("sharpe_p5", np.nan)
    p95, beats, pmean = placebo_beat(close, adv, full["sharpe_ann"])
    wf, n_trials = walk_forward(close, adv)
    wfs = summarise(wf, cfg.CRYPTO_PPY)["sharpe_ann"]
    b = s.dropna()
    dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0, n_trials * 4, 0.25 / cfg.CRYPTO_PPY)

    print(f"=== crypto·x-sect SPOT — SHIPPED config, honest cost (futures {COMMISSION:.0f}bps + "
          f"{HALF_SPREAD:.0f}bps spread + √-impact) ===")
    print(f"  {close.index.min().date()}→{close.index.max().date()}, {close.shape[1]} names, ~{live}/bar ranked")
    print(f"  full Sharpe {full['sharpe_ann']:+.2f}  DD {full['max_dd']:+.0%}  MC-P5 {p5:+.2f}")
    for a, b_, lab in [("2017-01", "2020-01", "2017-2019"), ("2020-01", "2023-01", "2020-2022"),
                       ("2023-01", "2027", "2023-2026")]:
        w = s.loc[a:b_]
        if len(w) > 60:
            print(f"    {lab}: Sharpe {summarise(w, cfg.CRYPTO_PPY)['sharpe_ann']:+.2f}")
    print(f"  PLACEBO: real {full['sharpe_ann']:+.2f} vs shuffled 95th {p95:+.2f} (mean {pmean:+.2f}) "
          f"-> {'BEATS' if beats else 'FAILS'}")
    print(f"  purged WF OOS: {wfs:+.2f}  (bar >0.5: {'PASS' if wfs > 0.5 else 'FAIL'})")
    print(f"  deflated Sharpe (N={n_trials * 4}): {dsr:.2f}")
    verdict = beats and (p5 > 0) and (wfs > 0.5)
    print(f"  ==> ACCEPTANCE: placebo {beats} · MC-P5>0 {p5 > 0} · WF>0.5 {wfs > 0.5}  ->  "
          f"{'PASS' if verdict else 'FAIL'}")
    s.rename("ret").to_frame().to_parquet(OUT / "xs_crypto_spot.parquet")
    print(f"saved {OUT / 'xs_crypto_spot.parquet'}")


if __name__ == "__main__":
    main()
