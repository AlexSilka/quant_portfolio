"""FX carry — the canonical carry trade, on the same cross-sectional machinery as crypto carry, so
the two asset classes are directly comparable. Rank currencies by 3-month rate, LONG high-rate /
SHORT low-rate, dollar-neutral; the book earns the rate spread plus FX moves. Validated with a
shuffled-rate placebo, per-year, skew (expect the carry-crash negative tail), cost sensitivity, and
its correlation to crypto carry (the diversification question).

    python scripts/carry/run_carry_fx.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.twelvedata import RateLimited  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.data.rates import short_rates  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_fx  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

PPY, TVOL, SEED = 252, VOL_TARGET_ANNUAL, SEED
rng = np.random.default_rng(SEED)


def vt(net):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def per_year(net):
    return {int(y): round(float(np.sqrt(PPY) * g.dropna().mean() / g.dropna().std(ddof=1)), 2)
            for y, g in net.groupby(net.index.year) if g.dropna().std(ddof=1) > 0}


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    fx_close = {}
    for pair in carry_fx.PAIR_MAP:
        try:
            fx_close[pair] = load_equity_daily(f"{pair}=X", start="2012-01-01")["close"]
        except RateLimited:
            raise                       # same reason as run_carry_equity: a refused fetch changes the
                                        # universe, not just the coverage, and does it silently.
        except Exception as e:
            print(f"skip {pair}: {e}")
    usd_value = carry_fx.usd_value_panel(fx_close)
    rates = short_rates()
    common = [c for c in usd_value.columns if c in rates.columns or c == "USD"]
    usd_value = usd_value[[c for c in common]]
    print(f"FX carry universe: {list(usd_value.columns)}  ({usd_value.index.min().date()}..{usd_value.index.max().date()})")
    print(f"latest 3M rates: {rates.iloc[-1].dropna().round(2).to_dict()}\n")

    # ---- grid: top-fraction x rebalance ----
    print("=== FX carry (dollar-neutral, long high-rate / short low-rate) ===")
    rows, best = [], None
    for tf in (0.25, 0.33, 0.5):
        for rb in (1, 5, 21):
            bk = carry_fx.fx_carry_book(usd_value, rates, top_frac=tf, rebalance=rb)
            net = vt(bk["ret"])
            s = summarise(net, PPY)
            p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan) if s["sharpe_ann"] > 0.2 else np.nan
            row = {"top": tf, "rebal": rb, "sharpe": round(s["sharpe_ann"], 2), "mc_p5": round(p5, 2) if p5 == p5 else np.nan,
                   "max_dd": round(s["max_dd"], 2), "skew": round(net.skew(), 2), "months+": round(s["months_in_profit"], 2),
                   "ann_carry_%": round(bk["carry"].mean() * PPY * 100, 1), "ann_fx_%": round(bk["fx"].mean() * PPY * 100, 1)}
            rows.append(row)
            if best is None or row["sharpe"] > best[0]:
                best = (row["sharpe"], tf, rb, bk, net)
            print(f"  top{int(tf*100)}_rb{rb:<2d} Sharpe {row['sharpe']:+.2f}  P5 {row['mc_p5']}  DD {row['max_dd']}  "
                  f"skew {row['skew']:+.1f}  carry {row['ann_carry_%']:+.1f}%/yr  fx {row['ann_fx_%']:+.1f}%/yr")

    _, tf, rb, bk, net = best
    print(f"\nheadline: top{int(tf*100)}_rb{rb}  Sharpe {best[0]:+.2f}")
    print(f"per-year: {per_year(net)}")

    # ---- placebo: shuffled rates (destroys the carry ranking) ----
    plac_rates = rates.copy()
    plac_rates[:] = rng.permutation(rates.values.ravel()).reshape(rates.shape)
    pnet = vt(carry_fx.fx_carry_book(usd_value, plac_rates, top_frac=tf, rebalance=rb)["ret"])
    print(f"placebo (shuffled rates) Sharpe {summarise(pnet, PPY)['sharpe_ann']:+.2f}")

    # ---- cost sensitivity ----
    def at(m):
        return vt(bk["fx"] + bk["carry"] - m * bk["cost"])
    costs = "  ".join(f"{m}x {summarise(at(m), PPY)['sharpe_ann']:+.2f}" for m in (1, 2, 3, 5))
    print(f"cost sensitivity: {costs}")

    # ---- correlation to crypto carry (diversification) ----
    try:
        cc = pd.read_parquet(CARRY_DIR / "carry_headline.parquet")["carry"]
        idx = net.index.intersection(cc.index)
        corr = float(pd.concat([net.reindex(idx), cc.reindex(idx)], axis=1).corr().iloc[0, 1]) if len(idx) > 60 else np.nan
        print(f"corr(FX carry, crypto carry) = {corr:+.2f}  (cross-asset carry diversification)")
    except Exception:
        corr = np.nan

    pd.DataFrame(rows).to_csv(CARRY_DIR / "carry_fx.csv", index=False)
    net.rename("ret").to_frame().to_parquet(CARRY_DIR / "carry_fx_headline.parquet")
    print("\nCARRY-FX OK")


if __name__ == "__main__":
    main()
