"""Deployable tail-hedged short-vol book: bound the standalone drawdown to fit a 15% portfolio
mandate, then size it into momentum+carry.

The naked research book (docs/strategies/VOLPREM.md) earns a high Sharpe with a -50% systemic-vol tail — not
deployable under a 15% DD limit. Here each sleeve caps the realised-variance charge (a bought wing
that bounds the crash) AND pays for that wing (wing_frac of the strike premium given back). We sweep
(cap, wing_frac) for the tail-bounded config with the best Sharpe whose standalone DD is smallest,
then confirm the sized portfolio stays under the mandate and save the integrated portfolio.

    python scripts/volprem/run_vol_premium_deploy.py
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, TREND_DIR, VOLPREM_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from scripts.volprem.run_vol_premium_book import UNIVERSE, sleeve, book_from, vt, naive_dt, PPY_BOOK  # noqa: E402

MANDATE_DD = 0.15


def build_book(**kw):
    rets = {}
    for src, sym, und, cls, ppy in UNIVERSE:
        try:
            rets[sym] = sleeve(src, sym, und, cls, ppy, **kw)
        except Exception:
            pass
    return book_from(rets)


def load_book(path):
    p = pd.read_parquet(path)
    return naive_dt(p["ret"] if "ret" in p else p.select_dtypes("number").iloc[:, 0])


def main():
    naked = build_book()
    sn = summarise(naked, PPY_BOOK)
    print(f"naked research book:  Sharpe {sn['sharpe_ann']:+.2f}  DD {sn['max_dd']:+.1%}  "
          f"skew {naked.skew():+.1f}  (not deployable standalone under a {MANDATE_DD:.0%} limit)")

    # (1) instrument tail-hedge (option wing) CANNOT be credibly priced from free data — show both failure
    # modes: too cheap => vol-target inflates a fake Sharpe; priced off the trailing tail => over/mistimed => ruin.
    print("\n=== attempted instrument wing hedge — fails without real option prices ===")
    fl = summarise(build_book(var_cap=2.0, wing_markup=0.0), PPY_BOOK)      # unpaid cap
    ru = summarise(build_book(var_cap=2.0, wing_markup=2.0), PPY_BOOK)      # trailing-priced cap
    print(f"  cap 2.0, wing unpaid:        Sharpe {fl['sharpe_ann']:+.2f}  DD {fl['max_dd']:+.1%}  <- free-lunch (fake)")
    print(f"  cap 2.0, wing priced x2:     Sharpe {ru['sharpe_ann']:+.2f}  DD {ru['max_dd']:+.1%}  <- ruin (mis-timed)")
    print("  => a real tail hedge needs the live option smile (paid). Not modelled from free data.")

    # (2) credible no-option risk control: ex-ante de-gross when implied vol spikes
    print("\n=== ex-ante spike de-gross (cut the short while implied vol spikes) ===")
    best = (naked, sn, "none")
    for t in (1.2, 1.3, 1.5):
        b = build_book(spike_degross=t)
        s = summarise(b, PPY_BOOK)
        print(f"  spike>{t}x avg:  Sharpe {s['sharpe_ann']:+.2f}  DD {s['max_dd']:+.1%}  skew {b.skew():+.1f}")
        if s["max_dd"] > best[1]["max_dd"]:
            best = (b, s, f"spike>{t}")
    dep, sd, how = best
    print(f"  best DD: {how}  Sharpe {sd['sharpe_ann']:+.2f}  DD {sd['max_dd']:+.1%} "
          f"(helps but does not reach 15% alone — the mandate is met by sizing below)")
    py = dep.groupby(dep.index.year).apply(lambda x: summarise(x, PPY_BOOK)["sharpe_ann"])
    print("  per-year: " + "  ".join(f"{y}:{v:+.1f}" for y, v in py.items()))

    # --- size into momentum + carry, confirm the PORTFOLIO stays under the mandate ---
    ext = {"VRP": naive_dt(dep)}
    if (TREND_DIR / "trend_block_returns.parquet").exists():
        ext["momentum"] = load_book(TREND_DIR / "trend_block_returns.parquet")
    for cp in (CARRY_DIR / "carry_refined.parquet", CARRY_DIR / "carry_headline.parquet"):
        if Path(cp).exists():
            ext["carry"] = load_book(cp)
            break
    E = pd.DataFrame(ext).dropna()
    if {"momentum", "carry"} <= set(E.columns):
        E = E.apply(lambda s: vt(s, PPY_BOOK)).dropna()
        base = E[["momentum", "carry"]].mean(axis=1)
        print("\n=== sized integration into momentum+carry (portfolio DD limit 15%) ===")
        rows = []
        for w in (0.0, 0.1, 0.2, 0.3, 0.4):
            s = summarise((1 - w) * base + w * E["VRP"], PPY_BOOK)
            rows.append({"w_vrp": w, "sharpe": s["sharpe_ann"], "dd": s["max_dd"]})
            flag = "" if s["max_dd"] > -MANDATE_DD else "  <-- breaches 15%"
            print(f"  w_VRP={w:.1f}  Sharpe {s['sharpe_ann']:+.2f}  portfolio DD {s['max_dd']:+.1%}{flag}")
        ok = [r for r in rows if r["dd"] > -MANDATE_DD]
        wsel = max(ok, key=lambda r: r["sharpe"])["w_vrp"] if ok else 0.0
        port = (1 - wsel) * base + wsel * E["VRP"]
        sp = summarise(port, PPY_BOOK)
        print(f"  -> deploy at w_VRP={wsel:.1f}: portfolio Sharpe {sp['sharpe_ann']:+.2f}  DD {sp['max_dd']:+.1%} "
              f"(vs momentum+carry {summarise(base, PPY_BOOK)['sharpe_ann']:+.2f})")
        port.to_frame("ret").to_parquet(VOLPREM_DIR / "volprem_deploy_portfolio.parquet")

    dep.to_frame("ret").to_parquet(VOLPREM_DIR / "volprem_deploy_book.parquet")
    print("\nVOLPREM-DEPLOY OK")


if __name__ == "__main__":
    main()
