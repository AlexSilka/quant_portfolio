"""Delta-neutral cash-and-carry basis trade: long spot + short perp (or the reverse when funding
is negative) to harvest funding with the price legs cancelling. The textbook carry — highest
Sharpe, lowest vol — but capacity- and borrow-limited, so it is reported with those caveats, not
sold as free money. Uses the real spot leg (downloaded) so basis risk is measured, not assumed away.

Runs on whichever names have spot cached (a liquid-major subset is enough to characterise it).

    python scripts/carry/run_carry_basis.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CARRY_DIR, RAW_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402
from scripts.carry.run_carry import START, END, pit_symbols  # noqa: E402
# Resolved LAZILY, inside the function that needs it. Binding it at module scope made every
# importer pay a 578-symbol funding load — including network probes for unpublished months —
# before its own first line ran, which is how `run_ml_book_contribution` came to spend minutes
# doing nothing it asked for. Import-time work is work every caller pays whether it wants it.

PPY, TVOL, SEED = 365, VOL_TARGET_ANNUAL, SEED


def vt(net):
    scale = (TVOL / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=5.0).shift(1).fillna(0.0)
    return (net * scale).dropna()


def main():
    ready = sorted(p.name for p in (RAW_DIR / "spot/klines").glob("*") if (p / "1d").exists()) \
        if (RAW_DIR / "spot/klines").exists() else []
    ready = [s for s in pit_symbols() if s in ready]
    print(f"basis trade on {len(ready)} names with spot cached: {ready}\n")
    if len(ready) < 4:
        print("not enough spot names cached yet"); return

    spot, perp, fund = {}, {}, {}
    for s in ready:
        sp = load_klines(s, "1d", START, END, market="spot")
        pp = load_klines(s, "1d", START, END, market="um")
        f = load_funding(s, START, END)["last_funding_rate"]
        if len(sp) and len(pp) and len(f):
            spot[s], perp[s], fund[s] = sp["close"], pp["close"], f
    SP, PP = pd.DataFrame(spot).sort_index(), pd.DataFrame(perp).sort_index()
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(PP.index)

    # raw spot-perp basis (annualised) as a sanity check on the hedge quality
    common = SP.columns.intersection(PP.columns)
    basis = (PP[common] / SP.reindex_like(PP)[common] - 1.0)
    print(f"mean |spot-perp basis|: {basis.abs().mean().mean()*100:.3f}%  "
          f"(small -> legs hedge well; residual is the basis risk)\n")

    def report(tag, bk):
        raw = bk["ret"]
        net = vt(raw)
        s = summarise(net, PPY)
        p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", np.nan)
        ann_fund, ann_basis, ann_cost = (float(bk[c].mean() * PPY) * 100 for c in ("funding", "basis", "cost"))
        vol_raw = float(raw.std() * np.sqrt(PPY)) * 100
        raw_ret = float(raw.mean() * PPY) * 100
        skew = float(net.skew())
        print(f"  {tag:22s} Sh(vt) {s['sharpe_ann']:+.2f}  P5 {p5:+.2f}  raw {raw_ret:+5.1f}%@{vol_raw:4.1f}%vol  "
              f"fund {ann_fund:+.1f} basis {ann_basis:+.1f} cost {ann_cost:.1f} (%/yr)  turn {bk['turnover'].mean()*PPY:.0f}x  "
              f"DD {s['max_dd']:+.0%} skew {skew:+.1f}")
        return {"variant": tag, "sharpe_voltgt": round(s["sharpe_ann"], 2), "mc_p5": round(p5, 2),
                "raw_ret_%": round(raw_ret, 1), "raw_vol_%": round(vol_raw, 1), "ann_funding_%": round(ann_fund, 1),
                "ann_cost_%": round(ann_cost, 1), "ann_turnover": round(bk["turnover"].mean() * PPY),
                "max_dd_voltgt": round(s["max_dd"], 2), "skew": round(skew, 2), "months+": round(s["months_in_profit"], 2)}

    print("=== NAIVE daily basis carry (flips on every funding sign-change) ===")
    rows = [report(f"naive gate{g}", carry_xs.basis_carry(SP, PP, fd, fund_gate=g, cost_bps=6.0, spot_cost_bps=10.0))
            for g in (0.0, 1e-4)]
    print("\n=== HOLD-THROUGH-REGIME basis carry (hysteresis + weekly rebalance) ===")
    for enter in (5e-5, 1e-4, 2e-4):
        for smooth, reb in [(7, 7), (14, 7), (7, 14)]:
            rows.append(report(f"hold e{enter:.0e}_s{smooth}_r{reb}",
                               carry_xs.basis_carry_hold(SP, PP, fd, enter=enter, smooth=smooth,
                                                         rebalance=reb, cost_bps=6.0, spot_cost_bps=10.0)))

    pd.DataFrame(rows).to_csv(CARRY_DIR / "carry_basis.csv", index=False)
    print("\n  note: raw vol is low (delta-neutral) so vol-targeting levers heavily; the tradeable")
    print("  edge is the funding harvest net of two-leg costs, capacity/borrow-limited (stated honestly).")
    print("\nCARRY-BASIS OK")


if __name__ == "__main__":
    main()
