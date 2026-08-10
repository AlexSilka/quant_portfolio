"""Marginal contribution of the breakout squeeze to the multi-family portfolio (READ-ONLY — reads
each family's published headline series, writes nothing the parallel sessions own).

Families (each family's current honest headline; the capped fake-Sharpe VRP column is deliberately
avoided): trend/momentum (trend_block_returns), carry (carry_breadth_headline), vol-premium (VRP baseline short),
cross-sectional momentum (crypto-50), and the breakout squeeze (bo_combined). Every series is
re-scaled to a common ~15% vol on trailing (lagged) vol so combining is risk-parity, then:
  - the cross-family correlation matrix (does breakout diversify, or duplicate a leg?),
  - portfolio WITHOUT vs WITH breakout, and the marginal-contribution curve (where it flattens),
  - breakout's share of P&L and the portfolio with the top contributor removed (§7).

    python scripts/breakout/run_bo_contribution.py
"""
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src import bo_common as bo  # noqa: E402
import scripts.run_master_book as mb  # noqa: E402  the assembler is the source of truth, not a copy
from src.config import BREAKOUT_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

PPY = 365
R = bo.REPORTS          # the family series live under reports/<family>/, not under this family's own
                        # sub-book — joining BREAKOUT with them looked for reports/breakout/trend/...

# (label, file, column) — the SAME canonical honest series as scripts/run_master_book.py (kept in
# sync so this breakout-local diagnostic matches the master book; run_master_book is the source of truth)
# Imported, not copied. The pasted list said it was "kept in sync" with run_master_book and was not:
# it named trend and carry, dropped from the book, and had never heard of crisis, global-macro or BAB.
FAMILIES = list(mb.FAMILIES)


# load / rescale are the assembler's own. The local copies dropped the timezone normalisation it does,
# so the moment this stopped reading only breakout-family files it could not join a tz-aware leg to a
# tz-naive one — a copy diverges by what it forgets, not only by what it changes.
load, rescale = mb.load, mb.rescale


def port(rets_df):
    return rets_df.mean(axis=1)          # equal risk (all re-scaled to common vol)


def main():
    raw = {lab: load(lab, f, c) for lab, f, c in FAMILIES}
    df = pd.DataFrame({k: rescale(v) for k, v in raw.items()}).dropna(how="all")
    df = df.loc[df.dropna().index.min():]            # common overlap window (volprem starts 2021)
    df = df.dropna()
    print(f"common window: {df.index.min().date()}..{df.index.max().date()}  ({len(df)} days)\n")

    print("standalone Sharpe (re-scaled to 15% vol):")
    solo = {c: summarise(df[c], PPY)["sharpe_ann"] for c in df.columns}
    for c, s in sorted(solo.items(), key=lambda kv: -kv[1]):
        print(f"    {c:16s} {s:+.2f}")

    print("\ncorrelation of breakout to each family:")
    for c in df.columns:
        if c != "breakout":
            print(f"    breakout vs {c:16s} {df['breakout'].corr(df[c]):+.2f}")

    others = [c for c in df.columns if c != "breakout"]
    p_wo, p_w = port(df[others]), port(df)
    s_wo, s_w = summarise(p_wo, PPY), summarise(p_w, PPY)
    mc_wo, mc_w = bootstrap_sharpe(p_wo, PPY, 2000, bo.SEED), bootstrap_sharpe(p_w, PPY, 2000, bo.SEED)
    print(f"\nportfolio WITHOUT breakout: Sharpe {s_wo['sharpe_ann']:+.2f}  maxDD {s_wo['max_dd']:+.1%}  "
          f"months+ {s_wo['months_in_profit']:.0%}  MC-P5 {mc_wo.get('sharpe_p5', float('nan')):+.2f}")
    print(f"portfolio WITH breakout   : Sharpe {s_w['sharpe_ann']:+.2f}  maxDD {s_w['max_dd']:+.1%}  "
          f"months+ {s_w['months_in_profit']:.0%}  MC-P5 {mc_w.get('sharpe_p5', float('nan')):+.2f}")
    print(f"-> breakout marginal: {s_w['sharpe_ann'] - s_wo['sharpe_ann']:+.2f} Sharpe, "
          f"{s_w['max_dd'] - s_wo['max_dd']:+.1%} maxDD")

    # marginal-contribution curve: add families in order of standalone Sharpe
    order = sorted(df.columns, key=lambda c: -solo[c])
    print("\nmarginal-contribution curve (added in Sharpe order):")
    marg = []
    for k in range(1, len(order) + 1):
        pk = port(df[order[:k]])
        sk = summarise(pk, PPY)["sharpe_ann"]
        marg.append({"n": k, "added": order[k - 1], "sharpe": sk})
        print(f"    +{order[k-1]:16s} -> {k} families, Sharpe {sk:+.2f}")

    # breakout's share of P&L + portfolio with the top contributor removed (§7)
    contrib = {c: float(df[c].mean()) for c in df.columns}
    tot = sum(v for v in contrib.values() if v > 0)
    share = contrib["breakout"] / tot if tot > 0 else float("nan")
    top = max(solo, key=solo.get)
    p_notop = port(df[[c for c in df.columns if c != top]])
    print(f"\nbreakout share of gross P&L: {share:+.0%}")
    print(f"portfolio with top contributor ({top}) removed: Sharpe {summarise(p_notop, PPY)['sharpe_ann']:+.2f} "
          f"(vs {s_w['sharpe_ann']:+.2f} full)")

    pd.DataFrame(marg).to_csv(BREAKOUT_DIR / "bo_contribution_marginal.csv", index=False)
    df.corr().to_csv(BREAKOUT_DIR / "bo_contribution_corr.csv")
    (BREAKOUT_DIR / "bo_contribution_summary.json").write_text(json.dumps({
        "window": [str(df.index.min().date()), str(df.index.max().date())],
        "standalone_sharpe": solo, "without_breakout": s_wo, "with_breakout": s_w,
        "breakout_marginal_sharpe": s_w["sharpe_ann"] - s_wo["sharpe_ann"],
        "breakout_pnl_share": share, "corr_to_breakout": {c: float(df["breakout"].corr(df[c]))
                                                          for c in others}}, indent=2, default=float))
    print("\nBO CONTRIBUTION OK")


if __name__ == "__main__":
    main()
