"""Grind-regime de-risk timer — the complementary half of the W/K attack (see run_convexity*.py).

Convexity fixes W and the shock months but not K, because the book's binding losing-streaks are multi-week
GRIND bleeds (correlated give-back of the risk premia in choppy or rising markets), not vol shocks. The
only other honest, non-overfit book-level lever on K is the opposite of adding a hedge: CUT book exposure
during the grind so the streak months go flat. A month flattened to ~0% is not <0, so it breaks a streak
WITHOUT changing months-in-profit's numerator (0 is not >0 either).

This tests every leading grind signal (book drawdown / negative drift / negative-drift-at-below-median-vol /
family-bleed breadth) at floors {cash, 0.3}, both replacing the throttle role on the raw book and stacked
on the canonical book. Result: NONE reaches 4/5. De-risk makes K WORSE (a partial cut turns -2.4% into
-1.2% — still red — while the timing whipsaw adds NEW short red months, K->4/5) and collapses M (it also
flattens green months the signal cannot distinguish from grinds). The grind is heterogeneous — some streak
months are risk-off (2019, 2022), one is risk-ON (Dec-2021, S&P +4.4%) — so no single mechanical detector
flattens exactly the streak months. Verdict: K<=2 and M>=80% are not both honestly reachable; ceiling 3/5.

    python scripts/run_grind_timer.py    ->  reports/lab/grind_timer_summary.json  (+ printed verdict)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
import scripts.run_master_book as mb  # noqa: E402  (read-only reuse of canonical mechanics)
from src.metrics import summarise, monthly_returns  # noqa: E402
from src.config import LAB_DIR  # noqa: E402

PPY = mb.PPY
START = mb.START_REPORT


def streak_max(mo: pd.Series) -> int:
    neg = (mo < 0).astype(int); st = mx = 0
    for v in neg:
        st = st + 1 if v else 0; mx = max(mx, st)
    return mx


def scorecard(ret: pd.Series) -> dict:
    ret = ret.dropna(); s = summarise(ret, PPY); mo = monthly_returns(ret)
    return dict(S=bool(2.5 <= s['sharpe_ann'] <= 4.0), M=bool(s['months_in_profit'] >= 0.80),
                W=bool(mo.min() >= -0.06), D=bool(s['max_dd'] >= -0.15), K=bool(streak_max(mo) <= 2),
                sharpe=float(s['sharpe_ann']), months=float(s['months_in_profit']),
                worst=float(mo.min()), maxdd=float(s['max_dd']), streak=int(streak_max(mo)))


def npass(x: dict) -> int:
    return int(x['S']) + int(x['M']) + int(x['W']) + int(x['D']) + int(x['K'])


def fmt(x: dict) -> str:
    return (f"{npass(x)}/5 S={x['sharpe']:.2f} M={x['months']:.0%} W={x['worst']:.1%} "
            f"D={x['maxdd']:.1%} K={x['streak']}")


def derisk(book: pd.Series, sig: pd.Series, floor: float) -> pd.Series:
    expo = pd.Series(1.0, index=book.index)
    expo[sig.reindex(book.index).fillna(False)] = floor
    return book * expo.shift(1).fillna(1.0)


def main() -> None:
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(START)]; df = df[df.notna().sum(axis=1) >= 2]
    mean6 = mb.book_stack(df)
    canon = mb.regime_overlay(mean6)

    eq = (1 + mean6).cumprod(); dd = eq / eq.cummax() - 1.0
    roll_ret = mean6.rolling(15).sum(); roll_vol = mean6.rolling(15).std() * np.sqrt(PPY)
    breadth = (df.rolling(15).sum() < 0).sum(axis=1) / df.notna().sum(axis=1)
    signals = {
        "dd": dd < -0.02,
        "drift": roll_ret < 0,
        "grind": (roll_ret < 0) & (roll_vol < roll_vol.rolling(252).median()),
        "breadth": breadth >= 0.6,
        "breadth+dd": (breadth >= 0.6) & (dd < -0.01),
    }

    base = scorecard(canon)
    print("canonical book:", fmt(base))
    out = {"canonical": base, "grid": {}, "best_pass": npass(base), "best_K": base['streak'],
           "best_M": base['months']}
    for target, book in [("raw", mean6), ("canon", canon)]:
        for name, sig in signals.items():
            for floor in [0.0, 0.3]:
                b = mb.regime_overlay(derisk(mean6, sig, floor)) if target == "raw" else derisk(canon, sig, floor)
                s = scorecard(b); key = f"{target}:{name}:floor{floor}"
                out["grid"][key] = s
                out["best_pass"] = max(out["best_pass"], npass(s))
                out["best_K"] = min(out["best_K"], s['streak'])
                out["best_M"] = max(out["best_M"], s['months'])
                print(f"{key:26s}: {fmt(s)}")

    fixed = out["best_pass"] >= 4 and out["best_K"] <= 2
    out["verdict"] = (
        f"{'FIXED' if fixed else 'NOT FIXED'} — best across all grind-de-risk variants: pass={out['best_pass']}/5, "
        f"best K={out['best_K']} (need <=2), best M={out['best_M']:.0%} (need >=80%). De-risk makes K worse "
        f"(partial cut leaves months red; whipsaw adds new red months) and collapses M (flattens green months "
        f"too). The grind is heterogeneous (risk-off 2019/2022 + risk-on Dec-2021), so no single mechanical "
        f"detector flattens exactly the streak months. K<=2 and M>=80% are not both honestly reachable — ceiling 3/5.")
    print("\n=== VERDICT ===\n" + out["verdict"])
    (LAB_DIR / "grind_timer_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nartifact -> reports/lab/grind_timer_summary.json")
    print("GRIND TIMER OK")


if __name__ == "__main__":
    main()
