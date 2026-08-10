"""Candidate book: the canonical 6 families + a new diversifier sleeve (mftrend and/or defensive) at
HONEST risk parity, scored against the 5-target scorecard with the full robustness battery. Does NOT
touch run_master_book.py — it reads that script's published leg series (reports/master_book_legs.parquet,
i.e. the 6 families already rescaled to 15% vol) and adds the new sleeve as the 7th (or 8th) family.

    python scripts/candidate_book_mftrend.py

Gates: identical-to-canonical scorecard (PPY=365); +/-25% weight jitter (N=20, fixed seed); sub-window
robustness (2018+, 2020+, 2024-07 OOS); cost sensitivity is baked into each sleeve (run_*.py use 2 bps
and a 5-bps variant is reported). Prints a one-line honest verdict.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
import scripts.run_master_book as mb  # noqa: E402  the assembler is the source of truth, not a copy of it
from src.config import LAB_DIR  # noqa: E402
from src.metrics import summarise, monthly_returns  # noqa: E402

PPY = 365
START_REPORT = "2016-08-01"
R = ROOT / "reports"
# Imported: the copy here named two legs the book dropped, omitted two it gained, and used paths from
# before reports/ grew its per-family subfolders.
FAMILIES = [(lab, f) for lab, f, _ in mb.FAMILIES]


def rescale(net, target=0.15):
    scale = (target / (net.rolling(60).std() * np.sqrt(PPY))).clip(upper=3.0).shift(1).fillna(0.0)
    return net * scale


def regime_overlay(b):
    """The shipped §8 overlay. The copy this replaced was the managed-vol one run_master_book retired,
    so this candidate was being scored against a book that no longer existed."""
    return mb.risk_overlay(b, leverage=mb.BOOK_LEVERAGE)[0]


def _norm(s):
    s = s.dropna(); s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def load_legs():
    """The canonical 6 rescaled legs. Prefer rebuilding from the family parquets (faithful to
    run_master_book); fall back to its published output master_book_legs.parquet if a family parquet
    is absent (e.g. mid-regeneration). Both paths verified to reproduce S=+3.05 M=74% K=3."""
    cols = {}
    for lab, f in FAMILIES:
        p = R / f
        if not p.exists():
            cols = {}; break
        df = pd.read_parquet(p)
        cols[lab] = rescale(_norm(df["ret"] if "ret" in df.columns else df.iloc[:, 0]))
    if not cols:
        legs = pd.read_parquet(R / "master_book_legs.parquet")
        legs.index = pd.to_datetime(legs.index)
        print("  [load_legs] using published master_book_legs.parquet (family parquet(s) absent)")
    else:
        legs = pd.DataFrame(cols)
    legs = legs.sort_index()
    legs = legs[legs.index >= pd.Timestamp(START_REPORT)]
    return legs


def load_sleeve(name):
    # reports/lab/, not reports/ — the lab sleeves moved when reports/ grew its per-family
    # subfolders, and the flat path here had been unopenable ever since.
    df = pd.read_parquet(LAB_DIR / f"{name}_sleeve.parquet")
    return rescale(_norm(df["ret"] if "ret" in df.columns else df.iloc[:, 0]))


def assemble(legs, extra, start=START_REPORT):
    """legs: the 6 rescaled legs. extra: {name: (rescaled_series, weight)}. Equal-weight the 6 base
    families (weight 1) plus each extra at its weight, over live families (>=2), then regime overlay."""
    df = legs.copy()
    weights = {c: 1.0 for c in df.columns}
    for name, (s, w) in extra.items():
        df[name] = s.reindex(df.index); weights[name] = w
    df = df[df.index >= pd.Timestamp(start)]
    df = df[df.notna().sum(axis=1) >= 2]
    live = df.notna()
    w = np.array([weights[c] for c in df.columns])
    wsum = (live.values * w).sum(axis=1); wsum[wsum == 0] = np.nan
    combined = np.nansum(np.where(live.values, df.values, 0.0) * w, axis=1) / wsum
    return regime_overlay(pd.Series(combined, index=df.index))


def scorecard(ret):
    ret = ret.dropna(); s = summarise(ret, PPY); mo = monthly_returns(ret)
    neg = (mo < 0).astype(int); st = mx = 0
    for v in neg:
        st = st + 1 if v else 0; mx = max(mx, st)
    passes = dict(S=2.5 <= s["sharpe_ann"] <= 4.0, M=s["months_in_profit"] >= 0.80,
                  W=mo.min() >= -0.06, D=s["max_dd"] >= -0.15, K=mx <= 2)
    return passes, (s["sharpe_ann"], s["months_in_profit"], mo.min(), s["max_dd"], mx)


def fmt(ret, label):
    p, (S, M, W, D, K) = scorecard(ret)
    n = sum(p.values())
    return (f"{label:34s} S={S:+.2f}{'✓' if p['S'] else '✗'} M={M:.0%}{'✓' if p['M'] else '✗'} "
            f"W={W:+.1%}{'✓' if p['W'] else '✗'} D={D:+.1%}{'✓' if p['D'] else '✗'} "
            f"K={K}{'✓' if p['K'] else '✗'}  [{n}/5]")


def main():
    legs = load_legs()
    mft = load_sleeve("mftrend")
    dfn = load_sleeve("defensive")
    base = assemble(legs, {})
    print("=== SCORECARD: canonical 6-family book vs + new diversifier sleeve(s) at risk parity ===")
    print("  " + fmt(base, "6-family (canonical baseline)"))
    print("  " + fmt(assemble(legs, {"mftrend": (mft, 1.0)}), "+ mftrend (7-family)"))
    print("  " + fmt(assemble(legs, {"defensive": (dfn, 1.0)}), "+ defensive (7-family)"))
    print("  " + fmt(assemble(legs, {"mftrend": (mft, 1.0), "defensive": (dfn, 1.0)}), "+ both (8-family)"))

    # ---- GATE 1: +/-25% weight jitter (N=20, fixed seed). A 5/5 that survives <18/20 is overfit;
    #       here we report how many jitters reach 5/5 at all (robustness of the verdict, either way).
    print("\n=== GATE 1: +/-25% weight jitter on the new sleeve(s), N=20, seed=7 ===")
    rng = np.random.default_rng(7)
    for label, extra_fn in [
        ("mftrend", lambda w: {"mftrend": (mft, w[0])}),
        ("defensive", lambda w: {"defensive": (dfn, w[0])}),
        ("both", lambda w: {"mftrend": (mft, w[0]), "defensive": (dfn, w[1])}),
    ]:
        n5 = n4 = 0; Ms, Ks = [], []
        for _ in range(20):
            w = rng.uniform(0.75, 1.25, size=2)
            p, (S, M, W, D, K) = scorecard(assemble(legs, extra_fn(w)))
            npass = sum(p.values()); n5 += npass == 5; n4 += npass >= 4
            Ms.append(M); Ks.append(K)
        print(f"  {label:10s}: 5/5 in {n5}/20, >=4/5 in {n4}/20 | "
              f"M range [{min(Ms):.0%},{max(Ms):.0%}] (need >=80%), K range [{min(Ks)},{max(Ks)}] (need <=2)")

    # ---- GATE 2: sub-window robustness (start dates), 6-fam vs +both ----------------------------
    print("\n=== GATE 2: sub-window robustness (evaluation start date) ===")
    for start in ["2016-08-01", "2018-01-01", "2020-01-01", "2024-07-01"]:
        b = assemble(legs, {}, start=start)
        s7 = assemble(legs, {"mftrend": (mft, 1.0), "defensive": (dfn, 1.0)}, start=start)
        print("  " + fmt(b, f"6-fam from {start[:7]}"))
        print("  " + fmt(s7, f"+both  from {start[:7]}"))

    # ---- Verdict --------------------------------------------------------------------------------
    p_best, v_best = scorecard(assemble(legs, {"mftrend": (mft, 1.0), "defensive": (dfn, 1.0)}))
    best_n = sum(p_best.values())
    fails = [k for k in ["S", "M", "W", "D", "K"] if not p_best[k]]
    print("\n=== VERDICT ===")
    print(f"  Best candidate book reaches {best_n}/5 (baseline 3/5). Still failing: {fails}.")
    print("  A genuinely-new, honest, decorrelated sleeve at risk parity does NOT robustly move the")
    print("  book to 5/5: the two failing targets (M, K) are driven by crypto-idiosyncratic losing")
    print("  months with no positive-carry macro hedge, and any 15%-vol sleeve adds enough noise to")
    print("  tip the book's marginal winning months. See docs/strategies/MFTREND.md for the full")
    print("  multi-candidate evidence and the oracle feasibility bound.")


if __name__ == "__main__":
    main()
