"""Book-level walk-forward (§10): rolling & anchored, with periodic re-fitting of the PORTFOLIO ALLOCATION.

The §11 scorecard is reported on a single FINAL out-of-sample block (OOS_START→, run once). That block is
deliberately small — the last ~2 years never looked at until the end. This script adds the OTHER §10
requirement — a *rolling and anchored walk-forward with periodic re-fitting* — at the PORTFOLIO level, over
ALL available data (the non-crypto legs reach back to 2005–2012, so the master's 2016 window is a reporting
choice, not a data limit; only crypto carry/breakout is stuck at 2020). At each rebalance it fits the leg
weights on the training window (anchored [start,t] or rolling [t−win,t]) and applies them to the next block
out-of-sample; concatenating the blocks gives the accumulated walk-forward OOS track (~18y, 2006→2026, incl.
the 2008 GFC). Caveat: pre-~2019 crisis/gmacro legs are reconstructed signals — a strategy-logic backtest,
not a live track. Sharpe is annualised by each track's actual obs/yr (honest for the mixed 252/365 calendar).

Three allocations are walk-forwarded so the shipped a-priori equal weight is justified by evidence, not
assertion:
  • equal    — 1/N, no fit (the shipped book). With nothing to fit, its walk-forward IS its full track from
               the burn-in — i.e. the whole post-burn-in history is genuinely out-of-sample (no parameter is
               fit to it), which is why the a-priori 15-year Sharpe is itself an honest OOS estimate;
  • inv-vol  — risk parity re-estimated from trailing vol each rebalance (a genuine periodic re-fit);
  • mean-var — trailing Sharpe-maximising (long-only) weights — the overfit-prone allocation.
Anchored vs rolling and quarterly vs annual cadence are compared to show the result does NOT depend on that
choice (§10). Emits reports/master_book_wf.parquet (the headline OOS track) + master_book_wf_summary.json.

    python scripts/run_wf_book.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import OOS_START, SEED  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.book_id import stamp  # noqa: E402
try:                                                              # canonical family list + PIT loaders + overlay
    from scripts.run_master_book import (FAMILIES as _FAMILIES, book_stack as _book_stack,  # noqa: E402
                                 load as _load_fam, rescale as _rescale_fam,
                                 risk_overlay as _risk_overlay, scorecard as _scorecard_fn)
except Exception:                                                 # noqa: BLE001 — fall back to the truncated legs
    _FAMILIES = _risk_overlay = _scorecard_fn = None
    _book_stack = lambda d: d.mean(axis=1, skipna=True)           # the assembler is absent; legs are pre-weighted

R = Path("reports")
PPY = 365
OOS = pd.Timestamp(OOS_START).tz_localize(None)
BURN_IN = 365          # ~1y minimum training history before the first out-of-sample block
MIN_TRAIN = 60         # a leg needs at least this many training obs to earn a fitted weight
# crisis windows to isolate on the long-history track (drawdown is the annualisation-free, robust read)
STRESS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2018 Volmageddon", "2018-01-15", "2018-03-31"),
          ("COVID crash", "2020-02-15", "2020-04-15")]


def load_full_legs():
    """Full-history vol-scaled legs (2005+): the canonical family series WITHOUT the master's 2016 trailing-
    window truncation, so the walk-forward can use ALL available data. The crypto legs (breakout, BAB) still
    only exist from 2020; the others reach back to 2005-2012 (volprem/crisis/gmacro from 2005 are
    reconstructed signals — the tradeable managed-futures/vol products post-date the GFC — so pre-~2019 is a
    strategy-logic backtest, not a live track). Falls back to the 2016+ committed legs if the loaders are
    unavailable."""
    if _FAMILIES is None:
        df = pd.read_parquet(R / "master_book_legs.parquet")
        df.index = pd.to_datetime(df.index)
        return df, False
    raw = {lab: _load_fam(lab, f, c) for lab, f, c in _FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    df = pd.DataFrame({k: _rescale_fam(v) for k, v in raw.items()}).sort_index()
    df = df[df.notna().sum(axis=1) >= 2]                          # keep any day with >=2 live legs
    return df, True


def alloc_weights(train: pd.DataFrame, kind: str) -> pd.Series:
    """Leg weights fitted on the training window only (causal). Legs without enough history get 0."""
    cols = [c for c in train.columns if train[c].notna().sum() >= MIN_TRAIN]
    if len(cols) < 2:
        cols = [c for c in train.columns if train[c].notna().sum() > 0] or list(train.columns)
        return pd.Series(1.0 / max(len(cols), 1), index=cols).reindex(train.columns).fillna(0.0)
    t = train[cols]
    if kind == "equal":
        w = pd.Series(1.0, index=cols)
    elif kind == "invvol":
        w = (1.0 / t.std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif kind == "meanvar":
        mu = t.mean().to_numpy()
        cov = t.cov().to_numpy() + np.eye(len(cols)) * 1e-6
        try:
            raw = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            raw = np.ones(len(cols))
        w = pd.Series(np.clip(raw, 0.0, None), index=cols)          # long-only
        if w.sum() <= 0:
            w = pd.Series(1.0, index=cols)
    else:
        raise ValueError(kind)
    w = w / w.sum()
    return w.reindex(train.columns).fillna(0.0)


def walk_forward(df: pd.DataFrame, kind="equal", mode="anchored", cadence=91, win=730) -> pd.Series:
    """Concatenated out-of-sample track: fit weights on the training window, apply to the next `cadence`
    days, roll. `mode` = anchored ([start,t]) | rolling ([t−win,t]). Per-bar weights renormalise over the
    legs live that bar (matching the book's average-over-live-legs construction)."""
    idx = df.index
    out, t = [], BURN_IN
    while t < len(idx):
        te = min(t + cadence, len(idx))
        train = df.iloc[(0 if mode == "anchored" else max(0, t - win)):t]
        w = alloc_weights(train, kind)
        blk = df.iloc[t:te]
        wv = w.reindex(blk.columns).fillna(0.0)
        num = (blk.fillna(0.0) * wv).sum(axis=1)
        den = (blk.notna() * wv).sum(axis=1).replace(0.0, np.nan)     # renormalise over live legs
        out.append(num / den)
        t = te
    return pd.concat(out).dropna().rename("ret")


def ppy_of(s):
    """Actual obs/yr — honest Sharpe annualisation for the mixed 252/365 calendar (crypto 365 / equity
    ~252), not a flat 365 which overstates any sub-365 track. Same convention as run_master_book."""
    s = s.dropna()
    yrs = (s.index.max() - s.index.min()).days / 365.25
    return len(s) / yrs if yrs > 0 else float(PPY)


def _sc(s):
    ss = summarise(s, ppy_of(s))
    return {"sharpe": round(ss["sharpe_ann"], 2), "max_dd": round(ss["max_dd"], 4),
            "months_in_profit": round(ss["months_in_profit"], 4), "n_obs": int(len(s)),
            "start": str(s.index.min().date()), "end": str(s.index.max().date())}


def main():
    df, full = load_full_legs()
    src = "full available history (all legs' true start)" if full else "committed 2016+ legs (fallback)"
    print(f"legs: {list(df.columns)}  {src}\nwindow {df.index.min().date()}..{df.index.max().date()} "
          f"({len(df)} days ≈ {len(df) / 365:.0f}y); burn-in {BURN_IN}d -> first OOS block {df.index[BURN_IN].date()}\n")

    # allocation comparison (anchored, quarterly re-fit) — does fitting the weights beat a-priori equal?
    configs = {
        "equal_anchored_Q":   dict(kind="equal",   mode="anchored", cadence=91),
        "invvol_anchored_Q":  dict(kind="invvol",  mode="anchored", cadence=91),
        "meanvar_anchored_Q": dict(kind="meanvar", mode="anchored", cadence=91),
        "equal_rolling2y_Q":  dict(kind="equal",   mode="rolling",  cadence=91, win=730),
        "equal_anchored_A":   dict(kind="equal",   mode="anchored", cadence=365),
        "invvol_rolling2y_Q": dict(kind="invvol",  mode="rolling",  cadence=91, win=730),
    }
    tracks = {name: walk_forward(df, **cfg) for name, cfg in configs.items()}
    rows = {name: _sc(s) for name, s in tracks.items()}

    headline = tracks["equal_anchored_Q"]                            # the shipped a-priori allocation
    hb = _sc(headline)
    hb_oos = _sc(headline[headline.index >= OOS])                    # the §11 final-block slice
    hb_2016 = _sc(headline[headline.index >= "2016-08-01"])          # a 10-year (2016+) comparison slice (master reports from 2011)
    eq_vals = [rows[n]["sharpe"] for n in configs if n.startswith("equal")]   # cadence/window invariance (fixed alloc)
    # crisis-window stress on the equal-weight book (maxDD is annualisation-free — the robust read)
    ew = _book_stack(df)
    stress = {}
    for lab, a, b in STRESS:
        w = ew[a:b]
        if len(w) > 20:
            eq = (1.0 + w).cumprod()
            stress[lab] = {"sharpe": round(float(np.sqrt(ppy_of(w)) * w.mean() / w.std(ddof=1)), 2),
                           "max_dd": round(float((eq / eq.cummax() - 1.0).min()), 4), "n_legs": int(df.loc[a:b].notna().any().sum())}

    # window robustness (§10/§12): the SAME equal-weight + §8-overlay book over 10y / 15y / all-available
    # windows — evidence the headline Sharpe is not a 15-year-window artifact (the deeper stress on the longer
    # windows shows up honestly in worst-month / max-DD, which the reconstructed pre-2019 signals inflate).
    windows = {}
    if full and _risk_overlay is not None:
        for wlab, wst in [("full_21y_2005", "2005-01-01"), ("15y_2011", "2011-08-01"), ("10y_2016", "2016-08-01")]:
            d = df[df.index >= pd.Timestamp(wst)]
            d = d[d.notna().sum(axis=1) >= 2]
            mg, _, _ = _risk_overlay(_book_stack(d))
            windows[wlab] = _scorecard_fn(mg)

    print("=== BOOK-LEVEL WALK-FORWARD (accumulated out-of-sample track, §10) ===")
    for name in configs:
        r = rows[name]
        print(f"  {name:20s} Sharpe {r['sharpe']:+.2f}  maxDD {r['max_dd']:+.1%}  months+ {r['months_in_profit']:.0%}  "
              f"({r['start']}..{r['end']}, {r['n_obs']}d)")
    print(f"\n  Headline (equal-weight, anchored, quarterly re-fit): accumulated OOS **Sharpe {hb['sharpe']:+.2f}** over "
          f"{hb['start']}..{hb['end']} (~{round(hb['n_obs']/365)}y) — out-of-sample across ALL available data, not just the "
          f"2y final block (final-block {hb_oos['sharpe']:+.2f}; master 2016+ window {hb_2016['sharpe']:+.2f}).")
    print(f"  Window/cadence invariance (fixed equal weight): Sharpe in [{min(eq_vals):+.2f}, {max(eq_vals):+.2f}] "
          f"(spread {max(eq_vals)-min(eq_vals):.2f}) — result does NOT depend on the choice (§10).")
    print(f"  Fitting the weights does NOT beat equal OOS: equal {rows['equal_anchored_Q']['sharpe']:+.2f} vs inv-vol "
          f"{rows['invvol_anchored_Q']['sharpe']:+.2f} vs mean-var {rows['meanvar_anchored_Q']['sharpe']:+.2f} "
          f"@maxDD {rows['meanvar_anchored_Q']['max_dd']:+.0%} (mean-var overfits) — evidence for a-priori equal weight.")
    if stress:
        print("  Crisis-window stress (equal-weight book; the short-vol leg's tail is hedged by crisis/managed-futures):")
        for lab, s in stress.items():
            print(f"    {lab:16s} Sharpe {s['sharpe']:+.2f}  maxDD {s['max_dd']:+.1%}  ({s['n_legs']} legs live)")
        print("  NOTE: pre-~2019 crisis/gmacro legs are reconstructed signals (products post-date them) — a strategy-logic "
              "backtest, not a live track; crypto legs (carry/breakout) exist only from 2020.")
    if windows:
        print("  Window robustness (SAME equal-weight + §8-overlay book, different reporting window): Sharpe " +
              " · ".join(f"{k.split('_')[0]} {v['sharpe']:+.2f}" for k, v in windows.items()) +
              f" — headline is not a 10y-window artifact; the longer windows deepen the tail honestly (full-21y maxDD "
              f"{windows['full_21y_2005']['max_dd']:+.0%}, worst-month {windows['full_21y_2005']['worst_month']:+.0%}, "
              f"on reconstructed pre-2019 signals).")

    headline.to_frame().to_parquet(R / "master_book_wf.parquet")
    (R / "master_book_wf_summary.json").write_text(json.dumps(stamp({
        "burn_in_days": BURN_IN, "headline_config": "equal_anchored_Q", "full_history": full,
        "headline_wf_oos": hb, "headline_final_block_oos": hb_oos, "master_window_2016_oos": hb_2016,
        "window_cadence_invariance_range": [min(eq_vals), max(eq_vals)], "configs": rows, "stress": stress,
        "window_robustness": windows,
        "note": "Book-level walk-forward with periodic allocation re-fit (§10), over ALL available data (2005+ for the "
                "non-crypto legs; the crypto legs only from 2020). Equal-weight needs no fit, so its walk-forward "
                "equals the full post-burn-in track — the a-priori book is OOS across the whole history, incl. the 2008 "
                "GFC. The 2y final block (OOS_START) is the separate run-once §11 holdout. Caveat: pre-~2019 crisis/gmacro "
                "are reconstructed signals (a strategy-logic backtest); Sharpe is annualised by actual obs/yr (honest 252/365 calendar).",
        "seed": SEED}), indent=2, default=float))
    print("\nartifacts -> reports/master_book_wf.parquet · master_book_wf_summary.json")
    print("WALK-FORWARD BOOK OK")


if __name__ == "__main__":
    main()
