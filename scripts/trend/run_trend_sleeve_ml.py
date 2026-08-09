"""Per-SLEEVE ML decisions inside the trend leg — is there a pocket where the gate pays, findable
without hindsight?

`run_trend_ml.py` answers "does the gate help the trend book" and finds it does not (best OOS +0.05
against an ungated +0.35). But a book average can hide a real, heterogeneous effect: on the fast
sleeves the same gate lifts SOL-1h (+1.03 -> +1.44, DD -15% -> -6%) and *breaks* ETH-1h (+0.23 ->
-0.09). So the open question is not "gate or not" but "gate WHICH sleeve", and that is the cut with
the highest overfitting risk in the whole project — 30 sleeves against a ~26-month final block.

Three arms, so the honest one is not flattered by the dishonest one:
  * gate NONE      — the shipped construction;
  * gate ALL       — a-priori, no selection (the anti-cherry-pick arm);
  * gate SELECTED  — per sleeve, chosen by an EXPANDING walk-forward: at each annual decision date a
                     sleeve is gated over the next year only if gating beat its own baseline on data
                     strictly before that date. No sleeve is ever picked using its own future.
  * gate ORACLE    — chosen on the full sample. NOT a result — it is the ceiling that says how much of
                     any "improvement" is hindsight, and the gap to SELECTED is the honest discount.

Judged on all five §11 targets AND on return, because a gate that cuts time-in-market can hold Sharpe
while cutting the money made (§5d).

    python scripts/trend/run_trend_sleeve_ml.py  ->  reports/trend/trend_sleeve_ml.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import OOS_START, TREND_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from scripts.trend.run_trend_ml import _daily, precompute, proba_cache  # noqa: E402
from src.pipeline import model_factory  # noqa: E402
from src.backtest.engine import positions_from_events  # noqa: E402

PPY = 365
OOS = pd.Timestamp(OOS_START).tz_localize(None)
DECIDE_EVERY = "YE"          # annual re-decision — a trend sleeve needs a year of trades to judge
THRESHOLD = 0.55             # the shipped gate cut, same as run_trend_ml's headline arm


def _naive(ix):
    ix = pd.DatetimeIndex(ix)
    return ix.tz_convert("UTC").tz_localize(None) if ix.tz is not None else ix


def sleeve_series(sleeves, proba, keys, threshold=THRESHOLD):
    """Daily returns per sleeve, gated for `keys` and raw for the rest."""
    out = {}
    for key, s in sleeves.items():
        if key in keys:
            p = proba[key]
            kept = p.index[p.values >= threshold]
            pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"], kept)
        else:
            pos = positions_from_events(s["px"].index, s["trades"]["side"], s["trades"]["t1"],
                                        s["trades"].index)
        r = _daily(s["px"], pos, s["fund"], s["adv"], s["tf"])
        r.index = _naive(r.index)
        out[key] = r
    return pd.DataFrame(out)


def book(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=1, skipna=True).dropna()


def card(s: pd.Series) -> dict:
    s = s.dropna()
    sc = summarise(s, PPY)
    m = (1 + s).resample("ME").prod() - 1
    neg = (m <= 0).astype(int).to_numpy()
    streak = mx = 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return {"sharpe": round(sc["sharpe_ann"], 2),
            "cagr": round(float((1 + s).prod() ** (1 / yrs) - 1), 3) if yrs > 0 else 0.0,
            "max_dd": round(sc["max_dd"], 3), "worst_month": round(float(m.min()), 3),
            "months_in_profit": round(float((m > 0).mean()), 3), "streak": int(mx)}


def walk_forward_keys(raw: pd.DataFrame, gat: pd.DataFrame) -> pd.Series:
    """Per-bar set of gated sleeves, decided only from the past.

    At each annual decision date, a sleeve is gated for the following year iff its gated series beat
    its raw series on [start, date) by Sharpe. The first period gates nothing (no history to judge)."""
    idx = raw.index
    dates = pd.date_range(idx.min(), idx.max(), freq=DECIDE_EVERY)
    chosen = pd.Series([frozenset()] * len(idx), index=idx)
    for i, d in enumerate(dates):
        hist = idx < d
        if hist.sum() < 365:
            continue
        keys = frozenset(
            k for k in raw.columns
            if summarise(gat[k][hist].dropna(), PPY)["sharpe_ann"]
            > summarise(raw[k][hist].dropna(), PPY)["sharpe_ann"])
        nxt = (idx >= d) & (idx < (dates[i + 1] if i + 1 < len(dates) else idx.max() + pd.Timedelta(days=1)))
        chosen[nxt] = [keys] * int(nxt.sum())
    return chosen


def blend(raw: pd.DataFrame, gat: pd.DataFrame, chosen: pd.Series) -> pd.Series:
    """Book where each bar takes the gated series for that bar's chosen sleeves, raw for the rest."""
    mask = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    for k in raw.columns:
        mask[k] = [k in s for s in chosen]
    mixed = gat.where(mask, raw)
    return book(mixed)


def main():
    print("=== per-SLEEVE ML decisions in the trend leg — does 'gate WHICH' beat 'gate none'? ===\n")
    sleeves = precompute()
    proba = proba_cache(sleeves, model_factory, weighted=True)   # the uniqueness-weighted
    # variant is the strongest gate run_trend_ml found, so the per-sleeve question is asked of it
    print(f"sleeves: {len(sleeves)}  ({', '.join(sorted({s['tf'] for s in sleeves.values()}))})\n")

    raw = sleeve_series(sleeves, proba, keys=set())
    gat = sleeve_series(sleeves, proba, keys=set(raw.columns))
    common = raw.index.intersection(gat.index)
    raw, gat = raw.loc[common], gat.loc[common]

    per = []
    for k in raw.columns:
        rb, gb = summarise(raw[k].dropna(), PPY), summarise(gat[k].dropna(), PPY)
        per.append({"sleeve": k, "sharpe_raw": round(rb["sharpe_ann"], 2), "sharpe_gated": round(gb["sharpe_ann"], 2),
                    "dd_raw": round(rb["max_dd"], 3), "dd_gated": round(gb["max_dd"], 3)})
    pf = pd.DataFrame(per).sort_values("sharpe_gated", ascending=False)
    helped = int((pf.sharpe_gated > pf.sharpe_raw).sum())
    print(f"per-sleeve: the gate helps {helped}/{len(pf)} sleeves in-sample\n")
    print(pf.to_string(index=False))

    arms = {"gate NONE (shipped)": book(raw), "gate ALL (a-priori)": book(gat)}
    chosen = walk_forward_keys(raw, gat)
    arms["gate SELECTED (walk-forward)"] = blend(raw, gat, chosen)
    oracle = frozenset(pf.sleeve[pf.sharpe_gated > pf.sharpe_raw])
    arms["gate ORACLE (hindsight, not a result)"] = book(gat[list(oracle)].join(raw[[c for c in raw.columns if c not in oracle]]))

    out = {}
    print("\n=== book-level, all five targets AND return ===")
    for tag, s in arms.items():
        cf, co = card(s), card(s[s.index >= OOS])
        out[tag] = {"full": cf, "oos": co}
        print(f"  {tag:38s} FULL Sh {cf['sharpe']:+.2f} CAGR {cf['cagr']:+.0%} DD {cf['max_dd']:+.1%} "
              f"worst {cf['worst_month']:+.1%} mo {cf['months_in_profit']:.0%} strk {cf['streak']}")
        print(f"  {'':38s}  OOS Sh {co['sharpe']:+.2f} CAGR {co['cagr']:+.0%} DD {co['max_dd']:+.1%} "
              f"worst {co['worst_month']:+.1%} mo {co['months_in_profit']:.0%} strk {co['streak']}")

    out["per_sleeve"] = pf.to_dict("records")
    out["walk_forward_avg_gated"] = round(float(np.mean([len(s) for s in chosen])), 2)
    out["oracle_gated"] = sorted(oracle)
    (TREND_DIR / "trend_sleeve_ml.json").write_text(json.dumps(out, indent=2))
    print(f"\nwalk-forward gates {out['walk_forward_avg_gated']:.1f} of {len(raw.columns)} sleeves on average")
    print(f"wrote {TREND_DIR / 'trend_sleeve_ml.json'}")


if __name__ == "__main__":
    main()
