"""Gate lab for the short-vol book — can the leg's drawdown be cut further than the shipped gate?

The shipped leg carries ONE timing rule: flat when the VIX curve is in backwardation (VIX3M/VIX < 1,
`src/risk/vol_regime.py`). This script asks whether a better rule exists, and answers it the only way
that is not curve-fitting: run every candidate through the SAME book, score all of them on the same
windows, and hold each against two null controls that a lucky rule must beat --

  * random gate  — same average exposure, dates shuffled in blocks (500 draws): does the TIMING matter,
    or would any rule that is flat 8% of the time have done this?
  * constant     — the same average exposure applied every day: is this de-risking rather than timing?

A candidate only counts if it beats the block-random distribution AND holds on the frozen OOS block.
Everything is causal (decided on the prior close, `shift(1)`); no candidate sees its own bar.

    python scripts/volprem/run_vol_premium_gates.py   ->  reports/volprem/volprem_gates.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[2]
from src.config import OOS_START, RAW_DIR, SEED, VOLPREM_DIR  # noqa: E402
from src.data.cboe import load_cboe_vol  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves.vol_premium import realized_vol  # noqa: E402

# reuse the shipped book's own universe / sleeve builder so the candidates are measured on the
# identical legs the deliverable uses — no re-implementation to drift from it
from scripts.volprem.run_vol_premium_book import (  # noqa: E402
    COST_BY_CLASS, PPY_BOOK, UNIVERSE, book_from, implied, naive_dt, sleeve, underlying_bars,
)
from src.config import VOL_TARGET_ANNUAL  # noqa: E402
from src.sleeves import vol_premium as vp  # noqa: E402

WINDOW_START = "2011-01-01"      # master-book window; VIX9D only exists from 2011-01-04
N_RANDOM = 500


# ----------------------------------------------------------------------------- signals
def _idx_series(sym: str, index) -> pd.Series:
    """A Cboe index aligned to `index`, forward-filled, information at t."""
    s = naive_dt(load_cboe_vol(sym))
    return s.reindex(pd.DatetimeIndex(index).normalize()).ffill()


def term_structure(index, near: str, far: str) -> pd.Series:
    """far/near ratio (>1 = contango over that segment)."""
    return _idx_series(far, index) / _idx_series(near, index)


def spx_vrp(index, lookback: int = 20) -> pd.Series:
    """VIX minus trailing realised vol of SPY, in vol points — the premium cushion, information at t."""
    spy = pd.read_parquet(RAW_DIR / "equity_td" / "SPY_1d.parquet")["close"]
    spy = naive_dt(spy)
    rv = realized_vol(spy, lookback=lookback, ppy=252) * 100.0
    nidx = pd.DatetimeIndex(index).normalize()
    return _idx_series("VIX", index) - rv.reindex(nidx).ffill()


def causal(sig: pd.Series, index) -> pd.Series:
    """Shift a decision variable one bar and restore the caller's index (never sees its own bar)."""
    out = sig.shift(1).fillna(1.0).clip(0.0, 1.0)
    out.index = pd.Index(index)
    return out


def persist(gate: pd.Series, k: int) -> pd.Series:
    """Re-enter only after `k` consecutive open bars — exits stay immediate (asymmetric hysteresis).

    Backwardation is a stress signal, not a return-to-normal signal: the curve flickers back to contango
    mid-crash. Requiring persistence on the way IN keeps the leg flat through that flicker."""
    return (gate.rolling(k, min_periods=k).min()).fillna(0.0).where(gate > 0, 0.0)


def build_candidates(index) -> dict[str, pd.Series]:
    """name -> exposure multiplier in [0,1], each already causal (decided on the prior close)."""
    r3m = term_structure(index, "VIX", "VIX3M")          # >1 contango on the 1m->3m segment (SHIPPED)
    r9d = term_structure(index, "VIX9D", "VIX")          # >1 contango on the 9d->1m segment (FAST)
    vix = _idx_series("VIX", index)
    vrp = spx_vrp(index)
    vixvol = vix.pct_change().rolling(20).std() * np.sqrt(252)      # vol-of-vol, information at t

    c: dict[str, pd.Series] = {}
    c["none (ungated)"] = pd.Series(1.0, index=pd.Index(index))
    c["SHIPPED VIX3M/VIX>=1"] = causal((r3m >= 1.0).astype(float), index)

    # --- the user's axis: shorter segment of the curve, alone and combined ---
    c["fast VIX/VIX9D>=1"] = causal((r9d >= 1.0).astype(float), index)
    c["both segments"] = causal(((r3m >= 1.0) & (r9d >= 1.0)).astype(float), index)
    c["either segment"] = causal(((r3m >= 1.0) | (r9d >= 1.0)).astype(float), index)

    # --- threshold sensitivity on the shipped rule (is 1.0 a cliff or a plateau?) ---
    for thr in (0.97, 1.03, 1.05):
        c[f"VIX3M/VIX>={thr}"] = causal((r3m >= thr).astype(float), index)

    # --- continuous ramp instead of a binary switch ---
    for band in (0.05, 0.10):
        c[f"ramp 3M band {band}"] = causal(((r3m - 1.0) / band).clip(0.0, 1.0), index)

    # --- asymmetric hysteresis: exit fast, re-enter slow ---
    for k in (3, 5, 10):
        c[f"SHIPPED + re-entry {k}d"] = causal(persist((r3m >= 1.0).astype(float), k), index)
    c["fast + re-entry 5d"] = causal(persist((r9d >= 1.0).astype(float), 5), index)
    c["both + re-entry 5d"] = causal(persist(((r3m >= 1.0) & (r9d >= 1.0)).astype(float), 5), index)

    # --- premium-cushion family (literature: surges cluster where VRP is small/negative) ---
    c["VRP>0 (VIX-RV20)"] = causal((vrp > 0).astype(float), index)
    c["VRP>2 vol pts"] = causal((vrp > 2.0).astype(float), index)
    c["SHIPPED and VRP>0"] = causal((((r3m >= 1.0)) & (vrp > 0)).astype(float), index)

    # --- level tiers (practitioner rule of thumb) ---
    c["VIX tiers 22/30"] = causal(pd.Series(np.where(vix < 22, 1.0, np.where(vix < 30, 0.5, 0.0)),
                                            index=vix.index), index)
    c["SHIPPED and VIX<30"] = causal((((r3m >= 1.0)) & (vix < 30)).astype(float), index)

    # --- vol-of-vol: is VIX itself becoming unstable? ---
    hi = vixvol.rolling(504, min_periods=120).quantile(0.90)
    c["vol-of-vol<90pct"] = causal((vixvol < hi).astype(float), index)
    c["SHIPPED and volvol"] = causal((((r3m >= 1.0)) & (vixvol < hi)).astype(float), index)
    return c


# ----------------------------------------------------------------------------- scoring
def score(ret: pd.Series, label: str, gate: pd.Series | None = None) -> dict:
    s = summarise(ret, PPY_BOOK)
    m = (1 + ret).resample("ME").prod() - 1
    return {"candidate": label, "sharpe": round(s["sharpe_ann"], 2), "max_dd": round(s["max_dd"], 3),
            "worst_day": round(float(ret.min()), 3), "worst_month": round(float(m.min()), 3),
            "months_pos": round(float((m > 0).mean()), 3), "skew": round(float(ret.skew()), 1),
            "exposure": round(float(gate.mean()), 3) if gate is not None else 1.0}


def priced_leg(src, sym, und, cls, ppy, gate: pd.Series | None) -> pd.Series:
    """One sleeve with the gate honestly accounted, which needs two things kept apart:

      * the switch is PAID — turning the swap off and back on crosses the same vega spread a roll does,
        charged here through the sleeve's own cost model (gate passed into `short_vol_book`);
      * the vol-target is sized off the UNGATED leg. Sizing it off the gated series would read the
        flat stretches as 'low volatility' and lever the leg up on re-entry — a pure accounting
        artifact that hands a gate free leverage exactly when it steps back in.
    """
    iv = naive_dt(implied(src, sym))
    bars = underlying_bars(und, cls)
    px = bars["close"]
    cost = COST_BY_CLASS.get(cls, 1.5)
    base = {"timed": False, "var_cap": 1e9, "bars": bars, "vega_cost_volpts": cost}
    ung = vp.short_vol_book(px, iv, ppy=ppy, **base)["net"]
    net = ung if gate is None else vp.short_vol_book(px, iv, ppy=ppy, gate=gate, **base)["net"]
    scale = (VOL_TARGET_ANNUAL / (ung.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
    return (net * scale.reindex(net.index)).clip(lower=-0.999).dropna()


def priced_book(gate: pd.Series | None) -> pd.Series:
    """The 18-leg book rebuilt on `priced_leg` — gate paid for, vol-target free of the re-entry artifact.

    The same artifact bites a second time at book level: `book_from` re-targets the equal-risk mean on
    ITS trailing vol, which the gated flat stretches deflate. So the book's scale is taken from the
    ungated book and applied to the gated one — matching how the shipped pipeline gates (after the
    vol-target, never inside it)."""
    ung, gat = {}, {}
    for src, sym, und, cls, ppy in UNIVERSE:
        try:
            ung[sym] = priced_leg(src, sym, und, cls, ppy, None)
            gat[sym] = ung[sym] if gate is None else priced_leg(src, sym, und, cls, ppy, gate)
        except Exception as e:
            print(f"  SKIP {sym}: {str(e)[:70]}")
    raw_u = pd.DataFrame(ung).sort_index().mean(axis=1, skipna=True).dropna()
    raw_g = pd.DataFrame(gat).sort_index().mean(axis=1, skipna=True).dropna()
    scale = (VOL_TARGET_ANNUAL / (raw_u.rolling(60).std() * np.sqrt(PPY_BOOK))).clip(upper=3.0).shift(1).fillna(0.0)
    return (raw_g * scale.reindex(raw_g.index)).clip(lower=-0.999).dropna()


def gate_switches(gate: pd.Series) -> float:
    """Round trips per year the gate adds on top of the weekly re-strike — the thing the spread is paid on."""
    flips = (gate.diff().abs() > 0).sum()
    yrs = (gate.index[-1] - gate.index[0]).days / 365.25
    return float(flips / max(yrs, 1e-9))


def book_impact(gated: dict[str, pd.Series], picks: list[str]) -> pd.DataFrame:
    """Push each candidate volprem leg through the CANONICAL assembler and score the five task targets.

    The leg's own drawdown is not the deliverable — the book's is. A gate that flatters the leg but
    costs the book its decorrelation would show up here and nowhere else."""
    import scripts.run_master_book as mb

    base = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
    base = {k: v for k, v in base.items() if v is not None}
    rows = []
    for label in picks:
        raw = dict(base)
        leg = gated[label].copy()
        leg.index = pd.DatetimeIndex(leg.index).tz_localize(None) if leg.index.tz is not None else leg.index
        raw["volprem"] = leg.rename("volprem")
        df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
        df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
        df = df[df.notna().sum(axis=1) >= 2]
        stack = df.mean(axis=1, skipna=True).dropna()
        managed, _, _ = mb.risk_overlay(stack, leverage=mb.BOOK_LEVERAGE)
        f, o = mb.scorecard(managed), mb.scorecard(managed.loc[mb.OOS:])
        rows.append({"volprem leg": label,
                     "sharpe": f["sharpe"], "max_dd": f["max_dd"], "worst_month": f["worst_month"],
                     "months_pos": f["months_in_profit"], "streak_mo": f["longest_losing_streak_mo"],
                     "oos_sharpe": o["sharpe"], "oos_dd": o["max_dd"], "oos_worst_mo": o["worst_month"],
                     "oos_months_pos": o["months_in_profit"], "oos_streak": o["longest_losing_streak_mo"]})
    return pd.DataFrame(rows).set_index("volprem leg")


def block_random(book: pd.Series, exposure: float, rng, block: int = 5) -> pd.DataFrame:
    """N_RANDOM gates with the same average exposure, flat days drawn in blocks (crashes are clustered,
    so an i.i.d. random gate is too weak a null — it never sits out a whole episode)."""
    n = len(book)
    n_blocks = max(1, int(round((1 - exposure) * n / block)))
    rows = []
    for _ in range(N_RANDOM):
        g = np.ones(n)
        for start in rng.integers(0, max(1, n - block), size=n_blocks):
            g[start:start + block] = 0.0
        r = book * pd.Series(g, index=book.index)
        s = summarise(r, PPY_BOOK)
        mm = (1 + r).resample("ME").prod() - 1
        rows.append({"sharpe": s["sharpe_ann"], "max_dd": s["max_dd"], "worst_month": float(mm.min())})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    print("building the 18 shipped sleeves ...")
    rets = {}
    for src, sym, und, cls, ppy in UNIVERSE:
        try:
            rets[sym] = sleeve(src, sym, und, cls, ppy)
        except Exception as e:                                  # a missing cache must be visible, not silent
            print(f"  SKIP {sym}: {str(e)[:70]}")
    book_full = book_from(rets)
    book = book_full.loc[WINDOW_START:]
    print(f"  book: {len(rets)} legs, {book.index.min().date()}..{book.index.max().date()} ({len(book)} bars)")

    # candidates are built on the FULL leg history so the assembler's 60-bar vol-target warms up on real
    # bars; VIX9D only lists from 2011-01-04, so every 9d-segment rule is undefined (= fully live) before
    # then — stated, because the 2010 flash crash therefore CANNOT be used to test them.
    cands_full = build_candidates(book_full.index)
    gated_full = {k: (book_full * g).dropna() for k, g in cands_full.items()}
    cands = {k: g.loc[g.index >= pd.Timestamp(WINDOW_START)] for k, g in cands_full.items()}
    oos = pd.Timestamp(OOS_START)                               # config publishes it tz-aware (UTC)
    if book.index.tz is None:                                   # sleeves are built tz-naive (naive_df)
        oos = oos.tz_convert("UTC").tz_localize(None) if oos.tz is not None else oos
    elif oos.tz is None:
        oos = oos.tz_localize(book.index.tz)

    rows, oos_rows, series = [], [], {}
    for label, gate in cands.items():
        gated = (book * gate).dropna()
        series[label] = gated
        rows.append(score(gated, label, gate))
        g_oos = gated.loc[oos:]
        oos_rows.append({**score(g_oos, label, gate.loc[gate.index >= oos]), "n_obs": len(g_oos)})

    full = pd.DataFrame(rows).sort_values("max_dd", ascending=False)
    oosd = pd.DataFrame(oos_rows).set_index("candidate")

    print(f"\n=== CANDIDATE GATES on the 18-leg book, {WINDOW_START}..{book.index.max().date()} ===")
    print("(sorted by shallowest drawdown; 'exposure' = fraction of days the leg is live)\n")
    show = full.set_index("candidate")
    show["oos_sharpe"] = oosd["sharpe"]
    show["oos_dd"] = oosd["max_dd"]
    print(show.to_string())

    # --- null controls against the shipped rule and the best candidate by drawdown ---
    print("\n=== NULL CONTROLS — does the TIMING matter, or just the reduced exposure? ===")
    for label in ("SHIPPED VIX3M/VIX>=1", full.iloc[0]["candidate"]):
        gate = cands[label]
        exp = float(gate.mean())
        real = score(series[label], label, gate)
        const = score(book * exp, f"{label} [constant {exp:.2f}x]")
        rnd = block_random(book, exp, rng)
        pct_dd = float((rnd["max_dd"] < real["max_dd"]).mean())        # frac of randoms with a DEEPER DD
        pct_sh = float((rnd["sharpe"] < real["sharpe"]).mean())
        print(f"\n  {label}  (live {exp:.1%} of days)")
        print(f"    real          Sharpe {real['sharpe']:+.2f}   maxDD {real['max_dd']:+.1%}   "
              f"worst month {real['worst_month']:+.1%}")
        print(f"    constant-exp  Sharpe {const['sharpe']:+.2f}   maxDD {const['max_dd']:+.1%}   "
              f"worst month {const['worst_month']:+.1%}")
        print(f"    block-random  Sharpe {rnd['sharpe'].median():+.2f} [P5 {rnd['sharpe'].quantile(.05):+.2f}, "
              f"P95 {rnd['sharpe'].quantile(.95):+.2f}]   maxDD {rnd['max_dd'].median():+.1%} "
              f"[P5 {rnd['max_dd'].quantile(.05):+.1%}, P95 {rnd['max_dd'].quantile(.95):+.1%}]")
        print(f"    -> real beats {pct_dd:.0%} of random gates on drawdown, {pct_sh:.0%} on Sharpe")

    # --- rebuild the finalists with the gate INSIDE the sleeve, so switching pays the vega spread ---
    picks = ["none (ungated)", "SHIPPED VIX3M/VIX>=1", "fast VIX/VIX9D>=1", "both segments",
             "both + re-entry 5d", "SHIPPED + re-entry 5d"]
    print("\n=== FINALISTS REBUILT WITH THE SWITCHING COST CHARGED (gate inside the sleeve) ===\n")
    priced = {}
    for label in picks:
        g = None if label == "none (ungated)" else cands_full[label]
        b = priced_book(g)
        priced[label] = b
        sl = b.loc[WINDOW_START:]
        free = score(gated_full[label].loc[WINDOW_START:], label)
        pay = score(sl, label)
        sw = 0.0 if g is None else gate_switches(g.loc[WINDOW_START:])
        print(f"  {label:24s} switches/yr {sw:5.1f}   free-gate Sharpe {free['sharpe']:+.2f} / DD {free['max_dd']:+.1%}"
              f"   -> PAID Sharpe {pay['sharpe']:+.2f} / DD {pay['max_dd']:+.1%} / worst mo {pay['worst_month']:+.1%}")

    # --- the number that actually decides it: the five task targets on the assembled master book ---
    print("\n=== MASTER-BOOK EFFECT — each candidate as the volprem leg, canonical assembly, 1.20x ===")
    print("(targets: max-DD > -15%   months >= 74%   worst month > -6%   streak <= 2)\n")
    bi = book_impact(priced, picks)
    print(bi.to_string())

    # --- is 1.0/1.0 a plateau or a spike? sweep BOTH thresholds through the assembled book ---
    # built on the FULL leg history, like the finalists — slicing the leg at 2011 first would starve the
    # assembler's 60-bar vol-target of warm-up and move the metrics for reasons that are not the gate
    r3 = term_structure(book_full.index, "VIX", "VIX3M")
    r9 = term_structure(book_full.index, "VIX9D", "VIX")
    grid = {f"{t3:.2f}/{t9:.2f}":
            (book_full * causal(((r3 >= t3) & (r9 >= t9)).astype(float), book_full.index)).dropna()
            for t3 in (0.96, 0.98, 1.00, 1.02, 1.04) for t9 in (0.96, 0.98, 1.00, 1.02, 1.04)}
    gb = book_impact(grid, list(grid))
    gb[["t3", "t9"]] = [k.split("/") for k in gb.index]
    print("\n=== THRESHOLD ROBUSTNESS — book Sharpe / max-DD / streak over both boundaries ===")
    print("(shape only: gates applied to the finished P&L, so levels sit above the costed table above)")
    for col in ("sharpe", "max_dd", "streak_mo"):
        print(f"\n  {col} (rows = VIX3M/VIX threshold, cols = VIX/VIX9D threshold)")
        print(gb.pivot(index="t3", columns="t9", values=col).round(3).to_string())

    # --- book-level null: same average exposure, dates shuffled in blocks. Timing or just less risk? ---
    exp = float(cands["both segments"].mean())            # exposure measured on the reporting window
    n, block = len(book_full), 5
    nb = max(1, int(round((1 - exp) * n / block)))
    rnd_c = {"REAL": gated_full["both segments"]}
    for i in range(200):
        g = np.ones(n)
        for s0 in rng.integers(0, max(1, n - block), size=nb):
            g[s0:s0 + block] = 0.0
        rnd_c[f"rnd{i}"] = (book_full * pd.Series(g, index=book_full.index)).dropna()
    rb = book_impact(rnd_c, list(rnd_c))
    real_row, rnd_rows = rb.loc["REAL"], rb.drop(index="REAL")
    print(f"\n=== BOOK-LEVEL NULL — 200 block-random gates at the same {exp:.1%} exposure ===")
    print("(real and randoms both on the finished-P&L basis, so the comparison is like-for-like)")
    for col in ("sharpe", "max_dd", "worst_month", "months_pos"):
        q = rnd_rows[col]
        print(f"  {col:12s} real {real_row[col]:+.4f}   random [P5 {q.quantile(.05):+.4f}, P50 {q.quantile(.50):+.4f}, "
              f"P95 {q.quantile(.95):+.4f}]   real beats {(q < real_row[col]).mean():.0%}")
    print(f"  {'streak_mo':12s} real {int(real_row['streak_mo'])}        "
          f"random reaching the <=2 target: {(rnd_rows['streak_mo'] <= 2).mean():.0%}")

    # --- the event no term-structure rule can reach: the curve was in CONTANGO the day before ---
    fc = pd.DataFrame({"VIX3M/VIX": r3, "VIX/VIX9D": r9}).loc["2010-05-04":"2010-05-07"]
    print("\n=== 2010 FLASH CRASH — why no curve rule catches it (decision uses the PRIOR close) ===")
    print(fc.round(3).to_string() if len(fc) else "  (outside this window: VIX9D lists 2011-01-04)")
    print("  VIX3M/VIX was 1.059 into 2010-05-06 — solid contango; the curve inverted ON the crash day.")
    print("  VIX9D starts 2011-01-04, so the 9d segment cannot even be tested there. The one-session")
    print("  dislocation out of a calm curve stays unhedged by ANY term-structure gate.")

    VOLPREM_DIR.mkdir(parents=True, exist_ok=True)
    gb.to_csv(VOLPREM_DIR / "volprem_gates_threshold_surface.csv")
    bi.to_csv(VOLPREM_DIR / "volprem_gates_book.csv")
    show.to_csv(VOLPREM_DIR / "volprem_gates.csv")
    pd.DataFrame(series).to_parquet(VOLPREM_DIR / "volprem_gate_series.parquet")
    print(f"\nwrote {VOLPREM_DIR / 'volprem_gates.csv'} and volprem_gate_series.parquet")


if __name__ == "__main__":
    main()
