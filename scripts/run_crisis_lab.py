"""Re-open the crisis-alpha sleeve: is the long-gamma leg worth its slot, and can it be built better?

The leg was added as the book's missing long-gamma hedge and has since gone quiet — Sharpe +0.56 over
21 years, −11.9% in 2025 and −2.6% so far in 2026. Two things have to be separated before anything is
changed:

  1. HOW MUCH OF THAT IS THE INDUSTRY. Kaminski-Wen (AlphaSimplex, Jun-2025) date the current trend
     drawdown from Apr-2024 at −21.8% on the SG Trend Index — the second-deepest since 2000, behind
     only "Trade War 1.0" (2015-2019, −23%). This sleeve's −11.9% is HALF the benchmark's. Trend
     drawdowns cluster in calm, rising-equity markets and unwind when equities break, which is the
     same statement as "the hedge is paid for in the good years". Fitting the sleeve to 2024-26 would
     be fitting the premium away, so no variant here is selected on the last two years.
  2. WHAT IS ACTUALLY BROKEN. Measured, not assumed: per-class the sleeve is +0.25 equity / +0.67
     commodity / +1.28 crypto but −0.27 BOND and −0.14 FX, and the bond/FX books turn over 91x and 70x
     a year against 9x for crypto. Costless they are +0.05 and +0.17 — the sign is a TURNOVER artifact,
     not a dead signal. The sign-blend rule flips a position by its full size on a one-day sign change,
     and on low-vol assets that position sits at the 3x leverage cap, so bonds pay ~28bps of Sharpe a
     year to trade a signal worth ~5bps.

So the levers tested are the ones that address (2), each defensible a-priori rather than picked off a
result: a continuous EWMAC signal with Baz et al.'s response function in place of the ±1 sign blend, a
no-trade band, signal smoothing, a tighter leverage cap, wider instrument breadth, and — separately from
construction — the SIZE of the slot, since a hedge handed an earner's risk budget pays for that budget
every calm month (the §6c long-gamma search reached the same conclusion about a different hedge).

Every variant is scored standalone AND as the book's crisis leg through the canonical assembler, on all
five §11 targets plus CAGR, on the full window and the frozen OOS block, against a rotated-path control.

    python scripts/run_crisis_lab.py   ->  reports/lab/crisis_lab.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_crisis as crisis  # noqa: E402
import scripts.run_master_book as mb  # noqa: E402
from src.config import LAB_DIR, OOS_START  # noqa: E402
from src.risk.stress import hedge_weight  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.book_id import stamp  # noqa: E402

OOS = pd.Timestamp(OOS_START).tz_localize(None)
COST_BPS = crisis.COST_BPS
STOCK_PPY, CRYPTO_PPY = crisis.STOCK_PPY, crisis.CRYPTO_PPY

# ── instrument sets. The shipped panels plus everything else in the local store that is a liquid macro
#    instrument — sectors, real estate, single-country, natural gas, miners, copper, inflation-linked and
#    aggregate bonds. Breadth is the one lever trend research is unanimous on, and it is free here: the
#    files are already pulled. Anything not present on disk is dropped by the loader, not faked. ───────
EQUITY_WIDE = crisis.EQUITY + ["XLU", "XLP", "XLV", "XLI", "XLY", "XLB", "XLRE", "EWZ", "FXI"]
COMMOD_WIDE = crisis.COMMOD + ["UNG", "GDX", "CPER"]
BOND_WIDE = crisis.BOND + ["TIP", "BND", "AGG"]

LOOKBACKS = crisis.LOOKBACKS                      # (10,20,40) (20,40,63) (40,63,120)
EWMAC_PAIRS = [(8, 24), (16, 48), (32, 96)]       # Baz-Granger-Harvey-Le Roux-Rattray halflife pairs


# ─────────────────────────────────────────── signals ────────────────────────────────────────────
def sig_sign(close: pd.DataFrame, lookbacks) -> pd.DataFrame:
    """The shipped rule: mean of the SIGN of the return over each lookback. ±1 per horizon, so the
    blend takes only four values and a one-day sign change moves the full position."""
    return sum(np.sign(close / close.shift(h) - 1.0) for h in lookbacks) / len(lookbacks)


def sig_ewmac(close: pd.DataFrame, pairs=EWMAC_PAIRS) -> pd.DataFrame:
    """Baz et al. (2015) CTA trend signal — the industry-standard construction, continuous by design.

    Per speed: x = EWMA(P, hl_fast) − EWMA(P, hl_slow); normalise by the 63-day price vol (y), then by
    y's own trailing year (z) so the signal is comparable across assets and eras; then the response
    function u = z·exp(−z²/4)/0.89, which is the piece that matters here — it damps ALREADY-extended
    trends instead of holding them at full size, so the position is largest mid-trend and fades into
    the exhaustion where sign-rules take their whipsaw losses. The three speeds are averaged."""
    out = 0.0
    for hl_f, hl_s in pairs:
        x = close.ewm(halflife=hl_f, min_periods=hl_f).mean() - close.ewm(halflife=hl_s, min_periods=hl_s).mean()
        y = x / close.rolling(63).std()
        z = y / y.rolling(252).std()
        out = out + z * np.exp(-z.pow(2) / 4.0) / 0.89
    return (out / len(pairs)).clip(-1.5, 1.5)


def sig_donchian(close: pd.DataFrame, lookbacks) -> pd.DataFrame:
    """Channel breakout, the other classical trend rule: go long on a new N-day high, short on a new
    N-day low, HOLD in between. The hold is the point — a position only changes on a genuine extreme,
    which is a different way of buying the same turnover reduction the response function buys."""
    out = 0.0
    for h in lookbacks:
        hi, lo = close.rolling(h).max(), close.rolling(h).min()
        raw = pd.DataFrame(np.where(close >= hi, 1.0, np.where(close <= lo, -1.0, np.nan)),
                           index=close.index, columns=close.columns)
        out = out + raw.ffill().fillna(0.0)
    return out / len(lookbacks)


# ────────────────────────────────────── position / book engine ──────────────────────────────────
def _apply_band(pos: pd.DataFrame, band: float, full: pd.DataFrame | None = None) -> pd.DataFrame:
    """No-trade band: hold the existing position until the target moves by more than `band` of a FULL
    position. The cheapest turnover control there is — it changes no signal, only when the signal is
    worth paying to act on. The band has to be measured against each asset's own full position (its
    vol-target leverage), not in raw position units: a 3x-levered bond and a 0.3x-levered coin would
    otherwise get bands that differ by a factor of ten for the same economic hesitation."""
    if band <= 0:
        return pos
    a = pos.to_numpy(copy=True)
    thr = (band * full.to_numpy()) if full is not None else np.full(a.shape, band)
    held = np.zeros(a.shape[1])
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        tgt, t_i = a[i], thr[i]
        move = np.abs(tgt - held) > np.where(np.isfinite(t_i), t_i, band)
        held = np.where(move & np.isfinite(tgt), tgt, held)
        out[i] = held
    return pd.DataFrame(out, index=pos.index, columns=pos.columns)


def leg(close: pd.DataFrame, ppy: int, *, signal="sign", lookbacks=None, cap=3.0, band=0.0,
        smooth=0, cost_bps=COST_BPS, live_norm=False) -> pd.Series:
    """One tranche: signal -> per-asset vol-targeted position -> equal-weighted, net of turnover cost.

    `live_norm=False` reproduces the shipped normalisation (divide by the panel's FULL width, so an
    instrument that has not listed yet dilutes the book); True divides by the instruments actually
    printing that day, which is what a wider panel needs to not be diluted by its own newest members."""
    lookbacks = lookbacks or LOOKBACKS[1]
    r = close.pct_change()
    vol = r.rolling(40).std()
    sig = {"sign": sig_sign, "ewmac": lambda c, _: sig_ewmac(c), "donchian": sig_donchian}[signal](close, lookbacks)
    if smooth:
        sig = sig.ewm(span=smooth, min_periods=1).mean()
    lev = (0.15 / np.sqrt(ppy) / vol).clip(upper=cap)
    pos = _apply_band(sig.shift(1) * lev, band, lev)
    n = close.notna().sum(axis=1).replace(0, np.nan) if live_norm else close.shape[1]
    gross = (pos * r).sum(axis=1) / n
    cost = (pos.diff().abs().sum(axis=1) / n) * cost_bps / 1e4
    return crisis._vol_target(gross - cost, ppy)


def class_book(close: pd.DataFrame, ppy: int, **kw) -> pd.Series | None:
    """Average the fast/medium/slow tranches, then vol-target the class — the shipped shape, with the
    tranche construction swappable. EWMAC carries its own three speeds, so it is built once."""
    if close is None or close.empty or close.shape[1] == 0:
        return None
    if kw.get("signal") == "ewmac":
        tranches = [crisis._vol_target(leg(close, ppy, **kw), ppy)]
    else:
        tranches = [crisis._vol_target(leg(close, ppy, lookbacks=lb, **kw), ppy) for lb in LOOKBACKS]
    return crisis._vol_target(pd.concat(tranches, axis=1).mean(axis=1).dropna(), ppy)


def turnover_of(close: pd.DataFrame, ppy: int, **kw) -> float:
    """Annual round-trip turnover of the medium tranche — the diagnostic the cost drag comes from."""
    lookbacks = kw.pop("lookbacks", LOOKBACKS[1])
    r = close.pct_change()
    vol = r.rolling(40).std()
    signal = kw.get("signal", "sign")
    sig = {"sign": sig_sign, "ewmac": lambda c, _: sig_ewmac(c), "donchian": sig_donchian}[signal](close, lookbacks)
    if kw.get("smooth"):
        sig = sig.ewm(span=kw["smooth"], min_periods=1).mean()
    lev = (0.15 / np.sqrt(ppy) / vol).clip(upper=kw.get("cap", 3.0))
    pos = _apply_band(sig.shift(1) * lev, kw.get("band", 0.0), lev)
    return float((pos.diff().abs().sum(axis=1) / close.shape[1]).mean() * ppy)


# ───────────────────────────────────────────── panels ───────────────────────────────────────────
_PANELS: dict[str, tuple[pd.DataFrame, int]] = {}


def panels(wide=False) -> dict[str, tuple[pd.DataFrame, int]]:
    """Load once — the crypto panel splices 20 symbols through the trend loader and is the slow part."""
    key = "wide" if wide else "base"
    if key not in _PANELS:
        eq = EQUITY_WIDE if wide else crisis.EQUITY
        cm = COMMOD_WIDE if wide else crisis.COMMOD
        bd = BOND_WIDE if wide else crisis.BOND
        if "crypto" not in _PANELS:
            _PANELS["crypto"] = (crisis._crypto_panel(crisis.CRYPTO_TOP), CRYPTO_PPY)
        _PANELS[key] = {
            "equity": (crisis._panel(eq, crisis._etf), STOCK_PPY),
            "commod": (crisis._panel(cm, crisis._etf), STOCK_PPY),
            "bond": (crisis._panel(bd, crisis._etf), STOCK_PPY),
            "fx": (crisis._panel(crisis.FX, crisis._fx), STOCK_PPY),
            "crypto": _PANELS["crypto"],
        }
    return _PANELS[key]


def build(wide=False, classes=None, **kw) -> pd.Series:
    """A full sleeve: the class books combined at equal risk and vol-targeted — the shipped assembly."""
    p = panels(wide)
    classes = classes or list(p)
    books = {k: class_book(px, ppy, **kw) for k, (px, ppy) in p.items() if k in classes}
    live = {k: v for k, v in books.items() if v is not None and len(v) > 100}
    df = pd.DataFrame(live).sort_index()
    return crisis._vol_target(df.mean(axis=1, skipna=True).dropna(), CRYPTO_PPY).rename("ret")


# ─────────────────────────────────── stress conditioning (sizing) ───────────────────────────────
def stress_scale(index: pd.Index, lo=0.5, hi=2.0) -> pd.Series:
    """The shipped sizing ramp, read from its owner rather than re-implemented here.

    This lab is what argued the hedge slot should follow market stress, so it must be scored against
    the same function the book runs — a second copy would let the published controls drift away from
    the rule they are supposed to be controlling. See src/risk/stress.py for the components and for the
    crypto-drawdown term that was built and rejected."""
    return hedge_weight(index, lo, hi)


# ──────────────────────────────────────────── scoring ───────────────────────────────────────────
def card(s: pd.Series) -> dict:
    s = s.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    sc = summarise(s, len(s) / yrs)
    m = (1 + s).resample("ME").prod() - 1
    neg, streak, mx = (m <= 0).astype(int).to_numpy(), 0, 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"sharpe": round(sc["sharpe_ann"], 2), "cagr": round(float((1 + s).prod() ** (1 / yrs) - 1), 3),
            "max_dd": round(sc["max_dd"], 3), "worst_month": round(float(m.min()), 3),
            "months_in_profit": round(float((m > 0).mean()), 3), "streak": int(mx),
            "skew": round(float(s.skew()), 1)}


_RAW_LEGS: dict[str, pd.Series] | None = None


def raw_legs() -> dict[str, pd.Series]:
    """Every family's published series, loaded once, BEFORE rescaling."""
    global _RAW_LEGS
    if _RAW_LEGS is None:
        raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
        _RAW_LEGS = {k: v for k, v in raw.items() if v is not None}
    return _RAW_LEGS


def book_with_crisis(sleeve: pd.Series | None, w: float = 1.0) -> pd.Series:
    """The canonical book with the crisis leg REPLACED by `sleeve` at `w` share of one equal-risk slot.

    This has to be the assembler's own arithmetic, not an approximation of it: the book is a UNION over
    families, so on a day when only two legs print it is the mean of those two, and blending a leg into
    a pre-computed stack would silently assume all six print every day (it overstated the book by ~4pp
    of CAGR when this script first did it that way). So the leg is swapped in at source and the
    weighted skipna-mean is taken over whoever prints — which reduces to `df.mean(axis=1, skipna=True)`
    exactly when w=1. w=1.0 is full parity (what ships), w=0 drops the leg, w=0.5 is half a slot.

    `w` may also be a per-day Series — the leg's weight is then a stated market-state rule rather than a
    constant, which is how a hedge stops paying an earner's risk budget in the calm months."""
    raw = dict(raw_legs())
    if sleeve is None or (np.isscalar(w) and w <= 0):
        raw.pop("crisis", None)
        w = 1.0
    else:
        raw["crisis"] = sleeve.rename("crisis")
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
    df = df[df.notna().sum(axis=1) >= 2]
    wts = pd.DataFrame(1.0, index=df.index, columns=df.columns)
    if "crisis" in df.columns:
        wts["crisis"] = float(w) if np.isscalar(w) else w.reindex(df.index).ffill().fillna(1.0)
    num = (df * wts).sum(axis=1, min_count=1)
    den = df.notna().mul(wts).sum(axis=1).replace(0, np.nan)
    return mb.risk_overlay((num / den).dropna(), leverage=mb.BOOK_LEVERAGE)[0]


def book_row(sleeve: pd.Series | None, w: float = 1.0) -> dict:
    b = book_with_crisis(sleeve, w)
    full, oos = card(b), card(b[b.index >= OOS])
    return {"full": {**full, "targets": mb.n_targets(full)}, "oos": {**oos, "targets": mb.n_targets(oos)}}


CRASH_WINDOWS = {"2008 GFC": ("2008-01-01", "2008-12-31"), "2011 selloff": ("2011-08-01", "2011-10-03"),
                 "2015 China": ("2015-08-17", "2015-09-30"), "2018 Q4": ("2018-10-01", "2018-12-31"),
                 "COVID crash": ("2020-02-19", "2020-03-23"), "2022 bear": ("2022-01-01", "2022-12-31"),
                 "yen unwind 2024": ("2024-07-25", "2024-08-09"), "2025 tariff": ("2025-02-19", "2025-04-30")}


def crash_table(s: pd.Series) -> dict:
    return {k: round(float((1 + s.loc[a:b]).prod() - 1), 3) for k, (a, b) in CRASH_WINDOWS.items() if len(s.loc[a:b])}


def halves(s: pd.Series) -> tuple[float, float]:
    """Sharpe on each half of the sleeve's own history — the cheapest stability read there is."""
    s = s.dropna()
    mid = s.index[len(s) // 2]
    return round(card(s[s.index < mid])["sharpe"], 2), round(card(s[s.index >= mid])["sharpe"], 2)


def main():
    print("=== crisis-alpha lab: is the long-gamma leg worth its slot, and can it be built better? ===\n")
    out: dict = {}

    # ---- 0. where the shipped sleeve leaks: per class, gross vs net, and the turnover behind it ----
    print("--- diagnosis: per-class gross vs net (the shipped sign-blend) ---")
    diag = {}
    for k, (px, ppy) in panels(False).items():
        g, n = class_book(px, ppy, cost_bps=0.0), class_book(px, ppy)
        diag[k] = {"gross_sharpe": card(g)["sharpe"], "net_sharpe": card(n)["sharpe"],
                   "turnover_yr": round(turnover_of(px, ppy), 1), "n_instruments": int(px.shape[1])}
        print(f"  {k:7s} gross {diag[k]['gross_sharpe']:+.2f}  net {diag[k]['net_sharpe']:+.2f}  "
              f"cost drag {diag[k]['gross_sharpe'] - diag[k]['net_sharpe']:+.2f}  turnover {diag[k]['turnover_yr']:6.1f}x")
    out["diagnosis_by_class"] = diag

    # ---- 1. the variants. Baseline first, then one lever at a time, then the combinations ----
    print("\n--- variants (standalone) ---")
    variants: dict[str, pd.Series] = {}
    variants["V0 shipped (sign blend)"] = crisis.build_crisis()
    variants["V1 EWMAC + response fn"] = build(signal="ewmac")
    variants["V2 sign + no-trade band"] = build(band=0.15)
    variants["V3 sign + smoothing"] = build(smooth=10)
    variants["V4 sign + tighter cap"] = build(cap=1.5)
    variants["V5 sign + wide panels"] = build(wide=True, live_norm=True)
    variants["V6 donchian channel"] = build(signal="donchian")
    variants["V7 EWMAC + wide"] = build(signal="ewmac", wide=True, live_norm=True)
    variants["V8 EWMAC + wide + band"] = build(signal="ewmac", wide=True, live_norm=True, band=0.15)
    variants["V9 EWMAC + wide + cap"] = build(signal="ewmac", wide=True, live_norm=True, cap=1.5)
    variants["V10 EWMAC + smoothing"] = build(signal="ewmac", smooth=10)
    variants["V11 EWMAC + band"] = build(signal="ewmac", band=0.15)

    rows = {}
    for name, s in variants.items():
        c = card(s)
        h1, h2 = halves(s)
        oos_c = card(s[s.index >= OOS])
        rows[name] = {**c, "h1": h1, "h2": h2, "oos_sharpe": oos_c["sharpe"], "oos_cagr": oos_c["cagr"],
                      "start": str(s.index.min().date()), "crash": crash_table(s)}
        print(f"  {name:26s} Sh {c['sharpe']:+.2f} (h1 {h1:+.2f} h2 {h2:+.2f} oos {oos_c['sharpe']:+.2f})  "
              f"CAGR {c['cagr']:+6.1%}  DD {c['max_dd']:+6.1%}  skew {c['skew']:+5.1f}  mo {c['months_in_profit']:.0%}")
    out["variants_standalone"] = rows

    # ---- 2. what each one does to the BOOK in the crisis slot, at full parity ----
    print("\n--- as the book's crisis leg, at FULL parity (the shipped sizing) ---")
    base_drop = book_row(None)
    print(f"  {'DROP the leg entirely':26s} FULL {base_drop['full']['targets']}/5 Sh {base_drop['full']['sharpe']:+.2f} "
          f"CAGR {base_drop['full']['cagr']:+.1%} worst {base_drop['full']['worst_month']:+.2%} "
          f"strk {base_drop['full']['streak']} | OOS {base_drop['oos']['targets']}/5 Sh {base_drop['oos']['sharpe']:+.2f} "
          f"CAGR {base_drop['oos']['cagr']:+.1%}")
    book_rows = {"DROP the leg entirely": base_drop}
    for name, s in variants.items():
        r = book_row(s)
        book_rows[name] = r
        print(f"  {name:26s} FULL {r['full']['targets']}/5 Sh {r['full']['sharpe']:+.2f} "
              f"CAGR {r['full']['cagr']:+.1%} worst {r['full']['worst_month']:+.2%} strk {r['full']['streak']} | "
              f"OOS {r['oos']['targets']}/5 Sh {r['oos']['sharpe']:+.2f} CAGR {r['oos']['cagr']:+.1%}")
    out["variants_in_book_full_parity"] = book_rows

    # ---- 3. SIZE. Construction is only half the question: the leg is held at a full equal-risk slot,
    #         which hands a hedge the same risk budget as an earner and makes it pay for that budget
    #         every calm month. Sweep the slot share for the shipped build and the best rebuild. ----
    print("\n--- slot size sweep (share of one equal-risk slot) ---")
    best = max(("V0 shipped (sign blend)", "V1 EWMAC + response fn", "V3 sign + smoothing", "V10 EWMAC + smoothing"),
               key=lambda k: rows[k]["sharpe"])
    print(f"  (best standalone rebuild: {best})")
    sweep = {}
    for name in ("V0 shipped (sign blend)", best):
        sweep[name] = {}
        for w in (0.0, 0.25, 0.50, 0.75, 1.0):
            r = book_row(variants[name] if w > 0 else None, w)
            sweep[name][f"{w:.2f}"] = r
            print(f"  {name:26s} w={w:.2f}  FULL {r['full']['targets']}/5 Sh {r['full']['sharpe']:+.2f} "
                  f"CAGR {r['full']['cagr']:+.1%} worst {r['full']['worst_month']:+.2%} mo {r['full']['months_in_profit']:.0%} "
                  f"strk {r['full']['streak']} | OOS {r['oos']['targets']}/5 Sh {r['oos']['sharpe']:+.2f} "
                  f"CAGR {r['oos']['cagr']:+.1%} mo {r['oos']['months_in_profit']:.0%}")
    out["slot_size_sweep"] = sweep

    # ---- 4. CONDITIONAL size: the same leg, weighted by a causal market-state ramp instead of a
    #         constant, so the calm months carry half a slot and the stressed months carry two. ----
    print("\n--- stress-conditional slot weight (VIX curve + equity drawdown, both t-1) ---")
    cond = {}
    for name in ("V0 shipped (sign blend)", best):
        s = variants[name]
        for lo, hi in ((0.5, 2.0), (0.25, 1.5), (0.0, 1.5)):
            sc = stress_scale(s.index, lo, hi)
            r = book_row(s, sc)
            cond[f"{name} [{lo}-{hi}x]"] = r
            print(f"  {name:26s} [{lo:.2f}-{hi:.2f}x] FULL {r['full']['targets']}/5 Sh {r['full']['sharpe']:+.2f} "
                  f"CAGR {r['full']['cagr']:+.1%} worst {r['full']['worst_month']:+.2%} strk {r['full']['streak']} | "
                  f"OOS {r['oos']['targets']}/5 Sh {r['oos']['sharpe']:+.2f} CAGR {r['oos']['cagr']:+.1%}")
    out["stress_conditional"] = cond

    # ---- 5. what the leg is bought FOR: its return through the crash windows, build by build ----
    print("\n--- crash-window returns (the hedge value — want POSITIVE) ---")
    print(f"  {'window':18s} " + "  ".join(f"{k.split()[0][:9]:>9s}" for k in ("V0 shipped", best)))
    for wname in CRASH_WINDOWS:
        cells = [rows[k]["crash"].get(wname) for k in ("V0 shipped (sign blend)", best)]
        print(f"  {wname:18s} " + "  ".join(f"{v:+9.1%}" if v is not None else f"{'—':>9s}" for v in cells))
    out["best_variant"] = best
    out.update(finalists(variants))

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    (LAB_DIR / "crisis_lab.json").write_text(json.dumps(stamp(out), indent=2, default=str))
    print(f"\nwrote {LAB_DIR / 'crisis_lab.json'}")
    return out, variants


def _lab_sleeve(fname: str) -> pd.Series | None:
    """An already-built lab sleeve (the §6c long-gamma candidates), re-judged here as a REPLACEMENT for
    the crisis slot rather than as a ninth family — a different question from the one they were rejected on."""
    p = LAB_DIR / f"{fname}.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p).iloc[:, 0].dropna()
    ix = pd.DatetimeIndex(s.index)
    s = pd.Series(np.asarray(s), index=ix.tz_convert("UTC").tz_localize(None) if ix.tz else ix)
    return s.groupby(level=0).last().sort_index().rename("ret")


def finalists(variants: dict[str, pd.Series]) -> dict:
    """Stage 2 — the candidates that survive stage 1, each at three sizings, with the controls that say
    whether the conditioner TIMES anything or merely holds less of the leg."""
    out: dict = {}
    cands = {"shipped sign-blend": variants["V0 shipped (sign blend)"],
             "donchian channel": variants["V6 donchian channel"],
             "sign + smoothing": variants["V3 sign + smoothing"],
             "EWMAC (best Sharpe)": variants["V1 EWMAC + response fn"]}
    for tag, f in (("term-timed long VIX", "convexity_sleeve"), ("haven basket", "defensive_sleeve")):
        s = _lab_sleeve(f)
        if s is not None:
            cands[tag] = s

    print("\n=== stage 2: finalists, each at three sizings ===")
    rows = {}
    for name, s in cands.items():
        sc = stress_scale(s.index, 0.25, 1.5)
        for size, w in (("full slot", 1.0), ("half slot", 0.5), ("stress-timed", sc)):
            r = book_row(s, w)
            rows[f"{name} | {size}"] = r
            print(f"  {name:22s} {size:13s} FULL {r['full']['targets']}/5 Sh {r['full']['sharpe']:+.2f} "
                  f"CAGR {r['full']['cagr']:+.1%} DD {r['full']['max_dd']:+.1%} worst {r['full']['worst_month']:+.2%} "
                  f"mo {r['full']['months_in_profit']:.0%} strk {r['full']['streak']} | OOS {r['oos']['targets']}/5 "
                  f"Sh {r['oos']['sharpe']:+.2f} CAGR {r['oos']['cagr']:+.1%} worst {r['oos']['worst_month']:+.2%} "
                  f"mo {r['oos']['months_in_profit']:.0%} strk {r['oos']['streak']}")
    out["finalists"] = rows

    # ---- controls. Diluting a book with ANY weakly-correlated series improves its tails; and holding
    #      LESS of a drag improves its return. Neither is a sleeve or a conditioner earning its slot.
    #      Control 1 rotates the sleeve's own path (keeps vol/skew/autocorrelation, destroys alignment).
    #      Control 2 rotates the STRESS signal, so the leg is still re-sized as often and as violently,
    #      just at the wrong times. Control 3 holds the conditioner's AVERAGE weight as a constant. ----
    print("\n--- controls on the stress-timed sizing (median of 40 rotations) ---")
    rng = np.random.default_rng(mb.SEED)
    ctrl = {}
    for name in ("shipped sign-blend", "donchian channel"):
        s = cands[name]
        sc = stress_scale(s.index, 0.25, 1.5)
        real = book_row(s, sc)
        arr, idx = sc.to_numpy(), sc.index
        draws = [book_row(s, pd.Series(np.roll(arr, int(k)), index=idx))
                 for k in rng.integers(1, len(arr) - 1, size=40)]
        med = {w: {m: round(float(np.median([d[w][m] for d in draws])), 4)
                   for m in ("sharpe", "cagr", "worst_month", "months_in_profit", "targets")}
               for w in ("full", "oos")}
        flat = book_row(s, float(sc.mean()))
        ctrl[name] = {"stress_timed": real, "rotated_conditioner_median": med,
                      "constant_at_mean_weight": flat, "mean_weight": round(float(sc.mean()), 3)}
        print(f"  {name}:  mean weight {sc.mean():.2f}")
        print(f"    stress-timed        FULL Sh {real['full']['sharpe']:+.2f} CAGR {real['full']['cagr']:+.1%} "
              f"worst {real['full']['worst_month']:+.2%} | OOS Sh {real['oos']['sharpe']:+.2f} CAGR {real['oos']['cagr']:+.1%}")
        print(f"    rotated conditioner FULL Sh {med['full']['sharpe']:+.2f} CAGR {med['full']['cagr']:+.1%} "
              f"worst {med['full']['worst_month']:+.2%} | OOS Sh {med['oos']['sharpe']:+.2f} CAGR {med['oos']['cagr']:+.1%}")
        print(f"    constant @ mean w   FULL Sh {flat['full']['sharpe']:+.2f} CAGR {flat['full']['cagr']:+.1%} "
              f"worst {flat['full']['worst_month']:+.2%} | OOS Sh {flat['oos']['sharpe']:+.2f} CAGR {flat['oos']['cagr']:+.1%}")
    out["controls"] = ctrl

    # ---- ramp sensitivity: the (lo, hi) pair is stated, not fitted, so show the neighbourhood ----
    print("\n--- conditioner ramp sensitivity (shipped sign-blend) ---")
    sens = {}
    s = cands["shipped sign-blend"]
    for lo, hi in ((0.0, 1.0), (0.25, 1.0), (0.25, 1.5), (0.5, 1.5), (0.25, 2.0), (0.5, 2.0)):
        r = book_row(s, stress_scale(s.index, lo, hi))
        sens[f"{lo}-{hi}"] = r
        print(f"  ramp {lo:.2f}-{hi:.2f}x  FULL {r['full']['targets']}/5 Sh {r['full']['sharpe']:+.2f} "
              f"CAGR {r['full']['cagr']:+.1%} worst {r['full']['worst_month']:+.2%} strk {r['full']['streak']} | "
              f"OOS {r['oos']['targets']}/5 Sh {r['oos']['sharpe']:+.2f} CAGR {r['oos']['cagr']:+.1%}")
    out["ramp_sensitivity"] = sens
    return out


if __name__ == "__main__":
    main()
