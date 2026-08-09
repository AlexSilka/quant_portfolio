"""Is the trend leg's crypto universe survivorship-free? (It is not — this measures the cost.)

The shipped trend leg trades a hard-coded `CORE10` of ten crypto names. Those are the majors as they
look TODAY, so the list is chosen with hindsight — and that is the one bias this project corrects
everywhere else and says so in the report: the x-sect deep-dive's headline finding is that a curated
50-coin list scores +1.06 against +0.70 for the honest point-in-time universe, and the carry leg
deliberately ships the WEAKER survivorship-free construction (+1.33) over the curated one (+1.47) on
that principle. Trend is the exception, and it is also the family the report leans on for breadth.

Two things are therefore suspect, not one:
  * the LEVEL — ten survivors trending well says little about ten names picked in 2020;
  * the deep-dive's "for crypto, fewer instruments is better" finding, which may be the same bias in
    disguise: of course a list of today's winners beats a broad universe that includes what died.

This rebuilds the crypto half of the leg on a point-in-time top-10 by trailing dollar volume — the
identical membership rule already used by the breakout and carry legs (`pit_members`: trailing median
volume, lagged, no look-ahead) — and prices the difference at leg and book level. Selection is scored
on the pre-OOS window; the frozen block is a read-out only (§10).

    python scripts/trend/run_trend_pit_universe.py  ->  reports/trend/trend_pit_universe.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

import scripts.run_master_book as mb  # noqa: E402
import scripts.trend.trend_common as T  # noqa: E402
from src import bo_common as bo  # noqa: E402
from src.config import OOS_START, RAW_DIR, TREND_DIR  # noqa: E402
from scripts.breakout.run_bo_xs_big import symbols_with_tf  # noqa: E402
from src.metrics import summarise  # noqa: E402

PPY = 365
OOS = pd.Timestamp(OOS_START).tz_localize(None)
SELECT_END = pd.Timestamp("2024-06-30")
TOP_N = 10                      # same count the shipped leg uses, so only the SELECTION RULE changes
LOOKBACK_D = 63                 # trailing window for the liquidity rank — the breakout leg's choice
SPEC = {"entry": "ema", "direction": "long_only", "exit": "reversal"}
BLOCK_TFS = ["1d", "4h"]        # the trend block's crypto timeframes (1h dropped, §timeframe finding)
# the hindsight arm the honest universe is scored against — kept here, where the A/B lives, so the
# shipped builder imports its universe rule from one place instead of carrying a list of its own
CORE10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
NONCRYPTO = {"BTCDOM", "DEFI", "BLUEBIRD"}
RAW_EQ = RAW_DIR / "equity_td"
FLOOR = pd.Timestamp("2017-01-01")   # Binance's own start; anything earlier is a corrupt timestamp


def _naive(s):
    """Tz-naive, float-typed, one row per timestamp — the shape every panel here is joined on.

    The float cast is load-bearing: a few perps come back object-dtype, and an object column silently
    poisons `DataFrame.mean(axis=1)` with a ufunc-divide TypeError only once the whole panel is built."""
    ix = pd.DatetimeIndex(s.index)
    s = pd.Series(np.asarray(s, dtype="float64"), index=ix.tz_convert("UTC").tz_localize(None) if ix.tz else ix)
    return s.groupby(level=0).last().sort_index()


def pool(tf: str, spec: dict | None = None):
    """Every crypto perp with usable history at `tf`: per-name trend returns + its dollar-volume."""
    rets, vol = {}, {}
    for sym in symbols_with_tf(tf):
        if sym[:-4] in NONCRYPTO:
            continue
        px = T.load_crypto_long(sym, tf)
        if px is None or "quote_volume" not in px or px["close"].notna().sum() < 250:
            continue
        try:
            _, r = T.eval_spec(px, spec or SPEC, tf, T.CRYPTO_TF[tf], T.CC,
                               fund=bo.safe_funding(sym), adv=T.crypto_adv(px))
        except Exception as e:                                   # a broken name must be visible
            print(f"    SKIP {sym} {tf}: {str(e)[:60]}")
            continue
        if r.std(ddof=1) > 0:
            rr, vv = _naive(r), _naive(px["quote_volume"].resample("D").sum())
            # a single corrupt bar timestamp (one perp carries an epoch-0 row) would otherwise drag the
            # union index back to 1970 and leave the rolling liquidity rank almost everywhere NaN,
            # silently shrinking a "top-10" universe to ~2 names. Floor both panels at the venue's start.
            rets[sym], vol[sym] = rr[rr.index >= FLOOR], vv[vv.index >= FLOOR]
    R = pd.DataFrame(rets).sort_index()
    V = pd.DataFrame(vol).reindex(R.index).sort_index()
    return R, V


def pit_members(vol: pd.DataFrame, n: int, win: int) -> pd.DataFrame:
    """Top-n by TRAILING median dollar volume, lagged — membership uses only past data."""
    tv = vol.rolling(win, min_periods=win // 2).median().shift(1)
    return tv.rank(axis=1, ascending=False) <= n


INDEX_ETFS = ["SPY", "QQQ", "IWM"]      # indices are a-priori: they always existed, nothing was picked
EQ_TOP_N = 7                            # same count the hand-picked single-name half used
MIN_EQ_DAYS = 500


def equity_legs(pit: bool, spec: dict | None = None) -> dict:
    """The equity half. `pit=False` reproduces the shipped hand-picked EQ_CORE; `pit=True` keeps the
    index ETFs (no selection possible) and replaces the seven single names with a point-in-time top-7 by
    trailing dollar volume out of the full local panel.

    The single-name half is the sharper version of the same bug the crypto core had: NVDA and META are in
    the list, and META did not IPO until May-2012 — a long-only trend sleeve handed the era's best large
    cap by hindsight is not a measurement of trend."""
    cols = {}
    for sym in INDEX_ETFS if pit else T.EQ_CORE:
        px = T.load_equity(sym)
        if px is None:
            continue
        adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
        _, r = T.eval_spec(px, spec or SPEC, "1d", T.EQUITY_TF["1d"], T.EC, fund=None, adv=adv, ppy_daily=252)
        if r.std(ddof=1) > 0:
            cols[f"{sym}_1d_eq"] = _naive(r)
    if not pit:
        return cols

    rets, vol = {}, {}
    for f in sorted((RAW_EQ).glob("*_1d.parquet")):
        sym = f.name[:-len("_1d.parquet")]
        if sym in INDEX_ETFS:
            continue
        # read the BROAD panel directly: trend_common.load_equity points at data/raw/equity, the
        # 113-name deep-history store the hand-picked core lives in. Selecting a point-in-time universe
        # out of that store would just be re-picking from an already-curated shortlist.
        px = pd.read_parquet(f)
        px = px[px.index >= pd.Timestamp("2012-01-01", tz="UTC")]
        if "volume" not in px or px["close"].notna().sum() < MIN_EQ_DAYS:
            continue
        adv = (px["close"] * px["volume"]).rolling(20).median().shift(1)
        try:
            _, r = T.eval_spec(px, spec or SPEC, "1d", T.EQUITY_TF["1d"], T.EC, fund=None, adv=adv, ppy_daily=252)
        except Exception:
            continue
        if r.std(ddof=1) > 0:
            rets[sym], vol[sym] = _naive(r), _naive(px["close"] * px["volume"])
    R = pd.DataFrame(rets).sort_index()
    V = pd.DataFrame(vol).reindex(R.index)
    mem = pit_members(V, EQ_TOP_N, LOOKBACK_D)
    ever = int((mem.sum() > 0).sum())
    print(f"  equity: pool {R.shape[1]} names, PIT top-{EQ_TOP_N}: {ever} distinct ever a member, "
          f"avg {mem.sum(axis=1).mean():.1f} live; the hand-picked seven hold "
          f"{mem[[c for c in T.EQ_CORE if c in mem.columns]].sum().sum() / max(mem.sum().sum(), 1):.0%} of member-days")
    for sym in R.columns:
        cols[f"{sym}_1d_eq"] = R[sym].where(mem[sym].reindex(R.index).fillna(False))
    return cols


def card(s: pd.Series) -> dict:
    s = s.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    sc = summarise(s, len(s) / yrs)
    m = (1 + s).resample("ME").prod() - 1
    neg, streak, mx = (m <= 0).astype(int).to_numpy(), 0, 0
    for v in neg:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    return {"sharpe": round(sc["sharpe_ann"], 2),
            "cagr": round(float((1 + s).prod() ** (1 / yrs) - 1), 3) if yrs > 0 else 0.0,
            "max_dd": round(sc["max_dd"], 3), "worst_month": round(float(m.min()), 3),
            "months_in_profit": round(float((m > 0).mean()), 3), "streak": int(mx)}


def n_targets(c: dict) -> int:
    return sum([2.5 <= c["sharpe"] <= 4.0, c["months_in_profit"] >= 0.80, c["max_dd"] >= -0.15,
                c["worst_month"] >= -0.06, c["streak"] <= 2])


def book_with(trend_leg: pd.Series, end=None) -> pd.Series:
    raw = {lab: mb.load(lab, f, c) for lab, f, c in mb.FAMILIES}
    raw = {k: v for k, v in raw.items() if v is not None}
    raw["trend_momentum"] = trend_leg.rename("trend_momentum")
    df = pd.DataFrame({k: mb.rescale(v) for k, v in raw.items()}).sort_index()
    df = df[df.index >= pd.Timestamp(mb.START_REPORT)]
    if end is not None:
        df = df[df.index <= end]
    df = df[df.notna().sum(axis=1) >= 2]
    return mb.risk_overlay(df.mean(axis=1, skipna=True).dropna(), leverage=mb.BOOK_LEVERAGE)[0]


def main():
    print("=== trend leg: hindsight lists vs point-in-time universes, crypto AND equity ===\n")
    eq_hind, eq_pit = equity_legs(pit=False), equity_legs(pit=True)
    hind, pit, churn = dict(eq_hind), dict(eq_pit), {}
    for tf in BLOCK_TFS:
        R, V = pool(tf)
        print(f"  {tf}: pool {R.shape[1]} perps, {R.index.min().date()}..{R.index.max().date()}")
        mem = pit_members(V, TOP_N, LOOKBACK_D)
        churn[tf] = {"avg_members": round(float(mem.sum(axis=1).mean()), 1),
                     "distinct_names_ever": int((mem.sum() > 0).sum()),
                     "core10_share_of_member_days": round(
                         float(mem[[c for c in CORE10 if c in mem.columns]].sum().sum() / max(mem.sum().sum(), 1)), 3)}
        for sym in R.columns:
            if sym in CORE10:
                hind[f"{sym}_{tf}"] = R[sym]
            pit[f"{sym}_{tf}"] = R[sym].where(mem[sym].reindex(R.index).fillna(False))
        print(f"      PIT universe: {churn[tf]['distinct_names_ever']} distinct names ever a member, "
              f"avg {churn[tf]['avg_members']:.1f} live; today's CORE10 hold "
              f"{churn[tf]['core10_share_of_member_days']:.0%} of member-days")

    legs = {"hindsight lists (crypto+equity)": pd.DataFrame(hind).mean(axis=1).dropna().rename("ret"),
            "PIT crypto only (previous ship)": pd.DataFrame({**eq_hind, **{k: v for k, v in pit.items()
                                                                          if not k.endswith("_1d_eq")}}
                                                           ).mean(axis=1).dropna().rename("ret"),
            "PIT crypto + PIT equity": pd.DataFrame(pit).mean(axis=1).dropna().rename("ret")}

    out = {"churn": churn}
    print("\nleg standalone:")
    for tag, s in legs.items():
        c = card(s)
        out[tag] = {"leg": c}
        print(f"  {tag:34s} Sharpe {c['sharpe']:+.2f}  CAGR {c['cagr']:+.1%}  DD {c['max_dd']:+.1%}  "
              f"months+ {c['months_in_profit']:.0%}")

    print("\nas the trend family in the canonical book at 1.15x — SELECTION WINDOW (pre-OOS):")
    for tag, s in legs.items():
        sel = card(book_with(s, SELECT_END))
        full = book_with(s, None)
        oos = card(full[full.index >= OOS])
        out[tag] |= {"book_selection": sel, "book_full": card(full), "book_oos": oos}
        print(f"  {tag:34s} Sh {sel['sharpe']:+.2f}  CAGR {sel['cagr']:+.1%}  DD {sel['max_dd']:+.1%}  "
              f"worst {sel['worst_month']:+.1%}  mo {sel['months_in_profit']:.0%}  strk {sel['streak']} "
              f"[{n_targets(sel)}/5]")
        print(f"  {'':34s} read-out OOS: Sh {oos['sharpe']:+.2f}  CAGR {oos['cagr']:+.0%}  "
              f"mo {oos['months_in_profit']:.0%}  strk {oos['streak']} [{n_targets(oos)}/5]")

    (TREND_DIR / "trend_pit_universe.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {TREND_DIR / 'trend_pit_universe.json'}")


if __name__ == "__main__":
    main()
