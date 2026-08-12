"""Global-macro diversifier sleeve — time-series momentum on EM FX + extended commodities.

A seventh, genuinely-new decorrelated leg built from asset classes the other families never trade.
Only the edges that survive a per-strategy out-of-sample test are kept: trend (TSMOM) on EM FX
(USD/TRY, BRL, INR, ZAR, PLN, CNH) and on a broad commodity set (metals, energy, agri, uranium).
Cross-sectional momentum and reversal were tested on the same universes and DROPPED — no OOS edge;
country-equity trend was tested and DROPPED — no standalone edge. So this is trend-only, and only on
the two classes where the edge holds in both halves of history (EM FX Sharpe ~0.9 h1/h2 +0.85/+0.89;
commodities ~0.6 h1/h2 +0.41/+0.83).

Value to the book: correlation ~+0.13 to the master, so it diversifies genuinely and improves the
worst month and Sharpe (the book's passing targets get more margin). It is a positive-skew trend leg,
so it does NOT lift months-in-profit — a hedge shrinks losing months, it cannot flip them to gains.

Construction per class: TSMOM sign-blend over three lookbacks (fast 10/20/40, medium 20/40/63, slow
40/63/120), per-asset vol-targeted, the three tranches averaged, the class vol-targeted to 15%; the two
class books combined at equal risk and vol-targeted to 15%. Signals fill t+2 and the vol scaler is lagged; ~2 bps turnover cost. EM crosses whose interest
differential this repo cannot price are dropped rather than traded unfunded.

    python scripts/run_gmacro.py   ->  <BOOK_DIR>/gmacro_sleeve.parquet
"""
from __future__ import annotations

import glob
import json
import os
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[1]
from src.config import BOOK_DIR  # noqa: E402
from src.data.twelvedata import _api_key  # noqa: E402
from src.log import get_logger  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.risk.sizing import vol_target_scale  # noqa: E402
from src.sleeves.trend_lab import tsmom_panel  # noqa: E402

EQ_STORE = ROOT / "data/raw/equity_td"
TD_DIR = ROOT / "data/raw/twelvedata"
EMFX = ["USD/TRY", "USD/BRL", "USD/INR", "USD/ZAR", "USD/PLN", "USD/CNH"]     # EM FX (risk-off diversifier)
COMMOD_TD = ["XAG/USD", "XPT/USD", "WTI/USD", "URA", "CORN", "WEAT", "SOYB"]  # metals/energy/agri (TwelveData)
COMMOD_LOCAL = ["GLD", "SLV", "USO", "DBC", "DBA"]                            # commodity ETFs (local store)
LOOKBACKS = [(10, 20, 40), (20, 40, 63), (40, 63, 120)]
PPY = 252
COST_BPS = 2.0        # per unit of turnover on liquid EM-FX / commodity futures — the shipped charge
log = get_logger("gmacro")


def _fetch_td(sym: str) -> pd.Series | None:
    """TwelveData daily close, cached under data/raw/twelvedata/<sym>_1day_..._gmacro.parquet."""
    cache = sorted(glob.glob(str(TD_DIR / f"{sym.replace('/', '-')}_1day_*_gmacro.parquet")))
    if cache:
        s = pd.read_parquet(cache[0])["close"]
    else:
        p = {"symbol": sym, "interval": "1day", "start_date": "2005-01-01", "outputsize": 5000,
             "order": "ASC", "apikey": _api_key()}
        url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(p)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.load(r)
        except Exception:
            return None
        v = d.get("values")
        if not v:
            return None
        df = pd.DataFrame(v).astype({"close": float}); df["datetime"] = pd.to_datetime(df["datetime"])
        s = df.set_index("datetime").sort_index()["close"]
        os.makedirs(TD_DIR, exist_ok=True)
        s.to_frame().to_parquet(TD_DIR / f"{sym.replace('/', '-')}_1day_2005-01-01_gmacro.parquet")
    s.index = pd.to_datetime(s.index)
    s = (s.tz_localize(None) if s.index.tz is not None else s).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _local(sym: str) -> pd.Series | None:
    p = EQ_STORE / f"{sym}_1d.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["close"]; s.index = pd.to_datetime(s.index)
    return (s.tz_localize(None) if s.index.tz is not None else s).sort_index()


def _panel(td_syms, local_syms=()):
    cols = {s: _fetch_td(s) for s in td_syms}
    cols.update({s: _local(s) for s in local_syms})
    return pd.DataFrame({k: v for k, v in cols.items() if v is not None}).sort_index()


def _vol_target(x, target=0.15, lb=60):
    lev = vol_target_scale(x, target, PPY, lookback=lb)
    return (x * lev).dropna()


def _tsmom(close, lookbacks, cost_bps=COST_BPS):
    """`trend_lab.tsmom_panel`, vol-targeted. This was a copy of the one in `run_crisis.py`
    and both carried the same two look-aheads — a fill at the signal bar's own close and an
    unlagged vol scaler. One implementation now, with both fixed."""
    return _vol_target(tsmom_panel(close, lookbacks, PPY, cost_bps))


CCY = {"USD": "US", "EUR": "EZ", "JPY": "JP", "GBP": "GB", "CHF": "CH", "AUD": "AU",
       "NZD": "NZ", "CAD": "CA", "MXN": "MX", "ZAR": "ZA", "NOK": "NO", "SEK": "SE"}


def _pair_ccy(pair):
    return pair.replace("-", "/").split("/")


def _rate(ccy, index):
    """3-month interbank rate (% p.a.) carried onto daily bars, or None if this repo has no series."""
    if ccy not in CCY:
        return None
    p = ROOT / "data/raw/rates" / f"IR3TIB01{CCY[ccy]}M156N.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["val"]
    i = pd.DatetimeIndex(s.index)
    s.index = i.tz_convert("UTC").tz_localize(None) if i.tz is not None else i
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def _class_book(close, cost_bps=COST_BPS):
    """Each tranche is vol-targeted ONCE — by `_tsmom` — and the blend once more.

    There used to be a second `_vol_target` wrapped around each tranche on top of the one `_tsmom`
    already applies, the same duplication `run_crisis._class_book` carried. Two nested targets do not
    cancel: each divides by its own trailing vol and each is capped at 3.0, so a quiet stretch could
    hand a tranche 9x. In the crisis sleeve it printed a single -35.3% day in a class targeted to 15%
    annualised; this is the same code path on FX and commodities.
    """
    if close.empty or close.shape[1] == 0:
        return None
    tranches = [_tsmom(close, lb, cost_bps) for lb in LOOKBACKS]
    return _vol_target(pd.concat(tranches, axis=1).mean(axis=1).dropna())


def _priceable_emfx(px):
    """EM crosses whose interest differential can actually be charged, and nothing else.

    A spot FX position is a funded position: long USD/TRY is holding dollars against borrowed lira,
    and under uncovered interest parity the lira's depreciation IS approximately what the borrowing
    costs. Measured on this panel the book is long USD/TRY on 69% of days while the cross drifts up
    21.8% a year — that whole drift was being booked as trend profit with the funding leg absent.
    `data/raw/rates` has a 3-month rate for ZAR and for none of TRY, BRL, INR, PLN or CNH, so five of
    the six crosses cannot be priced at all.

    They are dropped rather than kept unpriced — the same availability rule the vol-premium leg uses
    to exclude EVZ and VXXLE, and for the same reason: an instrument whose dominant cost this
    repository cannot measure does not belong in a book it reports a Sharpe for. One cross is not a
    tranche, so in practice this retires the EM-FX half and global-macro is its commodity book.
    """
    keep = [c for c in px.columns if all(_rate(x, px.index) is not None for x in _pair_ccy(c))]
    dropped = [c for c in px.columns if c not in keep]
    if dropped:
        log.warning("gmacro: dropping %s — no interest-rate series to charge the carry with; "
                    "kept %s", ", ".join(dropped), ", ".join(keep) or "nothing")
    return px[keep] if keep else pd.DataFrame()


def build_gmacro(cost_bps=COST_BPS) -> pd.Series:
    """`cost_bps` is a parameter only so the same book can be re-run costless — that pair is what §9's
    "cost as a share of gross P&L" is measured from; the shipped book always uses the default."""
    fx = _priceable_emfx(_panel(EMFX))
    books = {"emfx": _class_book(fx, cost_bps) if fx.shape[1] >= 3 else None,
             "commod": _class_book(_panel(COMMOD_TD, COMMOD_LOCAL), cost_bps)}
    live = {k: v for k, v in books.items() if v is not None and len(v) > 100}
    df = pd.DataFrame(live).sort_index()
    return _vol_target(df.mean(axis=1, skipna=True).dropna()).rename("ret")


def main():
    g = build_gmacro()
    g.to_frame().to_parquet(BOOK_DIR / "gmacro_sleeve.parquet")
    s = summarise(g, PPY)
    print(f"global-macro sleeve (EM-FX + commodities TSMOM): Sharpe {s['sharpe_ann']:+.2f}  "
          f"maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  {g.index.min().date()}..{g.index.max().date()}")
    print(f"RUN GMACRO OK -> {BOOK_DIR}/gmacro_sleeve.parquet")


if __name__ == "__main__":
    main()
