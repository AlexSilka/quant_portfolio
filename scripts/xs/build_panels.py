"""Assemble and cache close-price (and ADV) panels for cross-sectional research.

One parquet per (universe, timeframe): a wide bars×names frame of close prices, plus a
matching dollar-volume (ADV proxy) panel for the liquidity-aware cost term. Symbol lists are
read from what is actually on disk (klines dirs / cached daily files), never hard-coded, so
the panel is exactly the tradable universe. Run once; the sweeps read the cache.

    python scripts/xs/build_panels.py
"""
import json
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.config import CACHE_DIR, RAW_DIR, XS_DIR  # noqa: E402
from src.data.binance_bulk import load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.log import get_logger  # noqa: E402

log = get_logger("xs.panels")

START, END = "2020-01", "2026-07"
OUT = CACHE_DIR / "xs"
OUT.mkdir(parents=True, exist_ok=True)
KL = RAW_DIR / "futures/um/klines"
TD = RAW_DIR / "equity_td"


def crypto_symbols(interval: str) -> list[str]:
    """USDT perps that have this interval on disk (48-name intraday set, 68-name daily set)."""
    return sorted(p.parent.name for p in KL.glob(f"*/{interval}") if p.is_dir())


def equity_symbols() -> tuple[list[str], list[str]]:
    """(US equities/ETFs, FX pairs) from cached daily files."""
    names = sorted(p.name[:-11] for p in TD.glob("*_1d.parquet"))
    return [t for t in names if "=X" not in t], [t for t in names if "=X" in t]


BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96, "5m": 288}
MIN_DAILY_USD = 5e6      # tradable-universe floor on median daily $-volume
MAX_NAMES = 300          # cap panel width so the big universe stays tractable


def _record_universe(tag: str, keep: list, liq: dict) -> None:
    """Write the selected names to a TRACKED file, beside the git-ignored panel they describe.

    The panel lives in data/cache/, which is git-ignored, so the universe a published result was built
    on left no record anywhere — and the selection is a ranked cut: `keep` is the MAX_NAMES most liquid
    of whatever perps happen to be on disk, ranked on FULL-SAMPLE median volume. Download more symbols
    and the cut changes retroactively, rewriting the leg's whole history. That is how a rebuild during
    an audit moved the book's scored block by two targets with no code change and nothing to diff.

    This does not fix the selection — a full-sample rank over a growing disk is still a hindsight
    universe, and the honest fix is for the panel to stop pre-filtering and let the strategy's own
    TRAILING rank decide. It makes the input visible: the file is committed, so the next rebuild that
    moves a number shows exactly which names moved with it."""
    XS_DIR.mkdir(parents=True, exist_ok=True)
    (XS_DIR / f"{tag}_universe.json").write_text(json.dumps({
        "selected": keep, "n_selected": len(keep), "n_eligible": len(liq),
        "capped": len(liq) > MAX_NAMES, "max_names": MAX_NAMES, "min_daily_usd": MIN_DAILY_USD,
        "rank": "full-sample median daily $-volume (NOT point-in-time)",
        "median_daily_usd": {s: round(liq[s], 1) for s in keep},
    }, indent=1))


def build_crypto(interval: str) -> None:
    """Assemble the tradable crypto universe at this interval: every USDT perp on disk with
    ≥300 bars and median daily $-volume ≥ $5M, capped at the 300 most liquid — so the
    cross-section is broad but fillable, not padded with unfillable micro-caps."""
    bpd = BARS_PER_DAY[interval]
    close, advol, liq = {}, {}, {}
    for s in crypto_symbols(interval):
        try:
            px = load_klines(s, interval, START, END, market="um")
        except Exception as e:
            log.warning("xs: skipping crypto %s (%s)", s, e)
            continue
        if len(px) < 300:
            continue
        daily_usd = float(px["quote_volume"].median()) * bpd
        if daily_usd < MIN_DAILY_USD:
            continue
        close[s], advol[s], liq[s] = px["close"], px["quote_volume"], daily_usd
    keep = sorted(liq, key=liq.get, reverse=True)[:MAX_NAMES]
    _dump(f"crypto_{interval}", {s: close[s] for s in keep}, {s: advol[s] for s in keep})
    _record_universe(f"crypto_{interval}", keep, liq)


def build_equity() -> None:
    stocks, fx = equity_symbols()
    for tag, syms, has_vol in [("stocks_1d", stocks, True), ("fx_1d", fx, False)]:
        close, advol = {}, {}
        for s in syms:
            try:
                px = load_equity_daily(s, start="2012-01-01")
            except Exception as e:
                log.warning("xs: skipping equity %s (%s)", s, e)
                continue
            if len(px) < 300:
                continue
            close[s] = px["close"]
            if has_vol:
                advol[s] = px["close"] * px["volume"]
        _dump(tag, close, advol if has_vol else None)


_TD_ITV = {"1h": "1h", "4h": "4h", "15m": "15min", "5m": "5min"}


# a fixed liquid core for intraday cross-section (mega-caps + liquid index/sector ETFs) — the
# equity_td cache now holds ~700 names, far too many to fetch intraday; this is the tradable set.
INTRADAY_CORE = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
                 "TSLA", "AMD", "NFLX", "JPM", "BAC", "WFC", "GS", "V", "MA", "XOM", "CVX",
                 "JNJ", "PFE", "MRK", "UNH", "HD", "WMT", "COST", "PG", "KO", "DIS", "CRM",
                 "ORCL", "CSCO", "INTC", "QCOM", "IBM", "T", "VZ", "BA", "CAT", "GE",
                 "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]


def build_equity_intraday(intervals=("4h", "1h")) -> None:
    """Stocks/FX intraday panels via Twelve Data (from ~2020) — same bar contract as crypto.
    Restricted to a fixed liquid core (INTRADAY_CORE + the FX majors) — the tradable intraday set."""
    from src.data.twelvedata import load_bars
    from src.data.equity import _to_td_symbol
    _, fx = equity_symbols()
    stocks = INTRADAY_CORE
    for tf in intervals:
        for tag, syms, has_vol in [(f"stocks_{tf}", stocks, True), (f"fx_{tf}", fx, False)]:
            close, advol = {}, {}
            for s in syms:
                try:
                    px = load_bars(_to_td_symbol(s), _TD_ITV[tf], "2020-01-01", "2026-08-05")
                except Exception as e:
                    log.warning("xs: skipping equity-intraday %s (%s)", s, e)
                    continue
                if len(px) < 500:
                    continue
                close[s] = px["close"]
                if has_vol and "volume" in px:
                    advol[s] = px["close"] * px["volume"]
            _dump(tag, close, advol if has_vol and advol else None)


def _dump(tag: str, close: dict, advol: dict | None) -> None:
    cp = pd.DataFrame(close).sort_index()
    cp = cp.dropna(how="all").ffill(limit=5)
    cp.to_parquet(OUT / f"{tag}_close.parquet")
    if advol:
        ap = pd.DataFrame(advol).sort_index().reindex(cp.index)
        ap.to_parquet(OUT / f"{tag}_adv.parquet")
    span = f"{cp.index[0].date()}→{cp.index[-1].date()}" if len(cp) else "empty"
    print(f"{tag:16s} {cp.shape[0]:>7d} bars × {cp.shape[1]:>3d} names  {span}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "crypto"):
        for itv in ["1d", "4h", "1h", "15m", "5m"]:
            build_crypto(itv)
    if which in ("all", "equity"):
        build_equity()
    if which in ("all", "equity_intraday"):
        build_equity_intraday(("4h", "1h"))
    print("PANELS OK")


if __name__ == "__main__":
    main()
