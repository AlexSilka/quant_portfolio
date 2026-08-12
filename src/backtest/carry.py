"""Cost of HOLDING a position, as a property of the instrument rather than of the caller.

Commission, spread and impact are costs of *trading* and every backtest here charges them. Carry is
the cost of *holding* — perpetual funding, stock borrow — and it kept being the caller's job, because
`xs_backtest` and `bab_backtest` are handed a price panel and cannot tell a Binance perp from a cash
equity. So the panel backtests charged nothing, each strategy was supposed to remember, and the ones
that forgot looked better than they were. It was found and patched in the x-sect leg, then found
again in the lottery sleeve, then again in BAB — three patches of one hole, with three separate
copies of the same funding panel left behind (`xs/portfolio._funding_panel`,
`lottery/run_lottery._funding_daily`, `breakout/run_bo_xs_tf.funding_panel`). The equity mirror was
open just as long: `xs/broad.run_cfg` builds the shipped broad-equity sleeve, shorts a decile of 692
names, and never passed a borrow rate.

The fix is not a fourth copy. It is to make the instrument answer the question:

  * `for_panel(px)` asks, per name, whether Binance published a funding series for it. That is not a
    heuristic on the ticker string — it is the fact itself, and it is self-maintaining: a symbol is a
    perp exactly when the venue settled funding on it. A panel with no such names is cash, and a
    dollar-neutral cash book borrows every share it shorts, so its default is the config's borrow
    rate rather than zero.
  * the panel backtests call it when the caller does not pass a carry model, so **forgetting now
    charges instead of charging nothing**, and says so in the log.
  * opting out is `NoCarry()` — a decision that appears in a diff, unlike an omission.

Sign convention, one place: a LONG pays a positive funding rate and a SHORT receives it, so the
book's carry P&L is −Σ wᵢ·fᵢ. Borrow is the equity mirror — only the short leg pays, at an annual
rate accrued per bar. Both return a per-bar Series aligned to the weights, to be subtracted from
gross return exactly like any other cost.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EQUITY_BORROW_BPS_ANNUAL, RAW_DIR
from src.log import get_logger

log = get_logger("backtest.carry")

_FUNDING_DIR = Path(RAW_DIR) / "futures/um" / "fundingRate"
_RAW: dict[str, pd.Series] = {}          # per-symbol settlement series, read once per process
_PERPS: set[str] | None = None           # the archive's symbol list, scanned once per process
_SAID: set[tuple] = set()                # log lines already emitted, so a sweep says it once
_PANEL: dict[tuple, pd.DataFrame] = {}   # bar-binned funding panels, memoised per request shape


def perp_symbols() -> set[str]:
    """Every symbol the venue has ever settled funding on, from the offline archive.

    This is what makes "is this a perp?" a lookup rather than a guess about the ticker's spelling.
    Scanned once and held: it is ~830 directory probes, and a sweep asks per backtest.
    """
    global _PERPS
    if _PERPS is None:
        _PERPS = {p.name for p in _FUNDING_DIR.glob("*") if p.is_dir() and any(p.glob("[0-9]*.parquet"))}
    return _PERPS


def settlements(sym: str) -> pd.Series:
    """Raw 8h settlement series for one symbol. Strictly offline — a network fetch inside a sweep
    both stalls it and spends metered quota, so a symbol with nothing cached is simply absent."""
    if sym not in _RAW:
        files = sorted((_FUNDING_DIR / sym).glob("[0-9]*.parquet"))
        if not files:
            _RAW[sym] = pd.Series(dtype=float)
        else:
            df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
            df = df[~df.index.duplicated(keep="first")]
            col = "last_funding_rate" if "last_funding_rate" in df else df.columns[0]
            _RAW[sym] = df[col].sort_index()
    return _RAW[sym]


def funding_panel(symbols, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Funding accrued per BAR for a panel of perps, aligned to `index` and to `symbols`.

    Settlements land on an 8h grid, so each is summed into the bar it falls in. A plain reindex keeps
    only the settlement whose timestamp equals the bar's and silently drops the other two inside a
    daily bar — which is how a 1d book ends up charged a third of its funding.
    """
    cols = list(symbols)
    bar = index.to_series().diff().dropna().median() if len(index) > 1 else None
    # Memoised on the shape of the request, not just on the disk read: a construction sweep calls this
    # once per config, and re-binning ~580 settlement series onto the bar grid every time is the
    # difference between a sweep that runs and one that does not.
    key = (tuple(cols), bar, index[0], index[-1], len(index))
    hit = _PANEL.get(key)
    if hit is not None:
        return hit
    out = {}
    for sym in cols:
        s = settlements(sym)
        if not len(s):
            continue
        out[sym] = s.resample(bar, origin=index[0]).sum() if bar is not None else s
    F = (pd.DataFrame(0.0, index=index, columns=cols) if not out else
         pd.DataFrame(out).reindex(index).fillna(0.0).reindex(columns=cols).fillna(0.0))
    if len(_PANEL) < 32:                      # a handful of panels per process; never an unbounded map
        _PANEL[key] = F
    return F


# ── the models ────────────────────────────────────────────────────────────────────────────────
class NoCarry:
    """The instrument costs nothing to hold. An explicit statement, not an omission."""

    label = "none"

    def pnl(self, weights: pd.DataFrame, ppy: float = 365) -> pd.Series:
        return pd.Series(0.0, index=weights.index)


class PerpFunding:
    """USD-M perpetuals: a long pays the funding rate at every settlement, a short receives it."""

    label = "perp funding"

    def __init__(self, symbols=None):
        self._symbols = None if symbols is None else list(symbols)

    def pnl(self, weights: pd.DataFrame, ppy: float = 365) -> pd.Series:
        f = funding_panel(self._symbols or list(weights.columns), weights.index)
        return -(weights * f.reindex_like(weights).fillna(0.0)).sum(axis=1)


class Borrow:
    """Cash equity: only the short leg pays, at `bps_annual` accrued per bar."""

    label = "borrow"

    def __init__(self, bps_annual: float):
        self.bps_annual = float(bps_annual)

    def pnl(self, weights: pd.DataFrame, ppy: float = 252) -> pd.Series:
        return -weights.clip(upper=0.0).abs().sum(axis=1) * (self.bps_annual / 1e4) / ppy


class Both:
    """A panel that mixes venues, or a perp book that also pays a financing spread."""

    label = "combined"

    def __init__(self, *models):
        self.models = models

    def pnl(self, weights: pd.DataFrame, ppy: float = 365) -> pd.Series:
        return sum(m.pnl(weights, ppy) for m in self.models)


def for_panel(px: pd.DataFrame, *, borrow_bps_annual: float | None = None):
    """The carry model a price panel implies, decided by what the venue actually settles.

    A panel whose names have funding archives is a perp book and is charged funding. One whose names
    have none is cash — and a dollar-neutral cash book borrows every share it shorts, so the default
    there is `EQUITY_BORROW_BPS_ANNUAL`, not zero. That half of the hole was open for the same reason
    the funding half was: the shipped broad-equity sleeve shorts a decile of 692 names and
    `scripts/xs/broad.run_cfg` simply never passed a rate. `Borrow` only charges the short leg, so a
    long-only book is unaffected and it is safe as a default.

    `borrow_bps_annual=None` means "not decided — use the panel's own default"; an explicit `0.0`
    means "this book pays none", which is a statement and is honoured. FX is neither case: shorting a
    pair costs the interest differential, not stock borrow, so it is left uncharged and said out loud
    rather than charged with the wrong model. A MIXED panel is a modelling error rather than a case to
    handle silently — it is charged both and logged, because the honest fix is to split it.
    """
    names = list(px.columns)
    perps = perp_symbols()
    n_perp = sum(1 for c in names if c in perps)
    if n_perp == 0:
        if borrow_bps_annual is None:
            if names and all("=X" in str(c) for c in names):
                log.info("carry: FX panel (%d names) — the short pays an interest differential, not "
                         "stock borrow, and this repo does not model it; charging nothing", len(names))
                return NoCarry()
            borrow_bps_annual = EQUITY_BORROW_BPS_ANNUAL
        return Borrow(borrow_bps_annual) if borrow_bps_annual else NoCarry()
    borrow_bps_annual = borrow_bps_annual or 0.0
    if n_perp < len(names):
        log.warning("carry: panel mixes %d perps with %d non-perps — charging both models; "
                    "split the panel by venue instead", n_perp, len(names) - n_perp)
    if borrow_bps_annual:
        # A perp pays funding, not stock borrow, so this combination is a caller error — but an
        # explicit parameter is never dropped on the floor without saying so, because a silently
        # ignored instruction is exactly how the defect this module exists for got in.
        log.warning("carry: borrow_bps_annual=%.1f passed on a panel with %d perps — honouring it "
                    "ON TOP of funding; perps do not pay stock borrow", borrow_bps_annual, n_perp)
        return Both(PerpFunding(names), Borrow(borrow_bps_annual))
    return PerpFunding(names)


def resolve(carry, px: pd.DataFrame, *, borrow_bps_annual: float | None = None, where: str = ""):
    """What a backtest calls: honour an explicit model, otherwise derive one and say so.

    The log line is the point. Silence is what let three books hold perps for years with no funding
    charged; a caller that never thought about carry now gets the right number AND a line saying the
    decision was made for it.
    """
    if carry is not None:
        return carry
    model = for_panel(px, borrow_bps_annual=borrow_bps_annual)
    key = (where, model.label, len(px.columns))
    if not isinstance(model, NoCarry) and key not in _SAID:
        _SAID.add(key)                      # a sweep calls this thousands of times; say it once
        log.info("carry: %s not given a model, charging %s on %d names",
                 where or "backtest", model.label, len(px.columns))
    return model
