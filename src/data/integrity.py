"""Panel integrity checks — is a series still being *published*, or merely still being *returned*?

Four defects found on 2026-08-10, all the same shape: a data source kept answering after the thing it
described had stopped existing, and every one of them was invisible to a backtest because the numbers
that came back looked like numbers.

  1. Coin Metrics entitlement was inferred from a batched 403, so free exchange-flow metrics were
     recorded as pay-walled (a multi-metric call fails whole if any one metric is paid).
  2. Delisted US tickers in the survivorship-free equity panel resolved to *foreign companies* —
     ANTM came back as Aneka Tambang on the Jakarta exchange, in rupiah, ranked first by dollar volume.
  3. Cboe stopped publishing three vol indices in Feb-2022; an unbounded forward fill kept the legs
     selling variance at a frozen strike for three years.
  4. Binance keeps emitting daily bars for a *settled* perpetual — a constant price at zero volume,
     407 of them — so dead contracts counted as survivors in the survivorship test itself.

The checks below are what would have caught each one, written to run over any cached panel. They
report; they do not silently drop anything. Screens that change a result belong next to that result
(`onchain.live_universe`, `sp500_membership.truncate_after_exit`, `xsect.top_n_liquid`), where the
choice is visible in the run that depends on it.

    python -m src.data.integrity            # sweep every cached price panel
    python -m src.data.integrity --verbose  # list the offending names, not just the counts
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CACHE_DIR  # noqa: E402

XS = CACHE_DIR / "xs"


def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run — the shape all three "it stopped" checks reduce to."""
    run = best = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _returns(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change(fill_method=None)


def _traded_bars(panel: pd.DataFrame, max_flat_share: float = 0.8) -> pd.Series:
    """Bars on which the cross-section actually moved. A bar where almost nothing changed is a
    non-trading day the calendar dragged in (this repo's equity panel carries Sundays with the
    previous close repeated), and charging every name for it makes the whole universe look dead."""
    flat = _returns(panel) == 0.0
    share = flat.sum(axis=1).div(flat.count(axis=1).replace(0, np.nan))
    return share.fillna(1.0) < max_flat_share


def stale_series(panel: pd.DataFrame, max_stale_days: int = 30) -> pd.DataFrame:
    """Names whose last observation is well behind the panel's own end — a series that stopped."""
    end = panel.index.max()
    last = panel.apply(lambda c: c.last_valid_index())
    days = last.map(lambda d: (end - d).days if d is not None else np.inf)
    out = days[days > max_stale_days].sort_values(ascending=False)
    return pd.DataFrame({"stale_days": out, "last": last.reindex(out.index)})


def interior_gaps(panel: pd.DataFrame, min_gap_bars: int = 60) -> pd.DataFrame:
    """Names that stop and then come back. This is the one a staleness check misses entirely: the
    series is current, so it looks healthy, and the hole sits in the middle of the backtest."""
    rows = {}
    for c in panel.columns:
        s = panel[c]
        lo, hi = s.first_valid_index(), s.last_valid_index()
        if lo is None or hi is None:
            continue
        inner = s.loc[lo:hi]
        missing = inner.isna()
        if not missing.any():
            continue
        best = _longest_run(missing.to_numpy())
        if best >= min_gap_bars:
            rows[c] = {"longest_gap_bars": best, "missing_bars": int(missing.sum()),
                       "span": f"{lo.date()}..{hi.date()}"}
    return pd.DataFrame(rows).T.sort_values("longest_gap_bars", ascending=False) if rows else pd.DataFrame()


def frozen_tape(panel: pd.DataFrame, volume: pd.DataFrame | None = None,
                max_flat_frac: float = 0.25, min_flat_run: int = 30) -> pd.DataFrame:
    """Names whose price stops moving. Counted only on bars the market traded, so a Sunday-padded
    calendar does not indict everything. When volume is supplied, a flat tape carrying *reported
    volume* is called out separately — that contradiction is what a mis-resolved ticker looks like,
    while flat-and-zero-volume is an ordinary dead instrument."""
    traded = _traded_bars(panel)
    flat = (_returns(panel) == 0.0) & traded.to_numpy()[:, None]   # bool, no object dtype
    counted = panel.notna() & traded.to_numpy()[:, None]
    frac = flat.sum() / counted.sum().replace(0, np.nan)
    runs = pd.Series({c: _longest_run(flat[c].to_numpy()) for c in panel.columns})
    hit = frac[(frac > max_flat_frac) | (runs > min_flat_run)].sort_values(ascending=False)
    out = pd.DataFrame({"flat_frac": hit.round(3), "longest_flat_run": runs.reindex(hit.index)})
    if volume is not None:
        v = volume.reindex_like(panel).tail(252).fillna(0.0).sum()
        out["still_reports_volume"] = (v.reindex(hit.index) > 0)
    return out


def scale_outliers(panel: pd.DataFrame, factor: float = 20.0) -> pd.DataFrame:
    """Names whose level dwarfs the rest of the panel by an implausible factor. Deliberately a
    *report*, never a screen: on a homogeneous panel this isolates a mis-resolved ticker quoted in
    another currency, but on a panel with one dominant member (BTC in a wide spot set) the same rule
    would throw out the most important name. A human decides."""
    q99 = panel.quantile(0.99, axis=1)
    ratio = panel.div(q99, axis=0).max()
    hit = ratio[ratio > factor].sort_values(ascending=False)
    return pd.DataFrame({"peak_vs_panel_99th": hit.round(1)})


CHECKS = ("stale", "gaps", "frozen", "scale")


def check_panel(panel: pd.DataFrame, volume: pd.DataFrame | None = None,
                label: str = "panel", verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Run every check and print a one-line verdict per check."""
    if panel.index.tz is None:
        panel = panel.copy(); panel.index = panel.index.tz_localize("UTC")
    res = {"stale": stale_series(panel), "gaps": interior_gaps(panel),
           "frozen": frozen_tape(panel, volume), "scale": scale_outliers(panel)}
    n = {k: len(v) for k, v in res.items()}
    print(f"  {label:30s} {panel.shape[1]:4d} names | stale {n['stale']:3d} | "
          f"interior-gaps {n['gaps']:3d} | frozen-tape {n['frozen']:3d} | scale-outliers {n['scale']:2d}")
    if verbose:
        for k, v in res.items():
            if len(v):
                print(f"      [{k}]\n" + v.head(12).to_string().replace("\n", "\n      "))
    return res


def main(verbose: bool = False) -> None:
    print("panel integrity sweep — a series that answers is not the same as a series that is published\n")
    for close in sorted(XS.glob("*_close.parquet")):
        panel = pd.read_parquet(close)
        adv = close.with_name(close.name.replace("_close", "_adv"))
        vol = pd.read_parquet(adv) if adv.exists() else None
        check_panel(panel, vol, label=close.stem.replace("_close", ""), verbose=verbose)
    print("\nnothing here drops a name — screens live next to the result they change "
          "(onchain.live_universe, sp500_membership.truncate_after_exit, xsect.top_n_liquid).")


if __name__ == "__main__":
    import sys
    main(verbose="--verbose" in sys.argv)
