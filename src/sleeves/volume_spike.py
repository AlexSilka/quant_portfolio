"""Volume-spike sleeve: long-only intraday altcoin entry on an anomalous volume burst.

Ported from the author's live Rust bot (spike_bot, strategy VolumeSpike). The thesis: on an
altcoin perp, a sudden inflow of money prints an anomalously large bar volume *before* price has
finished moving; buying that burst (while price is not falling) rides the pump that follows. The
signal is the volume spike; the price condition is a weak confirmation, not the driver.

Faithful to the bot's entry (docs/STRATEGY_VIABILITY.md, lib/strategy/src/signal.rs):
  fire long at the close of bar t iff BOTH
    (1) quote_volume[t] >= k_vol * mean(quote_volume[t-vol_win : t])     # trigger bar EXCLUDED
    (2) (close[t]/close[t-1] - 1) * 100 >= min_price_chg_pct            # close-to-close, signed
Volume is quote (USDT) volume; the average is a simple SMA of the vol_win bars strictly before the
trigger (a ratio threshold, not a z-score). Optional fractal/BTC-alignment gates from the bot are
deliberately omitted (the user asked for the simplest entry).

This module returns only the persistent entry side (+1 on a spike bar, 0 otherwise). Exit, execution
delay, vol-targeting, liquidity-aware costs and funding are supplied downstream by the shared harness
(src/bo_common.py) exactly as for the breakout/mean-reversion sleeves, so results are directly
comparable to the book. Everything here is computable at bar t (backward-only rolling, no shift into
the future); the t+2 execution delay is applied by the engine, never here.
"""
from __future__ import annotations

import pandas as pd


def primary_side(close: pd.Series, quote_volume: pd.Series, *, k_vol: float = 2.0,
                 vol_win: int = 10, min_price_chg_pct: float = 0.5) -> pd.Series:
    """Long-only entry side in {0, +1}: +1 on a volume-spike bar that is also non-falling.

    k_vol             : spike multiple vs the trailing volume SMA (bot default 2.0).
    vol_win           : SMA window in bars, trigger excluded (bot default 10).
    min_price_chg_pct : min close-to-close % move to confirm (bot default 0.5).
    """
    # SMA of the vol_win bars STRICTLY BEFORE t (shift(1) drops the trigger bar from its own average)
    avg_vol = quote_volume.shift(1).rolling(vol_win).mean()
    vol_spike = quote_volume >= k_vol * avg_vol

    price_chg_pct = (close / close.shift(1) - 1.0) * 100.0
    non_falling = price_chg_pct >= min_price_chg_pct

    side = pd.Series(0.0, index=close.index)
    side[(vol_spike & non_falling).fillna(False)] = 1.0
    return side
