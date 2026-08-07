"""Background: spot 1d klines from 2017 for the book's crypto universe — the deep history (2017-08+)
that perp funding (2020+) lacks. Used to test whether a spot-native cross-sectional reversal proxy for
carry's price-leg works on the pre-2020 window. Loader skips months before each coin's listing."""
import sys, warnings
warnings.filterwarnings("ignore")
from src.config import REPORTS_DIR  # noqa: E402
from src.data.binance_bulk import load_klines
crypto = open(REPORTS_DIR / "crypto_universe.txt").read().strip().split(",")
print(f"downloading spot 1d (2017-08+) for {len(crypto)} names", flush=True)
done = 0
for i, s in enumerate(crypto):
    try:
        df = load_klines(s, "1d", "2017-08", "2026-07", market="spot")
        if len(df):
            done += 1
            print(f"  {s}: {len(df)} spot days from {df.index.min().date()}", flush=True)
    except Exception as e:
        print(f"  FAIL {s}: {e}", flush=True)
print(f"DONE spot-2017 download: {done}/{len(crypto)}", flush=True)
