"""Background: spot klines (1d/4h/1h) for the established liquid perp subset (>=30mo funding). Gives a
spot leg for every liquid perp -> enables basis / cash-and-carry, spot-perp relative value, and
spot-only strategies on the wide, survivorship-honest universe. Names without a spot listing (index
perps, 1000-scaled memes) 404 and are skipped."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import RAW_DIR  # noqa: E402
from src.data.binance_bulk import load_klines
fdir = RAW_DIR / "futures/um/fundingRate"
established = [s.name for s in sorted(fdir.iterdir()) if s.is_dir() and len(list(s.glob("*.parquet"))) >= 30]
print(f"{len(established)} established symbols -> downloading spot 1d/4h/1h", flush=True)
for tf in ("1d", "4h", "1h"):
    done = 0
    for i, s in enumerate(established):
        try:
            if len(load_klines(s, tf, "2019-09", "2026-07", market="spot")): done += 1
        except Exception as e:
            print(f"  FAIL spot {s} {tf}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {tf}: {i+1}/{len(established)} processed, {done} with spot", flush=True)
    print(f"DONE spot {tf}: {done}/{len(established)} have spot", flush=True)
print("DONE spot-wide download (1d/4h/1h)", flush=True)
