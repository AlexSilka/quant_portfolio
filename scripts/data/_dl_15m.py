"""Background: 15m klines for the established liquid subset (>=30mo funding) of the expanded universe."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import RAW_DIR  # noqa: E402
from src.data.binance_bulk import load_klines
fdir = RAW_DIR / "futures/um/fundingRate"
established = [s.name for s in sorted(fdir.iterdir()) if s.is_dir() and len(list(s.glob("*.parquet"))) >= 30]
print(f"{len(established)} established symbols -> downloading 15m klines", flush=True)
done = 0
for i, s in enumerate(established):
    try:
        if len(load_klines(s, "15m", "2019-09", "2026-07", market="um")): done += 1
    except Exception as e:
        print(f"  FAIL {s}: {e}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(established)} processed, {done} with 15m data", flush=True)
print(f"DONE 15m download: {done}/{len(established)}", flush=True)
