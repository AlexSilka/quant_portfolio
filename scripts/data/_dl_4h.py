"""Background: 4h klines for the established liquid subset (>=30 months funding history) of the
expanded perp universe. 4h aligns better with the 8h funding grid than 1d -> enables the precise
delta-neutral basis trade and an intraday carry-robustness check on the wider, survivorship-honest set."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import RAW_DIR  # noqa: E402
from src.data.binance_bulk import load_klines

fdir = RAW_DIR / "futures/um/fundingRate"
syms = sorted(s.name for s in fdir.iterdir() if s.is_dir())
# establishment filter: >=30 monthly funding files (>=2.5y), survivorship-honest (incl. later-delisted)
established = [s for s in syms if len(list((fdir / s).glob("*.parquet"))) >= 30]
print(f"{len(established)} established symbols (>=30mo funding) -> downloading 4h klines", flush=True)
done = 0
for i, s in enumerate(established):
    try:
        k = load_klines(s, "4h", "2019-09", "2026-07", market="um")
        if len(k): done += 1
    except Exception as e:
        print(f"  FAIL {s}: {e}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(established)} processed, {done} with 4h data", flush=True)
print(f"DONE 4h download: {done}/{len(established)}", flush=True)
