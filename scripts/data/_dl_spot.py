"""Background: download spot 1d klines for the 50-name perp panel (for the delta-neutral
cash-and-carry basis trade — needs spot leg to measure real basis risk, not assume a perfect hedge)."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.data.binance_bulk import load_klines
CRYPTO = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","TRXUSDT","DOGEUSDT","ZECUSDT",
 "ADAUSDT","XMRUSDT","LINKUSDT","XLMUSDT","BCHUSDT","LTCUSDT","HBARUSDT","1000SHIBUSDT",
 "AVAXUSDT","SUIUSDT","UNIUSDT","NEARUSDT","DOTUSDT","AAVEUSDT","1000PEPEUSDT","ICPUSDT",
 "ETCUSDT","QNTUSDT","ALGOUSDT","ATOMUSDT","FILUSDT","ARBUSDT","APTUSDT","INJUSDT",
 "DASHUSDT","VETUSDT","FETUSDT","CRVUSDT","1000LUNCUSDT","STXUSDT","LDOUSDT","IMXUSDT",
 "XTZUSDT","JASMYUSDT","CFXUSDT","OPUSDT","1000FLOKIUSDT","ENSUSDT","COMPUSDT","GRTUSDT",
 "IOTAUSDT","RUNEUSDT"]
ok=0
for s in CRYPTO:
    for tf in ("1d","4h"):
        try:
            df = load_klines(s, tf, "2020-01", "2026-07", market="spot")
            if len(df): ok+=1
            print(f"spot {s} {tf}: {len(df)} bars", flush=True)
        except Exception as e:
            print(f"FAIL spot {s} {tf}: {e}", flush=True)
print(f"DONE spot download, {ok} series", flush=True)
