"""Background: expand the crypto universe. Download funding + 1d klines for a broad set of Binance
USD-M perps (incl. delisted names in the archive) so cross-sectional carry can be tested point-in-time
across a much wider, survivorship-honest cross-section. 1d + funding only (small); finer TFs later if needed."""
import sys, warnings, urllib.request, urllib.parse, xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")
from src.data.binance_bulk import load_funding, load_klines

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
syms, token = [], None
for _ in range(10):
    q = {"list-type": "2", "prefix": "data/futures/um/monthly/fundingRate/", "delimiter": "/", "max-keys": "1000"}
    if token: q["continuation-token"] = token
    root = ET.fromstring(urllib.request.urlopen(f"{S3}?{urllib.parse.urlencode(q)}", timeout=30).read())
    for p in root.iter(NS + "CommonPrefixes"):
        syms.append(p.find(NS + "Prefix").text.split("/")[-2])
    tk = root.find(NS + "NextContinuationToken")
    if tk is None or not tk.text: break
    token = tk.text
usdt = sorted(s for s in syms if s.endswith("USDT"))
print(f"downloading funding + 1d klines for {len(usdt)} USDT perps (survivorship-honest, incl. delisted)", flush=True)
done = 0
for i, s in enumerate(usdt):
    try:
        f = load_funding(s, "2019-09", "2026-07")
        k = load_klines(s, "1d", "2019-09", "2026-07", market="um")
        if len(f) and len(k): done += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(usdt)} processed, {done} with data", flush=True)
    except Exception as e:
        print(f"  FAIL {s}: {e}", flush=True)
print(f"DONE universe download: {done}/{len(usdt)} symbols with funding+1d", flush=True)
