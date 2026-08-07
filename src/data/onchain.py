"""On-chain / network-data loader — the one genuinely new information source for crypto (H3).

Two free, no-key sources, each probed live (2026-08) before wiring in:

- **Coin Metrics community API** (`community-api.coinmetrics.io/v4`). Cross-asset daily network
  metrics. The community tier is a strict subset of the full catalog — most of the catalog 403s
  at query time. The *free* set actually usable here, confirmed by per-(asset,metric) probing:
  `PriceUSD, CapMrktCurUSD, AdrActCnt, TxCnt, TxTfrCnt, SplyCur, HashRate, CapMVRVCur`.
  Pay-walled (so **not** used): exchange net-flows (`Flow*ExNtv`), adjusted transfer value
  (`TxTfrValAdjUSD`, the NVT numerator), fees, miner revenue, realized cap, 7d/30d active addrs.
  So H3 here is **network-activity & valuation** on-chain, not exchange-flow on-chain — stated
  honestly in docs/strategies/ONCHAIN.md.

- **blockchain.com charts** (`api.blockchain.info/charts`). BTC-only, free, full history since 2009.
  Supplies exactly the two series CM pay-walls that a BTC *time-series* test wants:
  `estimated-transaction-volume-usd` (→ NVT) and `miners-revenue` (→ Puell multiple).

**Point-in-time discipline.** Every metric dated `t` is an end-of-day-`t` aggregate over that day's
(immutable) block data, finalised shortly after 00:00 UTC on `t+1`. Network-activity counts are not
revised (block data cannot change); realized-cap-derived MVRV carries a minor methodology-revision
risk, flagged in the report. Panels are stamped at `t` and consumed with the engine's `exec_lag ≥ 2`
(t+2 execution), so a signal never sees data it could not have had — the same delay every sleeve uses.

    python -m src.data.onchain          # build/refresh the cache under data/cache/onchain/
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from src.config import CACHE_DIR  # noqa: E402

CACHE = CACHE_DIR / "onchain"
CM_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BC_BASE = "https://api.blockchain.info/charts"

# ── universe: Binance symbol (repo price panel) → Coin Metrics asset code ─────────────────────
# Resolved by discover_universe(): the intersection of (repo crypto_1d_close panel) ∩ (names with a
# free CM AdrActCnt series ≥200d). Tokenised gold (paxg) and zero-liquidity/delisted names
# (xem, mkr, ren, omg, bal, eos) are dropped — they have on-chain rows but no tradable price here.
UNIVERSE: dict[str, str] = {
    "BTCUSDT": "btc", "ETHUSDT": "eth", "XRPUSDT": "xrp", "DOGEUSDT": "doge", "ZECUSDT": "zec",
    "ADAUSDT": "ada", "LINKUSDT": "link", "BCHUSDT": "bch", "AAVEUSDT": "aave", "LTCUSDT": "ltc",
    "DOTUSDT": "dot", "UNIUSDT": "uni", "XLMUSDT": "xlm", "TRXUSDT": "trx", "1000SHIBUSDT": "shib_eth",
    "CRVUSDT": "crv", "ETCUSDT": "etc", "DASHUSDT": "dash", "ICPUSDT": "icp", "LDOUSDT": "ldo",
    "ALGOUSDT": "algo", "POLUSDT": "pol_eth", "VETUSDT": "vet_eth", "SNXUSDT": "snx",
    "COMPUSDT": "comp", "MANAUSDT": "mana", "XTZUSDT": "xtz", "SUSHIUSDT": "sushi", "NEOUSDT": "neo",
    "ZILUSDT": "zil_eth", "BATUSDT": "bat", "1INCHUSDT": "1inch", "YFIUSDT": "yfi", "ZRXUSDT": "zrx",
    "QTUMUSDT": "qtum_eth", "LRCUSDT": "lrc_eth", "KNCUSDT": "knc",
}
# Free CM metrics used, and the wide-panel filename each lands in.
CM_METRICS = ["PriceUSD", "CapMrktCurUSD", "AdrActCnt", "TxCnt", "TxTfrCnt", "SplyCur", "CapMVRVCur"]
STABLES = {"usdt": "usdt", "usdc": "usdc", "dai": "dai", "busd": "busd"}  # SplyCur → aggregate supply
START = "2019-01-01"


# ── Coin Metrics ──────────────────────────────────────────────────────────────────────────────
def _cm_get(asset: str, metrics: list[str], start: str, end: str, retries: int = 4) -> pd.DataFrame:
    """One CM community call → long frame (time, metric cols). A 403 (a pay-walled metric for this
    asset) raises PermissionError so the caller can retry with a narrower metric set; other errors
    back off and retry, then log-and-return-empty rather than aborting the whole build."""
    params = dict(assets=asset, metrics=",".join(metrics), start_time=start, end_time=end,
                  frequency="1d", page_size=10000)
    for attempt in range(retries):
        try:
            r = requests.get(CM_BASE, params=params, timeout=40)
        except requests.RequestException as e:
            print(f"    ! CM {asset} network error ({e}); retry {attempt + 1}/{retries}")
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code == 403:
            raise PermissionError(r.json().get("error", {}).get("message", "forbidden"))
        if r.status_code == 429:
            print(f"    ! CM {asset} rate-limited; backing off"); time.sleep(2.0 * (attempt + 1)); continue
        if r.status_code != 200:
            print(f"    ! CM {asset} HTTP {r.status_code}; retry {attempt + 1}/{retries}")
            time.sleep(1.0 * (attempt + 1)); continue
        rows, tok = r.json().get("data", []), r.json().get("next_page_token")
        while tok:  # defensive: single-asset daily is one page, but honour pagination
            rp = requests.get(CM_BASE, params={**params, "next_page_token": tok}, timeout=40)
            rows += rp.json().get("data", []); tok = rp.json().get("next_page_token")
        if not rows:
            print(f"    ! CM {asset} returned no rows for {metrics}")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.normalize()
        keep = [m for m in metrics if m in df.columns]
        return df.set_index("time")[keep].apply(pd.to_numeric, errors="coerce")
    print(f"    ! CM {asset} exhausted retries for {metrics}")
    return pd.DataFrame()


def _cm_asset_all(asset: str) -> pd.DataFrame:
    """All free CM metrics for one asset. Tries the full set in one call; on a 403 (one metric this
    asset lacks, e.g. TRX has no MVRV) falls back to fetching each metric alone and keeping what works."""
    try:
        return _cm_get(asset, CM_METRICS, START, _today())
    except PermissionError:
        cols = {}
        for m in CM_METRICS:
            try:
                d = _cm_get(asset, [m], START, _today())
                if not d.empty:
                    cols[m] = d[m]
            except PermissionError:
                pass  # this metric is pay-walled for this asset — expected, skip
            time.sleep(0.05)
        return pd.DataFrame(cols)


def _today() -> str:
    return pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")


# ── blockchain.com (BTC only) ───────────────────────────────────────────────────────────────────
def blockchain_chart(chart: str, retries: int = 3) -> pd.Series:
    """One blockchain.com chart → daily UTC Series (BTC). Free, no key, full history."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{BC_BASE}/{chart}", params=dict(timespan="all", format="json",
                                                               sampled="false"), timeout=40)
            if r.status_code != 200:
                print(f"    ! blockchain.com {chart} HTTP {r.status_code}; retry"); time.sleep(1.5); continue
            v = r.json().get("values", [])
            if not v:
                print(f"    ! blockchain.com {chart} empty"); return pd.Series(dtype=float)
            s = pd.Series({pd.Timestamp(p["x"], unit="s", tz="UTC").normalize(): p["y"] for p in v})
            return s.sort_index()
        except requests.RequestException as e:
            print(f"    ! blockchain.com {chart} error ({e}); retry"); time.sleep(1.5)
    return pd.Series(dtype=float)


# ── build the cache ─────────────────────────────────────────────────────────────────────────────
def build(force: bool = False) -> None:
    """Fetch the universe's free on-chain metrics + stablecoin supply + BTC blockchain.com series,
    pivot to wide (date × Binance-symbol) panels, and cache each metric to parquet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "AdrActCnt.parquet"
    if done.exists() and not force:
        print(f"on-chain cache present ({done}); pass force=True to refresh"); return

    per_metric: dict[str, dict[str, pd.Series]] = {m: {} for m in CM_METRICS}
    print(f"Coin Metrics: fetching {len(UNIVERSE)} assets × {len(CM_METRICS)} free metrics…")
    for i, (sym, code) in enumerate(UNIVERSE.items(), 1):
        d = _cm_asset_all(code)
        got = list(d.columns)
        for m in got:
            per_metric[m][sym] = d[m]
        print(f"  [{i:2d}/{len(UNIVERSE)}] {sym:12s}({code:9s}) {len(d):5d}d  {got}")
        time.sleep(0.08)
    for m, cols in per_metric.items():
        if cols:
            pd.DataFrame(cols).sort_index().to_parquet(CACHE / f"{m}.parquet")
    print(f"  → wrote {len([m for m in per_metric if per_metric[m]])} metric panels")

    print("Coin Metrics: stablecoin supply (SSR / risk-on)…")
    stab = {}
    for name, code in STABLES.items():
        try:
            d = _cm_get(code, ["SplyCur"], START, _today())
            if not d.empty:
                stab[name] = d["SplyCur"]
        except PermissionError:
            print(f"    ! {name} SplyCur forbidden")
        time.sleep(0.08)
    if stab:
        pd.DataFrame(stab).sort_index().to_parquet(CACHE / "stablecoin_supply.parquet")
        print(f"  → stablecoins: {list(stab)}")

    print("blockchain.com: BTC transaction value + miner revenue + hashrate…")
    bc = {c: blockchain_chart(c) for c in
          ("estimated-transaction-volume-usd", "miners-revenue", "hash-rate",
           "n-unique-addresses", "n-transactions")}
    bc_df = pd.DataFrame({k: v for k, v in bc.items() if not v.empty}).sort_index()
    if not bc_df.empty:
        bc_df.to_parquet(CACHE / "btc_blockchain.parquet")
        print(f"  → BTC series: {list(bc_df.columns)}  {bc_df.index.min().date()}..{bc_df.index.max().date()}")
    print("on-chain cache build DONE")


def load(metric: str) -> pd.DataFrame:
    """Load a cached wide on-chain panel (date × Binance-symbol), tz-aware UTC index."""
    df = pd.read_parquet(CACHE / f"{metric}.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


if __name__ == "__main__":
    import sys
    build(force="--force" in sys.argv)
