"""Chain-fundamentals loader (DefiLlama) — the on-chain axis Coin Metrics cannot reach.

Coin Metrics' free tier covers network *activity* on 33 mostly-legacy coins and carries no network
metrics at all for the modern high-throughput chains (SOL, SUI, TON, APT, SEI, TIA, ARB, OP — market
data only). DefiLlama covers exactly those, free and without a key, on a different economic axis:
what the chain actually *earns*.

  - **fees** — total paid by users on the chain, daily USD (`/overview/fees/{chain}`)
  - **revenue** — the slice that accrues to the token itself (burn / validator take), daily USD
  - **TVL** — capital parked on the chain, daily USD (`/v2/historicalChainTvl/{chain}`)

Together these give a cash-flow valuation for a token — fee yield is the crypto earnings yield, and
market-cap-to-TVL is its price-to-book — rather than the address-count proxies H3 had to settle for.

**Point-in-time caveat, stated because it is the weak spot.** These series are *revised*: DefiLlama
adds protocol adapters over time and backfills them, so today's history for a chain is not what an
observer saw live. Fees and TVL for a large chain move mostly with its dominant protocols, which are
integrated early, but the tail is backfilled. There is no free vintaged snapshot to correct this, so
the honest handling is the shared `exec_lag` plus a stated assumption that revision noise is small
relative to the cross-sectional spread — and a result read with that in mind. A coverage-driven
backfill would, if anything, flatter a fee-growth signal.

    python -m src.data.defillama          # build/refresh data/cache/fundamentals/
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from src.config import CACHE_DIR  # noqa: E402
from src.data import onchain as oc  # noqa: E402  (Coin Metrics is where the market cap lives)

CACHE = CACHE_DIR / "fundamentals"
DL_FEES = "https://api.llama.fi/overview/fees/{chain}"
DL_TVL = "https://api.llama.fi/v2/historicalChainTvl/{chain}"

# Binance symbol (repo price panel) → DefiLlama chain. Only L1/L2 tokens whose chain has its own fee
# series: the token is then a claim on that chain's economics, which is what makes the ratio a
# valuation. DeFi governance tokens are deliberately out — mapping a token to a protocol's fees needs
# a per-protocol judgement (which deployment, which fee split) that no automatic mapping gets right.
UNIVERSE: dict[str, str] = {
    "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana", "BNBUSDT": "BSC",
    "AVAXUSDT": "Avalanche", "POLUSDT": "Polygon", "TRXUSDT": "Tron", "TONUSDT": "TON",
    "APTUSDT": "Aptos", "SUIUSDT": "Sui", "SEIUSDT": "Sei", "NEARUSDT": "Near",
    "ARBUSDT": "Arbitrum", "OPUSDT": "OP Mainnet", "ADAUSDT": "Cardano", "XLMUSDT": "Stellar",
    "LTCUSDT": "Litecoin", "ALGOUSDT": "Algorand", "HBARUSDT": "Hedera", "FILUSDT": "Filecoin",
    "ICPUSDT": "ICP", "XTZUSDT": "Tezos", "INJUSDT": "Injective", "TIAUSDT": "Celestia",
    "STXUSDT": "Stacks", "RUNEUSDT": "Thorchain", "FTMUSDT": "Fantom", "SUSDT": "Sonic",
}
# Polkadot and Kaspa are mapped nowhere on purpose: DefiLlama returns an empty fee chart for Polkadot
# and a 500 for Kaspa, so neither has a series to load.

# Market cap is the denominator every valuation ratio needs, and DefiLlama does not serve it
# historically (nor does CoinGecko's free tier, which caps at 365 days). Coin Metrics does:
# `CapMrktEstUSD` is on the community tier for 27 of these 28 chains — including the modern ones its
# *network* metrics exclude. TON is absent from Coin Metrics altogether, so it carries no valuation.
CM_MCAP = {
    "BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol", "BNBUSDT": "bnb", "AVAXUSDT": "avax",
    "POLUSDT": "pol", "TRXUSDT": "trx", "APTUSDT": "apt", "SUIUSDT": "sui", "SEIUSDT": "sei",
    "NEARUSDT": "near", "ARBUSDT": "arb", "OPUSDT": "op", "ADAUSDT": "ada", "XLMUSDT": "xlm",
    "LTCUSDT": "ltc", "ALGOUSDT": "algo", "HBARUSDT": "hbar", "FILUSDT": "fil", "ICPUSDT": "icp",
    "XTZUSDT": "xtz", "INJUSDT": "inj", "TIAUSDT": "tia", "STXUSDT": "stx", "RUNEUSDT": "rune",
    "FTMUSDT": "ftm", "SUSDT": "s",
}

STALE_DAYS = 30  # a chain whose newest observation is older than this is no longer being tracked


def _get(url: str, params: dict | None = None, retries: int = 4) -> dict | list | None:
    """One DefiLlama call. Free and unauthenticated, so the only failure modes worth handling are
    transport and rate limiting; anything else is logged and skipped rather than aborting a build."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, timeout=60)
        except requests.RequestException as e:
            print(f"    ! {url} network error ({e}); retry {attempt + 1}/{retries}")
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code == 429:
            print(f"    ! {url} rate-limited; backing off"); time.sleep(3.0 * (attempt + 1)); continue
        if r.status_code != 200:
            print(f"    ! {url} HTTP {r.status_code}; retry {attempt + 1}/{retries}")
            time.sleep(1.0 * (attempt + 1)); continue
        return r.json()
    print(f"    ! {url} exhausted retries")
    return None


def _chart_to_series(rows: list) -> pd.Series:
    """DefiLlama's [[unix, value], …] or [{date, tvl}, …] → daily UTC Series."""
    if not rows:
        return pd.Series(dtype=float)
    if isinstance(rows[0], dict):
        pairs = [(r.get("date"), r.get("tvl")) for r in rows]
    else:
        pairs = [(r[0], r[1]) for r in rows]
    s = pd.Series({pd.Timestamp(int(t), unit="s", tz="UTC").normalize(): v
                   for t, v in pairs if t is not None and v is not None}, dtype=float)
    return s.sort_index()


def chain_fees(chain: str, kind: str = "dailyFees") -> pd.Series:
    """Daily USD fees (or revenue) for one chain."""
    d = _get(DL_FEES.format(chain=chain),
             {"dataType": kind, "excludeTotalDataChartBreakdown": "true"})
    return _chart_to_series((d or {}).get("totalDataChart", []))


def chain_tvl(chain: str) -> pd.Series:
    """Daily USD TVL for one chain."""
    return _chart_to_series(_get(DL_TVL.format(chain=chain)) or [])


def build(force: bool = False) -> None:
    """Fetch fees / revenue / TVL for the mapped chains and cache one wide panel per metric."""
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "fees.parquet"
    if done.exists() and not force:
        print(f"fundamentals cache present ({done}); pass force=True to refresh"); return

    panels: dict[str, dict[str, pd.Series]] = {"fees": {}, "revenue": {}, "tvl": {}, "mcap": {}}
    print(f"DefiLlama: fetching {len(UNIVERSE)} chains × fees/revenue/TVL…")
    for i, (sym, chain) in enumerate(UNIVERSE.items(), 1):
        got = {}
        for name, s in (("fees", chain_fees(chain, "dailyFees")),
                        ("revenue", chain_fees(chain, "dailyRevenue")),
                        ("tvl", chain_tvl(chain))):
            if s.empty:
                continue
            panels[name][sym] = s
            got[name] = (len(s), s.index.max().date())
        if not got:
            print(f"  [{i:2d}/{len(UNIVERSE)}] {sym:10s}({chain:12s}) NO DATA — skipped")
            continue
        print(f"  [{i:2d}/{len(UNIVERSE)}] {sym:10s}({chain:12s}) "
              + "  ".join(f"{k}={v[0]}d→{v[1]}" for k, v in got.items()))
        time.sleep(0.15)

    print(f"Coin Metrics: market cap for {len(CM_MCAP)} chains (the valuation denominator)…")
    mcap: dict[str, pd.Series] = {}
    entitled = oc.free_metrics(sorted(set(CM_MCAP.values())))
    for sym, code in CM_MCAP.items():
        if "CapMrktEstUSD" not in entitled.get(code, set()):
            print(f"    ! {sym} ({code}): CapMrktEstUSD not on the community tier — no valuation ratio")
            continue
        d = oc._cm_get(code, ["CapMrktEstUSD"], "2019-01-01", oc._today())
        if d.empty:
            print(f"    ! {sym} ({code}): market-cap query returned nothing")
            continue
        mcap[sym] = d["CapMrktEstUSD"]
        time.sleep(0.08)
    if mcap:
        panels["mcap"] = mcap
        print(f"  → market cap: {len(mcap)} chains")

    for name, cols in panels.items():
        if cols:
            pd.DataFrame(cols).sort_index().to_parquet(CACHE / f"{name}.parquet")
            print(f"  → {name}: {len(cols)} chains")
    print("fundamentals cache build DONE")


def load(metric: str) -> pd.DataFrame:
    """Load a cached wide panel (date × Binance symbol), tz-aware UTC index."""
    df = pd.read_parquet(CACHE / f"{metric}.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def live_universe(as_of: pd.Timestamp | None = None, stale_days: int = STALE_DAYS) -> list[str]:
    """Chains DefiLlama is still tracking. Tezos, for one, stops in Jan-2025: a series that ends
    mid-panel would otherwise hold its last value forward and quietly become a constant tilt — the
    same failure mode as the dead ERC-20 shells in the Coin Metrics universe."""
    fees = load("fees")
    end = as_of or fees.index.max()
    last = fees.apply(lambda c: c.last_valid_index())
    dead = sorted(s for s in fees.columns
                  if last[s] is None or (end - last[s]).days > stale_days)
    if dead:
        print(f"  fundamentals: dropping {len(dead)} stale chains (no data in {stale_days}d): "
              + ", ".join(f"{s} last {last[s].date()}" for s in dead))
    return [s for s in fees.columns if s not in dead]


if __name__ == "__main__":
    import sys
    build(force="--force" in sys.argv)
