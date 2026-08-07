"""Does excluding the compressed-funding megacaps help carry? Test the MID-CAP liquidity BAND
(ranks lo..hi by trailing dollar volume, PIT survivorship-free) vs the top-N baseline, using the
carry session's own refined construction (run_carry / carry_xs). Data is read OFFLINE (cached monthly
parquets only) so the 829-symbol load never stalls on a network probe.

    python scripts/carry/check_carry_band.py
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from scripts.carry.run_carry_breadth import PPY, SEED, run_carry, vt  # noqa: E402
from src.config import RAW_DIR  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import carry_xs  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

_K = RAW_DIR / "futures/um/klines"
_F = RAW_DIR / "futures/um/fundingRate"
_LO = pd.Timestamp("2020-01-01", tz="UTC")
_HI = pd.Timestamp("2026-08-01", tz="UTC")


def _read(d, cols):
    files = sorted(d.glob("[0-9]*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[(df.index >= _LO) & (df.index < _HI)]


def load_offline():
    close, dvol, fund = {}, {}, {}
    for s in sorted(p.name for p in _K.iterdir() if p.is_dir()):
        if s in carry_xs.NON_CRYPTO_NATIVE or not (_F / s).exists():
            continue
        px = _read(_K / s / "1d", ["close", "quote_volume"])
        fr = _read(_F / s, ["last_funding_rate"])
        if px is None or fr is None or len(px) < 120:
            continue
        close[s], dvol[s], fund[s] = px["close"], px["quote_volume"], fr["last_funding_rate"]
    C = pd.DataFrame(close).sort_index()
    V = pd.DataFrame(dvol).reindex(C.index)
    fd = carry_xs.funding_daily(pd.DataFrame(fund)).reindex(C.index)
    return C, V, fd


def score(C, fd, sig, elig, btc, label):
    net = vt(run_carry(C, fd, sig, elig, beta_ret=btc)[0])
    s = summarise(net, PPY)
    p5 = bootstrap_sharpe(net, PPY, 500, SEED).get("sharpe_p5", float("nan")) if s["sharpe_ann"] > 0.2 else float("nan")
    py = {int(y): round(float((g.dropna().mean() / g.dropna().std(ddof=1)) * (PPY ** 0.5)), 1)
          for y, g in net.groupby(net.index.year) if g.dropna().std(ddof=1) > 0}
    print(f"  {label:16s} Sharpe {s['sharpe_ann']:+.2f}  MC-P5 {p5:+.2f}  DD {s['max_dd']:+.0%}  "
          f"months+ {s['months_in_profit']:.0%}  per-yr {py}", flush=True)


def main():
    C, V, fd = load_offline()
    btc = C["BTCUSDT"].pct_change() if "BTCUSDT" in C else C.iloc[:, 0].pct_change()
    sig = carry_xs.signal_level(fd, 7)
    print(f"loaded {C.shape[1]} symbols (incl. delisted)\n=== top-N baseline (PIT) ===")
    score(C, fd, sig, carry_xs.pit_eligible(V, 50), btc, "top-50")
    score(C, fd, sig, carry_xs.pit_eligible(V, 100), btc, "top-100")
    print("\n=== MID-CAP BANDS (exclude compressed-funding megacaps) ===")
    for lo, hi in [(10, 110), (20, 120), (30, 130), (20, 100), (20, 150), (40, 140)]:
        score(C, fd, sig, carry_xs.pit_eligible_band(V, lo, hi), btc, f"band {lo}-{hi}")
    print("\nCARRY-BAND OK")


if __name__ == "__main__":
    main()
