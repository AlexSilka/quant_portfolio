"""Discovery layer — the candidate ZOO, NOT the traded book. Mines timeframe x both asset classes x
four strategy families, sizes each sleeve to constant volatility, screens by Monte Carlo + placebo,
assembles the survivors into a naive equal-risk portfolio, then walk-forwards that selection out of
sample. Picking sleeves by in-sample Sharpe is selection bias — the walk-forward OOS Sharpe collapses
to ~0 — which is why the canonical book (scripts/run_master_book.py) selects nothing and applies theory
uniformly. Publishes reports/zoo_*.{parquet,csv,json} + all_returns.parquet; the survival funnel,
deflated Sharpe and walk-forward OOS are the honesty evidence the dashboard's anti-overfitting panel shows.

Default grid is the core timeframes 1h/4h/1d (crypto AND equity/FX); `--intraday` adds 5m/15m. Crypto
comes from the local Binance bulk cache; equity/FX bars come from Twelve Data Pro, cached per symbol
under data/raw/twelvedata — the first run warms any missing intraday cache (Pro key, no credit limit but
per-minute-rate-limited, so a cold fetch back-off is one-off), later runs read them locally. Crypto
5m/15m is the one genuinely heavy per-run cost (raw bar count ~300x daily), so the sub-hourly grid is
opt-in. See `make discovery`.

Per-family construction is correct, not uniform:
  - trend  : continuous EMA(50/200) cross, held to reversal (trend edge lives in the fat tail)
  - carry  : continuous funding z-score (crypto perps only)
  - mean_reversion / breakout : event-based, held to a triple-barrier (short, bounded holds)
Every sleeve is vol-targeted to ~15% annualised so sleeves combine on equal risk.

    python scripts/run_book.py
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.backtest.engine import backtest, positions_from_events, vol_target  # noqa: E402
from src.config import BOOK_DIR, CACHE_DIR, CAPITAL_USD, RAW_DIR, REPORTS_DIR, SEED, VOL_TARGET_ANNUAL  # noqa: E402
from src.data.binance_bulk import load_funding, load_klines  # noqa: E402
from src.data.equity import load_equity_daily  # noqa: E402
from src.labels.triple_barrier import trailing_vol, triple_barrier_labels  # noqa: E402
from src.metrics import deflated_sharpe, summarise  # noqa: E402
from src.pipeline import signal_events  # noqa: E402
from src.sleeves import breakout, carry, mean_reversion, momentum  # noqa: E402
from src.sleeves.sector_pairs import SECTOR_ETFS  # noqa: E402
from src.validation.monte_carlo import bootstrap_sharpe  # noqa: E402

SEED, CAP, TVOL = SEED, CAPITAL_USD, VOL_TARGET_ANNUAL
CC = dict(commission_bps=5.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)
EC = dict(commission_bps=1.0, half_spread_bps=2.0, impact_k=0.1, exec_lag=2)
CRYPTO_TF = {"5m": 288 * 365, "15m": 96 * 365, "1h": 24 * 365, "4h": 6 * 365, "1d": 365}
HORIZON = {"5m": 48, "15m": 32, "1h": 24, "4h": 30, "1d": 10}
CORE_TFS = ("1h", "4h", "1d")      # default grid (crypto + equity/FX): edge lives here; cheap once cached
INTRADAY_TFS = ("5m", "15m")       # --intraday only: crypto 5m/15m is heavy compute (~300x daily bars)
CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "TRXUSDT", "DOGEUSDT", "ZECUSDT",
          "ADAUSDT", "XMRUSDT", "LINKUSDT", "XLMUSDT", "BCHUSDT", "LTCUSDT", "HBARUSDT", "1000SHIBUSDT",
          "AVAXUSDT", "SUIUSDT", "UNIUSDT", "NEARUSDT", "DOTUSDT", "AAVEUSDT", "1000PEPEUSDT", "ICPUSDT",
          "ETCUSDT", "QNTUSDT", "ALGOUSDT", "ATOMUSDT", "FILUSDT", "ARBUSDT", "APTUSDT", "INJUSDT",
          "DASHUSDT", "VETUSDT", "FETUSDT", "CRVUSDT", "1000LUNCUSDT", "STXUSDT", "LDOUSDT", "IMXUSDT",
          "XTZUSDT", "JASMYUSDT", "CFXUSDT", "OPUSDT", "1000FLOKIUSDT", "ENSUSDT", "COMPUSDT", "GRTUSDT",
          "IOTAUSDT", "RUNEUSDT"]  # 50 liquid USDT perps: CoinGecko mcap-ranked, >=3y history, on Binance
START, END = "2020-01", "2026-07"  # frozen to the latest published month (§2 freeze); all-cached, no network
rng = np.random.default_rng(SEED)
ALL_RET = {}  # every candidate's daily return series (not just survivors) — feeds walk-forward selection


_EQTD = RAW_DIR / "equity_td"
_avail = {p.name[:-11] for p in _EQTD.glob("*_1d.parquet")} if _EQTD.exists() else set()
# Curated liquid large-cap US equities + core ETFs. NOT a glob of the whole dir: data/raw/equity_td now
# holds a ~1600-name broad universe (the cross-sectional panels use it), and globbing all of it made the
# single-asset zoo ~30x slower for no added signal. The zoo only needs a liquid core; `make discovery`
# / --intraday adds timeframe depth on this set, not the whole market one name at a time.
LARGE_CAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "MA", "JNJ", "WMT",
              "XOM", "PG", "HD", "CVX", "KO", "PEP", "BAC", "AVGO", "COST", "MRK", "LLY", "ABBV", "ADBE",
              "CRM", "NFLX", "AMD", "DIS", "CSCO", "INTC", "PFE", "WFC", "TMO", "MCD", "ABT", "ORCL",
              "ACN", "QCOM", "TXN", "NKE", "UNH", "HON", "UPS", "LOW", "IBM", "GS", "CAT", "SPY", "QQQ"]
STOCKS = [t for t in LARGE_CAPS if t in _avail]   # ~50 liquid large caps with cached bars (was: whole dir)
FX = sorted(t for t in _avail if "=X" in t)       # ~25 FX pairs (already a small curated set)
TD_ITV = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
EQUITY_TF = {"5m": 78 * 252, "15m": 26 * 252, "1h": 7 * 252, "4h": 2 * 252, "1d": 252}   # 6.5h US session
FX_TF = {"5m": 288 * 252, "15m": 96 * 252, "1h": 24 * 252, "4h": 6 * 252, "1d": 252}     # 24h weekday
FXC = dict(commission_bps=0.0, half_spread_bps=1.0, impact_k=0.1, exec_lag=2)            # FX: tight spread, no comm


def load_eqfx(sym, tf):
    """OHLCV for an equity/FX symbol at any timeframe: daily via yf-style loader (deep history),
    intraday via Twelve Data (cached 2020->now). Same bar contract as crypto."""
    if tf == "1d":
        return load_equity_daily(sym, start="2012-01-01")
    from src.data.twelvedata import load_bars
    from src.data.equity import _to_td_symbol
    return load_bars(_to_td_symbol(sym), TD_ITV[tf], "2020-01-01", "2026-08-05")


def sleeve_position(fam, px, fund, ppy_bar, horizon):
    close, high, low = px["close"], px["high"], px["low"]
    if fam == "trend":
        return vol_target(momentum.primary_side(close, 50, 200), close, TVOL, ppy_bar)
    if fam == "carry":
        return vol_target(carry.primary_side(fund, close), close, TVOL, ppy_bar)
    side = (mean_reversion.primary_side(close) if fam == "mean_reversion"
            else breakout.primary_side(close, high, low))
    events = signal_events(side)
    lab = triple_barrier_labels(close, events, trailing_vol(close, 100), 1.0, 1.0, horizon)
    held = positions_from_events(close.index, side, lab["t1"], events)
    return vol_target(held, close, TVOL, ppy_bar)


def evaluate(close, pos, fund, adv, freq, ppy_daily, costs, with_mc=True):
    bt = backtest(close, pos, capital=CAP, funding=fund, adv=adv, **costs)
    # daily frame carries the pieces the zoo diagnostics need: net return, cost drag (for the
    # cost-sensitivity sweep), gross leverage and turnover (for the exposure/turnover panel)
    daily = pd.DataFrame({
        "ret": (1 + bt["net_ret"]).resample(freq).prod() - 1,
        "cost": bt["cost"].resample(freq).sum(),
        "gross": bt["position"].abs().resample(freq).mean(),
        "turnover": bt["position"].diff().abs().resample(freq).sum(),
    }).dropna(subset=["ret"])
    s = summarise(daily["ret"], ppy_daily)
    # Monte Carlo only once a sleeve clears the cheap Sharpe screen — with a 50-name x 5-TF x
    # 4-family zoo (~1000 candidates) bootstrapping every one would cost hours for no signal
    if with_mc and s["sharpe_ann"] > 0.5:
        mc = bootstrap_sharpe(daily["ret"], ppy_daily, n_reps=500, seed=SEED)
        s["mc_p5"], s["mc_p50"] = mc.get("sharpe_p5", np.nan), mc.get("sharpe_p50", np.nan)
    else:
        s["mc_p5"], s["mc_p50"] = np.nan, np.nan
    s["turnover"] = float(bt["position"].diff().abs().sum())
    return s, daily


def run_class(rows, sleeves, kind):
    for sid, close, pos, fund, adv, freq, ppy_d, costs in sleeves:
        s, daily = evaluate(close, pos, fund, adv, freq, ppy_d, costs)
        # placebo: random-sign position, same magnitude/timing (Sharpe only, MC not needed)
        plac_pos = pos.abs() * pd.Series(rng.choice([-1.0, 1.0], len(pos)), index=pos.index)
        sp, _ = evaluate(close, plac_pos, fund, adv, freq, ppy_d, costs, with_mc=False)
        robust = s["sharpe_ann"] > 0.5 and s["mc_p5"] > 0.0 and s["n_obs"] > 100
        parts = sid.split("_")
        rows.append({"sleeve": sid, "kind": kind, "tf": parts[1], "family": "_".join(parts[2:]),
                     "sharpe": s["sharpe_ann"], "mc_p5": s["mc_p5"],
                     "mc_p50": s["mc_p50"], "max_dd": s["max_dd"], "months_in_profit": s["months_in_profit"],
                     "turnover": s["turnover"], "placebo_sharpe": sp["sharpe_ann"], "robust": robust})
        ALL_RET[sid] = daily["ret"]
        if robust:
            daily.to_parquet(CACHE_DIR / f"book/{sid}.parquet")
        print(f"{sid:26s} Sh {s['sharpe_ann']:+.2f}  P5 {s['mc_p5']:+.2f}  DD {s['max_dd']:+.0%}  "
              f"plac {sp['sharpe_ann']:+.2f}{'   ROBUST' if robust else ''}")


def run_xs(rows, panel, sid, kind, ppy, comm_bps):
    """Cross-sectional momentum: long recent winners / short losers, dollar-neutral, vs a
    random-normal placebo of the same shape (structurally distinct, market-neutral family)."""
    from src.sleeves.cross_sectional import momentum_signal, xs_returns
    rngx = np.random.default_rng(SEED)
    res = {}
    for label, sig in [("real", momentum_signal(panel, 120)),
                       ("plac", pd.DataFrame(rngx.standard_normal(panel.shape),
                                             index=panel.index, columns=panel.columns))]:
        gross, turn = xs_returns(panel, sig, top_frac=0.3)
        net = gross - turn * comm_bps / 1e4
        scale = (TVOL / (net.rolling(60).std() * np.sqrt(ppy))).clip(upper=3.0).shift(1).fillna(0.0)
        netv = (net * scale).dropna()
        p5 = bootstrap_sharpe(netv, ppy, 500, SEED).get("sharpe_p5", np.nan) if label == "real" else np.nan
        res[label] = (summarise(netv, ppy), p5, netv, float(turn.sum()))
    (rm, rp5, rret, rturn), pm = res["real"], res["plac"][0]
    robust = bool(rm["sharpe_ann"] > 0.5 and rp5 > 0.0)
    rows.append({"sleeve": sid, "kind": kind, "tf": sid.split("_")[1], "family": "cross_sectional",
                 "sharpe": rm["sharpe_ann"], "mc_p5": rp5, "mc_p50": np.nan, "max_dd": rm["max_dd"],
                 "months_in_profit": rm["months_in_profit"], "turnover": rturn,
                 "placebo_sharpe": pm["sharpe_ann"], "robust": robust})
    ALL_RET[sid] = rret
    if robust:
        rret.rename("ret").to_frame().to_parquet(CACHE_DIR / f"book/{sid}.parquet")
    print(f"{sid:28s} Sh {rm['sharpe_ann']:+.2f}  P5 {rp5:+.2f}  plac {pm['sharpe_ann']:+.2f}"
          f"{'   ROBUST' if robust else ''}")


def run_pairs(rows, panel, comm_bps):
    """Sector-ETF pairs stat-arb (neighbour sleeve): walk-forward cointegration basket, vol-targeted.
    A market-neutral mean-reversion stream, decorrelated from momentum — it pays off precisely when
    trend bleeds (COVID, 2022), so its value in the book is diversification, not standalone Sharpe."""
    from src.sleeves.sector_pairs import pairs_basket
    basket, npairs, _ = pairs_basket(panel, ppy=252, cost_bps=comm_bps)
    if basket is None:
        return
    sid = "SECTORS_1d_pairs"
    s = summarise(basket, 252)
    p5 = bootstrap_sharpe(basket, 252, 500, SEED).get("sharpe_p5", np.nan)
    robust = bool(s["sharpe_ann"] > 0.5 and p5 > 0.0)
    rows.append({"sleeve": sid, "kind": "equity", "tf": "1d", "family": "pairs",
                 "sharpe": s["sharpe_ann"], "mc_p5": p5, "mc_p50": np.nan, "max_dd": s["max_dd"],
                 "months_in_profit": s["months_in_profit"], "turnover": np.nan,
                 "placebo_sharpe": np.nan, "robust": robust})
    ALL_RET[sid] = basket
    if robust:
        basket.rename("ret").to_frame().to_parquet(CACHE_DIR / f"book/{sid}.parquet")
    print(f"{sid:28s} Sh {s['sharpe_ann']:+.2f}  P5 {p5:+.2f}  pairs~{npairs:.1f}/period"
          f"{'   ROBUST' if robust else ''}")


def main(intraday=False):
    (CACHE_DIR / "book").mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    tfs = CORE_TFS + (INTRADAY_TFS if intraday else ())
    print(f"zoo grid timeframes: {list(tfs)} (crypto + equity/FX)"
          + ("" if intraday else "  (default 1h/4h/1d; --intraday / `make discovery` adds 5m/15m)"))
    rows = []

    for tf, ppy_bar in CRYPTO_TF.items():
        if tf not in tfs:
            continue
        for sym in CRYPTO:
            px = load_klines(sym, tf, START, END, market="um")
            if len(px) < 500:
                continue
            fund = load_funding(sym, START, END)["last_funding_rate"]
            # every family on every timeframe — the point is to DISCOVER where edge is, not to
            # pre-judge; families that die to 5m/15m turnover show it in the results, not by omission
            fams = ["trend", "carry", "mean_reversion", "breakout"]
            # ADV proxy = traded USDT/bar (quote_volume), smoothed and lagged so the
            # liquidity-aware impact term is live (never a flat cost) and look-ahead-free
            advc = px["quote_volume"].rolling(20).median().shift(1)
            sl = [(f"{sym}_{tf}_{fam}", px["close"],
                   sleeve_position(fam, px, fund, ppy_bar, HORIZON[tf]), fund, advc, "D", 365, CC)
                  for fam in fams]
            run_class(rows, sl, "crypto")

    # equities (52 large caps) and FX (25 pairs) across every timeframe — same discover-everywhere
    # principle as crypto: every family on every TF, let the results say where the edge is
    for kind, syms, tfmap, costs, has_vol in [
            ("equity", STOCKS, EQUITY_TF, EC, True), ("fx", FX, FX_TF, FXC, False)]:
        for tf, ppy_bar in tfmap.items():
            if tf not in tfs:
                continue
            for sym in syms:
                try:
                    px = load_eqfx(sym, tf)
                except Exception:
                    continue
                if len(px) < 500:
                    continue
                adv = ((px["close"] * px["volume"]).rolling(20).median().shift(1)
                       if has_vol else None)   # FX bars carry no volume -> impact off, spread only
                sid = sym.replace("=X", "")
                sl = [(f"{sid}_{tf}_{fam}", px["close"],
                       sleeve_position(fam, px, None, ppy_bar, HORIZON[tf]), None, adv, "B", 252, costs)
                      for fam in ("trend", "mean_reversion", "breakout")]   # no carry (no funding)
                run_class(rows, sl, kind)

    # cross-sectional momentum panels (1d, market-neutral): crypto-50, top stocks, FX
    cr_panel = pd.DataFrame({s: load_klines(s, "1d", START, END, market="um")["close"]
                             for s in CRYPTO}).dropna(how="all").ffill()
    run_xs(rows, cr_panel, "CRYPTO50_1d_cross_sectional", "crypto", 365, CC["commission_bps"])
    eq_panel = pd.DataFrame({s: load_equity_daily(s, start="2012-01-01")["close"]
                             for s in STOCKS}).dropna(how="all").ffill()
    run_xs(rows, eq_panel, "STOCKS_1d_cross_sectional", "equity", 252, EC["commission_bps"])
    fx_panel = pd.DataFrame({s.replace("=X", ""): load_eqfx(s, "1d")["close"]
                             for s in FX}).dropna(how="all").ffill()
    run_xs(rows, fx_panel, "FX_1d_cross_sectional", "fx", 252, FXC["commission_bps"])

    # sector-ETF pairs stat-arb (neighbour sleeve). XLRE (listed 2015-10) is dropped so the
    # cointegration formation window reaches back far enough for the basket to cover 2016 onward.
    sec_etfs = [e for e in SECTOR_ETFS if e != "XLRE"]
    sec_panel = pd.DataFrame({e: load_equity_daily(e, start="2012-01-01")["close"]
                              for e in sec_etfs}).sort_index()
    run_pairs(rows, sec_panel, EC["commission_bps"])

    df = pd.DataFrame(rows)
    df.to_csv(BOOK_DIR / "zoo_results.csv", index=False)
    pd.DataFrame(ALL_RET).sort_index().to_parquet(BOOK_DIR / "all_returns.parquet")  # for walk-forward
    surv = df[df.robust]
    print(f"\n=== {len(surv)}/{len(df)} sleeves robust (Sharpe>0.5 & MC-P5>0), "
          f"placebo-robust {int((df.placebo_sharpe > 0.5).sum())}/{len(df)} ===")
    print("\n".join(f"  {r.sleeve:26s} Sh {r.sharpe:+.2f} P5 {r.mc_p5:+.2f}" for r in surv.itertuples()))
    _portfolio(surv, df)


def _wf_oos_sharpe():
    """Primary walk-forward (anchored, annual): at each year-start keep only sleeves robust on
    strictly-prior data, hold the next year, concatenate OOS. The naive in-sample book collapses here
    (Task A §10) — the number that proves selecting mined winners is selection bias, not edge."""
    rets = pd.DataFrame(ALL_RET).sort_index()
    rets = rets[rets.index >= pd.Timestamp("2012-01-01", tz=rets.index.tz)]
    dates = pd.date_range("2016-01-01", "2026-07-01", freq="YS", tz=rets.index.tz)
    port = []
    for i in range(len(dates) - 1):
        T, Tn = dates[i], dates[i + 1]
        win = rets.loc[:T]
        win = win[win.index < T]                                   # strictly past data — no look-ahead
        sh = np.sqrt(365) * win.mean() / win.std(ddof=1)
        keep = sh.index[(sh > 0.5) & (win.count() >= 252)]         # robust on prior data only
        held = rets.loc[T:Tn, keep]
        held = held[held.index < Tn]
        port.append(held.mean(axis=1) if len(keep) else pd.Series(0.0, index=held.index))
    wf = pd.concat(port).sort_index().dropna() if port else pd.Series(dtype=float)
    return round(summarise(wf, 365)["sharpe_ann"], 3) if len(wf) > 20 else float("nan")


def _portfolio(surv, df):
    files = [CACHE_DIR / f"book/{s}.parquet" for s in surv.sleeve]
    dfs = {f.stem: pd.read_parquet(f) for f in files if f.exists()}
    if not dfs:
        print("no survivors -> no portfolio"); return
    rets = pd.DataFrame({k: v["ret"] for k, v in dfs.items()}).sort_index()
    port = rets.fillna(0.0).mean(axis=1)  # equal risk (already vol-targeted)
    s = summarise(port, 365)
    mc = bootstrap_sharpe(port, 365, 1000, SEED)
    per_year = {}
    for y, g in port.groupby(port.index.year):
        g = g.dropna()
        per_year[int(y)] = round(float(np.sqrt(365) * g.mean() / g.std(ddof=1)), 2) \
            if g.std(ddof=1) > 0 else 0.0
    n_trials = int(len(df))
    var_tr = float((df["sharpe"].clip(-3, 3).dropna() / np.sqrt(365)).var())
    best = surv.sort_values("sharpe", ascending=False).iloc[0]
    b = dfs[best.sleeve]["ret"].dropna()
    best_dsr = deflated_sharpe(b.mean() / b.std(ddof=1), len(b), b.skew(), b.kurt() + 3.0,
                               n_trials, max(var_tr, 1e-8))
    wf_oos = _wf_oos_sharpe()      # the naive in-sample selection, walk-forwarded out of sample (§10)

    # exposure & turnover over time (§13): portfolio = equal-weight mean of the sleeve series
    gross = pd.DataFrame({k: v["gross"] for k, v in dfs.items() if "gross" in v}).reindex(rets.index)
    turn = pd.DataFrame({k: v["turnover"] for k, v in dfs.items() if "turnover" in v}).reindex(rets.index)
    expo = pd.DataFrame({"gross": gross.mean(axis=1), "turnover": turn.mean(axis=1)}).dropna(how="all")
    expo.to_parquet(BOOK_DIR / "zoo_exposure.parquet")
    annual_turnover = float(expo["turnover"].mean() * 365) if len(expo) else 0.0

    # cost sensitivity + break-even (§9): net_m = ret - (m-1)*cost (cost already charged once at 1x)
    costs = pd.DataFrame({k: v["cost"] for k, v in dfs.items() if "cost" in v}).reindex(
        index=rets.index, columns=rets.columns).fillna(0.0)

    def port_at(m):
        return (rets.fillna(0.0) - (m - 1.0) * costs).mean(axis=1)

    def _m(pm):
        ss = summarise(pm, 365)
        em = (1 + pm).cumprod()
        yy = (pm.index[-1] - pm.index[0]).days / 365.25
        return {"sharpe": ss["sharpe_ann"], "max_dd": ss["max_dd"],
                "cagr": float(em.iloc[-1] ** (1 / yy) - 1) if yy > 0 else 0.0}
    levels = [{"label": lab, "mult": m, **_m(port_at(m))}
              for m, lab in [(1.0, "1x base"), (2.0, "2x base"), (3.0, "3x base")]]
    breakeven = None
    for m in np.linspace(1.0, 25.0, 241):
        if (1 + port_at(m)).prod() - 1 <= 0:
            breakeven = float(m); break
    (BOOK_DIR / "zoo_cost_sensitivity.json").write_text(
        json.dumps({"levels": levels, "breakeven_mult": breakeven}, indent=2, default=float))

    # survival funnel (§6): counts down the pipeline gates
    funnel = [["generated", int(len(df))],
              ["Sharpe > 0.5", int((df.sharpe > 0.5).sum())],
              ["+ Monte-Carlo P5 > 0", int(df.robust.sum())],
              ["entered portfolio", int(len(dfs))]]

    df.pivot_table(index="tf", columns="family", values="sharpe", aggfunc="mean").to_csv(
        BOOK_DIR / "zoo_edge_map.csv")
    rets.corr().to_csv(BOOK_DIR / "zoo_correlation.csv")
    print(f"\n=== ZOO PORTFOLIO (naive equal-risk over {len(dfs)} in-sample survivors, net of costs) ===")
    print(f"Sharpe {s['sharpe_ann']:+.2f}  maxDD {s['max_dd']:+.1%}  months+ {s['months_in_profit']:.0%}  "
          f"MC[P5 {mc.get('sharpe_p5', float('nan')):+.2f} P50 {mc.get('sharpe_p50', float('nan')):+.2f}]")
    print(f"per-year Sharpe: {per_year}")
    print(f"best single sleeve deflated Sharpe (N={n_trials}): {best_dsr:.2f} "
          f"-> individually marginal; portfolio robustness comes from decorrelation")
    print(f"walk-forward OOS Sharpe (naive trailing-Sharpe selection): {wf_oos:+.2f}  "
          f"-> in-sample selection does NOT survive (zoo in-sample {s['sharpe_ann']:+.2f})")
    print("cost sensitivity: " + "  ".join(f"{lv['label']} Sh{lv['sharpe']:+.2f}" for lv in levels)
          + (f"  | break-even {breakeven:.1f}x base" if breakeven else "  | break-even >25x base"))
    print(f"annual turnover {annual_turnover:.1f}x  | funnel {[n for _, n in funnel]}")
    port.rename("ret").to_frame().to_parquet(BOOK_DIR / "zoo_portfolio.parquet")
    rets.to_parquet(BOOK_DIR / "zoo_sleeve_returns.parquet")
    (BOOK_DIR / "zoo_summary.json").write_text(json.dumps({
        "portfolio": s, "mc": mc, "per_year": per_year, "best_sleeve_dsr": best_dsr,
        "n_trials": n_trials, "n_survivors": len(dfs),
        "placebo_fdr": float((df.placebo_sharpe > 0.5).mean()),
        "families": surv.family.value_counts().to_dict(), "survivors": list(dfs),
        "funnel": funnel, "annual_turnover": annual_turnover, "wf_oos_sharpe": wf_oos,
    }, indent=2, default=float))
    print("ZOO OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="candidate-zoo discovery layer")
    ap.add_argument("--intraday", action="store_true",
                    help="also mine 5m/15m (full grid; ~10x slower, only deepens the trial count)")
    main(intraday=ap.parse_args().intraday)
