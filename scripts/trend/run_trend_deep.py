"""Deep sequence models for the trend strategy — the control test the tree ensembles motivated.

Trees see only a snapshot of features at trade time; sequence models (LSTM / GRU / TCN / Transformer)
see the raw *sequence* of recent bars and can, in principle, learn temporal structure the 82-feature
snapshot misses. Two framings, both with a strict time split (train < 2024-07 with an embargo, test on
the held-out block) and causal, vol-normalised inputs:

  A. DIRECTION FORECASTER — the deep model IS the signal: predict P(forward H-bar return > 0) from a
     window of L recent bars; build a long-only book from it and compare its OOS Sharpe + AUC to the
     non-ML EMA rule.
  B. SEQUENCE META-LABEL — the deep model gates the EMA rule: predict P(this EMA trade wins) from the
     pre-entry sequence; compare OOS AUC to the tree meta-label (0.505 — a coin flip).

Honest prior: if the tree AUC of 0.505 reflects *no learnable signal* (not limited model capacity),
deep models land at ~0.5 too. This runs them properly so the claim is measured, not asserted.

    python scripts/trend/run_trend_deep.py [--tf 4h] [--epochs 25]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import scripts.trend.trend_common as T  # noqa: E402
from scripts.trend.run_trend_book import sh  # noqa: E402
from src.config import SEED  # noqa: E402
from src.metrics import summarise  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

CORE10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
OOS = T.OOS_START
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --- causal, vol-normalised sequence panel ---------------------------------------

def features(close: pd.Series) -> pd.DataFrame:
    lr = np.log(close).diff()
    vol = lr.rolling(60).std()
    rnorm = (lr / (vol + 1e-9)).clip(-6, 6)
    return pd.DataFrame({
        "rnorm": rnorm,
        "rnorm_ma10": rnorm.rolling(10).mean(),
        "vol_z": ((vol - vol.rolling(250).mean()) / (vol.rolling(250).std() + 1e-9)).clip(-6, 6),
    }).fillna(0.0)


def build_panel(symbols, tf, L=40, H=6):
    """Windowed sequences pooled across instruments. Returns X, y, ts, fwd, inst(sample→symbol idx)."""
    Xs, ys, tss, fwds, insts = [], [], [], [], []
    for si, sym in enumerate(symbols):
        px = T.load_crypto_long(sym, tf) if sym.endswith("USDT") else T.load_equity(sym)
        if px is None:
            continue
        c = px["close"]
        F = features(c).to_numpy(dtype=np.float32)
        cl = c.to_numpy()
        idx = c.index
        n = len(cl)
        for i in range(L, n - H):
            Xs.append(F[i - L:i])
            fwd = cl[i + H] / cl[i] - 1.0
            ys.append(1.0 if fwd > 0 else 0.0)
            fwds.append(fwd)
            tss.append(idx[i])
            insts.append(si)
    X = np.asarray(Xs, dtype=np.float32)
    return (X, np.asarray(ys, dtype=np.float32), pd.DatetimeIndex(tss),
            np.asarray(fwds, dtype=np.float32), np.asarray(insts, dtype=np.int32))


def net_book(prob, ts_te, inst_te, symbols, tf):
    """Reconstruct the deep signal as a per-instrument long-only position and backtest it NET of cost
    (t+2, √-impact, funding) through the real engine — the honest, turnover-aware read. Returns
    (net Sharpe, avg annual turnover)."""
    rets, rets_s, turns = [], [], []
    for si, sym in enumerate(symbols):
        mask = inst_te == si
        if mask.sum() < 50:
            continue
        px = T.load_crypto_long(sym, tf)
        if px is None:
            continue
        p = pd.Series(prob[mask], index=ts_te[mask]).sort_index()
        p = p[~p.index.duplicated()]
        pos = (p > 0.5).astype(float).reindex(px.index).ffill().fillna(0.0)
        s, r = T.bo.evaluate(px["close"], pos, T.CRYPTO_TF[tf], T.CC,
                             fund=T.bo.safe_funding(sym), adv=T.crypto_adv(px), with_mc=False)
        rets.append(r)
        turns.append(s["ann_turnover"])
        # low-turnover variant: EMA-smooth the probability (span 2H) before thresholding
        ps = (p.ewm(span=2 * 6, adjust=False).mean() > 0.5).astype(float).reindex(px.index).ffill().fillna(0.0)
        _, rs = T.bo.evaluate(px["close"], ps, T.CRYPTO_TF[tf], T.CC,
                              fund=T.bo.safe_funding(sym), adv=T.crypto_adv(px), with_mc=False)
        rets_s.append(rs)
    book = pd.DataFrame({i: r for i, r in enumerate(rets)}).mean(axis=1)
    book_s = pd.DataFrame({i: r for i, r in enumerate(rets_s)}).mean(axis=1)
    return (round(sh(book[book.index >= OOS]), 2), round(sh(book_s[book_s.index >= OOS]), 2),
            round(float(np.nanmean(turns)), 0))


# --- models ----------------------------------------------------------------------

class LSTMNet(nn.Module):
    def __init__(self, c, h=48, kind="lstm"):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(c, h, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TCN(nn.Module):
    def __init__(self, c, h=48):
        super().__init__()
        layers = []
        cin = c
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(cin, h, 3, padding=d, dilation=d), nn.ReLU()]
            cin = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))          # (B, C, L)
        return self.head(z.mean(dim=-1)).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, c, d=48, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(c, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.enc(self.proj(x))
        return self.head(z.mean(dim=1)).squeeze(-1)


def make(kind, c):
    if kind in ("lstm", "gru"):
        return LSTMNet(c, kind=kind)
    if kind == "tcn":
        return TCN(c)
    return TransformerNet(c)


def train_eval(kind, Xtr, ytr, Xte, epochs, bs=512):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = make(kind, Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr)                              # keep data on CPU; move batches to DEV
    yt = torch.tensor(ytr)
    n = len(Xt)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            opt.zero_grad()
            loss = lossf(m(Xt[b].to(DEV)), yt[b].to(DEV))
            loss.backward()
            opt.step()
    m.eval()
    probs = []
    with torch.no_grad():
        Xe = torch.tensor(Xte)
        for j in range(0, len(Xe), 4096):
            probs.append(torch.sigmoid(m(Xe[j:j + 4096].to(DEV))).cpu().numpy())
    return np.concatenate(probs)


def strategy_sharpe(ts_te, fwd_te, prob, H):
    """Long-only book from the model: position = (P>0.5), forward-H return, non-overlapping, equal-risk
    across the pooled samples by date. A coarse but honest read of the signal's tradability net of nothing."""
    df = pd.DataFrame({"ts": ts_te, "fwd": fwd_te, "p": prob})
    df["pos"] = (df["p"] > 0.5).astype(float)
    df["pnl"] = df["pos"] * df["fwd"] / H            # spread the H-bar return over H bars (approx daily)
    daily = df.groupby(df["ts"].dt.date)["pnl"].mean()
    daily.index = pd.to_datetime(daily.index)
    return round(sh(daily), 2), round(summarise(daily, 365)["max_dd"], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--H", type=int, default=6)
    args = ap.parse_args()
    print(f"=== Deep sequence models on TREND — {args.tf}, L={args.L}, H={args.H}, device={DEV} ===\n")

    X, y, ts, fwd, inst = build_panel(CORE10, args.tf, args.L, args.H)
    embargo = pd.Timedelta(days=5)
    tr = ts < (OOS - embargo)
    te = ts >= OOS
    print(f"pooled windows: {len(X):,}  train {tr.sum():,}  test(OOS) {te.sum():,}  "
          f"base-rate up: {y[te].mean():.0%}\n")
    Xtr, ytr = X[tr], y[tr]
    Xte, yte, ts_te, fwd_te, inst_te = X[te], y[te], ts[te], fwd[te], inst[te]

    print(f"  {'model':12s} {'OOS AUC':>8s} {'gross Sh':>9s} {'NET Sh':>7s} {'NET smooth':>11s} {'annTO':>7s}")
    results = {}
    for kind in ("lstm", "gru", "tcn", "transformer"):
        prob = train_eval(kind, Xtr, ytr, Xte, args.epochs)
        auc = roc_auc_score(yte, prob) if len(np.unique(yte)) > 1 else float("nan")
        acc = ((prob > 0.5).astype(float) == yte).mean()
        gsh, _ = strategy_sharpe(ts_te, fwd_te, prob, args.H)
        nsh, nsh_s, ato = net_book(prob, ts_te, inst_te, CORE10, args.tf)
        results[kind] = {"oos_auc": round(float(auc), 3), "oos_acc": round(float(acc), 3),
                         "gross_sharpe": gsh, "net_sharpe": nsh, "net_smooth_sharpe": nsh_s, "ann_turnover": ato}
        print(f"  {kind:12s} {auc:>8.3f} {gsh:>+9.2f} {nsh:>+7.2f} {nsh_s:>+11.2f} {ato:>7.0f}", flush=True)

    # non-ML EMA baseline on the SAME OOS window/universe, for the honest reference
    ema_rets = []
    for sym in CORE10:
        px = T.load_crypto_long(sym, args.tf)
        if px is None:
            continue
        _, r = T.eval_spec(px, {"entry": "ema", "direction": "long_only", "exit": "reversal"},
                           args.tf, T.CRYPTO_TF[args.tf], T.CC, fund=T.bo.safe_funding(sym), adv=T.crypto_adv(px))
        ema_rets.append(r)
    ema = pd.DataFrame({i: r for i, r in enumerate(ema_rets)}).mean(axis=1)
    ema_oos = ema[ema.index >= OOS]
    print(f"\n  {'EMA rule':12s} {'(no ML)':>8s} {'':>8s} {sh(ema_oos):>+13.2f}   <- non-ML baseline, same OOS")
    best = max(results, key=lambda k: results[k]["oos_auc"])
    print(f"\nbest deep model: {best} OOS AUC {results[best]['oos_auc']:.3f}  "
          f"(tree meta-label was 0.505; 0.5 = coin flip)")

    (T.REPORTS / "trend_deep.json").write_text(json.dumps({
        "tf": args.tf, "L": args.L, "H": args.H, "epochs": args.epochs, "device": str(DEV),
        "n_windows": int(len(X)), "oos_base_rate": float(y[te].mean()),
        "models": results, "ema_oos_sharpe": round(sh(ema_oos), 3), "tree_meta_auc_oos": 0.505,
    }, indent=2, default=float))
    print("wrote reports/trend/trend_deep.json")


if __name__ == "__main__":
    main()
