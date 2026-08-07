"""Focused 15m cross-sectional probe — does the edge survive the fastest timeframe we hold?

The 15m panel (230k bars × 51 names) is too large for the full grid, so this tests only a
handful of monthly-cadence (low-turnover) constructions plus a random-signal placebo — enough to
answer "alive or dead at 15m" and complete the timeframe coverage.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)      # deprecations only; correctness warnings (pandas SettingWithCopy, numpy) still surface
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.config import CACHE_DIR, SEED  # noqa: E402
from src.metrics import summarise
from src.sleeves.xsect import mom, risk_adj_mom, blend_rank, xs_backtest, vol_target
from src.validation.monte_carlo import bootstrap_sharpe

px = pd.read_parquet(CACHE_DIR / "xs/crypto_15m_close.parquet")
adv = pd.read_parquet(CACHE_DIR / "xs/crypto_15m_adv.parquet")
bpd, ppy, cost = 96, 96*365, 6.0
print(f"crypto_15m panel {px.shape[0]}×{px.shape[1]}")

def sig(kind, lb):
    lbb=lb*bpd
    if kind=="raw": return mom(px, lbb)
    if kind=="riskadj": return risk_adj_mom(px, lbb)
    return blend_rank([risk_adj_mom(px, max(2,int(lbb*f)),0) for f in (0.5,1,2)])

configs = [("riskadj",20,0.3,21),("riskadj",30,0.3,21),("blend",20,0.2,21),
           ("blend",45,0.2,21),("riskadj",20,0.2,10),("blend",30,0.3,10)]
print(f"\n{'config':28s} {'Sharpe':>7s} {'MC-P5':>7s} {'DD':>6s} {'turn/yr':>8s}")
best=[]
for kind,lb,tf,rb in configs:
    bt = xs_backtest(px, sig(kind,lb), top_frac=tf, weighting="equal", rebal=rb*bpd,
                     cost_bps=cost, adv=adv, impact_k=0.1)
    netv = vol_target(bt["net"], ppy).dropna()
    s = summarise(netv, ppy)
    p5 = bootstrap_sharpe(netv, ppy, 300, SEED).get("sharpe_p5", float("nan")) if s["sharpe_ann"]>0.5 else float("nan")
    turn = bt["turnover"].sum()/(len(px)/ppy)
    print(f"{kind+'/'+str(lb)+'d/tf'+str(tf)+'/reb'+str(rb):28s} {s['sharpe_ann']:+7.2f} {p5:+7.2f} {s['max_dd']:+6.0%} {turn:8.0f}")
    best.append(s["sharpe_ann"])
# placebo
pmax=[]
for i in range(8):
    plc=pd.DataFrame(np.random.default_rng(200+i).standard_normal(px.shape),index=px.index,columns=px.columns)
    bt=xs_backtest(px,plc,top_frac=0.3,weighting="equal",rebal=21*bpd,cost_bps=cost)
    pmax.append(summarise(vol_target(bt["net"],ppy).dropna(),ppy)["sharpe_ann"])
print(f"\nplacebo (8 random signals): max {max(pmax):+.2f}, mean {np.mean(pmax):+.2f}")
print(f"best real {max(best):+.2f}  ->  {'ALIVE (beats placebo)' if max(best)>max(pmax)+0.3 else 'MARGINAL'}")
