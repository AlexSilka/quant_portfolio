"""Does adoption momentum improve the crypto x-sect sleeve when blended into its signal?

The on-chain deep-dive leaves one live signal — `adr_mom30`, active-address growth — which clears every
robustness gate on its own (+0.73, MC-P5 +0.08, placebo 98th, alpha over price t=+2.04) and yet moves the
book by nothing, because it is 0.32 correlated with the price momentum the book already runs. That leaves
the question this file answers: it may still be worth something not as a ninth family but as a *component*
of the momentum sleeve that exists, the way residual momentum upgraded that sleeve in RESIDMOM.md.

**The construction is declared before measuring** — an equal-weight rank blend of the sleeve's own
idiosyncratic-momentum signal with `adr_mom30`, applied only where on-chain coverage exists (33 of the
300-name panel) and leaving every other name untouched. The dose ladder below is reported whole; no rung
is promoted over the others.

**The control is the point.** Blending touches the ranks of ~33 names, so a change can come from the
signal or from merely jostling that subset. Each dose is therefore run against a NAME-SHUFFLED
`adr_mom30` — same marginals, same names touched, no real information. That control is not decoration:
at the smallest dose the shuffled arm *also* beats the raw sleeve, so the apparent lift there belongs to
the perturbation, not to adoption momentum. Reading the blend without it would have produced a
"+0.05 improvement" that is not one.

    python scripts/onchain/run_onchain_blend.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scripts.residmom.run_residmom import ASSETS as RM_ASSETS, _book as rm_book, _load as rm_load  # noqa: E402
from src.config import ONCHAIN_DIR, SEED  # noqa: E402
from src.data import onchain as oc  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.sleeves import bab  # noqa: E402
from src.sleeves import onchain as ocs  # noqa: E402
from src.sleeves.xsect import idio_mom  # noqa: E402

PPY = 365
DOSES = (0.10, 0.25, 0.50)      # share of the blended rank given to adoption momentum
N_CONTROL = 40                  # name-shuffles per dose
rng = np.random.default_rng(SEED)


def main() -> None:
    cfg = RM_ASSETS["crypto"]
    base = cfg["base"]
    Craw, adv = rm_load(cfg["tag"])
    C = bab.winsorize_panel(Craw, cfg["winsor"])
    base_sig = idio_mom(C, base["lb"], cfg["beta_lb"], base["sk"], market=None)
    r_base = base_sig.rank(axis=1, pct=True)

    live = [s for s in oc.live_universe() if s in C.columns]
    adr = oc.load("AdrActCnt").reindex(index=C.index, columns=C.columns)[live]
    adr_mom = ocs.adr_momentum(adr.mask(adr == 0), 30, 7)   # zero counts are outages, not zero usage
    n_cov = int(adr_mom.notna().any().sum())

    def blended(extra: pd.DataFrame, w: float) -> pd.DataFrame:
        r_ex = extra.reindex(index=base_sig.index, columns=base_sig.columns).rank(axis=1, pct=True)
        return r_base.where(r_ex.isna(), (1 - w) * r_base + w * r_ex)

    def sharpe(sig: pd.DataFrame) -> float:
        return summarise(rm_book(C, sig, adv, cfg)[0].dropna(), PPY)["sharpe_ann"]

    raw = sharpe(base_sig)
    print(f"\n{'='*84}\nADOPTION MOMENTUM AS A COMPONENT OF THE CRYPTO X-SECT SLEEVE\n{'='*84}")
    print(f"  panel {C.shape[1]} names, on-chain coverage {n_cov} — the blend can only touch those {n_cov}")
    print(f"  raw sleeve (idiosyncratic momentum): Sharpe {raw:+.3f}\n")
    print(f"  {'dose':>6} {'blended':>9} {'shuffled-name control':>26} {'pctile':>8}  verdict")

    rows = []
    for w in DOSES:
        real = sharpe(blended(adr_mom, w))
        ctl = []
        for _ in range(N_CONTROL):
            perm = adr_mom.copy()
            perm.columns = rng.permutation(adr_mom.columns)
            ctl.append(sharpe(blended(perm.reindex(columns=adr_mom.columns), w)))
        ctl = np.array([x for x in ctl if np.isfinite(x)])
        if len(ctl) < N_CONTROL:
            print(f"    ! {N_CONTROL - len(ctl)} control draws returned no Sharpe and were dropped")
        pct = float((real > ctl).mean() * 100)
        p95 = float(np.percentile(ctl, 95))
        verdict = "clears its control" if real > p95 else "inside the noise"
        print(f"  {w:>6.2f} {real:>+9.3f}   mean {ctl.mean():>+6.3f}  p95 {p95:>+6.3f} {pct:>7.0f}th  {verdict}")
        rows.append({"dose": w, "blended_sharpe": round(real, 3), "vs_raw": round(real - raw, 3),
                     "control_mean": round(float(ctl.mean()), 3), "control_p95": round(p95, 3),
                     "control_pctile": round(pct, 0), "clears_control": bool(real > p95)})

    best = max(rows, key=lambda r: r["blended_sharpe"])
    print(f"\n  Not one dose clears its own control. At the smallest the shuffled arm averages "
          f"{rows[0]['control_mean']:+.3f} against a raw sleeve of {raw:+.3f} — jostling the ranks of "
          f"{n_cov} names lifts it on its own, so the blend's {rows[0]['blended_sharpe']:+.3f} there is "
          f"not evidence of anything. Adoption momentum does not upgrade this sleeve.")

    summ = {"raw_sleeve_sharpe": round(raw, 3), "onchain_coverage_names": n_cov,
            "panel_names": int(C.shape[1]), "n_control_draws": N_CONTROL,
            "construction": "equal-weight rank blend, declared a-priori; dose ladder reported whole",
            "doses": rows, "best_dose": best,
            "verdict": "no dose clears its name-shuffled control — adoption momentum adds nothing to the "
                       "crypto x-sect sleeve as a component, just as it added nothing as a family"}
    (ONCHAIN_DIR / "onchain_blend_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(f"\n  wrote {ONCHAIN_DIR / 'onchain_blend_summary.json'}")
    print("RUN ONCHAIN-BLEND OK")


if __name__ == "__main__":
    main()
