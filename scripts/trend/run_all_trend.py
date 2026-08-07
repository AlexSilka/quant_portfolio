"""One command to reproduce the trend deep-dive headline results end to end.

Runs the edge-map sweep, the headline book (EMA) + the low-drawdown variant (blend), the
walk-forward suite, the ML overlay, and the dashboard — in dependency order. Each stage writes to
reports/trend/; stages are independently runnable (see their module docstrings). Assumes the data
is present (run scripts/trend/fetch_data.py first for the spot-2017 history + MKR intraday).

    python scripts/trend/run_all_trend.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [
    ("edge-map sweep", ["run_trend_sweep.py"]),
    ("headline book (EMA)", ["run_trend_book.py", "--entry", "ema", "--tfs", "1d,4h,1h"]),
    ("robust variant (blend)", ["run_trend_book.py", "--entry", "blend", "--tfs", "1d,4h,1h"]),
    ("walk-forward suite", ["run_trend_wfo.py", "--entry", "ema", "--tfs", "1d,4h,1h"]),
    ("ML overlay", ["run_trend_ml.py"]),
    ("crypto breadth scaling", ["run_trend_breadth.py"]),
    ("feature families", ["run_trend_features.py"]),
    ("regime-filter study", ["run_trend_regime.py"]),
    ("universe-selection bias", ["run_trend_universe.py"]),
    ("block composition", ["run_trend_composition.py"]),
    ("risk overlays", ["run_trend_risk.py"]),
    ("OOS trade log + targets", ["run_trend_trades.py"]),
    ("portfolio integration", ["run_trend_in_portfolio.py"]),
    ("dashboard", ["make_trend_report.py"]),
]


def main():
    for name, cmd in STAGES:
        print(f"\n{'='*70}\n### {name}: {' '.join(cmd)}\n{'='*70}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]])
        if r.returncode != 0:
            print(f"[stop] stage '{name}' failed (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\nALL TREND STAGES OK — see docs/strategies/TREND.md and trend_dashboard.html")


if __name__ == "__main__":
    main()
