"""Reproduce the headline end to end: discovery -> validation -> master portfolio -> dashboard.

  validate_sessions.py §3 equity NYSE session/half-day integrity via pandas_market_calendars
  run_book.py          discovery/zoo layer: mine the grid, screen, walk-forward the naive selection OOS
                       -> survival funnel + deflated Sharpe + OOS Sharpe (the dashboard's honesty panel)
  feature_report.py    §4 per-feature IC / stability / redundancy-cluster analysis + the reduction
  run_meta_overlay.py  the ML meta-label confidence gate (incremental-value measurement)
  run_crisis/gmacro.py the two diversifier sleeves (crisis-alpha, global-macro)
  run_master_book.py   THE canonical portfolio: equal-weight risk parity over the eight families' honest
                       published series + the §8 drawdown-ladder risk overlay (deep-dives rebuilt by the Makefile)
  run_wf_book.py       §10 book-level walk-forward: rolling & anchored, periodic allocation re-fit -> the
                       accumulated out-of-sample track (whole history, not just the final block)
  run_cscv.py          §6 probability of backtest overfitting (CSCV) on the full trial set
  make_oos_ledger.py   §13 portfolio-level out-of-sample trade/position ledger
  make_figures.py      §13 all nine required charts as standalone PNGs
  make_report.py       dashboard

Each step's output is streamed to the console AND captured to logs/run_all_<start>.log (named by the
run's start date-time), so a run is always reproducible from its log.

    python scripts/run_all.py
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("validate_sessions.py", "run_book.py", "feature_report.py", "run_meta_overlay.py", "run_crisis.py",
         "run_gmacro.py", "run_master_book.py", "run_wf_book.py", "run_cscv.py", "make_oos_ledger.py",
         "make_figures.py", "make_report.py")


def main() -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    logf = logs / f"run_all_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}    # line-buffer children so the tee is live
    print(f"logging to {logf.relative_to(ROOT)}")
    with logf.open("w", buffering=1) as log:          # line-buffered so the log is live
        for step in STEPS:
            banner = f"\n{'=' * 60}\n== {step}  ({datetime.now():%H:%M:%S})\n{'=' * 60}"
            print(banner, flush=True)
            log.write(banner + "\n")
            log.flush()
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            if proc.wait() != 0:
                log.flush()
                raise SystemExit(f"\n{step} FAILED (exit {proc.returncode}) — full log: {logf}")
    print(f"\nALL DONE — master: reports/master_book_summary.json · dashboard: reports/dashboard.html"
          f"\nlog: {logf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
