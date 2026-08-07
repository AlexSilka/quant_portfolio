"""Minimal logging helper — one auditable place, used by the library modules and by any
off-happy-path branch (a dropped symbol, a cache miss) that must stay operator-visible.

Research CLIs keep using print() for their intended stdout tables/summaries; this logger is for
diagnostics that would otherwise vanish silently (e.g. a universe name that fails to load and is
skipped — a silent drop is a survivorship hazard). Level defaults to INFO; override with
CROSS_ASSET_LOG=DEBUG|WARNING|... in the environment.
"""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "cross_asset_alpha") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("CROSS_ASSET_LOG", "INFO").upper())
        logger.propagate = False
    return logger
