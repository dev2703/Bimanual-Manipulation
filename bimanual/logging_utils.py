"""Shared logging configuration.

One place to configure format/level so it doesn't drift per-script, and so
remote runs (Modal training/eval jobs, Phase 6+) get structured, leveled
output instead of bare prints -- prints to stdout get lost/interleaved
across parallel workers with no timestamp or severity to filter on.

Usage: `log = get_logger(__name__)` at module scope, then `log.info(...)`,
`log.warning(...)`, `log.exception(...)` (inside an except block, to
capture the traceback) instead of `print(...)`.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("BIMANUAL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
