"""§10 semantic adjudication (owner: Opus 4). Contract entry point: `adjudicate`."""
from __future__ import annotations

from .aggregate import decide, recall_verdict
from .alignment import Alignment, align
from .core import AdjudicationError, adjudicate, deterministic_verdicts, recompute_without_family
from .geo import CgxIndex, route
from .protocol import (JudgeConfigError, evaluate_drift, judge_config, judge_config_hash, run_drift_probe,
                       select_judges)

__all__ = [n for n in dir() if not n.startswith("_")]
