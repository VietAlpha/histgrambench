"""Optional process pool for the evaluator's pure CPU-bound steps: CGX extraction, alignment and the deterministic
verdicts (EC2 MS5 2026-10-01). The final run is one process whose cell threads share the GIL, and a profile of it
put 93% of its time in these steps while compile slots sat idle. `run` calls the same function with the same
arguments in a worker process and returns its result, so outputs are unchanged; with no pool configured
(`batch.yaml` `cpu_workers` <= 1, the default) it calls the function inline."""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

_POOL: ProcessPoolExecutor | None = None


def _lower_priority() -> None:
    if hasattr(os, "nice"):
        os.nice(10)        # sandbox compiles (C24: CPU and wall time) and the run process keep priority


def configure(workers: int) -> None:
    """More workers than cores on purpose: a few cells take minutes of alignment, and the OS must keep
    interleaving the light ones (as the GIL did) instead of queueing them behind the heavy ones."""
    global _POOL
    if workers > 1 and _POOL is None:
        # spawn: the run process has hundreds of threads, and fork would copy their held locks
        _POOL = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_lower_priority)


def active() -> bool:
    return _POOL is not None


def run(fn, *args, **kwargs):
    if _POOL is None:
        return fn(*args, **kwargs)
    return _POOL.submit(fn, *args, **kwargs).result()
