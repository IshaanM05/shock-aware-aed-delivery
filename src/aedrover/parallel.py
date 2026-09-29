"""Small, dependency-free process-parallel map with progress and stable ordering.

MuJoCo is single-threaded per model, so throughput scales almost linearly with processes.
Workers are spawned (Windows-safe) and each keeps its own compiled-model cache.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_workers(reserve: int = 4) -> int:
    return max(1, (os.cpu_count() or 2) - reserve)


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def pmap(fn: Callable[[T], R], items: Iterable[T] | Sequence[T], *, workers: int | None = None,
         chunksize: int = 1, desc: str = "", progress: bool = True) -> list[R]:
    """Apply ``fn`` to every item in parallel; results keep the input order."""
    items = list(items)
    n = len(items)
    workers = default_workers() if workers is None else workers
    if workers <= 1 or n <= 1:
        return [fn(x) for x in items]
    out: list[Any] = [None] * n
    t0, done, next_report = time.perf_counter(), 0, 0.1
    with ProcessPoolExecutor(max_workers=min(workers, n), initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(fn, items, chunksize=chunksize)):
            out[i] = r
            done += 1
            if progress and done / n >= next_report:
                next_report += 0.1
                el = time.perf_counter() - t0
                print(f"  [{desc or 'pmap'}] {done}/{n} ({100 * done / n:.0f}%)  "
                      f"{el:.0f}s elapsed, ~{el / done * (n - done):.0f}s left", file=sys.stderr, flush=True)
    return out
