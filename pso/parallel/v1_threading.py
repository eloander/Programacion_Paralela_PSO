"""V1 — Concurrent fitness evaluation with threads (ThreadPoolExecutor).

CPython's Global Interpreter Lock (GIL) prevents true parallel execution of
pure-Python code in threads.  However:

* If the objective function calls NumPy/SciPy routines that release the GIL
  (most do), threads achieve real parallelism for the C-extension work.
* If the objective involves I/O (file, network), threads overlap the wait time.
* For purely Python-numeric objectives the GIL limits benefit to minor
  scheduling gains.  Overhead from thread creation can *hurt* for fast evals.

Conclusion: V1 helps when eval is moderately expensive (NumPy-heavy or I/O).
For trivially fast functions like Sphere, V0 can outperform V1 due to overhead.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from pso.parallel.base import FitnessEvaluator


class ThreadingEvaluator(FitnessEvaluator):
    """Evaluate particles concurrently using a thread pool.

    Parameters
    ----------
    objective:
        The fitness callable.
    max_workers:
        Thread-pool size.  Defaults to ``os.cpu_count()``.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        max_workers: int | None = None,
    ) -> None:
        self.objective = objective
        self.max_workers = max_workers or os.cpu_count()

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self.objective, positions))
        return np.array(results)
