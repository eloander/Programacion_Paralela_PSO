"""V5 — Parallel fitness evaluation with joblib.

joblib's ``loky`` backend maintains a *persistent* worker process pool across
calls, unlike ``ProcessPoolExecutor`` (V2) which spawns and tears down workers
on every ``with`` block (i.e., every PSO iteration).  This makes V5 more
efficient than V2 for expensive objectives where the per-iteration overhead of
process management dominates.

Why joblib over raw multiprocessing?
-------------------------------------
* **Process reuse**: loky keeps workers alive between ``evaluate()`` calls,
  eliminating repeated spawn + import costs.
* **Auto-batching**: ``batch_size="auto"`` lets joblib decide how to group
  tasks, reducing IPC round-trips for fast objectives.
* **Robustness**: loky handles crashes, timeouts, and serialisation more
  gracefully than ``ProcessPoolExecutor``.
* **Alternative backends**: pass ``backend="threading"`` to switch to threads
  (useful when the objective releases the GIL, e.g. NumPy C code).

Limitations
-----------
* The objective must still be picklable (same constraint as V2).
* For very cheap objectives V5 may still be slower than V0 due to IPC.
* ``n_jobs=-1`` uses all available CPU cores.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from joblib import Parallel, delayed

from pso.parallel.base import FitnessEvaluator


class JoblibEvaluator(FitnessEvaluator):
    """Evaluate particles in parallel using joblib.

    Parameters
    ----------
    objective:
        Picklable fitness callable ``f(x: ndarray) -> float``.
    n_jobs:
        Number of parallel workers.  ``-1`` uses all CPU cores.
    backend:
        joblib backend: ``'loky'`` (default, persistent processes),
        ``'threading'``, or ``'multiprocessing'``.
    batch_size:
        Tasks per worker per dispatch.  ``'auto'`` lets joblib decide.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        n_jobs: int = -1,
        backend: str = "loky",
        batch_size: int | str = "auto",
    ) -> None:
        self.objective = objective
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate all *positions* in parallel; return shape ``(n,)``."""
        results = Parallel(
            n_jobs=self.n_jobs,
            backend=self.backend,
            batch_size=self.batch_size,
        )(delayed(self.objective)(pos) for pos in positions)
        return np.array(results, dtype=float)
