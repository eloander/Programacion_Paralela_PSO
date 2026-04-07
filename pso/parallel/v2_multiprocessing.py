"""V2 — Parallel fitness evaluation with processes (ProcessPoolExecutor).

Bypasses the GIL by spawning separate Python interpreter processes.
Each worker has its own memory space; data is exchanged via pickle/IPC.

Key considerations
------------------
* **Pickling**: the objective function must be picklable.  Module-level
  functions (like the benchmarks in pso.objectives) satisfy this.  Lambdas
  and closures do not — use functools.partial or a class with __call__.
* **IPC overhead**: spawning processes and serialising data has a fixed cost
  per batch.  For very fast objectives this overhead dominates and V2 is
  *slower* than V0.  For expensive objectives the parallel speedup wins.
* **Batching**: instead of one subprocess task per particle, we group
  particles into batches to reduce the number of IPC round-trips.  Tune
  *batch_size* according to evaluation cost vs. overhead.
* **Windows**: uses "spawn" start method — the ``if __name__ == '__main__':``
  guard is required in entry-point scripts to prevent recursive spawning.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np

from pso.parallel.base import FitnessEvaluator


# Module-level helpers — must be at module scope to be picklable on Windows.

def _eval_single(args: tuple) -> float:
    func, position = args
    return func(position)


def _eval_batch(args: tuple) -> list[float]:
    func, batch = args
    return [func(pos) for pos in batch]


class MultiprocessingEvaluator(FitnessEvaluator):
    """Evaluate particles in parallel using a process pool.

    Parameters
    ----------
    objective:
        Picklable fitness callable (use module-level functions).
    max_workers:
        Number of worker processes.  Defaults to ``os.cpu_count()``.
    batch_size:
        Particles per subprocess task.  ``1`` → one task per particle;
        higher values reduce IPC overhead at the cost of load-balance.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        max_workers: int | None = None,
        batch_size: int = 5,
    ) -> None:
        self.objective = objective
        self.max_workers = max_workers or os.cpu_count()
        self.batch_size = max(1, batch_size)

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        n = len(positions)
        if self.batch_size == 1:
            args = [(self.objective, pos) for pos in positions]
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(_eval_single, args))
            return np.array(results)

        # Batched evaluation
        batches = [
            positions[i : i + self.batch_size]
            for i in range(0, n, self.batch_size)
        ]
        batch_args = [(self.objective, b) for b in batches]
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            batch_results = list(pool.map(_eval_batch, batch_args))
        flat = [val for br in batch_results for val in br]
        return np.array(flat)
