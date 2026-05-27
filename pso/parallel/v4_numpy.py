"""V4 — Vectorized fitness evaluation using NumPy matrix operations.

Instead of calling the objective function *n* times (one per particle), a
*vectorized* variant accepts the entire ``(n_particles, d)`` position matrix
and returns all ``n`` fitness values in a single call.  This exploits NumPy's
internal C/Fortran kernels (BLAS, LAPACK) and eliminates Python-level loop
overhead entirely.

Comparison with other strategies
---------------------------------
* **V0** — Python ``for`` loop: *n* Python calls, full interpreter overhead.
* **V1** — Threading: concurrent Python calls; limited by the GIL unless the
  objective releases it (e.g., NumPy C code).
* **V2** — Multiprocessing: true parallelism, but IPC + process spawn overhead.
* **V4** — Single vectorized NumPy call: no IPC, no GIL, minimal overhead.
  Best when all particles can be evaluated with broadcasting.

Fallback behaviour
------------------
If no ``vec_objective`` is provided, the evaluator falls back to
``np.apply_along_axis``, which still runs in C rather than Python but calls
the scalar function row-by-row.  This is faster than a Python ``for`` loop
but slower than a truly vectorized implementation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from pso.parallel.base import FitnessEvaluator


class VectorizedEvaluator(FitnessEvaluator):
    """Evaluate all particles in one vectorized NumPy call.

    Parameters
    ----------
    objective:
        Scalar fitness callable ``f(x: ndarray) -> float``.  Used as fallback
        when *vec_objective* is ``None``.
    vec_objective:
        Vectorized fitness callable ``f(X: ndarray) -> ndarray`` where *X* has
        shape ``(n, d)`` and the return has shape ``(n,)``.  When provided,
        evaluation is a single matrix operation with no Python loop overhead.
        When ``None``, falls back to ``np.apply_along_axis(objective, 1, X)``.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        vec_objective: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.objective = objective
        self.vec_objective = vec_objective

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate all *positions* and return fitness array of shape ``(n,)``."""
        if self.vec_objective is not None:
            return np.asarray(self.vec_objective(positions), dtype=float)
        # Fallback: C-level row iteration (no Python loop, but not fully vectorized)
        return np.apply_along_axis(self.objective, 1, positions).astype(float)
