"""V0 — Sequential fitness evaluation (baseline).

Evaluates each particle one at a time in a plain Python loop.
No parallelism; establishes the wall-time baseline for speedup calculations.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from pso.parallel.base import FitnessEvaluator


class SequentialEvaluator(FitnessEvaluator):
    """Evaluate each particle sequentially (no parallelism)."""

    def __init__(self, objective: Callable[[np.ndarray], float]) -> None:
        self.objective = objective

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        return np.array([self.objective(pos) for pos in positions])
