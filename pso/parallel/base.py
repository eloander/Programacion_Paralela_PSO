"""Abstract interface for fitness evaluators.

All four strategies (V0–V3) implement FitnessEvaluator so the PSO engine
remains agnostic to the evaluation backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FitnessEvaluator(ABC):
    """Evaluate a batch of particle positions and return their fitness values.

    Every concrete subclass wraps a callable ``f: ndarray → float`` and
    evaluates it on *all* rows of the positions matrix, returning a 1-D
    result array.  The internal strategy (loop, threads, processes, async)
    is an implementation detail hidden from the PSO engine.
    """

    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate fitness for a batch of positions.

        Parameters
        ----------
        positions:
            Shape ``(n_particles, d)``.

        Returns
        -------
        fitness:
            Shape ``(n_particles,)``.
        """
