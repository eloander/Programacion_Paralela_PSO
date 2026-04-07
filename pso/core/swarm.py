"""Data containers for PSO state and results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SwarmState:
    """Mutable snapshot of the swarm at a given iteration."""

    positions: np.ndarray       # (n_particles, d)
    velocities: np.ndarray      # (n_particles, d)
    fitness: np.ndarray         # (n_particles,)
    pbest_positions: np.ndarray # (n_particles, d)
    pbest_fitness: np.ndarray   # (n_particles,)


@dataclass
class IterationRecord:
    """Metrics collected at the end of each PSO iteration."""

    iteration: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    elapsed_s: float    # wall time from run() start
    eval_s: float       # time spent in fitness evaluation
    update_s: float     # time spent in velocity/position update


@dataclass
class PSOResult:
    """Final output of a PSO run."""

    best_position: np.ndarray
    best_fitness: float
    history: list[IterationRecord]
    total_s: float
    n_iter: int
    converged: bool          # True if stopped early (stagnation)
    config: dict[str, Any]
    seed: int | None
    trajectories: np.ndarray | None = field(default=None, repr=False)
    # trajectories shape: (n_iter, n_particles, d) — optional

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (no numpy arrays)."""
        return {
            "best_position": self.best_position.tolist(),
            "best_fitness": float(self.best_fitness),
            "total_s": float(self.total_s),
            "n_iter": self.n_iter,
            "converged": self.converged,
            "config": self.config,
            "seed": self.seed,
            "history": [
                {
                    "iteration": r.iteration,
                    "best_fitness": r.best_fitness,
                    "mean_fitness": r.mean_fitness,
                    "std_fitness": r.std_fitness,
                    "elapsed_s": r.elapsed_s,
                    "eval_s": r.eval_s,
                    "update_s": r.update_s,
                }
                for r in self.history
            ],
        }
