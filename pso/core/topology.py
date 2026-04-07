"""Swarm topology strategies.

GlobalTopology (gBest): every particle is attracted to the single best
position ever found by the whole swarm.  Fast convergence; higher risk
of premature convergence.

RingTopology (lBest): each particle is attracted to the best position
within a small ring-neighbourhood of radius *k*.  Slower convergence
but better exploration and lower premature-convergence risk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Topology(ABC):
    """Abstract swarm topology."""

    @abstractmethod
    def update(
        self,
        pbest_positions: np.ndarray,  # (n_particles, d)
        pbest_fitness: np.ndarray,    # (n_particles,)
    ) -> None:
        """Recompute topology state from updated personal bests."""

    @abstractmethod
    def get_social_best(self) -> np.ndarray:
        """Social-best position for every particle, shape (n_particles, d)."""

    @property
    @abstractmethod
    def gbest_position(self) -> np.ndarray:
        """Best position found by the whole swarm, shape (d,)."""

    @property
    @abstractmethod
    def gbest_fitness(self) -> float:
        """Fitness of the global best position."""


class GlobalTopology(Topology):
    """All particles share a single global best (gBest)."""

    def __init__(self) -> None:
        self._gbest_pos: np.ndarray | None = None
        self._gbest_fit: float = np.inf
        self._n: int = 0

    def update(
        self,
        pbest_positions: np.ndarray,
        pbest_fitness: np.ndarray,
    ) -> None:
        self._n = len(pbest_positions)
        idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[idx] < self._gbest_fit:
            self._gbest_fit = float(pbest_fitness[idx])
            self._gbest_pos = pbest_positions[idx].copy()

    def get_social_best(self) -> np.ndarray:
        return np.tile(self._gbest_pos, (self._n, 1))

    @property
    def gbest_position(self) -> np.ndarray:
        return self._gbest_pos

    @property
    def gbest_fitness(self) -> float:
        return self._gbest_fit


class RingTopology(Topology):
    """Ring (lBest) topology — neighbourhood radius *k* (2k+1 particles per neighbourhood)."""

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self._lbest_pos: np.ndarray | None = None
        self._gbest_pos: np.ndarray | None = None
        self._gbest_fit: float = np.inf

    def update(
        self,
        pbest_positions: np.ndarray,
        pbest_fitness: np.ndarray,
    ) -> None:
        n = len(pbest_positions)
        self._lbest_pos = np.empty_like(pbest_positions)
        for i in range(n):
            neighbours = [(i + j) % n for j in range(-self.k, self.k + 1)]
            local_fits = pbest_fitness[neighbours]
            best_local = neighbours[int(np.argmin(local_fits))]
            self._lbest_pos[i] = pbest_positions[best_local]

        # Maintain global best for reporting
        idx = int(np.argmin(pbest_fitness))
        if pbest_fitness[idx] < self._gbest_fit:
            self._gbest_fit = float(pbest_fitness[idx])
            self._gbest_pos = pbest_positions[idx].copy()

    def get_social_best(self) -> np.ndarray:
        return self._lbest_pos.copy()

    @property
    def gbest_position(self) -> np.ndarray:
        return self._gbest_pos

    @property
    def gbest_fitness(self) -> float:
        return self._gbest_fit


_TOPOLOGIES: dict[str, type[Topology]] = {
    "global": GlobalTopology,
    "ring": RingTopology,
}


def get_topology(name: str, **kwargs: object) -> Topology:
    """Factory — *name* is ``'global'`` or ``'ring'``."""
    try:
        return _TOPOLOGIES[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown topology '{name}'. Choose from {list(_TOPOLOGIES)}"
        )
