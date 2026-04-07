"""Boundary handling strategies for PSO.

Chosen default: ClampStrategy.
Rationale: simple, stable, zero risk of position escaping bounds after application.
Velocity is negated and damped on impact (absorbing-wall effect), which helps
reduce oscillations near boundaries without fully stopping exploration.

ReflectStrategy is provided as an alternative: it mirrors the overshoot about
the boundary, preserving particle "momentum" better at the cost of more
complex behaviour for large overshoots (handled by iterative reflection + clamp).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BoundsStrategy(ABC):
    """Abstract boundary constraint handler."""

    @abstractmethod
    def apply(
        self,
        positions: np.ndarray,   # (n_particles, d)
        velocities: np.ndarray,  # (n_particles, d)
        bounds: np.ndarray,      # (d, 2) — columns [lb, ub]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return *(positions, velocities)* after enforcing constraints."""


class ClampStrategy(BoundsStrategy):
    """Clamp positions to [lb, ub]; negate and halve velocity on impact."""

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lb, ub = bounds[:, 0], bounds[:, 1]
        violation = (positions < lb) | (positions > ub)
        velocities = np.where(violation, -0.5 * velocities, velocities)
        positions = np.clip(positions, lb, ub)
        return positions, velocities


class ReflectStrategy(BoundsStrategy):
    """Reflect position and negate velocity at boundaries.

    Applied iteratively up to *max_reflections* times, then clamped for safety.
    """

    def __init__(self, max_reflections: int = 5) -> None:
        self.max_reflections = max_reflections

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lb, ub = bounds[:, 0], bounds[:, 1]
        for _ in range(self.max_reflections):
            low_viol = positions < lb
            high_viol = positions > ub
            if not (low_viol.any() or high_viol.any()):
                break
            positions = np.where(low_viol, 2.0 * lb - positions, positions)
            velocities = np.where(low_viol, -velocities, velocities)
            positions = np.where(high_viol, 2.0 * ub - positions, positions)
            velocities = np.where(high_viol, -velocities, velocities)
        positions = np.clip(positions, lb, ub)
        return positions, velocities


_STRATEGIES: dict[str, type[BoundsStrategy]] = {
    "clamp": ClampStrategy,
    "reflect": ReflectStrategy,
}


def get_bounds_strategy(name: str, **kwargs: object) -> BoundsStrategy:
    """Factory — *name* is ``'clamp'`` or ``'reflect'``."""
    try:
        return _STRATEGIES[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown bounds strategy '{name}'. Choose from {list(_STRATEGIES)}"
        )
