"""Standard benchmark objective functions for PSO evaluation.

All functions are *module-level* so they can be pickled (required for
multiprocessing on Windows).  Each accepts a 1-D NumPy array and returns
a float.  Global minima are at *x = 0* (except Rosenbrock: *x = 1*).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    """f(x) = Σ xᵢ².  Global min = 0 at origin."""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """f(x) = Σ [100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²].  Global min = 0 at x=(1,…,1)."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """f(x) = 10d + Σ [xᵢ² − 10·cos(2πxᵢ)].  Global min = 0 at origin."""
    A = 10.0
    return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function.  Global min = 0 at origin."""
    d = len(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
    return float(term1 + term2 + 20.0 + np.e)


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Benchmark:
    """A named benchmark problem instance."""

    name: str
    func: Callable[[np.ndarray], float]
    lb: float   # lower bound (same for every dimension)
    ub: float   # upper bound (same for every dimension)
    known_minimum: float = 0.0

    def bounds_array(self, d: int) -> np.ndarray:
        """Return ``(d, 2)`` bounds array for dimension *d*."""
        return np.array([[self.lb, self.ub]] * d, dtype=float)


BENCHMARKS: dict[str, Benchmark] = {
    "sphere":     Benchmark("sphere",     sphere,     -5.12,   5.12,  0.0),
    "rosenbrock": Benchmark("rosenbrock", rosenbrock, -2.048,  2.048, 0.0),
    "rastrigin":  Benchmark("rastrigin",  rastrigin,  -5.12,   5.12,  0.0),
    "ackley":     Benchmark("ackley",     ackley,     -32.768, 32.768, 0.0),
}


def get_benchmark(name: str) -> Benchmark:
    """Look up a benchmark by name."""
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from {list(BENCHMARKS)}")


# ---------------------------------------------------------------------------
# Reproducible instance set for experiments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkInstance:
    """A fully-specified, reproducible benchmark run configuration."""

    benchmark_name: str
    dim: int
    seed: int


def make_instances(
    functions: list[str] | None = None,
    dims: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[BenchmarkInstance]:
    """Generate a Cartesian product of (function, dim, seed) instances."""
    if functions is None:
        functions = list(BENCHMARKS)
    if dims is None:
        dims = [2, 10, 30]
    if seeds is None:
        seeds = list(range(5))
    return [
        BenchmarkInstance(fn, d, s)
        for fn in functions
        for d in dims
        for s in seeds
    ]
