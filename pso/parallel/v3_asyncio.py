"""V3 — Concurrent fitness evaluation with asyncio.

Asyncio is cooperative, single-threaded concurrency.  It is NOT suitable for
CPU-bound work (no parallelism), but shines when tasks are I/O-bound: many
tasks can wait concurrently, so total wall time ≈ max(latencies) instead of
Σ(latencies).

Motivating scenario
-------------------
Imagine the "fitness function" is actually a call to an external scoring
service (REST API, database query, remote simulator).  Each evaluation has an
unpredictable network latency.  With asyncio.gather we fire all requests at
once and collect results as they arrive — drastically cutting wall time
compared to sequential requests.

Implementation
--------------
AsyncioEvaluator wraps the objective with a simulated random latency
(asyncio.sleep) to make the demo self-contained.  In a real system, replace
_evaluate_one with an actual async HTTP call (e.g., aiohttp).

The synchronous evaluate() method calls asyncio.run() to launch the event
loop — this is safe from the PSO engine's synchronous main loop.
"""
from __future__ import annotations

import asyncio
from typing import Callable

import numpy as np

from pso.parallel.base import FitnessEvaluator


class AsyncioEvaluator(FitnessEvaluator):
    """Evaluate all particles concurrently via asyncio.gather.

    Parameters
    ----------
    objective:
        Synchronous fitness callable ``f(x) -> float``.
    latency_range:
        ``(lo, hi)`` in seconds — uniform random latency added to each
        particle evaluation to simulate async I/O.
    seed:
        Seed for the latency RNG (ensures reproducible timing in tests).
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        latency_range: tuple[float, float] = (0.005, 0.03),
        seed: int | None = None,
    ) -> None:
        self.objective = objective
        self.latency_range = latency_range
        self._lat_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Synchronous entry-point (called by the PSO engine)
    # ------------------------------------------------------------------

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        # Generate latencies before entering the event loop so the RNG
        # state advances deterministically with each call.
        latencies = self._lat_rng.uniform(
            self.latency_range[0], self.latency_range[1], size=len(positions)
        )
        return asyncio.run(self._gather(positions, latencies))

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _gather(
        self, positions: np.ndarray, latencies: np.ndarray
    ) -> np.ndarray:
        tasks = [
            self._evaluate_one(pos, lat)
            for pos, lat in zip(positions, latencies)
        ]
        results = await asyncio.gather(*tasks)
        return np.array(results)

    async def _evaluate_one(self, position: np.ndarray, latency: float) -> float:
        """Simulate an async I/O call: sleep then compute."""
        await asyncio.sleep(latency)
        return self.objective(position)
