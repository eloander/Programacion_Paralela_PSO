"""Core PSO engine.

Single entry-point class PSO that accepts any FitnessEvaluator, any
BoundsStrategy, and any Topology.  The algorithm itself is always
sequential (velocity/position updates are cheap Python/NumPy operations);
only the *fitness evaluation* step is handed off to the evaluator, which
is where V0-V3 differ.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from pso.core.bounds import BoundsStrategy, get_bounds_strategy
from pso.core.swarm import IterationRecord, PSOResult, SwarmState
from pso.core.topology import Topology, get_topology
from pso.parallel.base import FitnessEvaluator
from pso.parallel.v0_sequential import SequentialEvaluator

logger = logging.getLogger("pso.core")


class PSO:
    """Canonical PSO for minimising continuous functions.

    Parameters
    ----------
    bounds:
        Array of shape ``(d, 2)`` — each row is ``[lb, ub]`` for one dimension.
    n_particles:
        Swarm size.
    w:
        Inertia weight.
    c1, c2:
        Cognitive and social acceleration coefficients.
    max_iter:
        Hard upper bound on iterations.
    topology:
        ``'global'`` (gBest) or ``'ring'`` (lBest, k=2).
    bounds_strategy:
        ``'clamp'`` (default) or ``'reflect'``.
    tol:
        Minimum absolute improvement to reset the stagnation counter.
    stagnation_iter:
        Stop early if the global best does not improve by *tol* for this
        many consecutive iterations.
    seed:
        Integer seed for the internal NumPy RNG — guarantees reproducibility.
    evaluator:
        A :class:`~pso.parallel.base.FitnessEvaluator` instance.  Pass
        ``None`` together with *objective* to use the sequential evaluator.
    objective:
        Fallback callable ``f(x: ndarray) -> float`` used when *evaluator*
        is ``None``.
    record_trajectories:
        If ``True``, store particle positions at every iteration.  Can be
        memory-intensive for large swarms / many iterations.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        n_particles: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iter: int = 500,
        topology: str | Topology = "global",
        bounds_strategy: str | BoundsStrategy = "clamp",
        tol: float = 1e-8,
        stagnation_iter: int = 50,
        seed: int | None = None,
        evaluator: FitnessEvaluator | None = None,
        objective: Callable[[np.ndarray], float] | None = None,
        record_trajectories: bool = False,
    ) -> None:
        if evaluator is None:
            if objective is None:
                raise ValueError("Provide either 'evaluator' or 'objective'.")
            evaluator = SequentialEvaluator(objective)

        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(self.bounds)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.tol = tol
        self.stagnation_iter = stagnation_iter
        self.seed = seed
        self.evaluator = evaluator
        self.record_trajectories = record_trajectories

        self.topology: Topology = (
            get_topology(topology) if isinstance(topology, str) else topology
        )
        self.bounds_strategy: BoundsStrategy = (
            get_bounds_strategy(bounds_strategy)
            if isinstance(bounds_strategy, str)
            else bounds_strategy
        )
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PSOResult:
        """Execute the PSO and return a :class:`~pso.core.swarm.PSOResult`."""
        t_start = time.perf_counter()

        state = self._initialise()
        self.topology.update(state.pbest_positions, state.pbest_fitness)

        history: list[IterationRecord] = []
        trajectories: list[np.ndarray] = [] if self.record_trajectories else []
        stagnation_count = 0
        prev_best = self.topology.gbest_fitness
        converged = False

        logger.info(
            "PSO started | n=%d d=%d max_iter=%d seed=%s strategy=%s",
            self.n_particles, self.dim, self.max_iter,
            self.seed, type(self.evaluator).__name__,
        )

        for it in range(self.max_iter):
            # --- velocity & position update ---
            t_upd = time.perf_counter()
            social_best = self.topology.get_social_best()   # (n, d)
            r1 = self._rng.random((self.n_particles, self.dim))
            r2 = self._rng.random((self.n_particles, self.dim))
            state.velocities = (
                self.w * state.velocities
                + self.c1 * r1 * (state.pbest_positions - state.positions)
                + self.c2 * r2 * (social_best - state.positions)
            )
            state.positions = state.positions + state.velocities
            state.positions, state.velocities = self.bounds_strategy.apply(
                state.positions, state.velocities, self.bounds
            )
            dt_upd = time.perf_counter() - t_upd

            # --- fitness evaluation (the hot-spot; parallelised in V1-V3) ---
            t_eval = time.perf_counter()
            state.fitness = self.evaluator.evaluate(state.positions)
            dt_eval = time.perf_counter() - t_eval

            # --- update personal and global bests ---
            improved = state.fitness < state.pbest_fitness
            state.pbest_positions = np.where(
                improved[:, np.newaxis], state.positions, state.pbest_positions
            )
            state.pbest_fitness = np.where(
                improved, state.fitness, state.pbest_fitness
            )
            self.topology.update(state.pbest_positions, state.pbest_fitness)

            # --- record ---
            elapsed = time.perf_counter() - t_start
            rec = IterationRecord(
                iteration=it,
                best_fitness=self.topology.gbest_fitness,
                mean_fitness=float(np.mean(state.fitness)),
                std_fitness=float(np.std(state.fitness)),
                elapsed_s=elapsed,
                eval_s=dt_eval,
                update_s=dt_upd,
            )
            history.append(rec)
            if self.record_trajectories:
                trajectories.append(state.positions.copy())

            logger.debug(
                "iter=%4d  gbest=%.6e  mean=%.6e  eval_ms=%5.1f  upd_ms=%4.1f",
                it,
                rec.best_fitness,
                rec.mean_fitness,
                rec.eval_s * 1e3,
                rec.update_s * 1e3,
            )

            # --- stopping criterion ---
            delta = abs(self.topology.gbest_fitness - prev_best)
            stagnation_count = 0 if delta >= self.tol else stagnation_count + 1
            prev_best = self.topology.gbest_fitness
            if stagnation_count >= self.stagnation_iter:
                converged = True
                logger.info(
                    "Early stop at iter=%d | stagnation=%d | gbest=%.6e",
                    it, self.stagnation_iter, self.topology.gbest_fitness,
                )
                break

        total_s = time.perf_counter() - t_start
        logger.info(
            "PSO done | n_iter=%d converged=%s gbest=%.6e total_s=%.3f",
            len(history), converged, self.topology.gbest_fitness, total_s,
        )

        trajs = (
            np.stack(trajectories, axis=0) if (self.record_trajectories and trajectories)
            else None
        )

        return PSOResult(
            best_position=self.topology.gbest_position.copy(),
            best_fitness=self.topology.gbest_fitness,
            history=history,
            total_s=total_s,
            n_iter=len(history),
            converged=converged,
            config=self._config_dict(),
            seed=self.seed,
            trajectories=trajs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialise(self) -> SwarmState:
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        positions = self._rng.uniform(lb, ub, size=(self.n_particles, self.dim))
        v_max = (ub - lb) * 0.5
        velocities = self._rng.uniform(-v_max, v_max, size=(self.n_particles, self.dim))
        fitness = self.evaluator.evaluate(positions)
        return SwarmState(
            positions=positions,
            velocities=velocities,
            fitness=fitness.copy(),
            pbest_positions=positions.copy(),
            pbest_fitness=fitness.copy(),
        )

    def _config_dict(self) -> dict[str, Any]:
        return {
            "n_particles": self.n_particles,
            "dim": self.dim,
            "w": self.w,
            "c1": self.c1,
            "c2": self.c2,
            "max_iter": self.max_iter,
            "topology": type(self.topology).__name__,
            "bounds_strategy": type(self.bounds_strategy).__name__,
            "tol": self.tol,
            "stagnation_iter": self.stagnation_iter,
            "seed": self.seed,
            "evaluator": type(self.evaluator).__name__,
        }
