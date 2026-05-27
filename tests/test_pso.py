"""Unit tests for the PSO engine.

Tests required by the spec:
1. Reproducibility by seed.
2. Bounds handling (positions always inside box with clamp strategy).
3. Monotonic global best (never worsens).
4. Correctness: Sphere converges near 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from pso.core.pso import PSO
from pso.objectives.benchmarks import ackley, rastrigin, rosenbrock, sphere, sphere_vec
from pso.parallel.v0_sequential import SequentialEvaluator
from pso.parallel.v1_threading import ThreadingEvaluator
from pso.parallel.v2_multiprocessing import MultiprocessingEvaluator
from pso.parallel.v3_asyncio import AsyncioEvaluator
from pso.parallel.v4_numpy import VectorizedEvaluator
from pso.parallel.v5_joblib import JoblibEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pso(
    objective=sphere,
    dim: int = 5,
    n_particles: int = 20,
    max_iter: int = 100,
    seed: int = 0,
    strategy: str = "v0",
    record_trajectories: bool = False,
    **kwargs,
) -> PSO:
    bounds = np.array([[-5.12, 5.12]] * dim)
    evaluator_map = {
        "v0": lambda: SequentialEvaluator(objective),
        "v1": lambda: ThreadingEvaluator(objective, max_workers=2),
        "v2": lambda: MultiprocessingEvaluator(objective, max_workers=2, batch_size=5),
        "v3": lambda: AsyncioEvaluator(objective, latency_range=(0.0, 0.001), seed=seed),
        "v4": lambda: VectorizedEvaluator(objective, vec_objective=sphere_vec),
        "v5": lambda: JoblibEvaluator(objective, n_jobs=2),
    }
    ev = evaluator_map[strategy]()
    return PSO(
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        seed=seed,
        evaluator=ev,
        record_trajectories=record_trajectories,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result_v0(self):
        r1 = _make_pso(seed=42).run()
        r2 = _make_pso(seed=42).run()
        assert r1.best_fitness == r2.best_fitness
        np.testing.assert_array_equal(r1.best_position, r2.best_position)

    def test_different_seed_different_result(self):
        r1 = _make_pso(seed=0).run()
        r2 = _make_pso(seed=99).run()
        # Not guaranteed to differ, but extremely likely for any stochastic algo
        assert r1.best_fitness != r2.best_fitness or not np.array_equal(
            r1.best_position, r2.best_position
        )

    def test_v0_v1_same_fitness(self):
        """V1 evaluates same function as V0; optimization quality must be equal."""
        r0 = _make_pso(seed=7, strategy="v0").run()
        r1 = _make_pso(seed=7, strategy="v1").run()
        # Same seed → same RNG path → same positions → same fitness
        assert abs(r0.best_fitness - r1.best_fitness) < 1e-12

    def test_v0_v2_same_fitness(self):
        r0 = _make_pso(seed=7, strategy="v0").run()
        r2 = _make_pso(seed=7, strategy="v2").run()
        assert abs(r0.best_fitness - r2.best_fitness) < 1e-12

    def test_v0_v3_same_fitness(self):
        """V3 adds latency but must compute the same fitness values."""
        r0 = _make_pso(seed=7, strategy="v0").run()
        r3 = _make_pso(seed=7, strategy="v3").run()
        assert abs(r0.best_fitness - r3.best_fitness) < 1e-12

    def test_v0_v4_same_fitness(self):
        """V4 vectorized must reach same best fitness as V0 (same RNG path)."""
        r0 = _make_pso(seed=7, strategy="v0").run()
        r4 = _make_pso(seed=7, strategy="v4").run()
        assert abs(r0.best_fitness - r4.best_fitness) < 1e-10

    def test_v0_v5_same_fitness(self):
        """V5 joblib must reach same best fitness as V0 (same scalar function)."""
        r0 = _make_pso(seed=7, strategy="v0").run()
        r5 = _make_pso(seed=7, strategy="v5").run()
        assert abs(r0.best_fitness - r5.best_fitness) < 1e-12


# ---------------------------------------------------------------------------
# 2. Boundary handling
# ---------------------------------------------------------------------------

class TestBounds:
    def test_clamp_positions_always_in_bounds(self):
        pso = _make_pso(dim=10, max_iter=200, seed=1, record_trajectories=True)
        result = pso.run()
        bounds = np.array([[-5.12, 5.12]] * 10)
        traj = result.trajectories   # (n_iter, n_particles, dim)
        assert traj is not None
        assert np.all(traj >= bounds[:, 0])
        assert np.all(traj <= bounds[:, 1])

    def test_reflect_positions_always_in_bounds(self):
        bounds = np.array([[-5.12, 5.12]] * 5)
        from pso.core.bounds import ReflectStrategy
        pso = PSO(
            bounds=bounds,
            n_particles=15,
            max_iter=100,
            seed=2,
            objective=sphere,
            bounds_strategy=ReflectStrategy(),
            record_trajectories=True,
        )
        result = pso.run()
        traj = result.trajectories
        assert np.all(traj >= bounds[:, 0])
        assert np.all(traj <= bounds[:, 1])

    def test_positions_never_nan(self):
        result = _make_pso(dim=10, max_iter=100, seed=3, record_trajectories=True).run()
        assert not np.any(np.isnan(result.trajectories))


# ---------------------------------------------------------------------------
# 3. Monotonic global best
# ---------------------------------------------------------------------------

class TestMonotonicBest:
    @pytest.mark.parametrize("func", [sphere, rosenbrock, rastrigin, ackley])
    def test_gbest_never_worsens(self, func):
        bounds = np.array([[-5.12, 5.12]] * 5)
        pso = PSO(bounds=bounds, n_particles=15, max_iter=100, seed=0, objective=func)
        result = pso.run()
        fitnesses = [r.best_fitness for r in result.history]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-12, (
                f"Global best worsened at iter {i}: "
                f"{fitnesses[i - 1]} → {fitnesses[i]}"
            )

    def test_gbest_never_worsens_ring_topology(self):
        bounds = np.array([[-5.12, 5.12]] * 5)
        pso = PSO(
            bounds=bounds, n_particles=15, max_iter=100, seed=0,
            objective=sphere, topology="ring",
        )
        result = pso.run()
        fitnesses = [r.best_fitness for r in result.history]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] <= fitnesses[i - 1] + 1e-12


# ---------------------------------------------------------------------------
# 4. Convergence correctness
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_sphere_converges_to_zero(self):
        """PSO must reliably find near-zero fitness for Sphere d=5."""
        bounds = np.array([[-5.12, 5.12]] * 5)
        pso = PSO(
            bounds=bounds, n_particles=30, max_iter=500,
            w=0.7, c1=1.5, c2=1.5, seed=0,
            objective=sphere,
        )
        result = pso.run()
        assert result.best_fitness < 1e-4, (
            f"Sphere d=5 did not converge: best={result.best_fitness}"
        )

    def test_history_length_bounded_by_max_iter(self):
        result = _make_pso(max_iter=50, seed=0).run()
        assert len(result.history) <= 50

    def test_early_stopping_triggers(self):
        """With a very patient stagnation and simple function, early stop fires."""
        bounds = np.array([[-5.12, 5.12]] * 2)
        pso = PSO(
            bounds=bounds, n_particles=20, max_iter=5000,
            seed=0, objective=sphere,
            tol=1e-3, stagnation_iter=20,
        )
        result = pso.run()
        # Should stop long before max_iter
        assert result.converged
        assert result.n_iter < 5000

    def test_result_contains_correct_fields(self):
        result = _make_pso(dim=3, max_iter=10, seed=0).run()
        assert result.best_position.shape == (3,)
        assert isinstance(result.best_fitness, float)
        assert isinstance(result.total_s, float)
        assert result.total_s > 0
        assert len(result.history) > 0
        assert result.seed == 0


# ---------------------------------------------------------------------------
# 5. Evaluator interface contract
# ---------------------------------------------------------------------------

class TestEvaluators:
    def test_sequential_output_shape(self):
        ev = SequentialEvaluator(sphere)
        pos = np.random.default_rng(0).uniform(-5, 5, (10, 4))
        fit = ev.evaluate(pos)
        assert fit.shape == (10,)

    def test_threading_output_shape(self):
        ev = ThreadingEvaluator(sphere, max_workers=2)
        pos = np.random.default_rng(0).uniform(-5, 5, (10, 4))
        fit = ev.evaluate(pos)
        assert fit.shape == (10,)

    def test_multiprocessing_output_shape(self):
        ev = MultiprocessingEvaluator(sphere, max_workers=2, batch_size=3)
        pos = np.random.default_rng(0).uniform(-5, 5, (10, 4))
        fit = ev.evaluate(pos)
        assert fit.shape == (10,)

    def test_asyncio_output_shape(self):
        ev = AsyncioEvaluator(sphere, latency_range=(0.0, 0.001), seed=0)
        pos = np.random.default_rng(0).uniform(-5, 5, (10, 4))
        fit = ev.evaluate(pos)
        assert fit.shape == (10,)

    def test_vectorized_output_shape(self):
        ev = VectorizedEvaluator(sphere, vec_objective=sphere_vec)
        pos = np.random.default_rng(0).uniform(-5, 5, (10, 4))
        fit = ev.evaluate(pos)
        assert fit.shape == (10,)

    def test_joblib_output_shape(self):
        ev = JoblibEvaluator(sphere, n_jobs=2)
        pos = np.random.default_rng(0).uniform(-5, 5, (10, 4))
        fit = ev.evaluate(pos)
        assert fit.shape == (10,)

    def test_all_evaluators_agree(self):
        """All evaluators must return identical values for the same positions."""
        pos = np.random.default_rng(42).uniform(-5, 5, (12, 4))
        ref = SequentialEvaluator(sphere).evaluate(pos)
        for Ev, kw in [
            (ThreadingEvaluator,       {"max_workers": 2}),
            (MultiprocessingEvaluator, {"max_workers": 2, "batch_size": 4}),
            (AsyncioEvaluator,         {"latency_range": (0.0, 0.0), "seed": 0}),
            (VectorizedEvaluator,      {"vec_objective": sphere_vec}),
            (JoblibEvaluator,          {"n_jobs": 2}),
        ]:
            result = Ev(sphere, **kw).evaluate(pos)
            np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)
