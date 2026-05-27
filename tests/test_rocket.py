"""Tests for the rocket landing use case.

Covers:
1. Basic output contract (type, shape).
2. Scalar / vectorized consistency.
3. Picklability (required for V2 and V5).
4. Benchmark registry integration.
5. PSO convergence on the rocket objective.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

# Trigger rocket registration in BENCHMARKS before any test runs
import pso.objectives  # noqa: F401

from pso.objectives.benchmarks import get_benchmark
from pso.objectives.rocket_landing import (
    DEFAULT_CONFIG,
    ROCKET_BOUNDS,
    RocketConfig,
    rocket_landing_objective,
    rocket_landing_vec,
    simulate_trajectory,
)


# ---------------------------------------------------------------------------
# Representative controller params
#
# GOOD: thrust_gain < 1.0 → rocket descends at high altitude (thrust < gravity).
#       negative vertical_damping → term vert_damp*vy is positive when vy < 0
#       (falling), adding thrust when descending → braking works correctly.
GOOD_PARAMS = np.array([0.7, -1.5, 2.0, 5.0, 8.0])

# BAD:  thrust_gain=3.0 → nom_thrust = 3*9.81 = 29.43 N, clamped to max_thrust=25 N.
#       Net upward force: 25 - 9.81 = 15.19 N → rocket accelerates up; never lands.
BAD_PARAMS  = np.array([3.0, 2.0, 0.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_scalar_returns_float(self):
        result = rocket_landing_objective(GOOD_PARAMS)
        assert isinstance(result, float)

    def test_scalar_finite(self):
        assert np.isfinite(rocket_landing_objective(GOOD_PARAMS))

    def test_vec_output_shape(self):
        X = np.random.default_rng(0).uniform(0, 5, (8, 5))
        fit = rocket_landing_vec(X)
        assert fit.shape == (8,)

    def test_vec_all_finite(self):
        X = np.random.default_rng(1).uniform(0, 5, (10, 5))
        fit = rocket_landing_vec(X)
        assert np.all(np.isfinite(fit))

    def test_vec_non_negative(self):
        X = np.random.default_rng(2).uniform(0, 5, (10, 5))
        fit = rocket_landing_vec(X)
        assert np.all(fit >= 0.0)


# ---------------------------------------------------------------------------
# 2. Scalar / vectorized consistency
# ---------------------------------------------------------------------------

class TestScalarVecAgreement:
    def test_single_particle_agrees(self):
        s = rocket_landing_objective(GOOD_PARAMS)
        v = float(rocket_landing_vec(GOOD_PARAMS[np.newaxis, :])[0])
        assert abs(s - v) < 1e-10

    def test_batch_agrees_with_scalar(self):
        # Use GOOD_PARAMS with small perturbations to ensure stable (non-chaotic)
        # trajectories where scalar and vectorized give bit-equivalent results.
        rng = np.random.default_rng(7)
        X = np.tile(GOOD_PARAMS, (5, 1)).astype(float)
        X += rng.uniform(-0.1, 0.1, X.shape)
        for i, (lo, hi) in enumerate(ROCKET_BOUNDS):
            X[:, i] = np.clip(X[:, i], lo, hi)
        vec_results = rocket_landing_vec(X)
        for i, row in enumerate(X):
            scalar = rocket_landing_objective(row)
            assert abs(scalar - vec_results[i]) < 1e-8, (
                f"Disagreement at row {i}: scalar={scalar}, vec={vec_results[i]}"
            )


# ---------------------------------------------------------------------------
# 3. Picklability (required for V2/V5 evaluators)
# ---------------------------------------------------------------------------

class TestPicklability:
    def test_objective_picklable(self):
        data = pickle.dumps(rocket_landing_objective)
        obj = pickle.loads(data)
        result = obj(GOOD_PARAMS)
        assert isinstance(result, float)

    def test_vec_objective_picklable(self):
        data = pickle.dumps(rocket_landing_vec)
        fn = pickle.loads(data)
        result = fn(GOOD_PARAMS[np.newaxis, :])
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# 4. Benchmark registry
# ---------------------------------------------------------------------------

class TestBenchmarkRegistry:
    def test_rocket_registered(self):
        bench = get_benchmark("rocket")
        assert bench.name == "rocket"

    def test_bounds_override_shape(self):
        bench = get_benchmark("rocket")
        bounds = bench.bounds_array(5)
        assert bounds.shape == (5, 2)

    def test_bounds_override_values(self):
        bench = get_benchmark("rocket")
        bounds = bench.bounds_array(5)
        # thrust_gain ∈ [0.5, 3.0]
        assert bounds[0, 0] == pytest.approx(0.5)
        assert bounds[0, 1] == pytest.approx(3.0)

    def test_vec_func_present(self):
        bench = get_benchmark("rocket")
        assert bench.vec_func is not None

    def test_vec_func_callable(self):
        bench = get_benchmark("rocket")
        X = np.ones((3, 5))
        result = bench.vec_func(X)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# 5. Physics sanity checks
# ---------------------------------------------------------------------------

class TestPhysicsSanity:
    def test_good_params_better_than_bad(self):
        """Well-tuned controller should outperform a near-zero-thrust one."""
        good = rocket_landing_objective(GOOD_PARAMS)
        bad  = rocket_landing_objective(BAD_PARAMS)
        assert good < bad

    def test_simulate_trajectory_returns_dict(self):
        traj = simulate_trajectory(GOOD_PARAMS)
        assert "states" in traj
        assert "thrust" in traj
        assert "fuel" in traj
        assert "fitness" in traj
        assert "landed" in traj
        assert "t" in traj

    def test_simulate_trajectory_shapes(self):
        traj = simulate_trajectory(GOOD_PARAMS)
        T = len(traj["t"])
        assert traj["states"].shape == (T, 6)
        assert traj["thrust"].shape == (T,)
        assert traj["fuel"].shape == (T,)

    def test_simulate_trajectory_fitness_matches_scalar(self):
        traj = simulate_trajectory(GOOD_PARAMS)
        scalar = rocket_landing_objective(GOOD_PARAMS)
        assert abs(traj["fitness"] - scalar) < 1e-8

    def test_custom_config(self):
        """Tighter timestep should give consistent (not crash-different) results."""
        cfg_fine = RocketConfig(dt=0.01, t_max=12.0)
        f_default = rocket_landing_objective(GOOD_PARAMS)
        f_fine = float(rocket_landing_vec(GOOD_PARAMS[np.newaxis, :], cfg_fine)[0])
        # Both should be finite and reasonable (not penalty territory)
        assert np.isfinite(f_fine)
        assert f_fine < DEFAULT_CONFIG.crash_penalty


# ---------------------------------------------------------------------------
# 6. PSO integration
# ---------------------------------------------------------------------------

class TestPSOIntegration:
    def test_pso_improves_on_rocket(self):
        """PSO with modest budget should find better-than-crash solutions."""
        from pso.core.pso import PSO

        bench = get_benchmark("rocket")
        bounds = bench.bounds_array(5)
        pso = PSO(
            bounds=bounds,
            n_particles=20,
            max_iter=50,
            w=0.7, c1=1.5, c2=1.5,
            seed=0,
            objective=bench.func,
        )
        result = pso.run()
        # Must be strictly better than the crash penalty
        assert result.best_fitness < DEFAULT_CONFIG.crash_penalty

    def test_pso_v4_rocket(self):
        """V4 vectorized evaluator must work with the rocket objective."""
        from pso.core.pso import PSO
        from pso.parallel.v4_numpy import VectorizedEvaluator

        bench = get_benchmark("rocket")
        bounds = bench.bounds_array(5)
        ev = VectorizedEvaluator(bench.func, vec_objective=bench.vec_func)
        pso = PSO(bounds=bounds, n_particles=10, max_iter=20, seed=1, evaluator=ev)
        result = pso.run()
        assert result.best_fitness < DEFAULT_CONFIG.crash_penalty

    def test_pso_v5_rocket(self):
        """V5 joblib evaluator must work with the rocket objective."""
        from pso.core.pso import PSO
        from pso.parallel.v5_joblib import JoblibEvaluator

        bench = get_benchmark("rocket")
        bounds = bench.bounds_array(5)
        ev = JoblibEvaluator(bench.func, n_jobs=2)
        pso = PSO(bounds=bounds, n_particles=10, max_iter=20, seed=2, evaluator=ev)
        result = pso.run()
        assert result.best_fitness < DEFAULT_CONFIG.crash_penalty
