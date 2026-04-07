"""Unit tests for benchmark functions."""
from __future__ import annotations

import numpy as np
import pytest

from pso.objectives.benchmarks import (
    BENCHMARKS,
    Benchmark,
    ackley,
    get_benchmark,
    make_instances,
    rastrigin,
    rosenbrock,
    sphere,
)


# ---------------------------------------------------------------------------
# Minimum values at known points
# ---------------------------------------------------------------------------

class TestMinima:
    def test_sphere_at_zero(self):
        for d in [1, 5, 30]:
            assert sphere(np.zeros(d)) == pytest.approx(0.0)

    def test_sphere_positive_elsewhere(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.uniform(-5, 5, 10)
            if not np.allclose(x, 0):
                assert sphere(x) > 0

    def test_rosenbrock_at_ones(self):
        for d in [2, 5, 10]:
            assert rosenbrock(np.ones(d)) == pytest.approx(0.0, abs=1e-12)

    def test_rosenbrock_at_zero_not_minimum(self):
        # f(0,...,0) = d-1 for Rosenbrock — not the minimum
        assert rosenbrock(np.zeros(5)) > 0

    def test_rastrigin_at_zero(self):
        for d in [1, 5, 20]:
            assert rastrigin(np.zeros(d)) == pytest.approx(0.0, abs=1e-12)

    def test_ackley_at_zero(self):
        for d in [1, 5, 20]:
            assert ackley(np.zeros(d)) == pytest.approx(0.0, abs=1e-10)

    def test_ackley_positive_elsewhere(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = rng.uniform(-10, 10, 5)
            if not np.allclose(x, 0):
                assert ackley(x) > 0


# ---------------------------------------------------------------------------
# Input/output contract
# ---------------------------------------------------------------------------

class TestContract:
    @pytest.mark.parametrize("func", [sphere, rosenbrock, rastrigin, ackley])
    def test_returns_float(self, func):
        x = np.ones(5)
        result = func(x)
        assert isinstance(result, float)

    @pytest.mark.parametrize("func", [sphere, rosenbrock, rastrigin, ackley])
    def test_accepts_various_dims(self, func):
        for d in [2, 5, 10, 30]:
            result = func(np.ones(d))
            assert np.isfinite(result)

    @pytest.mark.parametrize("func", [sphere, rosenbrock, rastrigin, ackley])
    def test_no_nan_for_random_inputs(self, func):
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.uniform(-10, 10, 10)
            assert np.isfinite(func(x))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_benchmarks_registered(self):
        for name in ["sphere", "rosenbrock", "rastrigin", "ackley"]:
            bench = get_benchmark(name)
            assert isinstance(bench, Benchmark)

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent")

    def test_bounds_array_shape(self):
        bench = get_benchmark("sphere")
        for d in [2, 10, 30]:
            arr = bench.bounds_array(d)
            assert arr.shape == (d, 2)
            assert np.all(arr[:, 0] < arr[:, 1])

    def test_make_instances(self):
        instances = make_instances(
            functions=["sphere", "rastrigin"],
            dims=[2, 10],
            seeds=[0, 1, 2],
        )
        assert len(instances) == 2 * 2 * 3
        names = {inst.benchmark_name for inst in instances}
        assert names == {"sphere", "rastrigin"}
