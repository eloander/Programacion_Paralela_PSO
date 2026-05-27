from pso.objectives.benchmarks import (
    BENCHMARKS,
    Benchmark,
    BenchmarkInstance,
    ackley,
    ackley_vec,
    get_benchmark,
    make_instances,
    rastrigin,
    rastrigin_vec,
    rosenbrock,
    rosenbrock_vec,
    sphere,
    sphere_vec,
)
from pso.objectives.rocket_landing import (
    DEFAULT_CONFIG,
    ROCKET_BENCHMARK,
    ROCKET_BOUNDS,
    RocketConfig,
    rocket_landing_objective,
    rocket_landing_vec,
    simulate_trajectory,
)

# Register rocket in the global benchmark dict so get_benchmark("rocket") works
BENCHMARKS["rocket"] = ROCKET_BENCHMARK

__all__ = [
    # Benchmarks registry
    "BENCHMARKS",
    "Benchmark",
    "BenchmarkInstance",
    "get_benchmark",
    "make_instances",
    # Scalar objectives
    "ackley",
    "rastrigin",
    "rosenbrock",
    "sphere",
    # Vectorized objectives
    "ackley_vec",
    "rastrigin_vec",
    "rosenbrock_vec",
    "sphere_vec",
    # Rocket landing
    "DEFAULT_CONFIG",
    "ROCKET_BENCHMARK",
    "ROCKET_BOUNDS",
    "RocketConfig",
    "rocket_landing_objective",
    "rocket_landing_vec",
    "simulate_trajectory",
]
