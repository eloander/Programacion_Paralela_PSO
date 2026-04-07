"""Single-experiment runner.

Wires together a benchmark, an evaluator strategy, and PSO parameters into
one reproducible run, and returns the result together with timing metadata.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pso.core.pso import PSO
from pso.core.swarm import PSOResult
from pso.objectives.benchmarks import get_benchmark
from pso.parallel import build_evaluator

logger = logging.getLogger("pso.experiments")


@dataclass
class RunConfig:
    """Complete specification of a single PSO experiment."""

    # Benchmark
    benchmark: str = "sphere"
    dim: int = 10

    # PSO hyper-parameters
    n_particles: int = 30
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    max_iter: int = 500
    topology: str = "global"
    bounds_strategy: str = "clamp"
    tol: float = 1e-8
    stagnation_iter: int = 50
    seed: int | None = 42

    # Evaluator
    strategy: str = "v0"
    max_workers: int | None = None
    batch_size: int = 5
    latency_range: tuple[float, float] = (0.005, 0.03)

    # Output
    record_trajectories: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "dim": self.dim,
            "n_particles": self.n_particles,
            "w": self.w,
            "c1": self.c1,
            "c2": self.c2,
            "max_iter": self.max_iter,
            "topology": self.topology,
            "bounds_strategy": self.bounds_strategy,
            "tol": self.tol,
            "stagnation_iter": self.stagnation_iter,
            "seed": self.seed,
            "strategy": self.strategy,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "latency_range": list(self.latency_range),
        }


def run_experiment(cfg: RunConfig) -> PSOResult:
    """Execute one PSO run described by *cfg* and return the result."""
    bench = get_benchmark(cfg.benchmark)
    bounds = bench.bounds_array(cfg.dim)

    # Build evaluator kwargs depending on strategy
    eval_kwargs: dict[str, Any] = {}
    if cfg.strategy == "v1":
        eval_kwargs["max_workers"] = cfg.max_workers
    elif cfg.strategy == "v2":
        eval_kwargs["max_workers"] = cfg.max_workers
        eval_kwargs["batch_size"] = cfg.batch_size
    elif cfg.strategy == "v3":
        eval_kwargs["latency_range"] = cfg.latency_range

    evaluator = build_evaluator(cfg.strategy, bench.func, **eval_kwargs)

    pso = PSO(
        bounds=bounds,
        n_particles=cfg.n_particles,
        w=cfg.w,
        c1=cfg.c1,
        c2=cfg.c2,
        max_iter=cfg.max_iter,
        topology=cfg.topology,
        bounds_strategy=cfg.bounds_strategy,
        tol=cfg.tol,
        stagnation_iter=cfg.stagnation_iter,
        seed=cfg.seed,
        evaluator=evaluator,
        record_trajectories=cfg.record_trajectories,
    )

    logger.info(
        "Running | bench=%s d=%d strategy=%s seed=%s",
        cfg.benchmark, cfg.dim, cfg.strategy, cfg.seed,
    )
    result = pso.run()

    # Inject run config into result's config dict for persistence
    result.config.update(cfg.to_dict())
    return result
