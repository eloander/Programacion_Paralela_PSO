"""Grid search over PSO hyper-parameters.

Generates all combinations from a parameter grid, runs each (with multiple
seeds), collects results, and returns a summary DataFrame-style list of dicts.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from pso.core.swarm import PSOResult
from pso.experiments.runner import RunConfig, run_experiment

logger = logging.getLogger("pso.experiments.grid_search")


@dataclass
class GridSearchResult:
    """Aggregated metrics for one hyper-parameter combination."""

    params: dict[str, Any]
    best_fitness_mean: float
    best_fitness_std: float
    best_fitness_min: float
    total_s_mean: float
    n_iter_mean: float
    runs: list[PSOResult] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.params)
        d["best_fitness_mean"] = self.best_fitness_mean
        d["best_fitness_std"] = self.best_fitness_std
        d["best_fitness_min"] = self.best_fitness_min
        d["total_s_mean"] = self.total_s_mean
        d["n_iter_mean"] = self.n_iter_mean
        return d


def grid_search(
    benchmark: str,
    dim: int,
    param_grid: dict[str, list[Any]],
    seeds: list[int] | None = None,
    base_cfg: RunConfig | None = None,
    show_progress: bool = True,
) -> list[GridSearchResult]:
    """Run a full grid search.

    Parameters
    ----------
    benchmark:
        Name of the benchmark function (e.g. ``'sphere'``).
    dim:
        Problem dimensionality.
    param_grid:
        Dict mapping parameter names to lists of values.
        Allowed keys: ``w``, ``c1``, ``c2``, ``n_particles``,
        ``max_iter``, ``topology``, ``strategy``.
    seeds:
        List of integer seeds; each combination is evaluated once per seed.
    base_cfg:
        Base configuration to merge parameter combinations into.
    show_progress:
        Display a tqdm progress bar.

    Returns
    -------
    list[GridSearchResult]
        One entry per parameter combination, sorted by mean best fitness.
    """
    if seeds is None:
        seeds = list(range(5))
    if base_cfg is None:
        base_cfg = RunConfig(benchmark=benchmark, dim=dim)

    # Build all parameter combinations
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(combos) * len(seeds)

    logger.info(
        "Grid search | bench=%s d=%d combos=%d seeds=%d total_runs=%d",
        benchmark, dim, len(combos), len(seeds), total,
    )

    all_results: list[GridSearchResult] = []
    bar = tqdm(total=total, desc="grid search", disable=not show_progress)

    for combo in combos:
        params = dict(zip(keys, combo))
        runs: list[PSOResult] = []

        for seed in seeds:
            cfg = RunConfig(
                benchmark=benchmark,
                dim=dim,
                n_particles=params.get("n_particles", base_cfg.n_particles),
                w=params.get("w", base_cfg.w),
                c1=params.get("c1", base_cfg.c1),
                c2=params.get("c2", base_cfg.c2),
                max_iter=params.get("max_iter", base_cfg.max_iter),
                topology=params.get("topology", base_cfg.topology),
                bounds_strategy=base_cfg.bounds_strategy,
                tol=base_cfg.tol,
                stagnation_iter=base_cfg.stagnation_iter,
                strategy=params.get("strategy", base_cfg.strategy),
                max_workers=base_cfg.max_workers,
                batch_size=base_cfg.batch_size,
                latency_range=base_cfg.latency_range,
                seed=seed,
            )
            try:
                result = run_experiment(cfg)
                runs.append(result)
            except Exception as exc:
                logger.warning("Run failed for %s seed=%d: %s", params, seed, exc)
            bar.update(1)

        if runs:
            fitnesses = np.array([r.best_fitness for r in runs])
            times = np.array([r.total_s for r in runs])
            n_iters = np.array([r.n_iter for r in runs])
            all_results.append(
                GridSearchResult(
                    params=params,
                    best_fitness_mean=float(np.mean(fitnesses)),
                    best_fitness_std=float(np.std(fitnesses)),
                    best_fitness_min=float(np.min(fitnesses)),
                    total_s_mean=float(np.mean(times)),
                    n_iter_mean=float(np.mean(n_iters)),
                    runs=runs,
                )
            )

    bar.close()
    all_results.sort(key=lambda r: r.best_fitness_mean)
    return all_results
