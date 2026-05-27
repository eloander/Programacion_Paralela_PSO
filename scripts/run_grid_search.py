#!/usr/bin/env python
"""Run a grid search over PSO hyper-parameters.

Example
-------
python scripts/run_grid_search.py --benchmark sphere --dim 10 --strategy v0
python scripts/run_grid_search.py --benchmark rastrigin --dim 30 --seeds 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pso.objectives  # noqa: F401 — registers rocket in BENCHMARKS
from pso.experiments.grid_search import grid_search
from pso.experiments.runner import RunConfig
from pso.io.persistence import save_grid_search


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.WARNING),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search over PSO hyper-parameters.")
    p.add_argument("--benchmark", default="sphere",
                   choices=["sphere", "rosenbrock", "rastrigin", "ackley", "rocket"])
    p.add_argument("--dim", type=int, default=10,
                   help="Search dimension (ignored for rocket, which uses 5-D param space).")
    p.add_argument("--strategy", default="v0",
                   choices=["v0", "v1", "v2", "v3", "v4", "v5"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-particles", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    param_grid = {
        "w":  [0.4, 0.7, 0.9],
        "c1": [1.0, 1.5, 2.0],
        "c2": [1.0, 1.5, 2.0],
    }

    # Rocket has a fixed 5-D controller parameter space
    dim = 5 if args.benchmark == "rocket" else args.dim
    base = RunConfig(
        benchmark=args.benchmark,
        dim=dim,
        strategy=args.strategy,
        n_particles=args.n_particles,
        max_iter=args.max_iter,
    )

    seeds = list(range(args.seeds))
    print(
        f"Grid search | {args.benchmark} d={dim} | "
        f"{len(param_grid['w'])*len(param_grid['c1'])*len(param_grid['c2'])} combos "
        f"× {args.seeds} seeds"
    )

    results = grid_search(
        benchmark=args.benchmark,
        dim=dim,
        param_grid=param_grid,
        seeds=seeds,
        base_cfg=base,
        show_progress=True,
    )

    csv_path = save_grid_search(
        results, base_dir=args.out_dir,
        tag=f"gs_{args.benchmark}_d{args.dim}_{args.strategy}",
    )

    print(f"\nTop 5 configurations (sorted by mean best fitness):")
    print(f"{'w':>5} {'c1':>5} {'c2':>5} {'mean_fit':>14} {'std_fit':>12} {'mean_t':>8}")
    print("-" * 55)
    for r in results[:5]:
        print(
            f"{r.params.get('w','-'):>5}  "
            f"{r.params.get('c1','-'):>5}  "
            f"{r.params.get('c2','-'):>5}  "
            f"{r.best_fitness_mean:>14.4e}  "
            f"{r.best_fitness_std:>12.4e}  "
            f"{r.total_s_mean:>7.2f}s"
        )
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
