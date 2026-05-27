#!/usr/bin/env python
"""Rocket landing demo: compare all PSO strategies on the controller optimisation problem.

The rocket landing objective optimises 5 controller parameters of a 2-D physics
simulator.  Each particle represents a different controller; each PSO fitness
evaluation runs a full simulation.

Strategies compared
-------------------
V0 — sequential baseline
V1 — ThreadPoolExecutor (GIL-limited for CPU-bound sims)
V2 — ProcessPoolExecutor + batching (true parallelism, IPC overhead)
V3 — asyncio (simulates remote/async evaluation latency)
V4 — NumPy vectorized (all particles simulated simultaneously)
V5 — joblib loky (persistent process pool, lower overhead than V2)

Output
------
* Console: speedup table
* results/  : individual run result directories
* rocket_trajectory.png : flight path of the best controller found
* rocket_landing.gif    : animation of the best landing sequence

Example
-------
python scripts/run_rocket.py
python scripts/run_rocket.py --n-particles 40 --max-iter 300 --seed 0
python scripts/run_rocket.py --strategies v0 v2 v4   # subset of strategies
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    # Windows multiprocessing guard — must be before any pso imports
    import multiprocessing
    multiprocessing.freeze_support()

import numpy as np
from tqdm import tqdm

import pso.objectives  # noqa: F401 — registers rocket in BENCHMARKS
from pso.experiments.runner import RunConfig, run_experiment
from pso.io.persistence import save_result
from pso.objectives.benchmarks import get_benchmark
from pso.viz.rocket_viz import animate_landing, plot_trajectory


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.WARNING),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rocket landing PSO demo.")
    p.add_argument("--strategies", nargs="+",
                   default=["v0", "v1", "v2", "v3", "v4", "v5"],
                   choices=["v0", "v1", "v2", "v3", "v4", "v5"])
    p.add_argument("--n-particles", type=int, default=30)
    p.add_argument("--max-iter",    type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--batch-size",  type=int, default=5)
    p.add_argument("--out-dir",     default="results")
    p.add_argument("--no-viz",      action="store_true",
                   help="Skip trajectory and animation output.")
    p.add_argument("--log-level",   default="WARNING")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    bench = get_benchmark("rocket")

    print("\n" + "=" * 65)
    print("  PSO Rocket Landing Demo — Controller Optimisation")
    print("=" * 65)
    print(f"  Objective : rocket landing (5-D parameter space)")
    print(f"  Particles : {args.n_particles}")
    print(f"  Max iter  : {args.max_iter}")
    print(f"  Seed      : {args.seed}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print("=" * 65 + "\n")

    results = {}
    saved_dirs = {}

    for strategy in tqdm(args.strategies, desc="strategies"):
        cfg = RunConfig(
            benchmark="rocket",
            dim=5,
            n_particles=args.n_particles,
            max_iter=args.max_iter,
            seed=args.seed,
            strategy=strategy,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
        )
        t0 = time.perf_counter()
        result = run_experiment(cfg)
        wall_time = time.perf_counter() - t0
        results[strategy] = (result, wall_time)
        saved_dirs[strategy] = save_result(result, base_dir=args.out_dir)

    # ---- Speedup table ----
    baseline_time = results["v0"][0].total_s if "v0" in results else None

    print(f"\n{'Strategy':>10} {'Time (s)':>10} {'Speedup':>9} {'Best fitness':>14} "
          f"{'Iters':>7} {'Converged':>10}")
    print("-" * 65)
    for strategy in args.strategies:
        result, _ = results[strategy]
        speedup = (baseline_time / result.total_s) if baseline_time else float("nan")
        conv = "yes" if result.converged else "no"
        print(
            f"{strategy:>10} {result.total_s:>10.3f} {speedup:>9.2f}x "
            f"{result.best_fitness:>14.4f} {result.n_iter:>7d} {conv:>10}"
        )
    print("-" * 65)

    # ---- Best overall result ----
    best_strategy = min(results, key=lambda s: results[s][0].best_fitness)
    best_result, _ = results[best_strategy]
    best_params = best_result.best_position

    print(f"\nBest strategy   : {best_strategy}")
    print(f"Best fitness    : {best_result.best_fitness:.4f}")
    print(f"Best params     :")
    param_names = [
        "thrust_gain", "vertical_damping", "horizontal_correction",
        "rotation_gain", "braking_altitude",
    ]
    for name, val in zip(param_names, best_params):
        print(f"  {name:<28} = {val:.4f}")

    # ---- Visualisation ----
    if not args.no_viz:
        traj_path = "rocket_trajectory.png"
        anim_path = "rocket_landing.gif"
        print(f"\nSaving trajectory plot → {traj_path}")
        try:
            plot_trajectory(best_params, save_path=traj_path)
        except Exception as exc:
            print(f"  [warning] trajectory plot failed: {exc}")

        print(f"Saving landing animation → {anim_path}")
        try:
            animate_landing(best_params, save_path=anim_path, fps=20, skip=2)
        except Exception as exc:
            print(f"  [warning] animation failed: {exc}")

    print(f"\nResults saved to: {args.out_dir}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
