#!/usr/bin/env python
"""Run a single PSO experiment.

Examples
--------
# Sequential, sphere d=10
python scripts/run_pso.py --benchmark sphere --dim 10 --strategy v0

# Threading, rastrigin d=30
python scripts/run_pso.py --benchmark rastrigin --dim 30 --strategy v1 --max-workers 4

# Multiprocessing, ackley d=10, batched
python scripts/run_pso.py --benchmark ackley --dim 10 --strategy v2 --batch-size 8

# Asyncio, sphere d=2, with trajectory recording
python scripts/run_pso.py --benchmark sphere --dim 2 --strategy v3 --trajectories
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from pso.experiments.runner import RunConfig, run_experiment
from pso.io.persistence import save_result


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def _load_yaml_config(path: str | None) -> dict:
    if path is None:
        default = Path(__file__).parent.parent / "config" / "default.yaml"
        if default.exists():
            with default.open() as f:
                return yaml.safe_load(f) or {}
    else:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single PSO experiment.")
    p.add_argument("--config", default=None, help="Path to YAML config file.")

    # Benchmark
    p.add_argument("--benchmark", default=None, choices=["sphere","rosenbrock","rastrigin","ackley"])
    p.add_argument("--dim", type=int, default=None)

    # PSO
    p.add_argument("--n-particles", type=int, default=None)
    p.add_argument("--w",  type=float, default=None)
    p.add_argument("--c1", type=float, default=None)
    p.add_argument("--c2", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--topology", default=None, choices=["global","ring"])
    p.add_argument("--seed", type=int, default=None)

    # Evaluator
    p.add_argument("--strategy", default=None, choices=["v0","v1","v2","v3"])
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)

    # Output
    p.add_argument("--out-dir", default=None)
    p.add_argument("--trajectories", action="store_true", default=False)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    yaml_cfg = _load_yaml_config(args.config)

    pso_cfg  = yaml_cfg.get("pso", {})
    obj_cfg  = yaml_cfg.get("objective", {})
    str_cfg  = yaml_cfg.get("strategy", {})
    out_cfg  = yaml_cfg.get("output", {})

    cfg = RunConfig(
        benchmark       = args.benchmark  or obj_cfg.get("name", "sphere"),
        dim             = args.dim        or obj_cfg.get("dim", 10),
        n_particles     = args.n_particles or pso_cfg.get("n_particles", 30),
        w               = args.w          or pso_cfg.get("w", 0.7),
        c1              = args.c1         or pso_cfg.get("c1", 1.5),
        c2              = args.c2         or pso_cfg.get("c2", 1.5),
        max_iter        = args.max_iter   or pso_cfg.get("max_iter", 500),
        topology        = args.topology   or pso_cfg.get("topology", "global"),
        bounds_strategy = pso_cfg.get("bounds_strategy", "clamp"),
        tol             = pso_cfg.get("tol", 1e-8),
        stagnation_iter = pso_cfg.get("stagnation_iter", 50),
        seed            = args.seed       or pso_cfg.get("seed", 42),
        strategy        = args.strategy   or str_cfg.get("name", "v0"),
        max_workers     = args.max_workers or str_cfg.get("max_workers"),
        batch_size      = args.batch_size  or str_cfg.get("batch_size", 5),
        latency_range   = tuple(str_cfg.get("latency_range", [0.005, 0.03])),
        record_trajectories = args.trajectories or out_cfg.get("save_trajectories", False),
    )

    result = run_experiment(cfg)

    out_dir = args.out_dir or out_cfg.get("dir", "results")
    saved_to = save_result(
        result,
        base_dir=out_dir,
        save_trajectories=cfg.record_trajectories,
        compress=out_cfg.get("compress", True),
    )

    print(f"\n{'='*55}")
    print(f"  Benchmark : {cfg.benchmark}  d={cfg.dim}")
    print(f"  Strategy  : {cfg.strategy}")
    print(f"  Best fit  : {result.best_fitness:.6e}")
    print(f"  Iterations: {result.n_iter}  (converged={result.converged})")
    print(f"  Total time: {result.total_s:.3f}s")
    print(f"  Saved to  : {saved_to}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
