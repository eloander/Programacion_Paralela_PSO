#!/usr/bin/env python
"""Run the full benchmark suite across strategies, dimensions, and seeds.

For each (function × dimension × seed × strategy) combination, executes PSO
and saves results.  Prints a summary table at the end.

Example
-------
# Quick smoke-test: d=2 only, 2 seeds, V0 and V1
python scripts/run_benchmarks.py --dims 2 --seeds 2 --strategies v0 v1

# Full suite (takes several minutes)
python scripts/run_benchmarks.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import pso.objectives  # noqa: F401 — registers rocket in BENCHMARKS
from pso.experiments.runner import RunConfig, run_experiment
from pso.io.persistence import save_result
from pso.objectives.benchmarks import BENCHMARKS


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.WARNING),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark suite.")
    p.add_argument("--functions", nargs="+", default=list(BENCHMARKS),
                   choices=list(BENCHMARKS))
    p.add_argument("--dims", nargs="+", type=int, default=[2, 10, 30])
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds (0 … seeds-1).")
    p.add_argument("--strategies", nargs="+", default=["v0", "v1", "v2", "v3"],
                   choices=["v0", "v1", "v2", "v3", "v4", "v5"])
    p.add_argument("--n-particles", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    seeds = list(range(args.seeds))
    runs = []
    for fn in args.functions:
        dims = [5] if fn == "rocket" else args.dims  # rocket has fixed 5-D param space
        for d in dims:
            for s in seeds:
                for st in args.strategies:
                    runs.append((fn, d, s, st))

    print(f"Benchmark suite: {len(runs)} runs "
          f"({len(args.functions)} funcs × {len(args.dims)} dims × "
          f"{args.seeds} seeds × {len(args.strategies)} strategies)\n")

    summary_rows = []
    t_suite_start = time.perf_counter()

    for fn, dim, seed, strategy in tqdm(runs, desc="benchmarks"):
        cfg = RunConfig(
            benchmark=fn, dim=dim, seed=seed, strategy=strategy,
            n_particles=args.n_particles, max_iter=args.max_iter,
            max_workers=args.max_workers,
        )
        result = run_experiment(cfg)
        save_result(result, base_dir=args.out_dir)
        summary_rows.append({
            "fn": fn, "dim": dim, "seed": seed, "strategy": strategy,
            "best": result.best_fitness, "time_s": result.total_s,
            "n_iter": result.n_iter,
        })

    t_total = time.perf_counter() - t_suite_start
    print(f"\nSuite completed in {t_total:.1f}s\n")

    # Print compact summary table
    print(f"{'Function':<12} {'dim':>4} {'strategy':>10} {'mean_best':>14} {'mean_t':>8}")
    print("-" * 55)
    # Group by (fn, dim, strategy)
    from collections import defaultdict
    import statistics
    groups: dict = defaultdict(list)
    for row in summary_rows:
        key = (row["fn"], row["dim"], row["strategy"])
        groups[key].append((row["best"], row["time_s"]))

    for (fn, dim, strategy), vals in sorted(groups.items()):
        bests, times = zip(*vals)
        print(
            f"{fn:<12} {dim:>4} {strategy:>10} "
            f"{statistics.mean(bests):>14.4e} {statistics.mean(times):>7.2f}s"
        )


if __name__ == "__main__":
    main()
