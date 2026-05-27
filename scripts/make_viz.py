#!/usr/bin/env python
"""Generate visualisations from saved PSO results.

Examples
--------
# Animate a 2-D swarm (must have been run with --trajectories)
python scripts/make_viz.py --mode swarm2d --run-dir results/run_XYZ

# Convergence plot comparing strategies
python scripts/make_viz.py --mode convergence --runs-dir results

# Speedup bar chart
python scripts/make_viz.py --mode speedup --runs-dir results

# Rocket landing trajectory (runs fresh PSO if no --run-dir given)
python scripts/make_viz.py --mode rocket_traj
python scripts/make_viz.py --mode rocket_traj --run-dir results/run_XYZ --out rocket.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import pso.objectives  # noqa: F401 — registers rocket in BENCHMARKS
from pso.experiments.runner import RunConfig, run_experiment
from pso.io.persistence import list_runs, load_result
from pso.objectives.benchmarks import get_benchmark
from pso.viz.visualizer import (
    boxplot_fitness,
    convergence_plot,
    speedup_plot,
    swarm_animation_2d,
    swarm_animation_3d,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PSO visualisations.")
    p.add_argument("--mode", required=True,
                   choices=["swarm2d", "swarm3d", "convergence", "speedup",
                            "boxplot", "rocket_traj"])
    p.add_argument("--run-dir",   default=None, help="Single run directory (for swarm modes).")
    p.add_argument("--runs-dir",  default="results", help="Root results directory.")
    p.add_argument("--benchmark", default="sphere",
                   choices=["sphere", "rosenbrock", "rastrigin", "ackley", "rocket"])
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--out", default=None, help="Output file path.")
    return p.parse_args()


def _swarm2d(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        print("Running a fresh 2-D sphere run with trajectory recording …")
        cfg = RunConfig(
            benchmark="sphere", dim=2, strategy="v0",
            n_particles=25, max_iter=80, seed=42,
            record_trajectories=True,
        )
        result = run_experiment(cfg)
        trajectories = result.trajectories
        gbest_hist = [r.best_fitness for r in result.history]
    else:
        data = load_result(args.run_dir)
        trajectories = data.get("trajectories")
        if trajectories is None:
            print("No trajectories found. Re-run with --trajectories flag.")
            return
        gbest_hist = [r["best_fitness"] for r in data["history"]]

    bench = get_benchmark(args.benchmark if args.run_dir is None else
                          load_result(args.run_dir).get("config",{}).get("benchmark","sphere"))
    out = args.out or "swarm_2d.gif"
    d = trajectories.shape[2]
    bounds = bench.bounds_array(d)
    print(f"Saving 2-D animation → {out}")
    swarm_animation_2d(trajectories, bench.func, bounds, gbest_hist, out_path=out)


def _swarm3d(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        print("Running a fresh 3-D sphere run with trajectory recording …")
        cfg = RunConfig(
            benchmark="sphere", dim=3, strategy="v0",
            n_particles=20, max_iter=60, seed=42,
            record_trajectories=True,
        )
        result = run_experiment(cfg)
        trajectories = result.trajectories
        gbest_hist = [r.best_fitness for r in result.history]
    else:
        data = load_result(args.run_dir)
        trajectories = data.get("trajectories")
        gbest_hist = [r["best_fitness"] for r in data["history"]]

    out = args.out or "swarm_3d.gif"
    print(f"Saving 3-D animation → {out}")
    swarm_animation_3d(trajectories, gbest_hist, out_path=out)


def _convergence(args: argparse.Namespace) -> None:
    runs = list_runs(args.runs_dir)
    histories = {}
    for run_dir in runs[:20]:
        data = load_result(run_dir)
        label = run_dir.name
        if "config" in data:
            c = data["config"]
            label = f"{c.get('strategy','?')} {c.get('benchmark','?')} d={c.get('dim','?')}"
        if data.get("history"):
            histories[label] = data["history"]

    if not histories:
        print("No histories found in", args.runs_dir)
        return

    out = args.out or "convergence.png"
    print(f"Saving convergence plot → {out}")
    fig = convergence_plot(histories, out_path=out)


def _speedup(args: argparse.Namespace) -> None:
    """Collect mean total_s per strategy from results dir and plot speedup."""
    from collections import defaultdict
    import statistics

    runs = list_runs(args.runs_dir)
    times: dict[str, list[float]] = defaultdict(list)
    for run_dir in runs:
        data = load_result(run_dir)
        cfg = data.get("config", {})
        st  = cfg.get("strategy")
        res = data.get("result", {})
        t   = res.get("total_s")
        if st and t:
            times[st].append(t)

    if not times:
        print("No timing data found."); return

    mean_times = {st: statistics.mean(ts) for st, ts in times.items() if ts}
    out = args.out or "speedup.png"
    print(f"Saving speedup plot → {out}")
    speedup_plot(mean_times, out_path=out)


def _boxplot(args: argparse.Namespace) -> None:
    from collections import defaultdict

    runs = list_runs(args.runs_dir)
    fitnesses: dict[str, list[float]] = defaultdict(list)
    for run_dir in runs:
        data = load_result(run_dir)
        cfg = data.get("config", {})
        res = data.get("result", {})
        st  = cfg.get("strategy")
        bf  = res.get("best_fitness")
        if st and bf is not None:
            fitnesses[st].append(bf)

    if not fitnesses:
        print("No fitness data found."); return

    out = args.out or "boxplot.png"
    print(f"Saving boxplot → {out}")
    boxplot_fitness(fitnesses, out_path=out)


def _rocket_traj(args: argparse.Namespace) -> None:
    """Run PSO on the rocket objective and visualise the best trajectory."""
    from pso.viz.rocket_viz import animate_landing, plot_trajectory

    best_params = None

    if args.run_dir is not None:
        data = load_result(args.run_dir)
        cfg_dict = data.get("config", {})
        if cfg_dict.get("benchmark") == "rocket":
            res = data.get("result", {})
            best_params_list = res.get("best_position")
            if best_params_list is not None:
                best_params = np.array(best_params_list)

    if best_params is None:
        print("Running fresh PSO on rocket objective (v0, 30 particles, 200 iter) …")
        cfg = RunConfig(
            benchmark="rocket", dim=5, strategy="v0",
            n_particles=30, max_iter=200, seed=42,
            record_trajectories=False,
        )
        result = run_experiment(cfg)
        best_params = result.best_position
        print(f"  Best fitness: {result.best_fitness:.4f}")
        print(f"  Best params:  {best_params}")

    traj_out = args.out or "rocket_trajectory.png"
    anim_out = traj_out.replace(".png", "_landing.gif") if traj_out.endswith(".png") else "rocket_landing.gif"

    print(f"Saving trajectory plot -> {traj_out}")
    plot_trajectory(best_params, save_path=traj_out)

    print(f"Saving landing animation -> {anim_out}")
    animate_landing(best_params, save_path=anim_out)


def main() -> None:
    args = _parse_args()
    dispatch = {
        "swarm2d":     _swarm2d,
        "swarm3d":     _swarm3d,
        "convergence": _convergence,
        "speedup":     _speedup,
        "boxplot":     _boxplot,
        "rocket_traj": _rocket_traj,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
