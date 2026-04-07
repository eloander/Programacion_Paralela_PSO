# PSO — Particle Swarm Optimization with Parallel Strategies

Modular Python implementation of canonical PSO for minimising continuous
functions, with four interchangeable evaluation strategies (V0-V3) and full
experiment infrastructure (grid search, persistence, visualisation).

---

## Architecture

```
pso/
├── core/           PSO engine (pso.py), state (swarm.py), bounds, topology
├── objectives/     Benchmark functions: Sphere, Rosenbrock, Rastrigin, Ackley
├── parallel/       Evaluator strategies: V0 sequential · V1 threads · V2 processes · V3 asyncio
├── experiments/    Runner (RunConfig) and grid search
├── io/             Save/load results (YAML config, JSON summary, CSV history, NPZ trajectories)
└── viz/            Convergence plots, 2D/3D swarm animations, speedup/boxplot charts
scripts/
├── run_pso.py          Single run
├── run_benchmarks.py   Full benchmark suite
├── run_grid_search.py  Hyper-parameter grid search
└── make_viz.py         Generate all plots and animations
tests/
├── test_pso.py
└── test_benchmarks.py
config/default.yaml
```

**Dependency flow:**  
`scripts` → `experiments` → `core` + `parallel` + `objectives`  
`io` / `viz` are leaf modules with no internal deps.

---

## Install

```bash
# From the repo root
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.10, NumPy, Matplotlib, PyYAML, tqdm.

---

## Quick start

```bash
# Single run — sequential, Sphere d=10
python scripts/run_pso.py --benchmark sphere --dim 10 --strategy v0

# Same benchmark with thread pool
python scripts/run_pso.py --benchmark sphere --dim 10 --strategy v1

# Multiprocessing with batching
python scripts/run_pso.py --benchmark rastrigin --dim 30 --strategy v2 --batch-size 8

# Asyncio (simulated I/O latency)
python scripts/run_pso.py --benchmark ackley --dim 10 --strategy v3

# Record trajectories (needed for animations)
python scripts/run_pso.py --benchmark sphere --dim 2 --strategy v0 --trajectories
```

---

## Benchmark suite

```bash
# Quick test: d=2, 2 seeds, V0+V1
python scripts/run_benchmarks.py --dims 2 --seeds 2 --strategies v0 v1

# Full suite (all functions × d=2,10,30 × 5 seeds × 4 strategies)
python scripts/run_benchmarks.py
```

---

## Grid search

```bash
python scripts/run_grid_search.py --benchmark sphere --dim 10 --strategy v0
python scripts/run_grid_search.py --benchmark rastrigin --dim 30 --seeds 3
```

Grid: w ∈ {0.4, 0.7, 0.9} × c1 ∈ {1.0, 1.5, 2.0} × c2 ∈ {1.0, 1.5, 2.0} = 27 combinations.  
Output saved to `results/gs_<benchmark>_d<dim>_<strategy>_<ts>.csv`.

---

## Visualisation

```bash
# 2-D swarm animation (GIF)
python scripts/make_viz.py --mode swarm2d

# 3-D swarm animation
python scripts/make_viz.py --mode swarm3d

# Convergence curves from saved results
python scripts/make_viz.py --mode convergence --runs-dir results

# Speedup bar chart
python scripts/make_viz.py --mode speedup --runs-dir results

# Fitness boxplots per strategy
python scripts/make_viz.py --mode boxplot --runs-dir results
```

---

## Tests

```bash
pytest -v
```

Covered:
- Reproducibility by seed (all strategies)
- Positions always inside bounds (clamp + reflect)
- Global best is monotonically non-increasing
- Sphere d=5 converges to < 1e-4
- All evaluators return identical fitness values for the same positions

---

## Parallel strategies

| ID | Class | Mechanism | When it helps |
|----|-------|-----------|---------------|
| V0 | `SequentialEvaluator` | Plain Python loop | Baseline |
| V1 | `ThreadingEvaluator` | `ThreadPoolExecutor` | NumPy-heavy or I/O-bound evals (GIL released) |
| V2 | `MultiprocessingEvaluator` | `ProcessPoolExecutor` + batching | Expensive CPU-bound evals; picklable objective required |
| V3 | `AsyncioEvaluator` | `asyncio.gather` + simulated latency | I/O-bound evals (API calls, DB queries) |

**GIL note:** V1 achieves real parallelism only when the objective releases the GIL (NumPy/SciPy routines do; pure-Python loops do not).  
**IPC note:** V2 spawns processes (Windows: `spawn` start method). Scripts must use `if __name__ == '__main__':`. Each process pickle/unpickle data — batch size trades IPC overhead vs load balance.  
**Asyncio note:** V3 is single-threaded cooperative concurrency. It cannot speed up CPU-bound computation but cuts wall time by max(latencies) when evaluations involve genuine I/O waits.

---

## Configuration

All parameters are controlled via `config/default.yaml` and/or CLI flags.  
CLI flags override the config file.

Key PSO parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_particles` | 30 | Swarm size |
| `w` | 0.7 | Inertia weight |
| `c1` | 1.5 | Cognitive coefficient |
| `c2` | 1.5 | Social coefficient |
| `max_iter` | 500 | Hard iteration limit |
| `topology` | `global` | `global` (gBest) or `ring` (lBest, k=2) |
| `bounds_strategy` | `clamp` | `clamp` or `reflect` |
| `stagnation_iter` | 50 | Early-stop patience |
| `seed` | 42 | RNG seed for reproducibility |

---

## Output format

Each run creates `results/<run_id>/`:

```
config.yaml      — full config + hardware metadata + git commit hash
result.json      — best_position, best_fitness, total_s, n_iter, converged
history.csv      — per-iteration: best_fitness, mean_fitness, std_fitness, eval_s, update_s
trajectories.npz — (optional) particle positions shape (n_iter, n_particles, d)
```

Rationale: YAML for human-readable config, JSON for machine-readable summary, CSV for
easy analysis in Python/Excel/R, NPZ for efficient compressed array storage.

---

## Boundary strategy

**Default: `clamp`** — positions are clipped to `[lb, ub]`; velocity is negated and
halved on impact (absorbing-wall effect). Simple, always valid, no oscillation risk.

**Alternative: `reflect`** — position is mirrored about the violated bound; velocity
reverses sign. Better energy conservation but can produce oscillation near boundaries
for large overshoots (handled by iterative application + safety clamp).

---

## Reproducibility checklist

- Every run accepts `--seed`; seed is recorded in `config.yaml`.
- Evaluator strategies V1-V3 use the same PSO RNG path as V0 (only fitness evaluation differs).
- V2 worker seeds are independent but fitness is deterministic, so results match V0.
- V3 latency RNG is separate from PSO RNG; only timing changes, not fitness values.
