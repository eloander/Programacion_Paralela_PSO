# PSO — Particle Swarm Optimization with Parallel Strategies
## Use case: Autonomous Rocket Landing

Modular Python implementation of canonical PSO for minimising continuous
functions, with **six interchangeable evaluation strategies (V0–V5)** and a
complete experiment infrastructure (grid search, persistence, visualisation).

Applied to a real use case: optimising the 5 parameters of a PD controller
that guides a 2D rocket to a soft landing on a pad. Each fitness evaluation
is a full physics simulation, making this an ideal benchmark for measuring
real parallelism speedup.

---

## Parallel strategies

| ID | Class | Mechanism | Rocket speedup |
|----|-------|-----------|---------------|
| V0 | `SequentialEvaluator` | Plain Python loop | 1× (baseline) |
| V1 | `ThreadingEvaluator` | `ThreadPoolExecutor` | Limited by GIL |
| V2 | `MultiprocessingEvaluator` | `ProcessPoolExecutor` + batching | ~3–5× |
| V3 | `AsyncioEvaluator` | `asyncio.gather` + simulated latency | I/O only |
| V4 | `VectorizedEvaluator` | Single NumPy matrix call | **~20×** |
| V5 | `JoblibEvaluator` | joblib loky (persistent process pool) | ~5–8× |

**V4** passes the full `(n_particles, d)` position matrix to a vectorised
objective that evaluates all particles in one C-level call — no Python loop,
no IPC. For the rocket, this means simulating 30 controllers simultaneously
via NumPy matrix operations over a `(30, 6)` state matrix.

**V5** differs from V2 in that the `loky` backend keeps workers alive across
PSO iterations, eliminating repeated process-spawn overhead.

---

## Rocket landing use case

The rocket starts at x=5 m, y=50 m with a slight tilt and lateral drift.
PSO finds the 5 controller parameters that produce the softest, most accurate
landing:

| Parameter | Range | Role |
|-----------|-------|------|
| `thrust_gain` | [0.5, 3.0] | Scales nominal hover thrust |
| `vertical_damping` | [−2.0, 2.0] | Braking thrust proportional to vertical speed |
| `horizontal_correction` | [0.0, 5.0] | Target tilt angle toward the pad |
| `rotation_gain` | [0.0, 10.0] | PD gain for tilt stabilisation |
| `braking_altitude` | [0.5, 10.0] | Altitude below which braking activates |

Fitness = weighted sum of landing error, impact speed, tilt, fuel used and
instability, plus heavy penalties for crash, fuel depletion and out-of-bounds.

```bash
# Full rocket demo: runs V0–V5, prints speedup table, saves trajectory plot
python scripts/run_rocket.py

# Single run with V4 (fastest)
python scripts/run_pso.py --benchmark rocket --strategy v4 --seed 42

# Trajectory PNG + landing GIF of the optimised controller
python scripts/make_viz.py --mode rocket_traj
```

---

## Architecture

```
pso/
├── core/           PSO engine, state dataclasses, bounds strategies, topologies
├── objectives/
│   ├── benchmarks.py       Sphere, Rosenbrock, Rastrigin, Ackley + vectorised *_vec variants
│   └── rocket_landing.py   Physics simulator + scalar and vectorised objectives
├── parallel/
│   ├── v0_sequential.py    Python for-loop baseline
│   ├── v1_threading.py     ThreadPoolExecutor
│   ├── v2_multiprocessing.py  ProcessPoolExecutor + batching
│   ├── v3_asyncio.py       asyncio.gather with simulated I/O latency
│   ├── v4_numpy.py         Single vectorised NumPy call (no Python loop)
│   └── v5_joblib.py        joblib Parallel, loky backend, persistent pool
├── experiments/    RunConfig, run_experiment(), grid_search()
├── io/             Save/load: YAML config · JSON summary · CSV history · NPZ trajectories
└── viz/
    ├── visualizer.py   Convergence, 2D/3D swarm animations, speedup, boxplot
    └── rocket_viz.py   Trajectory plot and landing animation
scripts/
├── run_pso.py          Single experiment (all benchmarks, all strategies)
├── run_benchmarks.py   Full suite: functions × dims × seeds × strategies
├── run_grid_search.py  Hyper-parameter grid search
├── run_rocket.py       Rocket demo with speedup table
└── make_viz.py         Generate all plots and animations
tests/
├── test_benchmarks.py  23 tests
├── test_pso.py         26 tests
└── test_rocket.py      22 tests   (71 total, all passing)
config/default.yaml
```

---

## Install

```bash
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.10, NumPy, Matplotlib, PyYAML, tqdm, joblib.

---

## Quick start

```bash
# Sequential baseline — Sphere d=10
python scripts/run_pso.py --benchmark sphere --dim 10 --strategy v0

# NumPy vectorised — Rosenbrock d=30
python scripts/run_pso.py --benchmark rosenbrock --dim 30 --strategy v4

# Joblib parallel — Rocket landing
python scripts/run_pso.py --benchmark rocket --strategy v5

# Record trajectories (needed for swarm animations)
python scripts/run_pso.py --benchmark sphere --dim 2 --strategy v0 --trajectories
```

---

## Benchmark suite

```bash
# Quick test: d=2, 2 seeds, V0+V4
python scripts/run_benchmarks.py --dims 2 --seeds 2 --strategies v0 v4

# Full suite (all functions × d=2,10,30 × 5 seeds × V0–V5)
python scripts/run_benchmarks.py --strategies v0 v1 v2 v3 v4 v5
```

Rocket always runs at d=5 (its fixed parameter space) regardless of `--dims`.

---

## Grid search

```bash
python scripts/run_grid_search.py --benchmark sphere --dim 10 --strategy v0
python scripts/run_grid_search.py --benchmark rocket --strategy v4
```

Grid: w ∈ {0.4, 0.7, 0.9} × c1 ∈ {1.0, 1.5, 2.0} × c2 ∈ {1.0, 1.5, 2.0} = 27 combinations.  
Output: `results/gs_<benchmark>_d<dim>_<strategy>_<ts>.csv`

---

## Visualisation

```bash
# Rocket trajectory PNG + landing GIF
python scripts/make_viz.py --mode rocket_traj

# 2D swarm animation (requires --trajectories run first)
python scripts/make_viz.py --mode swarm2d --run-dir results/<run_id>

# 3D swarm animation
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
pytest -v   # 71 tests
```

Coverage:
- Reproducibility by seed for **all six strategies** (V0–V5)
- Positions always inside bounds (clamp and reflect)
- Global best is monotonically non-increasing
- Sphere d=5 converges to < 1e-4
- All evaluators return identical fitness values for the same positions
- Rocket: scalar/vectorised agreement, picklability, physics sanity, PSO integration

---

## Configuration

All parameters via `config/default.yaml` and/or CLI flags (CLI takes precedence).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_particles` | 30 | Swarm size |
| `w` | 0.7 | Inertia weight |
| `c1` | 1.5 | Cognitive coefficient |
| `c2` | 1.5 | Social coefficient |
| `max_iter` | 500 | Hard iteration limit |
| `topology` | `global` | `global` (gBest) or `ring` (lBest) |
| `bounds_strategy` | `clamp` | `clamp` or `reflect` |
| `stagnation_iter` | 50 | Early-stop patience (iterations without improvement) |
| `seed` | 42 | RNG seed for reproducibility |
| `strategy` | `v0` | Evaluator: `v0` · `v1` · `v2` · `v3` · `v4` · `v5` |

---

## Output format

Each run creates `results/<run_id>/`:

```
config.yaml      — full config + hardware metadata + git commit hash
result.json      — best_position, best_fitness, total_s, n_iter, converged
history.csv      — per-iteration: best_fitness, mean_fitness, eval_s, update_s
trajectories.npz — (optional) particle positions, shape (n_iter, n_particles, d)
```

---

## Reproducibility

- Every run accepts `--seed`; the seed is stored in `config.yaml`.
- V1–V5 follow the same PSO RNG path as V0 (only the evaluation step differs),
  so the same seed produces the same best fitness across all strategies.
- V4 may show tiny floating-point differences vs V0 on chaotic rocket trajectories
  (SIMD rounding), but this does not affect optimisation quality.
