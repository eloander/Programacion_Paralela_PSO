"""Persistence: save and load PSO results to/from disk.

Directory layout per run
------------------------
    results/<run_id>/
        config.yaml      — full run configuration (human-readable)
        result.json      — final best position, fitness, timing
        history.csv      — per-iteration metrics
        trajectories.npz — (optional) particle positions per iteration

Format rationale
----------------
* YAML for config: human-readable, easy to edit and diff.
* JSON for result summary: standard, no extra deps, fast for small payloads.
* CSV for history: simple, compatible with pandas / Excel / any spreadsheet.
* NPZ (compressed NumPy) for trajectories: compact, lossless, fast I/O.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pso.core.swarm import IterationRecord, PSOResult


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_result(
    result: PSOResult,
    run_id: str | None = None,
    base_dir: str | Path = "results",
    save_trajectories: bool = False,
    compress: bool = True,
) -> Path:
    """Persist a :class:`~pso.core.swarm.PSOResult` to disk.

    Parameters
    ----------
    result:
        The result to save.
    run_id:
        Sub-directory name.  Auto-generated from timestamp if ``None``.
    base_dir:
        Root output directory.
    save_trajectories:
        Write position trajectories (can be large).
    compress:
        Compress NPZ files with zlib.

    Returns
    -------
    Path
        The directory where files were written.
    """
    if run_id is None:
        run_id = f"run_{int(time.time() * 1000)}"

    out_dir = Path(base_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- config.yaml ----
    cfg_path = out_dir / "config.yaml"
    config_payload = dict(result.config)
    config_payload["_meta"] = _meta()
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_payload, f, default_flow_style=False, sort_keys=True)

    # ---- result.json ----
    res_path = out_dir / "result.json"
    summary = {
        "best_fitness": float(result.best_fitness),
        "best_position": result.best_position.tolist(),
        "total_s": float(result.total_s),
        "n_iter": result.n_iter,
        "converged": result.converged,
        "seed": result.seed,
    }
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- history.csv ----
    hist_path = out_dir / "history.csv"
    _write_history_csv(hist_path, result.history)

    # ---- trajectories.npz ----
    if save_trajectories and result.trajectories is not None:
        traj_path = out_dir / "trajectories.npz"
        if compress:
            np.savez_compressed(traj_path, trajectories=result.trajectories)
        else:
            np.savez(traj_path, trajectories=result.trajectories)

    return out_dir


def _write_history_csv(path: Path, history: list[IterationRecord]) -> None:
    fieldnames = [
        "iteration", "best_fitness", "mean_fitness", "std_fitness",
        "elapsed_s", "eval_s", "update_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in history:
            writer.writerow({
                "iteration":    rec.iteration,
                "best_fitness": rec.best_fitness,
                "mean_fitness": rec.mean_fitness,
                "std_fitness":  rec.std_fitness,
                "elapsed_s":    rec.elapsed_s,
                "eval_s":       rec.eval_s,
                "update_s":     rec.update_s,
            })


def _meta() -> dict[str, Any]:
    """Collect hardware / software metadata for reproducibility."""
    meta: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = "unavailable"
    return meta


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_result(run_dir: str | Path) -> dict[str, Any]:
    """Load a saved run back into memory as plain dicts/arrays.

    Returns a dict with keys: ``config``, ``result``, ``history``,
    ``trajectories`` (or ``None``).
    """
    run_dir = Path(run_dir)
    out: dict[str, Any] = {}

    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            out["config"] = yaml.safe_load(f)

    res_path = run_dir / "result.json"
    if res_path.exists():
        with res_path.open("r", encoding="utf-8") as f:
            out["result"] = json.load(f)

    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        out["history"] = _read_history_csv(hist_path)

    traj_path = run_dir / "trajectories.npz"
    if traj_path.exists():
        data = np.load(traj_path)
        out["trajectories"] = data["trajectories"]
    else:
        out["trajectories"] = None

    return out


def _read_history_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "iteration":    int(row["iteration"]),
                "best_fitness": float(row["best_fitness"]),
                "mean_fitness": float(row["mean_fitness"]),
                "std_fitness":  float(row["std_fitness"]),
                "elapsed_s":    float(row["elapsed_s"]),
                "eval_s":       float(row["eval_s"]),
                "update_s":     float(row["update_s"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------

def list_runs(base_dir: str | Path = "results") -> list[Path]:
    """Return sorted list of run directories."""
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def save_grid_search(
    gs_results: list,   # list[GridSearchResult]
    base_dir: str | Path = "results",
    tag: str = "grid",
) -> Path:
    """Save grid search summary to a single CSV file."""
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_{int(time.time())}.csv"

    if not gs_results:
        return path

    rows = [r.to_dict() for r in gs_results]
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
