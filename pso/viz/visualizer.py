"""Visualisation module.

Provides:
* convergence_plot  — fitness vs iteration for one or more runs.
* swarm_animation_2d — animated swarm on a 2-D contour map (saves GIF/MP4).
* swarm_animation_3d — animated swarm on a 3-D surface (saves GIF/MP4).
* speedup_plot       — bar chart of strategy speedups vs V0.
* boxplot_fitness    — boxplot of final best fitness across seeds.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Lazy matplotlib import with non-interactive backend fallback
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive; override with env var MPLBACKEND
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    warnings.warn("matplotlib not found; visualisation disabled.")


def _require_mpl() -> None:
    if not _MPL_OK:
        raise ImportError("matplotlib is required for visualisation.")


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def convergence_plot(
    histories: dict[str, list[dict[str, Any]]],
    title: str = "Convergence",
    log_scale: bool = True,
    out_path: str | Path | None = None,
) -> "plt.Figure":
    """Plot best-fitness vs iteration for multiple runs/strategies.

    Parameters
    ----------
    histories:
        Dict mapping a label (e.g. ``'V0 sequential'``) to a history list
        as returned by :func:`~pso.io.persistence.load_result`.
    title:
        Plot title.
    log_scale:
        Use log-scale on the y-axis.
    out_path:
        Save to file if given.
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        iters = [r["iteration"] for r in history]
        fits = [r["best_fitness"] for r in history]
        ax.plot(iters, fits, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 2-D swarm animation
# ---------------------------------------------------------------------------

def swarm_animation_2d(
    trajectories: np.ndarray,         # (n_iter, n_particles, 2)
    objective: Callable,
    bounds: np.ndarray,                # (2, 2)
    gbest_history: list[float],        # best fitness per iteration
    out_path: str | Path = "swarm_2d.gif",
    fps: int = 8,
    grid_pts: int = 80,
    interval: int = 120,
) -> None:
    """Save an animated GIF/MP4 of the swarm on a 2-D contour plot."""
    _require_mpl()
    lb0, ub0 = bounds[0]
    lb1, ub1 = bounds[1]
    xs = np.linspace(lb0, ub0, grid_pts)
    ys = np.linspace(lb1, ub1, grid_pts)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[objective(np.array([x, y])) for x in xs] for y in ys])

    fig, (ax_swarm, ax_fit) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Contour stays fixed
    ax_swarm.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax_swarm.contour(X, Y, Z, levels=30, colors="white", linewidths=0.3, alpha=0.4)
    ax_swarm.set_xlim(lb0, ub0)
    ax_swarm.set_ylim(lb1, ub1)
    ax_swarm.set_xlabel("x₀")
    ax_swarm.set_ylabel("x₁")

    scat = ax_swarm.scatter([], [], c="cyan", s=20, zorder=5, alpha=0.8)
    best_dot = ax_swarm.scatter([], [], c="red", s=80, marker="*", zorder=6)
    title = ax_swarm.set_title("")

    fit_line, = ax_fit.plot([], [], color="tab:blue")
    ax_fit.set_xlim(0, len(trajectories))
    y_min = min(f for f in gbest_history if f > 0) * 0.5 if any(f > 0 for f in gbest_history) else 1e-10
    ax_fit.set_ylim(max(y_min, 1e-12), max(gbest_history) * 1.1)
    ax_fit.set_yscale("log")
    ax_fit.set_xlabel("Iteration")
    ax_fit.set_ylabel("Best fitness")
    ax_fit.set_title("Convergence")
    ax_fit.grid(True, which="both", ls="--", alpha=0.4)

    def _update(frame: int):
        pts = trajectories[frame]          # (n_particles, 2)
        scat.set_offsets(pts)
        best_idx = np.argmin([objective(p) for p in pts])
        best_dot.set_offsets(pts[best_idx : best_idx + 1])
        title.set_text(f"iter={frame}  gbest={gbest_history[frame]:.4e}")
        fit_line.set_data(range(frame + 1), gbest_history[: frame + 1])
        return scat, best_dot, title, fit_line

    ani = animation.FuncAnimation(
        fig, _update, frames=len(trajectories), interval=interval, blit=False
    )
    out_path = Path(out_path)
    _save_animation(ani, out_path, fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3-D swarm animation
# ---------------------------------------------------------------------------

def swarm_animation_3d(
    trajectories: np.ndarray,   # (n_iter, n_particles, 3)
    gbest_history: list[float],
    out_path: str | Path = "swarm_3d.gif",
    fps: int = 6,
    interval: int = 160,
) -> None:
    """Save an animated GIF/MP4 of particle positions in 3-D space."""
    _require_mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_pts = trajectories.reshape(-1, 3)
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_xlabel("x₀"); ax.set_ylabel("x₁"); ax.set_zlabel("x₂")

    scat = ax.scatter([], [], [], c="cyan", s=15, alpha=0.7)
    title = ax.set_title("")

    def _update(frame: int):
        pts = trajectories[frame]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        title.set_text(f"iter={frame}  gbest={gbest_history[frame]:.4e}")
        return (scat,)

    ani = animation.FuncAnimation(
        fig, _update, frames=len(trajectories), interval=interval, blit=False
    )
    out_path = Path(out_path)
    _save_animation(ani, out_path, fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Speedup bar chart
# ---------------------------------------------------------------------------

def speedup_plot(
    strategy_times: dict[str, float],
    baseline: str = "v0",
    out_path: str | Path | None = None,
) -> "plt.Figure":
    """Bar chart of speedup vs baseline."""
    _require_mpl()
    t0 = strategy_times[baseline]
    labels = list(strategy_times.keys())
    speedups = [t0 / strategy_times[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, speedups, color=["tab:gray" if k == baseline else "tab:blue" for k in labels])
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.bar_label(bars, fmt="%.2f×")
    ax.set_ylabel("Speedup vs V0")
    ax.set_title("Strategy speedup")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Fitness boxplot
# ---------------------------------------------------------------------------

def boxplot_fitness(
    strategy_fitnesses: dict[str, list[float]],
    out_path: str | Path | None = None,
    title: str = "Final best fitness",
) -> "plt.Figure":
    """Boxplot of final best fitness across seeds for each strategy."""
    _require_mpl()
    labels = list(strategy_fitnesses.keys())
    data = [strategy_fitnesses[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylabel("Best fitness")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_animation(
    ani: "animation.FuncAnimation",
    path: Path,
    fps: int,
) -> None:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(str(path), writer=writer)
    else:
        # Default: GIF via Pillow
        ani.save(str(path), writer="pillow", fps=fps)
