"""Rocket landing visualisation.

Two entry points:
- ``plot_trajectory``: static PNG showing the full flight path.
- ``animate_landing``: GIF/MP4 animation of the landing sequence.

Both call ``simulate_trajectory`` from ``pso.objectives.rocket_landing`` so
the visualisation always matches the objective function's physics exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

from pso.objectives.rocket_landing import DEFAULT_CONFIG, RocketConfig, simulate_trajectory


# ---------------------------------------------------------------------------
# Static trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(
    params: np.ndarray,
    cfg: RocketConfig = DEFAULT_CONFIG,
    save_path: str | Path = "rocket_trajectory.png",
) -> None:
    """Plot the flight trajectory of the controller defined by *params*.

    Shows:
    * Flight path coloured by time (blue → red).
    * Velocity arrows sampled every N steps.
    * Landing pad marker at (0, 0).
    * Key events: launch, landing (or crash).
    * Side panels: altitude vs time, speed vs time, tilt vs time, fuel vs time.
    """
    traj = simulate_trajectory(params, cfg)
    states = traj["states"]         # (T, 6)
    thrust = traj["thrust"]         # (T,)
    fuel   = traj["fuel"]           # (T,)
    t_axis = traj["t"]              # (T,)
    landed = traj["landed"]
    fitness = traj["fitness"]

    x, y, vx, vy, theta, omega = states.T
    T = len(t_axis)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Rocket Landing Trajectory  |  fitness = {fitness:.2f}  |  "
        f"{'LANDED ✓' if landed else 'CRASH ✗'}",
        fontsize=13, fontweight="bold",
    )

    # ---- Left: trajectory in (x, y) space ----
    ax_traj = fig.add_subplot(1, 2, 1)
    colors = plt.cm.plasma(np.linspace(0, 1, T))
    for i in range(T - 1):
        ax_traj.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2)

    # Velocity arrows (sub-sampled)
    arrow_step = max(1, T // 15)
    for i in range(0, T, arrow_step):
        speed = (vx[i] ** 2 + vy[i] ** 2) ** 0.5
        if speed > 0.1:
            scale = min(3.0, speed)
            ax_traj.annotate(
                "", xy=(x[i] + vx[i] * scale * 0.3, y[i] + vy[i] * scale * 0.3),
                xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
            )

    # Landing pad
    pad_w = 3.0
    ax_traj.fill_between([-pad_w, pad_w], [-0.5, -0.5], [0, 0],
                         color="gray", alpha=0.6, label="Landing pad")
    ax_traj.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)

    # Start / end markers
    ax_traj.plot(x[0], y[0], "go", ms=10, label="Launch", zorder=5)
    marker = "b*" if landed else "rx"
    label  = "Landing" if landed else "Crash"
    ax_traj.plot(x[-1], y[-1], marker, ms=12, label=label, zorder=5)

    ax_traj.set_xlabel("Horizontal position (m)")
    ax_traj.set_ylabel("Altitude (m)")
    ax_traj.set_title("Flight path")
    ax_traj.legend(loc="upper right", fontsize=9)
    ax_traj.set_xlim(min(x) - 3, max(x) + 3)
    ax_traj.set_ylim(-2, max(y) + 5)

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=t_axis[0], vmax=t_axis[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_traj, label="Time (s)", shrink=0.7)

    # ---- Right: 4 time-series subplots ----
    axes_right = fig.add_subplot(4, 2, 2), fig.add_subplot(4, 2, 4), \
                 fig.add_subplot(4, 2, 6), fig.add_subplot(4, 2, 8)

    axes_right[0].plot(t_axis, y, "royalblue")
    axes_right[0].set_ylabel("Altitude (m)")
    axes_right[0].set_title("Altitude vs Time")

    speed = np.sqrt(vx ** 2 + vy ** 2)
    axes_right[1].plot(t_axis, speed, "crimson")
    axes_right[1].set_ylabel("Speed (m/s)")
    axes_right[1].set_title("Speed vs Time")

    axes_right[2].plot(t_axis, np.degrees(theta), "darkorange")
    axes_right[2].axhline(0, color="k", lw=0.7, ls="--")
    axes_right[2].set_ylabel("Tilt (°)")
    axes_right[2].set_title("Tilt vs Time")

    axes_right[3].plot(t_axis, fuel, "seagreen")
    axes_right[3].set_ylabel("Fuel remaining")
    axes_right[3].set_xlabel("Time (s)")
    axes_right[3].set_title("Fuel vs Time")

    for ax in axes_right:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Landing animation
# ---------------------------------------------------------------------------

def animate_landing(
    params: np.ndarray,
    cfg: RocketConfig = DEFAULT_CONFIG,
    save_path: str | Path = "rocket_landing.gif",
    fps: int = 20,
    skip: int = 1,
) -> None:
    """Animate the full landing sequence and save as GIF.

    Parameters
    ----------
    params:
        Controller parameter vector (5-D).
    cfg:
        Rocket configuration.
    save_path:
        Output path (should end in ``.gif``).
    fps:
        Frames per second in the output animation.
    skip:
        Render every *skip*-th simulation step (use > 1 to speed up the GIF).
    """
    traj = simulate_trajectory(params, cfg)
    states = traj["states"]
    thrust = traj["thrust"]
    fuel   = traj["fuel"]
    t_axis = traj["t"]
    landed = traj["landed"]
    fitness = traj["fitness"]

    x, y, vx, vy, theta, omega = states.T
    T = len(t_axis)

    # Sub-sample frames
    frame_idx = list(range(0, T, max(1, skip)))
    if frame_idx[-1] != T - 1:
        frame_idx.append(T - 1)

    # Bounds for axes
    x_min, x_max = min(x) - 5, max(x) + 5
    y_max = max(y) + 5

    # Rocket body dimensions
    r_w = 0.6   # half-width (m)
    r_h = 2.0   # height (m)

    fig, (ax_main, ax_metrics) = plt.subplots(
        1, 2, figsize=(12, 7),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0d1b2a")

    # ---- Main animation axes ----
    ax_main.set_facecolor("#0d1b2a")
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(-3, y_max)
    ax_main.set_xlabel("x (m)", color="white")
    ax_main.set_ylabel("y (m)", color="white")
    ax_main.tick_params(colors="white")
    for spine in ax_main.spines.values():
        spine.set_edgecolor("white")

    # Ground
    ax_main.fill_between([x_min, x_max], [-3, -3], [0, 0],
                         color="#3d2b1f", zorder=1)
    ax_main.axhline(0, color="#888", lw=1.5, zorder=2)

    # Landing pad
    pad_w = 3.0
    pad_patch = mpatches.FancyBboxPatch(
        (-pad_w, -0.4), 2 * pad_w, 0.4,
        boxstyle="round,pad=0.1", facecolor="#e0c060", edgecolor="white", lw=2, zorder=3,
    )
    ax_main.add_patch(pad_patch)
    ax_main.text(0, -1.5, "▼ PAD", color="#e0c060", ha="center", fontsize=11,
                 fontweight="bold", zorder=4)

    # Trajectory trace
    trace_line, = ax_main.plot([], [], color="cyan", alpha=0.5, lw=1.2, zorder=4)

    # Rocket body (polygon)
    rocket_poly = plt.Polygon(
        _rocket_polygon(0, 0, 0, r_w, r_h),
        closed=True, facecolor="silver", edgecolor="white", lw=1.5, zorder=6,
    )
    ax_main.add_patch(rocket_poly)

    # Thrust arrow
    thrust_arrow = FancyArrowPatch(
        (0, 0), (0, -1), color="orangered",
        arrowstyle="-|>", mutation_scale=15, lw=2, zorder=7,
    )
    ax_main.add_patch(thrust_arrow)

    # Status text
    status_text = ax_main.text(
        0.02, 0.97, "", transform=ax_main.transAxes,
        color="white", fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa"),
    )

    title_text = ax_main.set_title(
        "Rocket Landing — PSO Optimised Controller",
        color="white", fontsize=12, fontweight="bold",
    )

    # ---- Metrics axes ----
    ax_metrics.set_facecolor("#0d1b2a")
    ax_metrics.set_xlim(0, cfg.t_max)
    ax_metrics.set_ylim(0, cfg.y0 + 5)
    ax_metrics.set_xlabel("Time (s)", color="white")
    ax_metrics.set_ylabel("Altitude (m)", color="white")
    ax_metrics.set_title("Altitude trace", color="white", fontsize=10)
    ax_metrics.tick_params(colors="white")
    for spine in ax_metrics.spines.values():
        spine.set_edgecolor("white")
    ax_metrics.plot(t_axis, y, color="cyan", alpha=0.3, lw=1)
    metrics_dot, = ax_metrics.plot([], [], "o", color="cyan", ms=8, zorder=5)

    def _init():
        trace_line.set_data([], [])
        metrics_dot.set_data([], [])
        return trace_line, rocket_poly, thrust_arrow, status_text, metrics_dot

    def _update(frame: int):
        i = frame_idx[frame]
        xi, yi, vxi, vyi, thi, omi = states[i]
        thr = thrust[i]
        fu  = fuel[i]
        ti  = t_axis[i]

        # Update trajectory trace
        trace_line.set_data(x[:i+1], y[:i+1])

        # Update rocket body
        verts = _rocket_polygon(xi, yi, thi, r_w, r_h)
        rocket_poly.set_xy(verts)

        # Update thrust arrow
        thrust_len = max(0.5, thr / cfg.max_thrust * 4.0)
        dx = -np.sin(thi) * thrust_len
        dy = -np.cos(thi) * thrust_len
        thrust_arrow.set_positions((xi, yi), (xi + dx, yi + dy))
        thrust_arrow.set_visible(thr > 0.1)

        # Status
        speed = (vxi ** 2 + vyi ** 2) ** 0.5
        status_text.set_text(
            f"t = {ti:.2f}s\n"
            f"alt = {yi:.1f}m\n"
            f"speed = {speed:.2f}m/s\n"
            f"tilt = {np.degrees(thi):.1f}°\n"
            f"fuel = {fu:.1f}"
        )

        metrics_dot.set_data([ti], [yi])
        return trace_line, rocket_poly, thrust_arrow, status_text, metrics_dot

    anim = FuncAnimation(
        fig, _update, init_func=_init,
        frames=len(frame_idx), interval=1000 // fps, blit=True,
    )

    save_path = Path(save_path)
    writer = PillowWriter(fps=fps)
    anim.save(str(save_path), writer=writer)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rocket_polygon(
    cx: float, cy: float, theta: float,
    half_w: float, height: float,
) -> np.ndarray:
    """Return the 5 vertices of a rocket (rectangle + nose cone) centred at
    (cx, cy) and rotated by *theta* radians from vertical."""
    # Rocket body in local frame (bottom-centre at origin, pointing up)
    body = np.array([
        [-half_w,  0.0],
        [ half_w,  0.0],
        [ half_w,  height * 0.7],
        [ 0.0,     height],
        [-half_w,  height * 0.7],
    ])
    # Offset so geometric centre is at (cx, cy)
    body -= np.array([0.0, height / 2.0])
    # Rotate by theta (clockwise from vertical)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = body @ R.T
    return rotated + np.array([cx, cy])
