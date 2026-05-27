"""Rocket landing use case: PSO optimizes the parameters of a PD controller
that guides a 2-D rocket to a soft landing on a pad at the origin.

Physical model
--------------
State vector (per rocket): [x, y, vx, vy, theta, omega]
  x, y      : position (m). Landing pad is at (0, 0). y = 0 is ground level.
  vx, vy    : velocity (m/s).
  theta     : tilt from vertical (rad). 0 = perfectly upright.
  omega     : angular velocity (rad/s).

Forces (body frame, Euler-explicit integration):
  * Gravity: constant downward force m*g.
  * Main thrust: along the rocket body axis (direction = theta from vertical).
  * Rotational torque: corrects tilt.
  * Fuel is consumed proportionally to thrust magnitude.

Controller parameters (5-D optimisation problem)
-------------------------------------------------
particle = [thrust_gain, vertical_damping, horizontal_correction,
            rotation_gain, braking_altitude]

  thrust_gain          ∈ [0.5, 3.0]  : scales nominal hover thrust (T_nom = g*m)
  vertical_damping     ∈ [-2.0, 2.0] : adjusts thrust by vert_damp * vy (braking phase)
  horizontal_correction∈ [0.0, 5.0]  : determines target tilt angle toward pad
  rotation_gain        ∈ [0.0, 10.0] : PD gain for angle stabilisation
  braking_altitude     ∈ [0.5, 10.0] : altitude (m) below which braking mode activates

Fitness function
----------------
  fitness = w_dist  * |x|
          + w_speed * sqrt(vx^2 + vy^2)
          + w_tilt  * |theta|
          + w_fuel  * fuel_used
          + w_instab* mean(|omega|)

Plus heavy penalties for: never landing, fuel depletion, out-of-bounds.

The scalar objective is a module-level function (picklable for V0-V2 and V5).
The vectorized objective processes *all* particles simultaneously with NumPy
matrix operations — ideal for the V4 evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pso.objectives.benchmarks import Benchmark


# ---------------------------------------------------------------------------
# Physics / simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class RocketConfig:
    """Physical and simulation constants for the rocket landing problem."""

    # Physics
    gravity: float = 9.81
    mass: float = 1.0
    inertia: float = 0.05
    max_thrust: float = 25.0
    max_torque: float = 5.0
    angular_damping: float = 0.5  # passive damping on omega in torque formula

    # Fuel
    initial_fuel: float = 100.0

    # Simulation
    dt: float = 0.05      # timestep (s)
    t_max: float = 12.0   # maximum simulation duration (s)

    # Initial state
    x0: float = 5.0       # lateral offset from landing pad (m)
    y0: float = 50.0      # starting altitude (m)
    vx0: float = -1.0     # initial lateral drift (m/s)
    vy0: float = 0.0      # initial vertical velocity (m/s)
    theta0: float = 0.1   # initial tilt (rad) — slight lean
    omega0: float = 0.0   # initial angular velocity (rad/s)

    # Fitness weights
    w_dist: float = 10.0
    w_speed: float = 5.0
    w_tilt: float = 3.0
    w_fuel: float = 0.1
    w_instab: float = 2.0

    # Penalties
    crash_penalty: float = 1000.0   # rocket never reached ground
    fuel_empty_penalty: float = 200.0
    bounds_penalty: float = 500.0   # left simulation area


DEFAULT_CONFIG = RocketConfig()

# Bounds for each of the 5 controller parameters
ROCKET_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.5,  3.0),    # thrust_gain
    (-2.0, 2.0),    # vertical_damping
    (0.0,  5.0),    # horizontal_correction
    (0.0,  10.0),   # rotation_gain
    (0.5,  10.0),   # braking_altitude
)

# Spatial limits: rockets that leave this box are penalised
_X_LIMIT = 200.0
_Y_LIMIT = 200.0


# ---------------------------------------------------------------------------
# Vectorized simulation  (n particles simultaneously, no Python loop over n)
# ---------------------------------------------------------------------------

def rocket_landing_vec(
    params_matrix: np.ndarray,
    cfg: RocketConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Simulate *n* rocket controllers simultaneously.

    Parameters
    ----------
    params_matrix:
        Shape ``(n, 5)``.  Each row is one set of controller parameters:
        ``[thrust_gain, vertical_damping, horizontal_correction,
           rotation_gain, braking_altitude]``.
    cfg:
        Rocket physics and fitness configuration.

    Returns
    -------
    fitness : np.ndarray, shape ``(n,)``
        Lower is better.  Minimum ≈ 0 for a perfect landing.
    """
    n = len(params_matrix)
    n_steps = int(cfg.t_max / cfg.dt)

    # Unpack controller parameters — shape (n,) each
    thrust_gain   = params_matrix[:, 0]
    vert_damp     = params_matrix[:, 1]
    horiz_corr    = params_matrix[:, 2]
    rot_gain      = params_matrix[:, 3]
    brake_alt     = params_matrix[:, 4]

    # State matrix: (n, 6) = [x, y, vx, vy, theta, omega]
    states = np.tile(
        [cfg.x0, cfg.y0, cfg.vx0, cfg.vy0, cfg.theta0, cfg.omega0],
        (n, 1),
    ).astype(float)

    fuel = np.full(n, cfg.initial_fuel)
    instability_acc = np.zeros(n)   # ∫ |omega| dt over flight
    out_of_bounds = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)  # False after touching ground

    # Record the state *at the moment of landing* for each particle so that
    # post-landing gravity drift does not corrupt the fitness calculation.
    final_states = states.copy()
    final_fuel   = fuel.copy()
    final_instab = instability_acc.copy()

    for _ in range(n_steps):
        if not np.any(active):
            break

        x, y, vx, vy, theta, omega = states.T

        # ---- Controller (all n particles at once) ----
        # Nominal hover thrust to counteract gravity
        nom_thrust = thrust_gain * cfg.gravity * cfg.mass

        # Braking phase: below braking_altitude, dampen vertical speed
        braking = active & (y < brake_alt)
        thrust_cmd = nom_thrust + braking * (vert_damp * vy)
        thrust_cmd = np.clip(thrust_cmd, 0.0, cfg.max_thrust)
        thrust_cmd *= active  # cut thrust once landed

        # Target tilt angle toward landing pad: lean proportional to x
        target_angle = np.clip(
            -np.arctan(horiz_corr * x), -np.pi / 4.0, np.pi / 4.0,
        )
        # PD controller on angle (proportional + derivative on omega)
        torque_cmd = rot_gain * (target_angle - theta) - cfg.angular_damping * omega
        torque_cmd = np.clip(torque_cmd, -cfg.max_torque, cfg.max_torque)
        torque_cmd *= active

        # ---- Euler-explicit physics update ----
        # Thrust force in world frame (body axis rotated by theta)
        Fx = -thrust_cmd * np.sin(theta)
        Fy =  thrust_cmd * np.cos(theta) - cfg.mass * cfg.gravity
        ax = Fx / cfg.mass
        ay = Fy / cfg.mass
        alpha = torque_cmd / cfg.inertia

        states[:, 2] += ax    * cfg.dt      # vx
        states[:, 3] += ay    * cfg.dt      # vy
        states[:, 5] += alpha * cfg.dt      # omega
        states[:, 0] += states[:, 2] * cfg.dt  # x
        states[:, 1] += states[:, 3] * cfg.dt  # y
        states[:, 4] += states[:, 5] * cfg.dt  # theta

        # Fuel consumption
        fuel -= thrust_cmd * cfg.dt

        # Instability: accumulate |omega| weighted by dt
        instability_acc += np.abs(states[:, 5]) * cfg.dt

        # Spatial bounds check
        out_of_bounds |= (
            (np.abs(states[:, 0]) > _X_LIMIT) |
            (states[:, 1] > _Y_LIMIT)
        )

        # Landing detection: y ≤ 0 while still active
        newly_landed = active & (states[:, 1] <= 0.0)
        # Snapshot state at the exact landing step (before further gravity drift)
        if np.any(newly_landed):
            final_states[newly_landed] = states[newly_landed]
            final_fuel[newly_landed]   = fuel[newly_landed]
            final_instab[newly_landed] = instability_acc[newly_landed]
        active &= ~newly_landed

    # For never-landed particles, use end-of-sim state
    final_states[active] = states[active]
    final_fuel[active]   = fuel[active]
    final_instab[active] = instability_acc[active]

    # ---- Compute fitness ----
    x, y, vx, vy, theta, omega = final_states.T
    fitness = np.zeros(n)

    # Rockets that never reached the ground: heavy penalty + remaining altitude
    never_landed = active  # still "active" at end of sim
    fitness += never_landed * (cfg.crash_penalty + np.abs(y))

    # Rockets that landed: evaluate landing quality at the landing timestep
    landed = ~active
    if np.any(landed):
        dist   = np.abs(x)
        speed  = np.sqrt(vx ** 2 + vy ** 2)
        tilt   = np.abs(theta)
        f_used = cfg.initial_fuel - np.clip(final_fuel, 0.0, None)
        instab = final_instab / cfg.t_max   # normalise by sim duration

        fitness += landed * (
            cfg.w_dist   * dist   +
            cfg.w_speed  * speed  +
            cfg.w_tilt   * tilt   +
            cfg.w_fuel   * f_used +
            cfg.w_instab * instab
        )

    # Additional penalties
    fitness += (final_fuel < 0.0) * cfg.fuel_empty_penalty
    fitness += out_of_bounds * cfg.bounds_penalty

    return fitness


# ---------------------------------------------------------------------------
# Scalar objective  (module-level → picklable for V0-V2 and V5)
# ---------------------------------------------------------------------------

def rocket_landing_objective(params: np.ndarray) -> float:
    """Evaluate a single set of controller parameters.

    Wraps ``rocket_landing_vec`` with ``n = 1`` so there is a single
    implementation for both scalar and vectorized paths.

    Parameters
    ----------
    params : np.ndarray, shape ``(5,)``

    Returns
    -------
    float
        Fitness value (lower = better landing).
    """
    return float(rocket_landing_vec(params[np.newaxis, :], DEFAULT_CONFIG)[0])


# ---------------------------------------------------------------------------
# Simulation step-by-step recorder  (used by visualiser)
# ---------------------------------------------------------------------------

def simulate_trajectory(
    params: np.ndarray,
    cfg: RocketConfig = DEFAULT_CONFIG,
) -> dict:
    """Run a single simulation and record the full state trajectory.

    Returns a dict with keys:
      ``states``   — shape ``(T, 6)`` array [x, y, vx, vy, theta, omega]
      ``thrust``   — shape ``(T,)`` thrust commands applied each step
      ``fuel``     — shape ``(T,)`` remaining fuel at each step
      ``fitness``  — final scalar fitness
      ``landed``   — bool: did the rocket reach the ground?
      ``t``        — shape ``(T,)`` time axis
    """
    n_steps = int(cfg.t_max / cfg.dt)
    state = np.array(
        [cfg.x0, cfg.y0, cfg.vx0, cfg.vy0, cfg.theta0, cfg.omega0],
        dtype=float,
    )

    thrust_gain, vert_damp, horiz_corr, rot_gain, brake_alt = params

    fuel = cfg.initial_fuel
    instab_acc = 0.0
    landed = False
    out_of_bounds = False

    history_states: list[np.ndarray] = []
    history_thrust: list[float] = []
    history_fuel:   list[float] = []
    history_t:      list[float] = []

    for step in range(n_steps):
        history_states.append(state.copy())
        history_fuel.append(fuel)
        history_t.append(step * cfg.dt)

        x, y, vx, vy, theta, omega = state

        nom_thrust = thrust_gain * cfg.gravity * cfg.mass
        braking = y < brake_alt
        thrust_cmd = nom_thrust + (braking * vert_damp * vy)
        thrust_cmd = float(np.clip(thrust_cmd, 0.0, cfg.max_thrust))

        target_angle = float(np.clip(-np.arctan(horiz_corr * x), -np.pi / 4, np.pi / 4))
        torque_cmd = float(np.clip(
            rot_gain * (target_angle - theta) - cfg.angular_damping * omega,
            -cfg.max_torque, cfg.max_torque,
        ))

        history_thrust.append(thrust_cmd)

        Fx = -thrust_cmd * np.sin(theta)
        Fy =  thrust_cmd * np.cos(theta) - cfg.mass * cfg.gravity
        ax = Fx / cfg.mass
        ay = Fy / cfg.mass
        alpha = torque_cmd / cfg.inertia

        state[2] += ax    * cfg.dt
        state[3] += ay    * cfg.dt
        state[5] += alpha * cfg.dt
        state[0] += state[2] * cfg.dt
        state[1] += state[3] * cfg.dt
        state[4] += state[5] * cfg.dt

        fuel -= thrust_cmd * cfg.dt
        instab_acc += abs(state[5]) * cfg.dt

        if abs(state[0]) > _X_LIMIT or state[1] > _Y_LIMIT:
            out_of_bounds = True

        if state[1] <= 0.0:
            landed = True
            break

    # Append final state
    history_states.append(state.copy())
    history_fuel.append(fuel)
    history_thrust.append(0.0)
    history_t.append(len(history_t) * cfg.dt)

    states_arr = np.array(history_states)
    x, y, vx, vy, theta, omega = state

    # Compute fitness
    if not landed:
        fitness = cfg.crash_penalty + abs(y)
    else:
        dist   = abs(x)
        speed  = (vx ** 2 + vy ** 2) ** 0.5
        tilt   = abs(theta)
        f_used = cfg.initial_fuel - max(fuel, 0.0)
        instab = instab_acc / cfg.t_max
        fitness = (
            cfg.w_dist   * dist   +
            cfg.w_speed  * speed  +
            cfg.w_tilt   * tilt   +
            cfg.w_fuel   * f_used +
            cfg.w_instab * instab
        )

    if fuel < 0.0:
        fitness += cfg.fuel_empty_penalty
    if out_of_bounds:
        fitness += cfg.bounds_penalty

    return {
        "states":   states_arr,
        "thrust":   np.array(history_thrust),
        "fuel":     np.array(history_fuel),
        "fitness":  float(fitness),
        "landed":   landed,
        "t":        np.array(history_t),
    }


# ---------------------------------------------------------------------------
# Benchmark registration entry
# ---------------------------------------------------------------------------

ROCKET_BENCHMARK = Benchmark(
    name="rocket",
    func=rocket_landing_objective,
    lb=0.0,
    ub=10.0,                   # fallback only; bounds_override takes precedence
    known_minimum=0.0,
    vec_func=rocket_landing_vec,
    bounds_override=ROCKET_BOUNDS,
)
