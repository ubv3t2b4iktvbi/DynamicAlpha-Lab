from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


def rk4_step(func: Callable[[np.ndarray, float, dict], np.ndarray], x: np.ndarray, t: float, dt: float, params: dict) -> np.ndarray:
    k1 = func(x, t, params)
    k2 = func(x + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = func(x + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = func(x + dt * k3, t + dt, params)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz63_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    sigma = p.get("sigma", 10.0)
    rho = p.get("rho", 28.0)
    beta = p.get("beta", 8.0 / 3.0)
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz], dtype=float)


def rossler_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    a = p.get("a", 0.2)
    b = p.get("b", 0.2)
    c = p.get("c", 5.7)
    dx = -x[1] - x[2]
    dy = x[0] + a * x[1]
    dz = b + x[2] * (x[0] - c)
    return np.array([dx, dy, dz], dtype=float)


def duffing_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    delta = p.get("delta", 0.2)
    gamma = p.get("gamma", 0.3)
    omega = p.get("omega", 1.2)
    dx = x[1]
    dv = x[0] - x[0] ** 3 - delta * x[1] + gamma * np.cos(omega * t)
    return np.array([dx, dv], dtype=float)


def vanderpol_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    mu = p.get("mu", 5.0)
    dx = x[1]
    dv = mu * (1.0 - x[0] ** 2) * x[1] - x[0]
    return np.array([dx, dv], dtype=float)


def bistable_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    a = p.get("a", 1.0)
    k = p.get("k", 1.0)
    S = p.get("S", 0.5)
    n = int(p.get("n", 4))
    state = np.asarray(x, dtype=float)
    state_n = np.power(state, n)
    s_n = S**n
    activate = a * state_n / (s_n + state_n)
    restrict = a * s_n / (s_n + state_n)
    return -k * state + activate + restrict[::-1]


def fitzhugh_nagumo_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    a = p.get("a", 0.7)
    b = p.get("b", 0.8)
    eps = p.get("eps", 0.08)
    I = p.get("I", 0.5)
    slow_to_fast_h = p.get("slow_to_fast_h", 1.0)
    fast_to_slow_h = p.get("fast_to_slow_h", 1.0)
    v, w = x
    dv = v - v ** 3 / 3.0 - slow_to_fast_h * w + I
    dw = eps * (fast_to_slow_h * v + a - b * w)
    return np.array([dv, dw], dtype=float)


def hindmarsh_rose_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    a = p.get("a", 1.0)
    b = p.get("b", 3.0)
    c = p.get("c", 1.0)
    d = p.get("d", 5.0)
    r = p.get("r", 0.006)
    s = p.get("s", 4.0)
    x_r = p.get("x_r", -1.6)
    I = p.get("I", 3.25)
    slow_to_fast_h = p.get("slow_to_fast_h", 1.0)
    fast_to_slow_h = p.get("fast_to_slow_h", 1.0)
    v, y, z = x
    dv = y - a * v**3 + b * v**2 - slow_to_fast_h * z + I
    dy = c - d * v**2 - y
    dz = r * (fast_to_slow_h * s * (v - x_r) - z)
    return np.array([dv, dy, dz], dtype=float)


def lorenz96_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    F = p.get("F", 8.0)
    K = len(x)
    dx = np.zeros(K, dtype=float)
    for k in range(K):
        dx[k] = (x[(k + 1) % K] - x[k - 2]) * x[k - 1] - x[k] + F
    return dx


def lorenz96_twoscale_rhs(x: np.ndarray, t: float, p: dict) -> np.ndarray:
    K = p.get("K", 8)
    J = p.get("J", 4)
    F = p.get("F", 10.0)
    h = p.get("h", 1.0)
    fast_to_slow_h = p.get("fast_to_slow_h", h)
    slow_to_fast_h = p.get("slow_to_fast_h", h)
    c = p.get("c", 10.0)
    b = p.get("b", 10.0)
    X = x[:K]
    Y = x[K:].reshape(K, J)
    dX = np.zeros_like(X)
    dY = np.zeros_like(Y)
    for k in range(K):
        coupling = (fast_to_slow_h * c / b) * np.sum(Y[k])
        dX[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % K]) - X[k] + F - coupling
    for k in range(K):
        for j in range(J):
            jp1 = (j + 1) % J
            jp2 = (j + 2) % J
            jm1 = (j - 1) % J
            dY[k, j] = -c * b * Y[k, jp1] * (Y[k, jp2] - Y[k, jm1]) - c * Y[k, j] + (slow_to_fast_h * c / b) * X[k]
    return np.concatenate([dX, dY.reshape(-1)])


def lorenz96_default_x0(params: dict) -> np.ndarray:
    K = int(params.get("K", 8))
    return np.array([8.0 + 0.01 * i for i in range(K)], dtype=float)


def lorenz96_twoscale_default_x0(params: dict) -> np.ndarray:
    K = int(params.get("K", 8))
    J = int(params.get("J", 4))
    slow = np.array([8.0 + 0.01 * i for i in range(K)], dtype=float)
    fast = 0.01 * np.ones(K * J, dtype=float)
    return np.concatenate([slow, fast])


SYSTEMS: Dict[str, Dict[str, object]] = {
    "lorenz63": {"rhs": lorenz63_rhs, "default_x0": np.array([1.0, 1.0, 1.0], dtype=float)},
    "rossler": {"rhs": rossler_rhs, "default_x0": np.array([1.0, 0.0, 0.0], dtype=float)},
    "duffing": {"rhs": duffing_rhs, "default_x0": np.array([0.1, 0.0], dtype=float)},
    "vanderpol": {"rhs": vanderpol_rhs, "default_x0": np.array([2.0, 0.0], dtype=float)},
    "bistable": {"rhs": bistable_rhs, "default_x0": np.array([0.25, 1.25], dtype=float), "bounds": (0.0, 3.0)},
    "fitzhugh_nagumo": {"rhs": fitzhugh_nagumo_rhs, "default_x0": np.array([-1.0, 1.0], dtype=float)},
    "hindmarsh_rose": {"rhs": hindmarsh_rose_rhs, "default_x0": np.array([0.0, 0.0, 0.0], dtype=float)},
    "lorenz96": {"rhs": lorenz96_rhs, "default_x0": lorenz96_default_x0},
    "lorenz96_twoscale": {"rhs": lorenz96_twoscale_rhs, "default_x0": lorenz96_twoscale_default_x0},
}


@dataclass
class SimulationResult:
    states: np.ndarray
    obs: np.ndarray


@dataclass
class BenchmarkTask:
    name: str
    system: str
    dt: float
    n_train: int
    n_val: int
    n_test: int
    burn_in: int = 500
    process_noise_std: float = 0.0
    obs_noise_std: float = 0.0
    process_noise_volatility: float = 0.0
    obs_noise_volatility: float = 0.0
    noise_ema_span: float = 32.0
    noise_volatility_clip: float = 4.0
    match_obs_noise_energy: bool = False
    params: dict = field(default_factory=dict)
    x0: Optional[np.ndarray] = None
    obs_mode: str = "x0"
    obs_params: dict = field(default_factory=dict)
    eval_horizons: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    selection_horizons: tuple[int, ...] = (10, 50)
    stat_horizon: int = 256
    family: str = ""
    regime: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return self.n_train + self.n_val + self.n_test


def observe(states: np.ndarray, system: str, obs_mode: str, params: Optional[dict] = None, obs_params: Optional[dict] = None) -> np.ndarray:
    params = params or {}
    obs_params = obs_params or {}
    if obs_mode == "linear_proj":
        indices = np.asarray(obs_params.get("indices", [0]), dtype=int)
        weights = np.asarray(obs_params.get("weights", np.ones(len(indices))), dtype=float)
        if len(indices) != len(weights):
            raise ValueError("obs_params['indices'] and obs_params['weights'] must have same length")
        if obs_params.get("normalize", True):
            weights = weights / (np.linalg.norm(weights) + 1e-12)
        return states[:, indices] @ weights
    if system == "lorenz96_twoscale":
        K = int(params.get("K", 8))
        if obs_mode == "slow0":
            return states[:, 0]
        if obs_mode == "slow_mean":
            return np.mean(states[:, :K], axis=1)
    if obs_mode == "x0":
        return states[:, 0]
    if obs_mode == "x1":
        return states[:, 1]
    if obs_mode == "mean":
        return np.mean(states, axis=1)
    raise ValueError(f"Unsupported obs_mode={obs_mode} for system={system}")


def _metadata_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def task_metadata_columns(task: BenchmarkTask) -> dict[str, object]:
    payload = dict(task.metadata or {})
    row = {"task_metadata": json.dumps(payload, ensure_ascii=False, sort_keys=True)}
    for key, value in payload.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        row[key_str] = _metadata_value(value)
    return row


def _update_noise_scale(
    *,
    activity: float,
    ema: float | None,
    baseline: float | None,
    noise_alpha: float,
    baseline_alpha: float,
    volatility: float,
    clip: float,
) -> tuple[float, float, float]:
    activity = float(abs(activity))
    if ema is None or baseline is None:
        return 1.0, activity, max(activity, 1e-12)
    ema = (1.0 - noise_alpha) * ema + noise_alpha * activity
    baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * activity
    activity_ratio = ema / (baseline + 1e-12)
    scale = 1.0 + float(volatility) * min(max(activity_ratio - 1.0, 0.0), float(clip))
    return scale, ema, baseline


def _match_noise_rms(noise: np.ndarray, target_std: float) -> np.ndarray:
    target_std = float(target_std)
    if target_std <= 0.0:
        return np.asarray(noise, dtype=float)
    noise = np.asarray(noise, dtype=float)
    rms = float(np.sqrt(np.mean(np.square(noise))) + 1e-12)
    return noise * (target_std / rms)


def simulate_task(task: BenchmarkTask, seed: int) -> SimulationResult:
    rng = np.random.default_rng(seed)
    meta = SYSTEMS[task.system]
    rhs = meta["rhs"]
    default_x0 = meta["default_x0"]
    if task.x0 is not None:
        x0 = np.asarray(task.x0, dtype=float).copy()
    elif callable(default_x0):
        x0 = np.asarray(default_x0(task.params), dtype=float).copy()
    else:
        x0 = np.asarray(default_x0, dtype=float).copy()
    total_steps = task.total_steps + task.burn_in
    states = np.zeros((total_steps, len(x0)), dtype=float)
    x = x0.copy()
    t = 0.0
    noise_alpha = 2.0 / (max(float(task.noise_ema_span), 1.0) + 1.0)
    baseline_alpha = max(0.05 * noise_alpha, min(0.2 * noise_alpha, 0.05))
    process_ema = None
    process_baseline = None
    for i in range(total_steps):
        x_next = rk4_step(rhs, x, t, task.dt, task.params)
        if task.process_noise_std > 0.0:
            activity = float(np.linalg.norm(x_next - x) / np.sqrt(max(len(x), 1)))
            process_scale, process_ema, process_baseline = _update_noise_scale(
                activity=activity,
                ema=process_ema,
                baseline=process_baseline,
                noise_alpha=noise_alpha,
                baseline_alpha=baseline_alpha,
                volatility=task.process_noise_volatility,
                clip=task.noise_volatility_clip,
            )
            x_next = x_next + np.sqrt(task.dt) * task.process_noise_std * process_scale * rng.normal(size=x.shape)
        x = x_next
        states[i] = x_next
        t += task.dt
    states = states[task.burn_in:]
    obs_clean = observe(states, task.system, task.obs_mode, params=task.params, obs_params=task.obs_params).astype(float)
    obs = obs_clean.copy()
    if task.obs_noise_std > 0.0:
        obs_ema = None
        obs_baseline = None
        noise = np.zeros_like(obs)
        for i in range(len(obs)):
            activity = 0.0 if i == 0 else float(abs(obs_clean[i] - obs_clean[i - 1]))
            obs_scale, obs_ema, obs_baseline = _update_noise_scale(
                activity=activity,
                ema=obs_ema,
                baseline=obs_baseline,
                noise_alpha=noise_alpha,
                baseline_alpha=baseline_alpha,
                volatility=task.obs_noise_volatility,
                clip=task.noise_volatility_clip,
            )
            noise[i] = task.obs_noise_std * obs_scale * rng.normal()
        if task.match_obs_noise_energy:
            noise = _match_noise_rms(noise, target_std=task.obs_noise_std)
        obs = obs + noise
    return SimulationResult(states=states, obs=obs)


def split_series(y: np.ndarray, n_train: int, n_val: int, n_test: int) -> dict[str, np.ndarray]:
    assert len(y) >= n_train + n_val + n_test
    return {
        "train": y[:n_train].copy(),
        "val": y[n_train:n_train + n_val].copy(),
        "test": y[n_train + n_val:n_train + n_val + n_test].copy(),
    }
