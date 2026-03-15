from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov

from .systems import BenchmarkTask, SYSTEMS, simulate_task


NON_AUTONOMOUS_SYSTEMS = {"duffing"}


@dataclass(frozen=True)
class WSGAConfig:
    noise_strength: float = 0.01
    dt: float = 0.01
    steps: int = 2000
    rand_num: int = 128
    padding_fraction: float = 0.15
    convergence_fraction: float = 0.2
    convergence_tol_scale: float = 2.0
    cluster_radius_scale: float = 10.0
    jacobian_eps: float = 1e-4
    covariance_jitter: float = 1e-6

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AttractorPrior:
    task_name: str
    system: str
    noise_strength: float
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    counts: np.ndarray
    stability_real_parts: np.ndarray
    x_min: float
    x_max: float


def _nan_metrics() -> dict[str, float]:
    return {
        "wsga_attractor_count": float("nan"),
        "wsga_active_basin_count": float("nan"),
        "wsga_basin_separation": float("nan"),
        "wsga_basin_sep_gap": float("nan"),
        "wsga_centroid_dist_corr": float("nan"),
        "wsga_entropy_gap": float("nan"),
        "wsga_epr_loss": float("nan"),
        "wsga_epr_score": float("nan"),
    }


def _regularize_covariance(cov: np.ndarray, jitter: float) -> np.ndarray:
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, jitter)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _infer_bounds(task: BenchmarkTask, reference_states: np.ndarray | None, seed: int) -> tuple[float, float]:
    if reference_states is None:
        reference_states = simulate_task(task, seed=seed).states
    lo = float(np.min(reference_states))
    hi = float(np.max(reference_states))
    span = max(hi - lo, 1e-3)
    padding = 0.15 * span
    return lo - padding, hi + padding


def _numerical_jacobian(rhs, x: np.ndarray, dt: float, params: dict, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    dim = x.size
    jac = np.zeros((dim, dim), dtype=float)
    for j in range(dim):
        direction = np.zeros(dim, dtype=float)
        direction[j] = eps
        forward = np.asarray(rhs(x + direction, dt, params), dtype=float).reshape(-1)
        backward = np.asarray(rhs(x - direction, dt, params), dtype=float).reshape(-1)
        jac[:, j] = (forward - backward) / (2.0 * eps)
    return jac


def _log_gaussian_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    diff = np.asarray(points, dtype=float) - np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite")
    quad = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    dim = diff.shape[1]
    return -0.5 * (quad + logdet + dim * np.log(2.0 * np.pi))


def _mixture_log_responsibilities(points: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    log_terms = []
    for weight, mean, cov in zip(weights, means, covs):
        log_terms.append(np.log(max(float(weight), 1e-12)) + _log_gaussian_density(points, mean, cov))
    stacked = np.column_stack(log_terms)
    max_log = np.max(stacked, axis=1, keepdims=True)
    stabilized = stacked - max_log
    log_norm = max_log[:, 0] + np.log(np.sum(np.exp(stabilized), axis=1) + 1e-12)
    return stacked - log_norm[:, None]


def _mixture_score(points: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    log_resp = _mixture_log_responsibilities(points, weights, means, covs)
    resp = np.exp(log_resp)
    score = np.zeros_like(points, dtype=float)
    for k, (mean, cov) in enumerate(zip(means, covs)):
        diff = points - mean[None, :]
        inv_cov = np.linalg.inv(cov)
        score += resp[:, [k]] * (-(diff @ inv_cov.T))
    return score


def build_task_attractor_prior(
    task: BenchmarkTask,
    *,
    seed: int,
    config: WSGAConfig,
    reference_states: np.ndarray | None = None,
) -> AttractorPrior:
    if task.system in NON_AUTONOMOUS_SYSTEMS:
        raise ValueError(f"WSGA prior is disabled for non-autonomous system={task.system}.")
    rhs = SYSTEMS[task.system]["rhs"]
    x_min, x_max = _infer_bounds(task, reference_states=reference_states, seed=seed)
    rng = np.random.default_rng(seed)
    dim = int(np.asarray(reference_states[0] if reference_states is not None else SYSTEMS[task.system]["default_x0"], dtype=float).size)
    time = np.arange(config.steps + 1, dtype=float) * float(config.dt)
    padding = float(config.padding_fraction) * float(x_max - x_min)
    drift_threshold = float(config.convergence_tol_scale) * float(config.dt)
    cluster_radius = float(config.cluster_radius_scale) * float(config.dt)
    tail_start = max(0, int((1.0 - float(config.convergence_fraction)) * config.steps))

    centers: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []

    def ode_fun(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(rhs(np.asarray(y, dtype=float), t, task.params), dtype=float).reshape(-1)

    for _ in range(int(config.rand_num)):
        x0 = rng.uniform(x_min - padding, x_max + padding, size=dim)
        sol = solve_ivp(
            ode_fun,
            (0.0, float(config.steps) * float(config.dt)),
            x0,
            t_eval=time,
            method="RK45",
        )
        if sol.y.size == 0:
            continue
        path = sol.y.T
        tail = path[tail_start:]
        if tail.size == 0:
            continue
        terminal = path[-1]
        drift = float(np.linalg.norm(terminal - np.mean(tail, axis=0)))
        if drift >= drift_threshold:
            continue
        if not centers:
            centers.append(terminal.copy())
            sums.append(terminal.copy())
            counts.append(1)
            continue
        center_arr = np.stack(centers)
        dists = np.linalg.norm(center_arr - terminal[None, :], axis=1)
        index = int(np.argmin(dists))
        if float(dists[index]) > cluster_radius:
            centers.append(terminal.copy())
            sums.append(terminal.copy())
            counts.append(1)
        else:
            counts[index] += 1
            sums[index] = sums[index] + terminal
            centers[index] = sums[index] / float(counts[index])

    if not centers:
        raise ValueError(
            "WSGA did not find any stable fixed-point attractor. "
            "This task is likely oscillatory, chaotic, or incompatible with a fixed-point prior."
        )

    means = np.stack(centers).astype(float)
    hit_counts = np.asarray(counts, dtype=int)
    weights = hit_counts.astype(float) / float(config.rand_num)
    covariances = np.zeros((len(means), dim, dim), dtype=float)
    stability = np.zeros((len(means), dim), dtype=float)
    diffusion = 2.0 * float(config.noise_strength) * np.eye(dim, dtype=float)
    for i, mean in enumerate(means):
        jac = _numerical_jacobian(rhs, mean, dt=0.0, params=task.params, eps=float(config.jacobian_eps))
        eigvals = np.linalg.eigvals(jac)
        stability[i, : len(eigvals)] = np.sort(np.real(eigvals))
        cov = solve_continuous_lyapunov(jac, -diffusion)
        covariances[i] = _regularize_covariance(cov, jitter=float(config.covariance_jitter))
    stable_mask = np.max(stability, axis=1) < 0.0
    if not np.any(stable_mask):
        raise ValueError("WSGA only found non-stationary or unstable fixed points for this task.")
    means = means[stable_mask]
    hit_counts = hit_counts[stable_mask]
    covariances = covariances[stable_mask]
    stability = stability[stable_mask]
    weights = hit_counts.astype(float)
    weights = weights / np.sum(weights)
    return AttractorPrior(
        task_name=task.name,
        system=task.system,
        noise_strength=float(config.noise_strength),
        weights=weights,
        means=means,
        covariances=covariances,
        counts=hit_counts,
        stability_real_parts=stability,
        x_min=float(x_min),
        x_max=float(x_max),
    )


def assign_attractor_labels(states: np.ndarray, prior: AttractorPrior) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    log_resp = _mixture_log_responsibilities(states, prior.weights, prior.means, prior.covariances)
    return np.argmax(log_resp, axis=1).astype(int)


def _fit_label_gaussians(values: np.ndarray, labels: np.ndarray, num_labels: int, jitter: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    active = []
    means = []
    covs = []
    weights = []
    for label in range(num_labels):
        mask = labels == label
        count = int(np.sum(mask))
        if count < max(6, values.shape[1] + 2):
            continue
        subset = values[mask]
        mean = np.mean(subset, axis=0)
        cov = np.cov(subset, rowvar=False)
        if values.shape[1] == 1:
            cov = np.asarray([[float(cov)]], dtype=float)
        cov = _regularize_covariance(np.asarray(cov, dtype=float), jitter=jitter)
        active.append(label)
        means.append(mean)
        covs.append(cov)
        weights.append(float(count) / float(len(values)))
    if not active:
        raise ValueError("No attractor basin has enough samples in this coordinate.")
    return (
        np.asarray(active, dtype=int),
        np.asarray(weights, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(covs, dtype=float),
    )


def _pairwise_upper(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.asarray([], dtype=float)
    mat = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    tri = np.triu_indices(len(values), k=1)
    return np.asarray(mat[tri], dtype=float)


def _normalized_entropy(cov: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    eigvals = np.maximum(eigvals, 1e-12)
    probs = eigvals / np.sum(eigvals)
    entropy = -float(np.sum(probs * np.log(probs)))
    denom = np.log(max(len(eigvals), 2))
    return entropy / denom


def evaluate_coordinate_attractor_prior(
    z: np.ndarray,
    labels: np.ndarray,
    prior: AttractorPrior | None,
    *,
    dt: float,
    covariance_jitter: float = 1e-6,
) -> dict[str, float]:
    if prior is None:
        return _nan_metrics()

    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(z) != len(labels) or len(z) < 8:
        return _nan_metrics()

    try:
        active_labels, weights_z, means_z, covs_z = _fit_label_gaussians(
            z,
            labels,
            num_labels=len(prior.weights),
            jitter=float(covariance_jitter),
        )
    except ValueError:
        return _nan_metrics()

    within_scales = [float(np.sqrt(np.trace(cov) / cov.shape[0])) for cov in covs_z]
    between = _pairwise_upper(means_z)
    basin_separation = float(np.mean(between) / (np.mean(within_scales) + 1e-12)) if len(between) else float("nan")
    state_within_scales = [float(np.sqrt(np.trace(prior.covariances[label]) / prior.covariances[label].shape[0])) for label in active_labels]
    state_between = _pairwise_upper(prior.means[active_labels])
    state_separation = float(np.mean(state_between) / (np.mean(state_within_scales) + 1e-12)) if len(state_between) else float("nan")
    basin_sep_gap = float(abs(basin_separation - state_separation)) if np.isfinite(basin_separation) and np.isfinite(state_separation) else float("nan")

    state_pairwise = _pairwise_upper(prior.means[active_labels])
    coord_pairwise = _pairwise_upper(means_z)
    if len(state_pairwise) > 1 and len(coord_pairwise) == len(state_pairwise):
        centroid_corr = float(np.corrcoef(state_pairwise, coord_pairwise)[0, 1])
    else:
        centroid_corr = float("nan")

    entropy_gap = float(
        np.mean(
            [
                abs(_normalized_entropy(prior.covariances[label]) - _normalized_entropy(cov))
                for label, cov in zip(active_labels, covs_z)
            ]
        )
    )

    score = _mixture_score(z[:-1], weights_z, means_z, covs_z)
    grad_u = -float(prior.noise_strength) * score
    drift = np.diff(z, axis=0) / max(float(dt), 1e-12)
    epr_terms = np.sum((drift + grad_u) ** 2, axis=1)
    epr_loss = float(np.mean(epr_terms))
    drift_scale = float(np.mean(np.sum(drift**2, axis=1)) + 1e-12)
    epr_score = float(1.0 / (1.0 + epr_loss / drift_scale))

    return {
        "wsga_attractor_count": float(len(prior.weights)),
        "wsga_active_basin_count": float(len(active_labels)),
        "wsga_basin_separation": basin_separation,
        "wsga_basin_sep_gap": basin_sep_gap,
        "wsga_centroid_dist_corr": centroid_corr,
        "wsga_entropy_gap": entropy_gap,
        "wsga_epr_loss": epr_loss,
        "wsga_epr_score": epr_score,
    }
