from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .utils import ridge_solve


def build_poly_library(X: np.ndarray, feature_names: Sequence[str], poly_order: int = 2) -> Tuple[np.ndarray, List[str]]:
    X = np.asarray(X, dtype=float)
    X = np.clip(X, -50.0, 50.0)
    names = list(feature_names)
    n, d = X.shape
    cols = [np.ones((n, 1), dtype=float), X]
    lib_names = ["1"] + names.copy()
    if poly_order >= 2:
        for i in range(d):
            cols.append(X[:, i:i + 1] ** 2)
            lib_names.append(f"{names[i]}^2")
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i:i + 1] * X[:, j:j + 1])
                lib_names.append(f"{names[i]}*{names[j]}")
    Theta = np.hstack(cols)
    Theta = np.nan_to_num(Theta, nan=0.0, posinf=1e6, neginf=-1e6)
    return Theta, lib_names


def fit_stlsq(Theta: np.ndarray, target: np.ndarray, ridge: float, threshold: float, n_iter: int = 10) -> np.ndarray:
    coef = ridge_solve(Theta, target, ridge)
    for _ in range(n_iter):
        small = np.abs(coef) < threshold
        coef[small] = 0.0
        keep = np.where(~small)[0]
        if len(keep) == 0:
            break
        coef_sub = ridge_solve(Theta[:, keep], target, ridge)
        coef[:] = 0.0
        coef[keep] = coef_sub
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    return coef
