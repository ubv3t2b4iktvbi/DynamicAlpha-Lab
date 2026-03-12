import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ridge_solve(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=float)
    b = X.T @ y
    return np.linalg.solve(A, b)


def to_jsonable(obj: Any) -> str:
    if is_dataclass(obj):
        payload = asdict(obj)
    else:
        payload = obj
    return json.dumps(payload, ensure_ascii=False)


def safe_clip(x: np.ndarray | float, clip: float) -> np.ndarray | float:
    if clip <= 0:
        return x
    return np.clip(x, -clip, clip)
