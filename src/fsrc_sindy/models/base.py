from __future__ import annotations

import numpy as np


class BaseForecastModel:
    def fit(self, y_train: np.ndarray):
        raise NotImplementedError

    def rollout(self, y_hist: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def one_step_metrics(self, series: np.ndarray, burn_in: int):
        raise NotImplementedError

    def count_total_params(self) -> int:
        return 0

    def count_trained_params(self) -> int:
        return 0

    def effective_dim(self) -> int:
        return self.count_trained_params()
