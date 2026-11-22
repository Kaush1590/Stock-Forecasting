"""Last-observed-close (persistence) forecasting baseline.

For each forecast origin, predict that the next N trading closes equal the
most recently observed closing price. No fitting and no future information.
"""
from __future__ import annotations
import numpy as np


class NaiveLastClose:
    """Predict a constant close for every forecast horizon."""

    def __init__(self, horizon: int, target_idx: int):
        if horizon < 1 or target_idx < 0:
            raise ValueError("horizon must be >= 1 and target_idx must be >= 0")
        self.horizon = int(horizon)
        self.target_idx = int(target_idx)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Return scaled predictions shaped (samples, horizon).

        windows contains input features where Close shares the target scaler.
        """
        windows = np.asarray(windows)
        if windows.ndim != 3 or not len(windows):
            raise ValueError("Expected non-empty windows (samples, lookback, features)")
        if self.target_idx >= windows.shape[-1]:
            raise ValueError("target_idx outside feature dimension")
        last_close = windows[:, -1, self.target_idx]
        return np.repeat(last_close[:, None], self.horizon, axis=1)

    def predict_prices(self, last_close: float) -> np.ndarray:
        """Forecast in original units from one observed closing price."""
        value = float(last_close)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("last_close must be a finite positive price")
        return np.repeat(value, self.horizon)
