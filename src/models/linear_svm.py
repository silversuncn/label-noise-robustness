"""Small deterministic one-vs-rest linear hinge classifier."""

from __future__ import annotations

import numpy as np


class LinearSVMHingeLite:
    def __init__(self, epochs: int = 120, lr: float = 0.35, l2: float = 1e-4, random_state: int = 29) -> None:
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "LinearSVMHingeLite":
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0.0, 0.01, size=(x.shape[1], num_classes))
        self.bias = np.zeros(num_classes, dtype=np.float64)
        n = max(x.shape[0], 1)
        for _ in range(self.epochs):
            scores = x @ self.weights + self.bias
            grad_w = self.l2 * self.weights
            grad_b = np.zeros(num_classes, dtype=np.float64)
            for cls in range(num_classes):
                binary_y = np.where(y == cls, 1.0, -1.0)
                active = (binary_y * scores[:, cls]) < 1.0
                if np.any(active):
                    grad_w[:, cls] -= (x[active].T @ binary_y[active]) / n
                    grad_b[cls] -= float(binary_y[active].sum()) / n
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("model is not fitted")
        return np.argmax(x @ self.weights + self.bias, axis=1)
