"""Small deterministic multinomial logistic regression."""

from __future__ import annotations

import numpy as np


class SoftmaxLogRegLite:
    def __init__(self, epochs: int = 220, lr: float = 0.8, l2: float = 1e-4, random_state: int = 17) -> None:
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "SoftmaxLogRegLite":
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0.0, 0.01, size=(x.shape[1], num_classes))
        self.bias = np.zeros(num_classes, dtype=np.float64)
        y_one_hot = np.eye(num_classes, dtype=np.float64)[y]
        n = max(x.shape[0], 1)
        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            diff = probs - y_one_hot
            grad_w = (x.T @ diff) / n + self.l2 * self.weights
            grad_b = diff.mean(axis=0)
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("model is not fitted")
        logits = x @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)
