"""Small multinomial naive Bayes classifier."""

from __future__ import annotations

import numpy as np


class MultinomialNBLite:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.log_prior: np.ndarray | None = None
        self.log_likelihood: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "MultinomialNBLite":
        class_counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        self.log_prior = np.log((class_counts + self.alpha) / (class_counts.sum() + self.alpha * num_classes))
        feature_counts = np.full((num_classes, x.shape[1]), self.alpha, dtype=np.float64)
        for cls in range(num_classes):
            if np.any(y == cls):
                feature_counts[cls] += x[y == cls].sum(axis=0)
        self.log_likelihood = np.log(feature_counts / feature_counts.sum(axis=1, keepdims=True))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.log_prior is None or self.log_likelihood is None:
            raise RuntimeError("model is not fitted")
        return np.argmax(x @ self.log_likelihood.T + self.log_prior, axis=1)
