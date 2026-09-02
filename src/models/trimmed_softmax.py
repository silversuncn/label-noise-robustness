"""Trimmed two-stage softmax logistic regression."""

from __future__ import annotations

import numpy as np

from .softmax_logreg import SoftmaxLogRegLite


class TrimmedSoftmaxLogRegLite:
    def __init__(self, trim_fraction: float = 0.2) -> None:
        self.trim_fraction = trim_fraction
        self.initial: SoftmaxLogRegLite | None = None
        self.final: SoftmaxLogRegLite | None = None
        self.kept_fraction: float | None = None

    @staticmethod
    def _trimmed_indices(losses: np.ndarray, y: np.ndarray, num_classes: int, trim_fraction: float) -> np.ndarray:
        n = len(losses)
        keep_count = max(int(round(n * (1.0 - trim_fraction))), min(n, num_classes * 2))
        keep: set[int] = set(np.argsort(losses)[:keep_count].tolist())
        for cls in range(num_classes):
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) and not any(int(idx) in keep for idx in cls_indices):
                best_cls_idx = int(cls_indices[np.argmin(losses[cls_indices])])
                keep.add(best_cls_idx)
        return np.asarray(sorted(keep), dtype=np.int64)

    def fit(self, x: np.ndarray, y: np.ndarray, num_classes: int) -> "TrimmedSoftmaxLogRegLite":
        self.initial = SoftmaxLogRegLite(epochs=90, lr=0.7, l2=2e-4).fit(x, y, num_classes)
        probs = self.initial.predict_proba(x)
        losses = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))
        keep = self._trimmed_indices(losses, y, num_classes, self.trim_fraction)
        self.kept_fraction = len(keep) / max(len(y), 1)
        self.final = SoftmaxLogRegLite(epochs=220, lr=0.8, l2=1e-4).fit(x[keep], y[keep], num_classes)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.final is None:
            raise RuntimeError("model is not fitted")
        return self.final.predict(x)
