"""Fixed-count shared-mask label-noise generation."""

from __future__ import annotations

import hashlib
import random
from typing import Any


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def fixed_count_mask(n_items: int, noise_rate: float, seed: int) -> list[int]:
    n_corrupt = int(round(int(n_items) * float(noise_rate)))
    n_corrupt = max(0, min(int(n_items), n_corrupt))
    rng = random.Random(seed)
    selected = set(rng.sample(range(int(n_items)), n_corrupt)) if n_corrupt else set()
    return [1 if idx in selected else 0 for idx in range(int(n_items))]


def mask_sha256(mask: list[int]) -> str:
    payload = ",".join(str(value) for value in mask)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_fixed_count_noise(
    labels: list[int],
    num_classes: int,
    noise_rate: float,
    seed: int,
    noise_type: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    if noise_type not in {"symmetric", "class_conditional"}:
        raise ValueError(f"unsupported noise_type: {noise_type}")
    mask = fixed_count_mask(len(labels), noise_rate, seed)
    symmetric_rng = random.Random(seed + 104729)
    symmetric_labels: list[int] = []
    class_cond_labels: list[int] = []
    records: list[dict[str, Any]] = []

    for idx, clean in enumerate(labels):
        if mask[idx]:
            choices = [label for label in range(num_classes) if label != clean]
            symmetric_label = symmetric_rng.choice(choices)
            class_cond_label = (clean + 1) % num_classes
        else:
            symmetric_label = clean
            class_cond_label = clean
        symmetric_labels.append(symmetric_label)
        class_cond_labels.append(class_cond_label)
        records.append(
            {
                "sample_position": idx,
                "clean_label": int(clean),
                "is_corrupted": int(mask[idx]),
                "symmetric_label": int(symmetric_label),
                "class_cond_label": int(class_cond_label),
                "symmetric_changed": int(symmetric_label != clean),
                "class_cond_changed": int(class_cond_label != clean),
            }
        )

    noisy = symmetric_labels if noise_type == "symmetric" else class_cond_labels
    return noisy, records
