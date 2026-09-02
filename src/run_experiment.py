#!/usr/bin/env python3
"""Run one public lightweight label-noise experiment cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .data_loader import DatasetBundle, TfidfVectorizerLite, cap_test_per_class, load_dataset_bundle, sample_per_class
    from .models.linear_svm import LinearSVMHingeLite
    from .models.multinomial_nb import MultinomialNBLite
    from .models.softmax_logreg import SoftmaxLogRegLite
    from .models.trimmed_softmax import TrimmedSoftmaxLogRegLite
    from .noise_generator import apply_fixed_count_noise, mask_sha256, stable_seed
except ImportError:
    from data_loader import DatasetBundle, TfidfVectorizerLite, cap_test_per_class, load_dataset_bundle, sample_per_class
    from models.linear_svm import LinearSVMHingeLite
    from models.multinomial_nb import MultinomialNBLite
    from models.softmax_logreg import SoftmaxLogRegLite
    from models.trimmed_softmax import TrimmedSoftmaxLogRegLite
    from noise_generator import apply_fixed_count_noise, mask_sha256, stable_seed


METHOD_FAMILY = {
    "softmax_logreg": "linear_probabilistic",
    "linear_svm_hinge": "linear_margin",
    "multinomial_nb": "generative_nb",
    "trimmed_softmax_logreg": "bounded_robust_variant",
}


def make_model(method: str):
    if method == "softmax_logreg":
        return SoftmaxLogRegLite()
    if method == "linear_svm_hinge":
        return LinearSVMHingeLite()
    if method == "multinomial_nb":
        return MultinomialNBLite()
    if method == "trimmed_softmax_logreg":
        return TrimmedSoftmaxLogRegLite()
    raise ValueError(f"unsupported method: {method}")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores: list[float] = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def sample_order_sha256(labels: list[int]) -> str:
    payload = ",".join(str(label) for label in labels)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_single_cell(
    dataset: DatasetBundle,
    budget_per_class: int,
    noise_rate: float,
    noise_type: str,
    method: str,
    seed: int,
    test_per_class_cap: int = 200,
    max_features: int = 3000,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.time()
    train_records, train_counts = sample_per_class(dataset.train, budget_per_class, seed)
    test_records = cap_test_per_class(dataset.test, test_per_class_cap, seed + 999)
    clean_train_labels = [record.label for record in train_records]
    mask_seed = stable_seed(dataset.name, budget_per_class, noise_rate, seed, "mask")
    noisy_labels, mask_records = apply_fixed_count_noise(
        clean_train_labels,
        dataset.num_classes,
        noise_rate,
        mask_seed,
        noise_type,
    )

    vectorizer = TfidfVectorizerLite(max_features=max_features)
    x_train = vectorizer.fit_transform([record.text for record in train_records])
    x_test = vectorizer.transform([record.text for record in test_records])
    y_train = np.asarray(noisy_labels, dtype=np.int64)
    y_test = np.asarray([record.label for record in test_records], dtype=np.int64)
    model = make_model(method).fit(x_train, y_train, dataset.num_classes)
    pred = model.predict(x_test)
    n_corrupt = sum(record["is_corrupted"] for record in mask_records)
    order_hash = sample_order_sha256(clean_train_labels)
    fit_notes = ""
    if isinstance(model, TrimmedSoftmaxLogRegLite) and model.kept_fraction is not None:
        fit_notes = f"kept_fraction={model.kept_fraction:.4f}"

    row = {
        "dataset": dataset.name,
        "budget_per_class": budget_per_class,
        "budget_train_total": len(train_records),
        "noise_rate": noise_rate,
        "noise_type": noise_type,
        "method": method,
        "seed": seed,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "num_classes": dataset.num_classes,
        "n_corrupt": n_corrupt,
        "actual_noise_fraction": n_corrupt / max(len(train_records), 1),
        "accuracy": accuracy(y_test, pred),
        "macro_f1": macro_f1(y_test, pred, dataset.num_classes),
        "runtime_seconds": time.time() - started,
        "train_counts": json.dumps(train_counts, sort_keys=True),
        "source_kind": "user_supplied_csv",
        "method_family": METHOD_FAMILY[method],
        "fit_notes": fit_notes,
        "mask_sha256": mask_sha256([record["is_corrupted"] for record in mask_records]),
        "sample_order_sha256": order_hash,
    }
    predictions = [
        {
            "dataset": dataset.name,
            "budget_per_class": budget_per_class,
            "noise_rate": noise_rate,
            "noise_type": noise_type,
            "method": method,
            "seed": seed,
            "eval_id": idx,
            "gold_label": int(y_test[idx]),
            "predicted_label": int(pred[idx]),
            "correct": int(y_test[idx] == pred[idx]),
        }
        for idx in range(len(test_records))
    ]
    masks = [
        {
            "dataset": dataset.name,
            "budget_per_class": budget_per_class,
            "budget_train_total": len(train_records),
            "noise_rate": noise_rate,
            "seed": seed,
            "original_idx_in_dataset": idx,
            "mask_seed": mask_seed,
            "sample_order_sha256": order_hash,
            **record,
        }
        for idx, record in enumerate(mask_records)
    ]
    return row, predictions, masks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget-per-class", type=int, required=True)
    parser.add_argument("--noise-rate", type=float, required=True)
    parser.add_argument("--noise-type", choices=["symmetric", "class_conditional"], required=True)
    parser.add_argument("--method", choices=sorted(METHOD_FAMILY), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    dataset = load_dataset_bundle(args.data_root, args.dataset)
    row, predictions, masks = run_single_cell(
        dataset,
        args.budget_per_class,
        args.noise_rate,
        args.noise_type,
        args.method,
        args.seed,
    )
    stem = f"{args.dataset}__b{args.budget_per_class}__r{args.noise_rate:.2f}__{args.noise_type}__{args.method}__s{args.seed}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}_predictions.csv", predictions)
    write_csv(args.output_dir / f"{stem}_masks.csv", masks)
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
