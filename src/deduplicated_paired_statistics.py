#!/usr/bin/env python3
"""Recompute paired tests after deduplicating the realized clean condition."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


COMPARISONS = (
    ("linear_svm_hinge", "softmax_logreg"),
    ("multinomial_nb", "softmax_logreg"),
    ("trimmed_softmax_logreg", "softmax_logreg"),
)


def realized_noise_type(row: dict[str, str]) -> str:
    return "clean" if float(row["noise_rate"]) == 0.0 else row["noise_type"]


def paired_permutation_pvalue(deltas: list[float], n_perm: int = 4096, seed: int = 31415) -> float:
    if n_perm <= 0:
        return 1.0
    values = np.asarray(deltas, dtype=np.float64)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values))
        if abs(float((values * signs).mean())) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, min(1.0, (len(p_values) - rank) * p_value))
        adjusted[index] = running
    return adjusted


def paired_statistics(rows: list[dict[str, str]], n_perm: int = 4096) -> list[dict[str, float | int | str | bool]]:
    indexed = {}
    for row in rows:
        key = (
            row["dataset"],
            int(row["budget_per_class"]),
            float(row["noise_rate"]),
            realized_noise_type(row),
            int(row["seed"]),
            row["method"],
        )
        previous = indexed.get(key)
        if previous is not None and float(previous["macro_f1"]) != float(row["macro_f1"]):
            raise ValueError(f"clean-condition duplicate mismatch for {key}")
        indexed[key] = row

    result = []
    p_values = []
    for method, anchor in COMPARISONS:
        deltas = []
        for key, row in indexed.items():
            dataset, budget, noise_rate, noise_type, seed, row_method = key
            if row_method != method:
                continue
            anchor_row = indexed[(dataset, budget, noise_rate, noise_type, seed, anchor)]
            deltas.append(float(row["macro_f1"]) - float(anchor_row["macro_f1"]))
        p_value = paired_permutation_pvalue(deltas, n_perm=n_perm)
        p_values.append(p_value)
        result.append(
            {
                "comparison": f"{method}_minus_{anchor}",
                "metric": "macro_f1",
                "paired_units": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "median_delta": statistics.median(deltas),
                "permutation_p": p_value,
            }
        )
    for row, adjusted in zip(result, holm_adjust(p_values)):
        row["holm_p"] = adjusted
        row["significant_at_0_05"] = adjusted <= 0.05
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        stats = paired_statistics(list(csv.DictReader(handle)))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    args.output_json.write_text(json.dumps({"clean_condition_policy": "count once", "statistics": stats}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
