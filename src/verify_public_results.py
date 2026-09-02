#!/usr/bin/env python3
"""Verify the public shared-mask label-noise reproduction bundle.

The verifier intentionally uses only the Python standard library. It validates
the published aggregate matrix against the released corruption masks and
per-sample predictions, then recomputes the dual-estimand sign-flip analysis.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ANCHOR = "softmax_logreg"
COMPARATORS = ["linear_svm_hinge", "multinomial_nb", "trimmed_softmax_logreg"]
RAW_TOL = 1e-12
SIX_DECIMAL_TOL = 5e-7 + 1e-12
FOUR_DECIMAL_TOL = 5e-5 + 1e-12
MAX_PREFERRED_BYTES = 90 * 1024 * 1024
MAX_GITHUB_BYTES = 100 * 1024 * 1024

LEGACY_FILES = [
    "strengthened_merged_results.csv",
    "strengthened_merged_results_v2.csv",
    "strengthened_aggregate.csv",
    "strengthened_aggregate_v2.csv",
    "strengthened_degradation.csv",
    "strengthened_degradation_v2.csv",
    "strengthened_per_class_metrics.csv",
    "strengthened_per_class_metrics_v2.csv",
    "strengthened_noise_sensitivity_area.csv",
    "strengthened_noise_sensitivity_area_v2.csv",
    "strengthened_rank_stability.csv",
    "strengthened_rank_stability_v2.csv",
    "strengthened_paired_statistics.csv",
    "strengthened_third_results.csv",
    "strengthened_third_per_class_metrics.csv",
    "cluster_sign_flip_cluster_means_v2.csv",
    "cluster_sign_flip_statistics_v2.csv",
    "statistical_analysis_v2.json",
    "best_lightweight_distribution_v2.csv",
    "dataset_level_sensitivity_v2.csv",
    "maximum_degradation_table_v2.csv",
    "median_degradation_direction_table_v2.csv",
    "public_summary.json",
    "public_summary_v2.json",
    "deduplicated_paired_statistics.json",
    "distilbert_symmetric_aggregate.csv",
    "distilbert_symmetric_comparison_to_lightweight.csv",
    "distilbert_symmetric_comparison_to_lightweight_v2.csv",
    "distilbert_symmetric_degradation.csv",
    "distilbert_symmetric_results.csv",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tol: float) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def expected_corruption_count(n_train: int, noise_rate: float) -> int:
    return int(round(int(n_train) * float(noise_rate)))


def result_key(row: dict[str, str]) -> tuple[str, int, float, str, str, int]:
    return (
        row["dataset"],
        int(row["budget_per_class"]),
        round(float(row["noise_rate"]), 10),
        row["noise_type"],
        row["method"],
        int(row["seed"]),
    )


def mask_key(row: dict[str, str]) -> tuple[str, int, float, int]:
    return (
        row["dataset"],
        int(row["budget_per_class"]),
        round(float(row["noise_rate"]), 10),
        int(row["seed"]),
    )


def group_key(row: dict[str, str]) -> tuple[str, int, float, str, str]:
    return (
        row["dataset"],
        int(row["budget_per_class"]),
        round(float(row["noise_rate"]), 10),
        row["noise_type"],
        row["method"],
    )


def discover_prediction_files(data_dir: Path = DATA) -> list[Path]:
    direct = data_dir / "per_sample_predictions.csv"
    if direct.exists():
        return [direct]
    return sorted(data_dir.glob("per_sample_predictions__*.csv"))


def exact_sign_flip_pvalue(values: list[float]) -> float:
    observed = abs(sum(values))
    total = 1 << len(values)
    extreme = 0
    for mask in range(total):
        signed = 0.0
        for idx, value in enumerate(values):
            signed += value if ((mask >> idx) & 1) else -value
        if abs(signed) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def holm_correct(raw_ps: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(raw_ps.items(), key=lambda item: item[1])
    total = len(sorted_items)
    corrected: dict[str, float] = {}
    running_max = 0.0
    for rank, (method, p_value) in enumerate(sorted_items):
        adjusted = min(p_value * (total - rank), 1.0)
        running_max = max(running_max, adjusted)
        corrected[method] = running_max
    return corrected


def macro_f1_from_counts(stats: dict[str, Any], num_classes: int) -> float:
    scores: list[float] = []
    for cls in range(num_classes):
        tp = stats["tp"][cls]
        fp = stats["pred"][cls] - tp
        fn = stats["gold"][cls] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def verify_corruption_masks(results: list[dict[str, str]]) -> dict[str, Any]:
    path = DATA / "corruption_masks.csv"
    expected_keys: dict[tuple[str, int, float, int], dict[str, Any]] = {}
    for row in results:
        key = mask_key(row)
        expected_keys[key] = {
            "n_train": int(row["n_train"]),
            "n_corrupt": int(row["n_corrupt"]),
            "actual_noise_fraction": float(row["actual_noise_fraction"]),
            "mask_sha256": row["mask_sha256"],
            "sample_order_sha256": row["sample_order_sha256"],
        }

    groups: dict[tuple[str, int, float, int], dict[str, Any]] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            key = mask_key(row)
            group = groups.setdefault(
                key,
                {
                    "rows": 0,
                    "budget_train_total": int(row["budget_train_total"]),
                    "corrupted": 0,
                    "symmetric_changed": 0,
                    "class_cond_changed": 0,
                    "mask_sha256": row["mask_sha256"],
                    "sample_order_sha256": row["sample_order_sha256"],
                },
            )
            group["rows"] += 1
            group["corrupted"] += int(row["is_corrupted"])
            group["symmetric_changed"] += int(row["symmetric_changed"])
            group["class_cond_changed"] += int(row["class_cond_changed"])
            if group["mask_sha256"] != row["mask_sha256"]:
                group["mask_sha256_mismatch"] = True
            if group["sample_order_sha256"] != row["sample_order_sha256"]:
                group["sample_order_sha256_mismatch"] = True

    failures: dict[str, Any] = {}
    if set(groups) != set(expected_keys):
        failures["mask_key_set"] = {
            "missing": [str(key) for key in sorted(set(expected_keys) - set(groups))[:10]],
            "unexpected": [str(key) for key in sorted(set(groups) - set(expected_keys))[:10]],
        }

    max_fraction_error = 0.0
    for key, group in groups.items():
        expected = expected_keys.get(key)
        if expected is None:
            continue
        expected_count = expected_corruption_count(group["budget_train_total"], key[2])
        expected_fraction = expected_count / group["budget_train_total"] if group["budget_train_total"] else 0.0
        max_fraction_error = max(max_fraction_error, abs(expected_fraction - expected["actual_noise_fraction"]))
        local_failures: dict[str, Any] = {}
        if group["rows"] != group["budget_train_total"]:
            local_failures["row_count"] = {"expected": group["budget_train_total"], "actual": group["rows"]}
        if group["rows"] != expected["n_train"]:
            local_failures["n_train"] = {"expected": expected["n_train"], "actual": group["rows"]}
        if group["corrupted"] != expected_count:
            local_failures["fixed_count"] = {"expected": expected_count, "actual": group["corrupted"]}
        if group["corrupted"] != expected["n_corrupt"]:
            local_failures["result_n_corrupt"] = {"expected": expected["n_corrupt"], "actual": group["corrupted"]}
        if not close(expected_fraction, expected["actual_noise_fraction"], RAW_TOL):
            local_failures["actual_noise_fraction"] = {
                "expected": expected["actual_noise_fraction"],
                "actual": expected_fraction,
            }
        if group["symmetric_changed"] != group["corrupted"]:
            local_failures["symmetric_changed"] = {"expected": group["corrupted"], "actual": group["symmetric_changed"]}
        if group["class_cond_changed"] != group["corrupted"]:
            local_failures["class_cond_changed"] = {"expected": group["corrupted"], "actual": group["class_cond_changed"]}
        if group.get("mask_sha256_mismatch") or group["mask_sha256"] != expected["mask_sha256"]:
            local_failures["mask_sha256"] = {"expected": expected["mask_sha256"], "actual": group["mask_sha256"]}
        if group.get("sample_order_sha256_mismatch") or group["sample_order_sha256"] != expected["sample_order_sha256"]:
            local_failures["sample_order_sha256"] = {
                "expected": expected["sample_order_sha256"],
                "actual": group["sample_order_sha256"],
            }
        if local_failures:
            failures[str(key)] = local_failures
            if len(failures) > 20:
                break

    return {
        "status": "PASS" if not failures else "FAIL",
        "rows": row_count,
        "groups": len(groups),
        "max_actual_noise_fraction_abs_error": max_fraction_error,
        "failures": failures,
    }


def verify_prediction_macro_f1(results: list[dict[str, str]]) -> dict[str, Any]:
    expected: dict[tuple[str, int, float, str, str, int], dict[str, Any]] = {}
    for row in results:
        expected[result_key(row)] = {
            "n_test": int(row["n_test"]),
            "num_classes": int(row["num_classes"]),
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
        }

    prediction_files = discover_prediction_files(DATA)
    stats: dict[tuple[str, int, float, str, str, int], dict[str, Any]] = {
        key: {
            "n": 0,
            "correct": 0,
            "tp": [0] * values["num_classes"],
            "gold": [0] * values["num_classes"],
            "pred": [0] * values["num_classes"],
        }
        for key, values in expected.items()
    }

    failures: dict[str, Any] = {}
    total_rows = 0
    for path in prediction_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total_rows += 1
                key = result_key(row)
                if key not in stats:
                    failures.setdefault("unexpected_prediction_keys", []).append(str(key))
                    if len(failures["unexpected_prediction_keys"]) > 20:
                        break
                    continue
                item = stats[key]
                gold = int(row["gold_label"])
                predicted = int(row["predicted_label"])
                if gold >= len(item["gold"]) or predicted >= len(item["pred"]):
                    failures.setdefault("label_out_of_range", []).append(str(key))
                    continue
                correct = int(row["correct"])
                if correct != int(gold == predicted):
                    failures.setdefault("correct_flag", []).append(str(key))
                item["n"] += 1
                item["correct"] += correct
                item["gold"][gold] += 1
                item["pred"][predicted] += 1
                if gold == predicted:
                    item["tp"][gold] += 1

    max_accuracy_error = 0.0
    max_macro_f1_error = 0.0
    for key, values in expected.items():
        item = stats[key]
        local_failures: dict[str, Any] = {}
        if item["n"] != values["n_test"]:
            local_failures["n_test"] = {"expected": values["n_test"], "actual": item["n"]}
        accuracy = item["correct"] / item["n"] if item["n"] else 0.0
        macro_f1 = macro_f1_from_counts(item, values["num_classes"])
        max_accuracy_error = max(max_accuracy_error, abs(accuracy - values["accuracy"]))
        max_macro_f1_error = max(max_macro_f1_error, abs(macro_f1 - values["macro_f1"]))
        if not close(accuracy, values["accuracy"], RAW_TOL):
            local_failures["accuracy"] = {"expected": values["accuracy"], "actual": accuracy}
        if not close(macro_f1, values["macro_f1"], RAW_TOL):
            local_failures["macro_f1"] = {"expected": values["macro_f1"], "actual": macro_f1}
        if local_failures:
            failures[str(key)] = local_failures
            if len(failures) > 20:
                break

    file_sizes = {path.name: path.stat().st_size for path in prediction_files}
    oversized = {name: size for name, size in file_sizes.items() if size > MAX_GITHUB_BYTES}
    over_preferred = {name: size for name, size in file_sizes.items() if size > MAX_PREFERRED_BYTES}
    if oversized:
        failures["prediction_files_over_100mb"] = oversized

    return {
        "status": "PASS" if not failures else "FAIL",
        "files": [path.name for path in prediction_files],
        "file_sizes": file_sizes,
        "files_over_90mb": over_preferred,
        "rows": total_rows,
        "max_accuracy_abs_error": max_accuracy_error,
        "max_macro_f1_abs_error": max_macro_f1_error,
        "failures": failures,
    }


def aggregate_seed_mean(results: list[dict[str, str]]) -> dict[tuple[str, int, float, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in results:
        grouped[group_key(row)].append(row)
    out: dict[tuple[str, int, float, str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        acc = [float(row["accuracy"]) for row in rows]
        f1 = [float(row["macro_f1"]) for row in rows]
        out[key] = {
            "mean_accuracy": statistics.mean(acc),
            "std_accuracy": statistics.pstdev(acc),
            "mean_macro_f1": statistics.mean(f1),
            "std_macro_f1": statistics.pstdev(f1),
            "n_seeds": len(rows),
        }
    return out


def verify_aggregate_files(results: list[dict[str, str]]) -> dict[str, Any]:
    aggregate = read_csv(DATA / "aggregate_seed_mean_v3.csv")
    degradation = read_csv(DATA / "degradation_vs_clean_v3.csv")
    means = aggregate_seed_mean(results)
    failures: dict[str, Any] = {}
    max_error = 0.0

    for row in aggregate:
        key = group_key(row)
        expected = means.get(key)
        if expected is None:
            failures[f"aggregate_missing_source:{key}"] = row
            continue
        checks = {
            "mean_macro_f1": expected["mean_macro_f1"],
            "std_macro_f1": expected["std_macro_f1"],
            "n_seeds": expected["n_seeds"],
        }
        for field, actual in checks.items():
            reported = float(row[field]) if field != "n_seeds" else int(row[field])
            tol = SIX_DECIMAL_TOL if field != "n_seeds" else 0.0
            error = abs(float(actual) - float(reported))
            max_error = max(max_error, error)
            if error > tol:
                failures[f"aggregate:{key}:{field}"] = {"expected": actual, "actual": reported}

    for row in degradation:
        key = group_key(row)
        expected = means.get(key)
        if expected is None:
            failures[f"degradation_missing_source:{key}"] = row
            continue
        clean_key = (key[0], key[1], 0.0, key[3], key[4])
        clean = means[clean_key]
        expected_delta = expected["mean_macro_f1"] - clean["mean_macro_f1"]
        for field, actual in {
            "mean_macro_f1": expected["mean_macro_f1"],
            "delta_macro_f1_vs_clean": expected_delta,
        }.items():
            reported = float(row[field])
            error = abs(actual - reported)
            max_error = max(max_error, error)
            if error > SIX_DECIMAL_TOL:
                failures[f"degradation:{key}:{field}"] = {"expected": actual, "actual": reported}

    return {
        "status": "PASS" if not failures else "FAIL",
        "aggregate_rows": len(aggregate),
        "degradation_rows": len(degradation),
        "max_abs_error": max_error,
        "failures": failures,
    }


def result_lookup(results: list[dict[str, str]]) -> dict[tuple[str, int, float, str, str, int], float]:
    return {result_key(row): float(row["macro_f1"]) for row in results}


def grid_dims(results: list[dict[str, str]]) -> dict[str, list[Any]]:
    return {
        "datasets": sorted({row["dataset"] for row in results}),
        "budgets": sorted({int(row["budget_per_class"]) for row in results}),
        "noise_rates": sorted({round(float(row["noise_rate"]), 10) for row in results}),
        "noise_types": sorted({row["noise_type"] for row in results}),
        "methods": sorted({row["method"] for row in results}),
        "seeds": sorted({int(row["seed"]) for row in results}),
    }


def compute_dual_estimands(results: list[dict[str, str]]) -> dict[str, Any]:
    lookup = result_lookup(results)
    dims = grid_dims(results)
    nonzero_rates = [rate for rate in dims["noise_rates"] if rate > 0.0]
    output: dict[str, Any] = {}

    for estimand in ["performance", "robustness"]:
        raw_ps: dict[str, float] = {}
        details: dict[str, Any] = {}
        for method in COMPARATORS:
            cluster_means: list[float] = []
            cluster_labels: list[str] = []
            for dataset in dims["datasets"]:
                for seed in dims["seeds"]:
                    deltas: list[float] = []
                    for budget in dims["budgets"]:
                        for noise_type in dims["noise_types"]:
                            clean_method = lookup.get((dataset, budget, 0.0, noise_type, method, seed))
                            clean_anchor = lookup.get((dataset, budget, 0.0, noise_type, ANCHOR, seed))
                            for rate in nonzero_rates:
                                noisy_method = lookup.get((dataset, budget, rate, noise_type, method, seed))
                                noisy_anchor = lookup.get((dataset, budget, rate, noise_type, ANCHOR, seed))
                                if estimand == "performance":
                                    if noisy_method is not None and noisy_anchor is not None:
                                        deltas.append(noisy_method - noisy_anchor)
                                elif all(value is not None for value in [noisy_method, noisy_anchor, clean_method, clean_anchor]):
                                    deltas.append((noisy_method - clean_method) - (noisy_anchor - clean_anchor))
                    if deltas:
                        cluster_means.append(statistics.mean(deltas))
                        cluster_labels.append(f"{dataset}_seed{seed}")
            raw_p = exact_sign_flip_pvalue(cluster_means)
            raw_ps[method] = raw_p
            details[method] = {
                "n_clusters": len(cluster_means),
                "n_cells_per_cluster": len(nonzero_rates) * len(dims["budgets"]) * len(dims["noise_types"]),
                "overall_mean": statistics.mean(cluster_means),
                "raw_p": raw_p,
                "cluster_means": dict(zip(cluster_labels, cluster_means)),
            }
        holm = holm_correct(raw_ps)
        for method in COMPARATORS:
            details[method]["holm_p"] = holm[method]
        output[estimand] = details
    return output


def verify_dual_estimands(results: list[dict[str, str]]) -> dict[str, Any]:
    computed = compute_dual_estimands(results)
    reported = json.loads((DATA / "dual_estimand_sign_flip_v3.json").read_text(encoding="utf-8"))
    failures: dict[str, Any] = {}
    max_error = 0.0

    for estimand in ["performance", "robustness"]:
        for method in COMPARATORS:
            expected = computed[estimand][method]
            actual = reported[estimand][method]
            for field in ["n_clusters", "n_cells_per_cluster"]:
                if int(actual[field]) != int(expected[field]):
                    failures[f"{estimand}:{method}:{field}"] = {"expected": expected[field], "actual": actual[field]}
            for field in ["overall_mean", "raw_p", "holm_p"]:
                error = abs(float(actual[field]) - float(expected[field]))
                max_error = max(max_error, error)
                if error > SIX_DECIMAL_TOL:
                    failures[f"{estimand}:{method}:{field}"] = {"expected": expected[field], "actual": actual[field]}
            actual_clusters = actual["cluster_means"]
            for label, expected_value in expected["cluster_means"].items():
                if label not in actual_clusters:
                    failures[f"{estimand}:{method}:missing_cluster:{label}"] = expected_value
                    continue
                error = abs(float(actual_clusters[label]) - expected_value)
                max_error = max(max_error, error)
                if error > SIX_DECIMAL_TOL:
                    failures[f"{estimand}:{method}:cluster:{label}"] = {
                        "expected": expected_value,
                        "actual": actual_clusters[label],
                    }

    dataset_rows = read_csv(DATA / "per_dataset_estimands_v3.csv")
    lookup = result_lookup(results)
    dims = grid_dims(results)
    nonzero_rates = [rate for rate in dims["noise_rates"] if rate > 0.0]
    expected_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in dims["datasets"]:
        for method in COMPARATORS:
            perf: list[float] = []
            robust: list[float] = []
            for budget in dims["budgets"]:
                for noise_type in dims["noise_types"]:
                    for seed in dims["seeds"]:
                        clean_method = lookup.get((dataset, budget, 0.0, noise_type, method, seed))
                        clean_anchor = lookup.get((dataset, budget, 0.0, noise_type, ANCHOR, seed))
                        for rate in nonzero_rates:
                            noisy_method = lookup.get((dataset, budget, rate, noise_type, method, seed))
                            noisy_anchor = lookup.get((dataset, budget, rate, noise_type, ANCHOR, seed))
                            if noisy_method is not None and noisy_anchor is not None:
                                perf.append(noisy_method - noisy_anchor)
                            if all(value is not None for value in [noisy_method, noisy_anchor, clean_method, clean_anchor]):
                                robust.append((noisy_method - clean_method) - (noisy_anchor - clean_anchor))
            expected_dataset[(dataset, method)] = {
                "perf_mean": statistics.mean(perf),
                "robust_mean": statistics.mean(robust),
                "n_cells": len(perf),
            }
    for row in dataset_rows:
        key = (row["dataset"], row["method"])
        expected = expected_dataset.get(key)
        if expected is None:
            failures[f"per_dataset:unexpected:{key}"] = row
            continue
        if int(row["n_cells"]) != expected["n_cells"]:
            failures[f"per_dataset:{key}:n_cells"] = {"expected": expected["n_cells"], "actual": row["n_cells"]}
        for field in ["perf_mean", "robust_mean"]:
            error = abs(float(row[field]) - expected[field])
            max_error = max(max_error, error)
            if error > FOUR_DECIMAL_TOL:
                failures[f"per_dataset:{key}:{field}"] = {"expected": expected[field], "actual": row[field]}

    return {
        "status": "PASS" if not failures else "FAIL",
        "max_abs_error": max_error,
        "reported_estimands": sorted(reported),
        "per_dataset_rows": len(dataset_rows),
        "failures": failures,
    }


def verify_legacy_archival() -> dict[str, Any]:
    active = [name for name in LEGACY_FILES if (DATA / name).exists()]
    missing_archives = []
    for name in LEGACY_FILES:
        path = DATA / name
        if path.exists():
            continue
        stem = path.stem
        archive = DATA / f"{stem}_LEGACY_bernoulli{path.suffix}"
        if not archive.exists():
            missing_archives.append(archive.name)
    failures: dict[str, Any] = {}
    if active:
        failures["legacy_files_still_active"] = active
    if missing_archives:
        failures["missing_legacy_archives"] = missing_archives
    return {
        "status": "PASS" if not failures else "FAIL",
        "active_legacy_files": active,
        "missing_archives": missing_archives,
        "failures": failures,
    }


def verify() -> dict[str, Any]:
    results = read_csv(DATA / "results_shared_mask_v3.csv")
    dims = grid_dims(results)
    duplicate_keys = len(results) - len({result_key(row) for row in results})
    failures: dict[str, Any] = {}

    computed = {
        "result_rows": len(results),
        "datasets": dims["datasets"],
        "budgets_per_class": dims["budgets"],
        "noise_rates": dims["noise_rates"],
        "noise_types": dims["noise_types"],
        "methods": dims["methods"],
        "seeds": dims["seeds"],
        "duplicate_result_keys": duplicate_keys,
        "prediction_rows_from_decision_rows": sum(int(row["n_test"]) for row in results),
        "corruption_records_from_decision_rows": sum(int(row["n_train"]) for row in results),
        "corruption_mask_rows": 0,
        "prediction_rows": 0,
    }

    expected = {
        "result_rows": 3000,
        "datasets": ["ag_news", "dbpedia_14", "trec_qc"],
        "budgets_per_class": [8, 16, 32, 64, 128],
        "noise_rates": [0.0, 0.05, 0.1, 0.2, 0.3],
        "noise_types": ["class_conditional", "symmetric"],
        "methods": ["linear_svm_hinge", "multinomial_nb", "softmax_logreg", "trimmed_softmax_logreg"],
        "seeds": [101, 202, 303, 404, 505],
        "duplicate_result_keys": 0,
        "prediction_rows_from_decision_rows": 4100000,
        "corruption_records_from_decision_rows": 1182000,
    }
    for key, expected_value in expected.items():
        if computed[key] != expected_value:
            failures[key] = {"expected": expected_value, "actual": computed[key]}

    corruption_check = verify_corruption_masks(results)
    prediction_check = verify_prediction_macro_f1(results)
    aggregate_check = verify_aggregate_files(results)
    dual_check = verify_dual_estimands(results)
    legacy_check = verify_legacy_archival()

    computed["corruption_mask_rows"] = corruption_check["rows"]
    computed["prediction_rows"] = prediction_check["rows"]
    if corruption_check["rows"] != 147750:
        failures["corruption_mask_rows"] = {"expected": 147750, "actual": corruption_check["rows"]}
    if prediction_check["rows"] != 4100000:
        failures["prediction_rows"] = {"expected": 4100000, "actual": prediction_check["rows"]}

    for name, check in {
        "corruption_mask_check": corruption_check,
        "prediction_macro_f1_check": prediction_check,
        "aggregate_file_check": aggregate_check,
        "dual_estimand_check": dual_check,
        "legacy_archival_check": legacy_check,
    }.items():
        if check["status"] != "PASS":
            failures[name] = check["failures"]

    return {
        "status": "PASS" if not failures else "FAIL",
        "computed": computed,
        "expected": expected,
        "corruption_mask_check": corruption_check,
        "prediction_macro_f1_check": prediction_check,
        "aggregate_file_check": aggregate_check,
        "dual_estimand_check": dual_check,
        "legacy_archival_check": legacy_check,
        "failures": failures,
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
