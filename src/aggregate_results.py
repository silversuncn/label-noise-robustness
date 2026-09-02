#!/usr/bin/env python3
"""Aggregate public label-noise result rows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_FIELDS = ["dataset", "budget_per_class", "noise_rate", "noise_type", "method"]


def read_result_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".json":
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def group_key(row: dict[str, Any]) -> tuple[str, int, float, str, str]:
    return (
        str(row["dataset"]),
        int(row["budget_per_class"]),
        round(float(row["noise_rate"]), 10),
        str(row["noise_type"]),
        str(row["method"]),
    )


def aggregate_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        accuracies = [float(row["accuracy"]) for row in items]
        macro_f1 = [float(row["macro_f1"]) for row in items]
        output.append(
            {
                "dataset": key[0],
                "budget_per_class": key[1],
                "noise_rate": key[2],
                "noise_type": key[3],
                "method": key[4],
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.pstdev(accuracies),
                "mean_macro_f1": statistics.mean(macro_f1),
                "std_macro_f1": statistics.pstdev(macro_f1),
                "n_seeds": len(items),
            }
        )
    return output


def degradation_rows(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["dataset"],
            int(row["budget_per_class"]),
            row["noise_type"],
            row["method"],
            round(float(row["noise_rate"]), 10),
        ): row
        for row in aggregate
    }
    output: list[dict[str, Any]] = []
    for row in sorted(aggregate, key=group_key):
        clean = lookup[(row["dataset"], int(row["budget_per_class"]), row["noise_type"], row["method"], 0.0)]
        output.append(
            {
                "dataset": row["dataset"],
                "budget_per_class": int(row["budget_per_class"]),
                "noise_rate": float(row["noise_rate"]),
                "noise_type": row["noise_type"],
                "method": row["method"],
                "mean_macro_f1": float(row["mean_macro_f1"]),
                "delta_macro_f1_vs_clean": float(row["mean_macro_f1"]) - float(clean["mean_macro_f1"]),
            }
        )
    return output


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
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    rows = read_result_rows(args.inputs)
    aggregate = aggregate_result_rows(rows)
    degradation = degradation_rows(aggregate)
    write_csv(args.output_dir / "aggregate_seed_mean_v3.csv", aggregate)
    write_csv(args.output_dir / "degradation_vs_clean_v3.csv", degradation)
    print(json.dumps({"rows": len(rows), "aggregate_rows": len(aggregate), "degradation_rows": len(degradation)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
