#!/usr/bin/env python3
"""Create or execute the public lightweight label-noise experiment grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


DATASETS = ["ag_news", "dbpedia_14", "trec_qc"]
BUDGETS = [8, 16, 32, 64, 128]
NOISE_RATES = [0.0, 0.05, 0.1, 0.2, 0.3]
NOISE_TYPES = ["symmetric", "class_conditional"]
METHODS = ["softmax_logreg", "linear_svm_hinge", "multinomial_nb", "trimmed_softmax_logreg"]
SEEDS = [101, 202, 303, 404, 505]


def build_commands(data_root: Path, output_dir: Path, python: str = sys.executable) -> list[list[str]]:
    script = Path(__file__).resolve().parent / "run_experiment.py"
    commands: list[list[str]] = []
    for dataset, budget, rate, noise_type, method, seed in product(DATASETS, BUDGETS, NOISE_RATES, NOISE_TYPES, METHODS, SEEDS):
        commands.append(
            [
                python,
                str(script),
                "--data-root",
                str(data_root),
                "--dataset",
                dataset,
                "--budget-per-class",
                str(budget),
                "--noise-rate",
                str(rate),
                "--noise-type",
                noise_type,
                "--method",
                method,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--manifest", type=Path, default=Path("results/grid_commands.json"))
    parser.add_argument("--execute", action="store_true", help="Run the generated commands sequentially.")
    args = parser.parse_args()

    commands = build_commands(args.data_root, args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        for command in commands:
            subprocess.run(command, check=True)
    print(json.dumps({"commands": len(commands), "manifest": str(args.manifest), "executed": args.execute}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
