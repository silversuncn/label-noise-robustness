#!/usr/bin/env python3
"""Dual-estimand statistical analysis for the shared-mask label-noise grid.

Reads per-seed results and produces:
  aggregate_seed_mean_v3.csv     — mean/std over seeds (for figure generation)
  degradation_vs_clean_v3.csv    — delta macro-F1 vs matched clean cell
  dual_estimand_sign_flip_v3.json — 15-cluster exact sign-flip test results
  per_dataset_estimands_v3.csv   — per-dataset Performance + Robustness breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path

ANCHOR = "softmax_logreg"
COMPARATORS = ["linear_svm_hinge", "multinomial_nb", "trimmed_softmax_logreg"]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            r["macro_f1"] = float(r["macro_f1"])
            r["noise_rate"] = float(r["noise_rate"])
            r["budget_per_class"] = int(r["budget_per_class"])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def build_lookup(rows: list[dict]) -> dict[tuple, float]:
    lk = {}
    for r in rows:
        key = (r["dataset"], r["budget_per_class"], r["noise_rate"],
               r["noise_type"], r["method"], r["seed"])
        lk[key] = r["macro_f1"]
    return lk


def grid_dims(rows: list[dict]) -> dict[str, list]:
    return {
        "datasets": sorted(set(r["dataset"] for r in rows)),
        "budgets": sorted(set(r["budget_per_class"] for r in rows)),
        "noise_rates": sorted(set(r["noise_rate"] for r in rows)),
        "noise_types": sorted(set(r["noise_type"] for r in rows)),
        "methods": sorted(set(r["method"] for r in rows)),
        "seeds": sorted(set(r["seed"] for r in rows)),
    }


def aggregate_seed_mean(rows: list[dict], out: Path) -> None:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["budget_per_class"], r["noise_rate"],
               r["noise_type"], r["method"])
        groups[key].append(r["macro_f1"])
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "budget_per_class", "noise_rate", "noise_type",
                     "method", "mean_macro_f1", "std_macro_f1", "n_seeds"])
        for key in sorted(groups):
            vals = groups[key]
            ds, bud, nr, nt, meth = key
            m = statistics.mean(vals)
            s = statistics.pstdev(vals)
            w.writerow([ds, bud, nr, nt, meth, f"{m:.6f}", f"{s:.6f}", len(vals)])
    print(f"  aggregate_seed_mean: {len(groups)} cells -> {out.name}")


def degradation_vs_clean(rows: list[dict], out: Path) -> None:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["budget_per_class"], r["noise_rate"],
               r["noise_type"], r["method"])
        groups[key].append(r["macro_f1"])
    means = {k: statistics.mean(v) for k, v in groups.items()}
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "budget_per_class", "noise_rate", "noise_type",
                     "method", "mean_macro_f1", "delta_macro_f1_vs_clean"])
        for key in sorted(means):
            ds, bud, nr, nt, meth = key
            clean_key = (ds, bud, 0.0, nt, meth)
            delta = means[key] - means[clean_key]
            w.writerow([ds, bud, nr, nt, meth, f"{means[key]:.6f}", f"{delta:.6f}"])
    print(f"  degradation_vs_clean: {len(means)} cells -> {out.name}")


def exact_sign_flip(cluster_means: list[float]) -> float:
    n = len(cluster_means)
    observed = abs(sum(cluster_means))
    total = 2 ** n
    count = sum(
        1 for bits in range(total)
        if abs(sum(
            (1 if (bits >> i) & 1 else -1) * cluster_means[i]
            for i in range(n)
        )) >= observed
    )
    return count / total


def holm_correct(raw_ps: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(raw_ps.items(), key=lambda x: x[1])
    k = len(sorted_items)
    result = {}
    running_max = 0.0
    for rank, (method, p) in enumerate(sorted_items):
        adj = min(p * (k - rank), 1.0)
        running_max = max(running_max, adj)
        result[method] = running_max
    return result


def sign_flip_tests(lookup: dict[tuple, float], dims: dict) -> dict:
    datasets = dims["datasets"]
    budgets = dims["budgets"]
    noise_types = dims["noise_types"]
    seeds = dims["seeds"]
    nonzero = [r for r in dims["noise_rates"] if r > 0]

    results = {}
    for estimand in ["performance", "robustness"]:
        raw_ps = {}
        details = {}
        for method in COMPARATORS:
            cluster_means = []
            cluster_labels = []
            for ds in datasets:
                for seed in seeds:
                    deltas = []
                    for bud in budgets:
                        for nt in noise_types:
                            clean_m = lookup.get((ds, bud, 0.0, nt, method, seed))
                            clean_a = lookup.get((ds, bud, 0.0, nt, ANCHOR, seed))
                            for nr in nonzero:
                                noisy_m = lookup.get((ds, bud, nr, nt, method, seed))
                                noisy_a = lookup.get((ds, bud, nr, nt, ANCHOR, seed))
                                if estimand == "performance":
                                    if noisy_m is not None and noisy_a is not None:
                                        deltas.append(noisy_m - noisy_a)
                                else:
                                    if all(v is not None for v in [noisy_m, noisy_a, clean_m, clean_a]):
                                        deltas.append((noisy_m - clean_m) - (noisy_a - clean_a))
                    if deltas:
                        cluster_means.append(statistics.mean(deltas))
                        cluster_labels.append(f"{ds}_seed{seed}")
            p = exact_sign_flip(cluster_means)
            raw_ps[method] = p
            overall_mean = statistics.mean(cluster_means)
            details[method] = {
                "n_clusters": len(cluster_means),
                "n_cells_per_cluster": len(nonzero) * len(budgets) * len(noise_types),
                "overall_mean": round(overall_mean, 6),
                "raw_p": round(p, 6),
                "cluster_means": {lbl: round(v, 6) for lbl, v in zip(cluster_labels, cluster_means)},
            }
        holm = holm_correct(raw_ps)
        for method in COMPARATORS:
            details[method]["holm_p"] = round(holm[method], 6)
        results[estimand] = details
    return results


def per_dataset_estimands(lookup: dict[tuple, float], dims: dict, out: Path) -> None:
    datasets = dims["datasets"]
    budgets = dims["budgets"]
    noise_types = dims["noise_types"]
    seeds = dims["seeds"]
    nonzero = [r for r in dims["noise_rates"] if r > 0]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "perf_mean", "robust_mean", "n_cells"])
        for ds in datasets:
            for method in COMPARATORS:
                perf, robust = [], []
                for bud in budgets:
                    for nt in noise_types:
                        for seed in seeds:
                            clean_m = lookup.get((ds, bud, 0.0, nt, method, seed))
                            clean_a = lookup.get((ds, bud, 0.0, nt, ANCHOR, seed))
                            for nr in nonzero:
                                noisy_m = lookup.get((ds, bud, nr, nt, method, seed))
                                noisy_a = lookup.get((ds, bud, nr, nt, ANCHOR, seed))
                                if noisy_m is not None and noisy_a is not None:
                                    perf.append(noisy_m - noisy_a)
                                if all(v is not None for v in [noisy_m, noisy_a, clean_m, clean_a]):
                                    robust.append((noisy_m - clean_m) - (noisy_a - clean_a))
                w.writerow([ds, method,
                            f"{statistics.mean(perf):.4f}",
                            f"{statistics.mean(robust):.4f}",
                            len(perf)])
    print(f"  per_dataset_estimands: {len(datasets) * len(COMPARATORS)} rows -> {out.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/results_shared_mask_v3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    dims = grid_dims(rows)
    lookup = build_lookup(rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Read {len(rows)} rows: {dims['datasets']}, budgets={dims['budgets']}")

    aggregate_seed_mean(rows, out / "aggregate_seed_mean_v3.csv")
    degradation_vs_clean(rows, out / "degradation_vs_clean_v3.csv")
    per_dataset_estimands(lookup, dims, out / "per_dataset_estimands_v3.csv")

    print("  Running 15-cluster exact sign-flip tests (2^15 = 32768 per test)...")
    sf_results = sign_flip_tests(lookup, dims)
    sf_out = out / "dual_estimand_sign_flip_v3.json"
    with sf_out.open("w", encoding="utf-8") as f:
        json.dump(sf_results, f, indent=2, sort_keys=True)
    print(f"  sign_flip_tests -> {sf_out.name}")

    for est_name, est_data in sf_results.items():
        print(f"\n  === {est_name} ===")
        for method in COMPARATORS:
            d = est_data[method]
            print(f"    {method}: mean={d['overall_mean']:+.4f}, "
                  f"raw_p={d['raw_p']:.6f}, holm_p={d['holm_p']:.6f}")

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
