#!/usr/bin/env python3
"""Regenerate active label-noise figures from archived aggregate evidence."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ACTIVE_FIGURE_STEMS = ("strengthened_macro_f1_curves_budget64", "degradation_heatmap_symmetric_030")
DATASET_ORDER = ["ag_news", "dbpedia_14", "trec_qc"]
NOISE_ORDER = ["class_conditional", "symmetric"]
METHOD_ORDER = ["softmax_logreg", "multinomial_nb", "linear_svm_hinge", "trimmed_softmax_logreg"]
METHOD_LABELS = {
    "softmax_logreg": "Softmax LR",
    "multinomial_nb": "Multinomial NB",
    "linear_svm_hinge": "Linear SVM",
    "trimmed_softmax_logreg": "Trimmed LR",
}
COLORS = {
    "softmax_logreg": "#2563a6",
    "multinomial_nb": "#d97706",
    "linear_svm_hinge": "#2f855a",
    "trimmed_softmax_logreg": "#b43c39",
}
MARKERS = {"softmax_logreg": "o", "multinomial_nb": "s", "linear_svm_hinge": "^", "trimmed_softmax_logreg": "D"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def budget64_curves(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[tuple[float, float]]]:
    curves: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if int(row["budget_per_class"]) != 64:
            continue
        key = (row["dataset"], row["noise_type"], row["method"])
        curves[key].append((float(row["noise_rate"]), float(row["mean_macro_f1"])))
    return {key: sorted(points) for key, points in curves.items()}


def symmetric_degradation_grid(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str], list[list[float]]]:
    selected = {
        (row["dataset"], int(row["budget_per_class"]), row["method"]): float(row["delta_macro_f1_vs_clean"])
        for row in rows
        if row["noise_type"] == "symmetric" and float(row["noise_rate"]) == 0.3
    }
    budgets = sorted(set(int(row["budget_per_class"]) for row in rows if row["noise_type"] == "symmetric" and float(row["noise_rate"]) == 0.3))
    row_labels = [f"{dataset}:{budget}" for dataset in DATASET_ORDER for budget in budgets]
    values = [
        [selected[(dataset, budget, method)] for method in METHOD_ORDER]
        for dataset in DATASET_ORDER
        for budget in budgets
    ]
    return row_labels, [METHOD_LABELS[method] for method in METHOD_ORDER], values


def apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_both(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")


def render_figures(aggregate_csv: Path, degradation_csv: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = budget64_curves(read_csv(aggregate_csv))

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.75), sharex=True, sharey="row")
    for row_index, noise_type in enumerate(NOISE_ORDER):
        for col_index, dataset in enumerate(DATASET_ORDER):
            ax = axes[row_index][col_index]
            for method in METHOD_ORDER:
                points = curves[(dataset, noise_type, method)]
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker=MARKERS[method],
                    markersize=3.2,
                    color=COLORS[method],
                    label=METHOD_LABELS[method],
                )
            ax.set_title(f"{dataset.replace('_', ' ').title()} / {noise_type.replace('_', ' ')}")
            ax.set_xticks([0.0, 0.1, 0.2, 0.3])
            ax.grid(True, linewidth=0.35, alpha=0.28)
            if row_index == 1:
                ax.set_xlabel("Training-label noise rate")
            if col_index == 0:
                ax.set_ylabel("Mean clean-test macro-F1")
    axes[0][2].legend(loc="best", frameon=True, framealpha=0.94, borderpad=0.25, handletextpad=0.35)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    save_both(fig, output_dir, "strengthened_macro_f1_curves_budget64")
    plt.close(fig)

    row_labels, method_labels, values = symmetric_degradation_grid(read_csv(degradation_csv))
    fig_height = 2.2 + 0.22 * len(row_labels)
    fig, ax = plt.subplots(figsize=(3.5, fig_height))
    image = ax.imshow(values, cmap="RdYlGn", vmin=min(min(row) for row in values), vmax=0.0, aspect="auto")
    ax.set_xticks(range(len(method_labels)), method_labels, rotation=34, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_ylabel("Dataset : budget")
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            text_color = "white" if value < -0.19 else "black"
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("Delta macro-F1", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_both(fig, output_dir, "degradation_heatmap_symmetric_030")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--degradation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_figures(args.aggregate_csv, args.degradation_csv, args.output_dir)
    print(f"PASS: generated {len(ACTIVE_FIGURE_STEMS) * 2} files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
