# Label-Noise Robustness of Lightweight Text Classifiers under Limited Supervision

Yaowen Sun

## Overview

This repository is a public verification package with runnable lightweight
training code. It publishes the final shared-mask result matrix, fixed-count
corruption masks, split per-sample prediction evidence, analysis scripts, figure
scripts, and a changelog for the linear SVM result audit.

The raw text corpora are not bundled. To rerun the full training grid, provide
normalized train/test CSV files with `text` and `label` columns under
`data/raw/`. The released verification path does not require those raw text
files because all reported aggregate values can be recomputed from the published
result, mask, and prediction evidence.

## Repository Structure

```text
.
├── CHANGELOG_SVM.md
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── results_shared_mask_v3.csv
│   ├── aggregate_seed_mean_v3.csv
│   ├── degradation_vs_clean_v3.csv
│   ├── dual_estimand_sign_flip_v3.json
│   ├── per_dataset_estimands_v3.csv
│   ├── corruption_masks.csv
│   ├── per_sample_predictions__dataset-...__method-....csv
│   ├── svm_old_vs_new_comparison.csv
│   └── *_LEGACY_bernoulli.*
├── figures/
│   ├── degradation_heatmap_symmetric_030.pdf
│   ├── degradation_heatmap_symmetric_030.png
│   ├── strengthened_macro_f1_curves_budget64.pdf
│   └── strengthened_macro_f1_curves_budget64.png
├── src/
│   ├── data_loader.py
│   ├── noise_generator.py
│   ├── run_experiment.py
│   ├── launch_grid.py
│   ├── aggregate_results.py
│   ├── dual_estimand_analysis.py
│   ├── plot_figures.py
│   ├── verify_public_results.py
│   └── models/
│       ├── linear_svm.py
│       ├── multinomial_nb.py
│       ├── softmax_logreg.py
│       └── trimmed_softmax.py
└── tests/
    ├── test_public_results.py
    └── test_training_components.py
```

## Experimental Grid

| Dimension | Values | Count |
| --- | --- | ---: |
| Datasets | `ag_news`, `dbpedia_14`, `trec_qc` | 3 |
| Budgets per class | `8`, `16`, `32`, `64`, `128` | 5 |
| Noise rates | `0.0`, `0.05`, `0.1`, `0.2`, `0.3` | 5 |
| Noise types | `symmetric`, `class_conditional` | 2 |
| Methods | `softmax_logreg`, `linear_svm_hinge`, `multinomial_nb`, `trimmed_softmax_logreg` | 4 |
| Seeds | `101`, `202`, `303`, `404`, `505` | 5 |

The matrix contains:

```text
3 datasets x 5 budgets x 5 noise rates x 2 noise types x 4 methods x 5 seeds = 3000 result rows
4100000 released per-sample prediction rows
147750 released corruption-mask rows
```

`budget_per_class` is the requested per-class training budget. The realized
training total is stored in `budget_train_total` and `n_train`; it may be lower
than `budget_per_class * num_classes` when a dataset class has fewer available
training examples.

## Fixed-Count Shared-Mask Noise

For each `(dataset, budget_per_class, noise_rate, seed)` cell, the pipeline:

1. Samples training examples without replacement within each class.
2. Selects exactly `round(n_train * noise_rate)` training positions for
   corruption.
3. Reuses that same selected-position mask across all four lightweight methods.
4. Reuses that same mask for both label-noise mechanisms, changing only the
   replacement-label rule.

For symmetric noise, a corrupted label is replaced by a different class sampled
uniformly from the remaining labels. For class-conditional noise, a corrupted
label is replaced by `(clean_label + 1) mod num_classes`.

The published `corruption_masks.csv` verifies the selected positions, realized
corruption count, replacement labels, mask hash, and sample-order hash for every
unique `(dataset, budget_per_class, noise_rate, seed)` cell.

## Estimands

The analysis separates absolute performance from noise robustness.

Performance contrast:

```text
macro_f1(method, noisy cell) - macro_f1(softmax_logreg, noisy cell)
```

Robustness difference-in-differences:

```text
[macro_f1(method, noisy cell) - macro_f1(method, matched clean cell)]
- [macro_f1(softmax_logreg, noisy cell) - macro_f1(softmax_logreg, matched clean cell)]
```

The reported sign-flip diagnostics use 15 dataset-seed clusters. Each cluster
averages 40 condition deltas: 5 budgets x 4 nonzero noise rates x 2 noise types.

## Data Files

| File | Purpose |
| --- | --- |
| `data/results_shared_mask_v3.csv` | Per-seed result matrix with accuracy, macro-F1, realized training counts, corruption counts, and hashes. |
| `data/aggregate_seed_mean_v3.csv` | Mean and population standard deviation over the five seeds for each aggregate cell. |
| `data/degradation_vs_clean_v3.csv` | Mean macro-F1 and clean-matched degradation for every aggregate cell, including zero-noise rows. |
| `data/dual_estimand_sign_flip_v3.json` | Performance and robustness cluster-level exact sign-flip results with Holm correction. |
| `data/per_dataset_estimands_v3.csv` | Per-dataset performance and robustness means for each comparator method. |
| `data/corruption_masks.csv` | Fixed-count shared-mask evidence for training-label corruption. |
| `data/per_sample_predictions__dataset-...__method-....csv` | Split per-sample prediction evidence; files are split by dataset and method to stay below common repository size limits. |
| `data/svm_old_vs_new_comparison.csv` | Linear SVM comparison rows used by `CHANGELOG_SVM.md`. |
| `data/*_LEGACY_bernoulli.*` | Archived legacy Bernoulli-noise and removed-comparator outputs retained only for provenance; they are not active analysis inputs. |

## Verification

The main verifier uses only the Python standard library:

```bash
python src/verify_public_results.py
```

It checks:

- 3000 result rows with no duplicate primary keys.
- 147750 corruption-mask rows.
- Exact `round(n_train * noise_rate)` corruption counts.
- Shared mask and sample-order hashes against the result matrix.
- 4100000 per-sample prediction rows across 12 split files.
- Macro-F1 and accuracy recomputed from predictions for every result row.
- Aggregate, degradation, per-dataset estimand, and 15-cluster sign-flip outputs.
- Legacy files archived under `_LEGACY_bernoulli` names.

After installing optional numerical dependencies, run the full test suite:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

To regenerate analysis derivatives without overwriting released data, write to a
separate directory:

```bash
python src/dual_estimand_analysis.py --input data/results_shared_mask_v3.csv --output-dir regenerated
```

To regenerate figures:

```bash
python src/plot_figures.py \
  --aggregate-csv data/aggregate_seed_mean_v3.csv \
  --degradation-csv data/degradation_vs_clean_v3.csv \
  --output-dir regenerated_figures
```

## Training Pipeline

Prepare normalized raw text CSVs:

```text
data/raw/ag_news_train.csv
data/raw/ag_news_test.csv
data/raw/dbpedia_14_train.csv
data/raw/dbpedia_14_test.csv
data/raw/trec_qc_train.csv
data/raw/trec_qc_test.csv
```

Each CSV needs `text` and integer `label` columns. Then create the full grid
command manifest:

```bash
python src/launch_grid.py --data-root data/raw --output-dir results --manifest results/grid_commands.json
```

To execute the grid sequentially:

```bash
python src/launch_grid.py --data-root data/raw --output-dir results --execute
```

Aggregate produced cell result files:

```bash
python src/aggregate_results.py results/*.json --output-dir regenerated
```

## Linear SVM Audit

`CHANGELOG_SVM.md` documents the lightweight linear SVM implementation, the
comparison baseline, the new shared-mask matrix, numeric deltas, per-dataset
statistics, and the top-10 absolute-delta cells from
`data/svm_old_vs_new_comparison.csv`.

## Citation

```bibtex
@article{sun2026labelnoiserobustness,
  title = {Label-Noise Robustness of Lightweight Text Classifiers under Limited Supervision},
  author = {Sun, Yaowen},
  year = {2026}
}
```
