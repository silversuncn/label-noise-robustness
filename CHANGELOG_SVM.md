# Linear SVM Result Changelog

## Scope

This changelog documents the active lightweight linear SVM method in the
shared-mask matrix and the released comparison file:

```text
data/svm_old_vs_new_comparison.csv
```

The CSV contains 150 linear SVM rows:

- 120 rows are directly compared against the pre-shared-mask baseline.
- 30 rows are new because the shared-mask matrix adds the 32-per-class budget.
- The original private source-location column from the upstream comparison was
  removed from the public CSV.

## Previous Implementation

The comparison baseline used the lightweight method named `linear_svm_hinge`.
It was implemented as a deterministic one-vs-rest hinge-loss linear classifier,
not as `sklearn.svm.LinearSVC`, `sklearn.svm.SVC`, or
`sklearn.linear_model.SGDClassifier`.

Audited parameters:

| Parameter | Value |
| --- | ---: |
| `epochs` | `120` |
| `lr` | `0.35` |
| `l2` | `1e-4` |
| weight initialization RNG | `29` |
| kernel | not used |
| `C`, `tol`, `dual`, `class_weight`, `max_iter` | not used |

Known limitation of the baseline used for this comparison: it preceded the
final fixed-count shared-mask evidence package and did not contain the
32-per-class budget rows. It should be treated as a legacy comparison point, not
as an active analysis input.

## Current Implementation

The current package keeps the same lightweight `linear_svm_hinge` classifier and
publishes it in:

```text
src/models/linear_svm.py
```

Audited parameters remain:

| Parameter | Value |
| --- | ---: |
| `epochs` | `120` |
| `lr` | `0.35` |
| `l2` | `1e-4` |
| weight initialization RNG | `29` |

The active change is the experimental protocol:

- The noise count is fixed at `round(n_train * noise_rate)`.
- The same selected-position mask is shared across methods.
- The same selected-position mask is shared across symmetric and
  class-conditional mechanisms.
- A 32-per-class budget is included.

The execution environment used to create the shared-mask evidence recorded
`sklearn=1.8.0` for auditability. The lightweight linear SVM runner does not use
sklearn SVM classes or sklearn SVM hyperparameters.

## Numeric Difference Summary

Statistics below are computed from `data/svm_old_vs_new_comparison.csv`.

| Quantity | Value |
| --- | ---: |
| Total SVM rows in comparison file | `150` |
| Directly compared rows | `120` |
| Newly added 32-budget rows | `30` |
| Mean macro-F1 delta on compared rows | `0.000000` |
| Median macro-F1 delta on compared rows | `0.000000` |
| Minimum macro-F1 delta on compared rows | `0.000000` |
| Maximum macro-F1 delta on compared rows | `0.000000` |
| Mean absolute macro-F1 delta on compared rows | `0.000000` |

Per-dataset compared-row statistics:

| Dataset | Compared rows | Mean delta | Median delta | Min delta | Max delta | Mean absolute delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ag_news` | `40` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `dbpedia_14` | `40` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `trec_qc` | `40` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |

New rows by dataset:

| Dataset | New 32-budget rows |
| --- | ---: |
| `ag_news` | `10` |
| `dbpedia_14` | `10` |
| `trec_qc` | `10` |

## Top Absolute Delta Cells

All directly compared rows have zero macro-F1 delta in the supplied comparison
file. The top-10 absolute-delta rows are therefore ties:

| Dataset | Budget | Noise rate | Noise type | Seed | Old macro-F1 | New macro-F1 | Delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `ag_news` | `8` | `0.00` | `class_conditional` | `101` | `0.411099485208` | `0.411099485208` | `0.000000` |
| `ag_news` | `8` | `0.00` | `class_conditional` | `202` | `0.506824104406` | `0.506824104406` | `0.000000` |
| `ag_news` | `8` | `0.00` | `class_conditional` | `303` | `0.493967970965` | `0.493967970965` | `0.000000` |
| `ag_news` | `8` | `0.00` | `class_conditional` | `404` | `0.531992922922` | `0.531992922922` | `0.000000` |
| `ag_news` | `8` | `0.00` | `class_conditional` | `505` | `0.428435860186` | `0.428435860186` | `0.000000` |
| `ag_news` | `8` | `0.00` | `symmetric` | `101` | `0.411099485208` | `0.411099485208` | `0.000000` |
| `ag_news` | `8` | `0.00` | `symmetric` | `202` | `0.506824104406` | `0.506824104406` | `0.000000` |
| `ag_news` | `8` | `0.00` | `symmetric` | `303` | `0.493967970965` | `0.493967970965` | `0.000000` |
| `ag_news` | `8` | `0.00` | `symmetric` | `404` | `0.531992922922` | `0.531992922922` | `0.000000` |
| `ag_news` | `8` | `0.00` | `symmetric` | `505` | `0.428435860186` | `0.428435860186` | `0.000000` |

## Interpretation

The supplied comparison file does not show nonzero overlapping-cell SVM changes.
The active evidence difference relative to the legacy comparison baseline is the
publication of fixed-count shared-mask evidence and the addition of the
32-per-class budget rows. The active analysis should therefore use
`results_shared_mask_v3.csv`, `corruption_masks.csv`, split per-sample
prediction files, and the v3 dual-estimand outputs as the current source of
truth.
