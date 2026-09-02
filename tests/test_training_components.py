import unittest

import numpy as np

from src.aggregate_results import aggregate_result_rows, degradation_rows
from src.data_loader import TextRecord, TfidfVectorizerLite, sample_per_class
from src.models.linear_svm import LinearSVMHingeLite
from src.models.multinomial_nb import MultinomialNBLite
from src.models.softmax_logreg import SoftmaxLogRegLite
from src.models.trimmed_softmax import TrimmedSoftmaxLogRegLite
from src.noise_generator import apply_fixed_count_noise, fixed_count_mask
from src.run_experiment import DatasetBundle, run_single_cell


class TrainingComponentTest(unittest.TestCase):
    def test_fixed_count_mask_is_shared_and_uses_nearest_integer(self):
        mask_a = fixed_count_mask(48, 0.05, seed=17)
        mask_b = fixed_count_mask(48, 0.05, seed=17)
        self.assertEqual(mask_a, mask_b)
        self.assertEqual(sum(mask_a), 2)

    def test_noise_mechanisms_reuse_same_mask_positions(self):
        labels = [0, 1, 2, 3] * 12
        symmetric, sym_records = apply_fixed_count_noise(labels, 4, 0.05, 17, "symmetric")
        class_cond, cls_records = apply_fixed_count_noise(labels, 4, 0.05, 17, "class_conditional")
        self.assertEqual([r["is_corrupted"] for r in sym_records], [r["is_corrupted"] for r in cls_records])
        self.assertEqual(sum(r["is_corrupted"] for r in sym_records), 2)
        for clean, noisy, record in zip(labels, class_cond, cls_records):
            if record["is_corrupted"]:
                self.assertEqual(noisy, (clean + 1) % 4)

    def test_vectorizer_and_models_fit_small_separable_problem(self):
        texts = ["alpha alpha red", "alpha red", "beta beta blue", "beta blue"]
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        x = TfidfVectorizerLite(max_features=8).fit_transform(texts)
        self.assertEqual(x.shape[0], 4)
        models = [
            SoftmaxLogRegLite(epochs=40, lr=0.8),
            LinearSVMHingeLite(epochs=20, lr=0.4),
            MultinomialNBLite(),
            TrimmedSoftmaxLogRegLite(trim_fraction=0.25),
        ]
        for model in models:
            pred = model.fit(x, y, num_classes=2).predict(x)
            self.assertEqual(set(pred.tolist()), {0, 1})

    def test_run_single_cell_and_aggregation_emit_public_schema(self):
        train = [
            TextRecord("alpha red", 0),
            TextRecord("alpha news", 0),
            TextRecord("beta blue", 1),
            TextRecord("beta topic", 1),
        ]
        test = [TextRecord("alpha red", 0), TextRecord("beta blue", 1)]
        bundle = DatasetBundle(name="toy", train=train, test=test, num_classes=2)
        row, predictions, masks = run_single_cell(
            bundle,
            budget_per_class=2,
            noise_rate=0.0,
            noise_type="symmetric",
            method="multinomial_nb",
            seed=101,
        )
        self.assertEqual(row["n_train"], 4)
        self.assertEqual(row["n_test"], 2)
        self.assertEqual(row["n_corrupt"], 0)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(masks), 4)

        aggregate = aggregate_result_rows([row])
        degradation = degradation_rows(aggregate)
        self.assertEqual(aggregate[0]["n_seeds"], 1)
        self.assertEqual(degradation[0]["delta_macro_f1_vs_clean"], 0.0)

    def test_sample_per_class_is_seed_stable(self):
        records = [TextRecord(f"class {label} row {idx}", label) for label in (0, 1) for idx in range(4)]
        first, counts_first = sample_per_class(records, 2, seed=5)
        second, counts_second = sample_per_class(records, 2, seed=5)
        self.assertEqual(first, second)
        self.assertEqual(counts_first, {0: 2, 1: 2})
        self.assertEqual(counts_second, {0: 2, 1: 2})


if __name__ == "__main__":
    unittest.main()
