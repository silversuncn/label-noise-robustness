import unittest
from pathlib import Path

from src.verify_public_results import (
    discover_prediction_files,
    expected_corruption_count,
    verify,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAX_PREFERRED_BYTES = 90 * 1024 * 1024


class PublicResultVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = verify()

    def test_public_results_match_shared_mask_evidence(self):
        self.assertEqual(self.result["status"], "PASS", self.result.get("failures"))
        self.assertEqual(self.result["computed"]["result_rows"], 3000)
        self.assertEqual(self.result["computed"]["corruption_mask_rows"], 147750)
        self.assertEqual(self.result["computed"]["prediction_rows"], 4100000)
        self.assertLessEqual(self.result["prediction_macro_f1_check"]["max_macro_f1_abs_error"], 1e-12)
        self.assertEqual(self.result["dual_estimand_check"]["status"], "PASS")

    def test_fixed_count_uses_nearest_integer(self):
        self.assertEqual(expected_corruption_count(32, 0.05), 2)
        self.assertEqual(expected_corruption_count(48, 0.05), 2)
        self.assertEqual(expected_corruption_count(726, 0.30), 218)

    def test_prediction_evidence_is_split_for_github(self):
        prediction_files = discover_prediction_files(DATA)
        self.assertEqual(len(prediction_files), 12)
        self.assertFalse((DATA / "per_sample_predictions.csv").exists())
        oversized = [path.name for path in prediction_files if path.stat().st_size > MAX_PREFERRED_BYTES]
        self.assertEqual(oversized, [])

    def test_legacy_bernoulli_and_removed_comparator_files_are_archived(self):
        active_legacy_names = [
            "strengthened_merged_results.csv",
            "strengthened_aggregate.csv",
            "strengthened_degradation.csv",
            "strengthened_paired_statistics.csv",
            "distilbert_symmetric_results.csv",
            "distilbert_symmetric_aggregate.csv",
        ]
        still_active = [name for name in active_legacy_names if (DATA / name).exists()]
        self.assertEqual(still_active, [])

        archived = [
            "strengthened_merged_results_LEGACY_bernoulli.csv",
            "strengthened_aggregate_LEGACY_bernoulli.csv",
            "strengthened_degradation_LEGACY_bernoulli.csv",
            "strengthened_paired_statistics_LEGACY_bernoulli.csv",
            "distilbert_symmetric_results_LEGACY_bernoulli.csv",
            "distilbert_symmetric_aggregate_LEGACY_bernoulli.csv",
        ]
        missing_archives = [name for name in archived if not (DATA / name).exists()]
        self.assertEqual(missing_archives, [])

    def test_no_forbidden_terms_in_text_bundle(self):
        forbidden_terms = [
            "evo" + "scientist",
            "clau" + "de",
            "open" + "ai",
            "anthro" + "pic",
            "sk" + "-",
            "nsa" + "hub",
            "/home/" + "silver",
            "/users/" + "silver",
            "paper" + "-review",
            "run" + "_dir",
            "paper" + "25",
            "paper " + "25",
        ]
        offenders = []
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() not in {".png", ".pdf", ".pyc"}:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                for term in forbidden_terms:
                    if term in text:
                        offenders.append((str(path.relative_to(ROOT)), term))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
