import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_disparities import disparity_summary, prepare_candidates, segment_candidates
from generate_sample_data import generate


class DisparityAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate(rows=240)
        cls.clean = prepare_candidates(cls.raw)

    def test_preparation_creates_outcomes(self):
        self.assertIn("credential_ready", self.clean.columns)
        self.assertTrue(self.clean["subjects_passed"].between(0, 4).all())

    def test_regional_summary_preserves_population(self):
        summary = disparity_summary(self.clean, "region")
        self.assertEqual(int(summary["candidates"].sum()), len(self.clean))

    def test_segmentation_assigns_every_complete_candidate(self):
        segmented, profile = segment_candidates(self.clean)
        self.assertEqual(len(segmented), len(self.clean))
        self.assertEqual(profile["segment"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
