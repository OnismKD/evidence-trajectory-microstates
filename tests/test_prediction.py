from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evidence_microstates.prediction import NestedCVSettings, nested_classification, nested_regression


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(21)
        target = np.repeat([0, 1], 30)
        age = np.linspace(20.0, 70.0, target.size) + rng.normal(scale=2.0, size=target.size)
        self.frame = pd.DataFrame(
            {
                "target": target,
                "age": age,
                "hard_1": 0.35 * target + rng.normal(size=target.size),
                "hard_2": 0.02 * age + rng.normal(size=target.size),
                "trajectory_1": 0.75 * target + rng.normal(size=target.size),
                "trajectory_2": 0.04 * age + rng.normal(size=target.size),
            }
        )
        self.feature_sets = {
            "hard": ["hard_1", "hard_2"],
            "trajectory": ["trajectory_1", "trajectory_2"],
            "combined": ["hard_1", "hard_2", "trajectory_1", "trajectory_2"],
        }
        self.settings = NestedCVSettings(outer_splits=3, repeats=1, inner_splits=2)

    def test_nested_classification_returns_paired_scores(self) -> None:
        summary, folds = nested_classification(self.frame, self.feature_sets, "target", self.settings)
        self.assertEqual(set(summary["feature_family"]), set(self.feature_sets))
        self.assertEqual(
            folds.groupby("feature_family").size().to_dict(), {name: 3 for name in self.feature_sets}
        )
        self.assertTrue(np.isfinite(folds["roc_auc"]).all())

    def test_nested_regression_returns_paired_scores(self) -> None:
        summary, folds = nested_regression(self.frame, self.feature_sets, "age", self.settings)
        self.assertEqual(set(summary["feature_family"]), set(self.feature_sets))
        self.assertEqual(
            folds.groupby("feature_family").size().to_dict(), {name: 3 for name in self.feature_sets}
        )
        self.assertTrue(np.isfinite(folds[["r2", "mae", "rmse"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
