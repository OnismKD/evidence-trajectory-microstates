from __future__ import annotations

import unittest

import numpy as np

from evidence_microstates.peaks import PeakSequence
from evidence_microstates.workflow import ReadoutSettings, analyze_two_level_fixed_k


class WorkflowTests(unittest.TestCase):
    def test_two_level_kmeans_smoke(self) -> None:
        rng = np.random.default_rng(7)
        canonical = rng.normal(size=(4, 8))
        sequences = []
        for _ in range(6):
            labels = rng.integers(0, 4, size=100)
            maps = canonical[labels] + rng.normal(scale=0.2, size=(100, 8))
            maps *= rng.choice([-1.0, 1.0], size=(100, 1))
            sequences.append(
                PeakSequence(
                    maps=maps,
                    indices=np.arange(100),
                    weights_sec=np.full(100, 0.04),
                    block_ids=np.zeros(100, dtype=np.int32),
                    sfreq=200.0,
                    n_samples=800,
                )
            )
        frame, templates, metadata = analyze_two_level_fixed_k(
            sequences,
            [f"sub-{idx:02d}" for idx in range(6)],
            "kmeans",
            4,
            ReadoutSettings(min_duration_ms=0),
            random_state=11,
        )
        self.assertEqual(frame.shape[0], 6)
        self.assertEqual(templates.shape, (4, 8))
        self.assertEqual(metadata["n_subjects"], 6)
        self.assertFalse(frame["hard_global_mean_duration_ms"].isna().any())
        self.assertFalse(frame["null_lzc_norm"].isna().any())


if __name__ == "__main__":
    unittest.main()
