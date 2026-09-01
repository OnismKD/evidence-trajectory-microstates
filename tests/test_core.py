from __future__ import annotations

import unittest

import numpy as np

from evidence_microstates.complexity import (
    boundary_aware_lzc,
    lz76_complexity,
    normalized_lzc,
    percentile_max_null_sequence,
)
from evidence_microstates.descriptors import hard_label_descriptors, trajectory_descriptors
from evidence_microstates.evidence import sharpen_evidence, spatial_evidence, weighted_percentile_ranks
from evidence_microstates.peaks import global_field_power, midpoint_weights
from evidence_microstates.preprocessing import preprocess_eeg, preprocess_paper_ds004504_raw
from evidence_microstates.temporal import contiguous_segments, merge_short_segments


class CoreTests(unittest.TestCase):
    def test_preprocessing_returns_car_alpha_signal(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.normal(size=(4, 1000))
        filtered, sfreq = preprocess_eeg(data, 200.0, target_sfreq=100.0)
        self.assertEqual(filtered.shape, (4, 500))
        self.assertEqual(sfreq, 100.0)
        np.testing.assert_allclose(filtered.mean(axis=0), 0.0, atol=1e-6)

    def test_paper_preprocessing_profile_is_explicit_and_car_referenced(self) -> None:
        import mne

        rng = np.random.default_rng(13)
        raw = mne.io.RawArray(
            rng.normal(size=(4, 1000)),
            mne.create_info(["A", "B", "C", "D"], 200.0, "eeg"),
            verbose="ERROR",
        )
        filtered, sfreq = preprocess_paper_ds004504_raw(raw)
        self.assertEqual(sfreq, 200.0)
        self.assertEqual(filtered.shape, (4, 1000))
        np.testing.assert_allclose(filtered.mean(axis=0), 0.0, atol=1e-6)

    def test_midpoint_weights_cover_the_recording(self) -> None:
        weights = midpoint_weights(np.asarray([10, 30, 70]), n_samples=100, sfreq=100.0)
        np.testing.assert_allclose(weights, [0.2, 0.3, 0.5])
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_gfp_precision_can_match_the_frozen_paper_cache(self) -> None:
        data = np.asarray(
            [[1.0, 1.0 + 1e-7, -1.0], [-1.0, -1.0, 1.0]],
            dtype=np.float32,
        )
        gfp = global_field_power(data, dtype=np.float32)
        self.assertEqual(gfp.dtype, np.float32)
        np.testing.assert_array_equal(gfp, np.std(data, axis=0))

    def test_spatial_evidence_is_polarity_invariant(self) -> None:
        maps = np.asarray([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
        template = np.asarray([[1.0, 0.0, -1.0]])
        evidence = spatial_evidence(maps, template)
        np.testing.assert_allclose(evidence[:, 0], 1.0)
        second_state = np.full((evidence.shape[0], 1), 0.5)
        np.testing.assert_allclose(sharpen_evidence(np.c_[evidence, second_state], 2.0).sum(axis=1), 1.0)

    def test_weighted_percentile_ties_receive_midrank(self) -> None:
        ranks = weighted_percentile_ranks(np.asarray([1.0, 1.0, 2.0]), np.asarray([1.0, 1.0, 2.0]))
        np.testing.assert_allclose(ranks, [0.25, 0.25, 0.75])

    def test_boundary_aware_segments_do_not_join_blocks(self) -> None:
        labels = np.asarray([0, 0, 0, 0])
        weights = np.ones(4)
        blocks = np.asarray([0, 0, 1, 1])
        segments = contiguous_segments(labels, weights, blocks)
        self.assertEqual(len(segments), 2)
        self.assertEqual([segment[3] for segment in segments], [2.0, 2.0])

    def test_short_segment_merging_matches_analysis_rule(self) -> None:
        labels = np.asarray([0, 1, 0])
        smoothed = merge_short_segments(labels, np.asarray([0.10, 0.01, 0.10]), 0.03)
        np.testing.assert_array_equal(smoothed, [0, 0, 0])

    def test_lzc_is_finite_and_boundary_aware(self) -> None:
        sequence = np.asarray([0, 1, 0, 1, 2, 2, 1, 0])
        raw, normalized = normalized_lzc(sequence)
        self.assertEqual(raw, lz76_complexity(sequence))
        self.assertGreater(raw, 0)
        self.assertTrue(np.isfinite(normalized))
        block_raw, block_norm = boundary_aware_lzc(sequence, np.asarray([0] * 4 + [1] * 4))
        self.assertGreater(block_raw, 0)
        self.assertTrue(np.isfinite(block_norm))

    def test_percentile_null_sequence_uses_extra_symbol(self) -> None:
        evidence = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
        sequence, keep = percentile_max_null_sequence(evidence, np.ones(4), 0.70)
        self.assertTrue(np.any(~keep))
        self.assertTrue(np.all(sequence[~keep] == 2))

    def test_matched_descriptors_return_expected_fields(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 0])
        weights = np.full(6, 0.05)
        hard = hard_label_descriptors(labels, weights, n_states=2, min_duration_ms=0)
        evidence = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2], [0.9, 0.1]])
        trajectory = trajectory_descriptors(evidence, weights, gamma=1.0, percentile=0.5, min_duration_ms=0)
        self.assertAlmostEqual(hard["hard_global_mean_duration_ms"], 100.0)
        self.assertIn("trajectory_global_mean_duration_ms", trajectory)
        self.assertIn("trajectory_state1_occurrence_per_sec", trajectory)


if __name__ == "__main__":
    unittest.main()
