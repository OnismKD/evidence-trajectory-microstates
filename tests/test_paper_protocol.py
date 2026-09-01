from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from evidence_microstates.adaptive import build_polarity_invariant_knn_graph
from evidence_microstates.evidence import normalize_topographies
from evidence_microstates.paper_protocol import (
    PaperProtocolSettings,
    build_paper_feature_matrix,
    paper_grid_readouts,
    paper_repeated_classification,
    select_paper_features,
)
from evidence_microstates.peaks import PeakSequence


class PaperProtocolTests(unittest.TestCase):
    def test_paper_graph_recomputes_selected_float32_edge_weights(self) -> None:
        maps = np.random.default_rng(9).normal(size=(200, 19)).astype(np.float32)
        actual, _ = build_polarity_invariant_knn_graph(
            maps,
            knn_fraction=0.03,
            weight_mode="abs_cosine",
        )
        x = normalize_topographies(maps).astype(np.float32)
        n, k = x.shape[0], round(0.03 * x.shape[0])
        tree = cKDTree(x)
        _, positive = tree.query(x, k=k + 1)
        _, negative = tree.query(-x, k=k + 1)
        rows, columns, values = [], [], []
        for row in range(n):
            neighbors = np.unique(np.r_[positive[row], negative[row]])
            neighbors = neighbors[neighbors != row]
            if neighbors.size > k:
                candidates = np.abs(x[row] @ x[neighbors].T)
                neighbors = neighbors[np.argsort(candidates)[-k:]]
            selected = np.abs(x[row] @ x[neighbors].T)
            rows.extend([row] * neighbors.size)
            columns.extend(neighbors.tolist())
            values.extend(selected.tolist())
        expected = sparse.coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()
        expected = expected.maximum(expected.T)
        expected.setdiag(0)
        expected.eliminate_zeros()
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.data, expected.data)

    def test_grid_uses_paper_feature_names(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int32)
        evidence = np.column_stack([0.8 - 0.6 * labels, 0.2 + 0.6 * labels])
        sequence = PeakSequence(
            maps=np.zeros((labels.size, 3), dtype=np.float32),
            indices=np.arange(labels.size),
            weights_sec=np.full(labels.size, 0.05),
            block_ids=np.zeros(labels.size, dtype=np.int32),
            sfreq=200.0,
            n_samples=80,
        )
        settings = PaperProtocolSettings(gamma_values=(1.0,), percentiles=(0.5,), min_duration_ms=0)
        result = paper_grid_readouts(sequence, labels, evidence, settings)
        self.assertIn("hard_global_mean_duration_ms", result)
        self.assertIn("traj_g1_p50_global_mean_duration_ms", result)
        self.assertIn("traj_g1_p50_hardany_null_lzc_norm", result)
        self.assertIn("traj_g1_p50_hardself_null_lzc_norm", result)

    def test_selection_and_fixed_cv_build_matched_feature_families(self) -> None:
        rng = np.random.default_rng(12)
        rows = []
        for subject in range(30):
            group = "AD" if subject < 15 else "HC"
            target = float(group == "AD")
            rows.append(
                {
                    "dataset": "ds004504",
                    "subject_id": f"sub-{subject:03d}",
                    "group": group,
                    "age": 65.0,
                    "sex": "",
                    "method": "subject_kmeans_K4_fullpeaks",
                    "algorithm": "kmeans",
                    "level": "subject_native",
                    "K_fixed": 4,
                    "K_found": 4,
                    "hard_global_mean_duration_ms": target + rng.normal(scale=0.7),
                    "hard_lzc_norm": target + rng.normal(scale=0.7),
                    "traj_g1_p50_global_mean_duration_ms": 2 * target + rng.normal(scale=0.5),
                    "traj_g1_p50_hardany_null_lzc_norm": 2 * target + rng.normal(scale=0.5),
                }
            )
        features = pd.DataFrame(rows)
        selection = select_paper_features(features)
        matrix = build_paper_feature_matrix(features, selection)
        summary, folds = paper_repeated_classification(
            matrix,
            PaperProtocolSettings(cv_splits=3, cv_repeats=1),
        )
        self.assertEqual(set(summary["feature_family"]), {"Hard", "Trajectory", "Combined"})
        self.assertEqual(summary.set_index("feature_family")["n_features"].to_dict(), {
            "Hard": 2,
            "Trajectory": 2,
            "Combined": 4,
        })
        self.assertEqual(folds.groupby("feature_family").size().to_dict(), {
            "Hard": 3,
            "Trajectory": 3,
            "Combined": 3,
        })


if __name__ == "__main__":
    unittest.main()
