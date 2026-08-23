"""Predeclared gamma/percentile readout scans from a fixed state model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .complexity import boundary_aware_lzc, evidence_quantile_self_null_sequence, percentile_winner_labels
from .descriptors import trajectory_descriptors
from .peaks import PeakSequence
from .temporal import merge_short_segments


def scan_duration_grid(
    sequence: PeakSequence,
    evidence: np.ndarray,
    *,
    gammas: list[float],
    percentiles: list[float],
    min_duration_ms: float = 30.0,
) -> pd.DataFrame:
    """Return one row per ``(gamma, percentile)`` duration readout."""
    rows = []
    for gamma in gammas:
        for percentile in percentiles:
            features = trajectory_descriptors(
                evidence,
                sequence.weights_sec,
                gamma=float(gamma),
                percentile=float(percentile),
                block_ids=sequence.block_ids,
                min_duration_ms=min_duration_ms,
            )
            rows.append({"gamma": float(gamma), "percentile": float(percentile), **features})
    return pd.DataFrame(rows)


def scan_null_lzc_grid(
    sequence: PeakSequence,
    evidence: np.ndarray,
    *,
    gammas: list[float],
    percentiles: list[float],
    min_duration_ms: float = 30.0,
) -> pd.DataFrame:
    """Return the evidence-quantile-self null-LZC sensitivity grid."""
    labels, _ = percentile_winner_labels(evidence, sequence.weights_sec)
    labels = merge_short_segments(
        labels,
        sequence.weights_sec,
        min_duration_ms / 1000.0,
        sequence.block_ids,
    )
    total = float(sequence.weights_sec.sum())
    rows = []
    for gamma in gammas:
        for percentile in percentiles:
            null_sequence, keep = evidence_quantile_self_null_sequence(
                evidence,
                sequence.weights_sec,
                labels,
                float(percentile),
                float(gamma),
            )
            raw, normalized = boundary_aware_lzc(null_sequence, sequence.block_ids)
            rows.append(
                {
                    "gamma": float(gamma),
                    "percentile": float(percentile),
                    "null_lzc_raw": int(raw),
                    "null_lzc_norm": float(normalized),
                    "null_fraction": (
                        float(sequence.weights_sec[~keep].sum() / total) if total > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)
