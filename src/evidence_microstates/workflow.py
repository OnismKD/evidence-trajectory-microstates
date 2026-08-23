"""High-level functions shared by the command-line workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adaptive import AdaptiveAlgorithm, fit_adaptive_model
from .clustering import FixedAlgorithm, backfit_templates, fit_fixed_model, fit_two_level_templates
from .complexity import (
    boundary_aware_lzc,
    evidence_quantile_self_null_sequence,
    percentile_max_null_sequence,
    percentile_winner_labels,
)
from .descriptors import hard_label_descriptors, trajectory_descriptors
from .peaks import PeakSequence
from .temporal import merge_short_segments


@dataclass(frozen=True)
class ReadoutSettings:
    gamma: float = 1.0
    duration_percentile: float = 0.50
    lzc_percentile: float = 0.80
    min_duration_ms: float = 30.0
    null_mode: str = "evidence_quantile_self"


def matched_readouts(
    sequence: PeakSequence,
    labels: np.ndarray,
    evidence: np.ndarray,
    settings: ReadoutSettings,
) -> dict[str, float]:
    """Compute matched descriptors from one state model and peak sequence."""
    labels = merge_short_segments(
        labels,
        sequence.weights_sec,
        settings.min_duration_ms / 1000.0,
        sequence.block_ids,
    )
    hard = hard_label_descriptors(
        labels,
        sequence.weights_sec,
        n_states=evidence.shape[1],
        block_ids=sequence.block_ids,
        min_duration_ms=0.0,
    )
    trajectory = trajectory_descriptors(
        evidence,
        sequence.weights_sec,
        gamma=settings.gamma,
        percentile=settings.duration_percentile,
        block_ids=sequence.block_ids,
        min_duration_ms=settings.min_duration_ms,
    )
    percentile_labels, _ = percentile_winner_labels(evidence, sequence.weights_sec)
    percentile_labels = merge_short_segments(
        percentile_labels,
        sequence.weights_sec,
        settings.min_duration_ms / 1000.0,
        sequence.block_ids,
    )
    if settings.null_mode == "percentile_max":
        null_sequence, keep = percentile_max_null_sequence(
            evidence, sequence.weights_sec, settings.lzc_percentile
        )
    elif settings.null_mode == "evidence_quantile_self":
        null_sequence, keep = evidence_quantile_self_null_sequence(
            evidence,
            sequence.weights_sec,
            percentile_labels,
            settings.lzc_percentile,
            settings.gamma,
        )
    else:
        raise ValueError(f"Unknown null_mode: {settings.null_mode}")
    raw, norm = boundary_aware_lzc(null_sequence, sequence.block_ids)
    total = float(sequence.weights_sec.sum())
    null = {
        "null_lzc_raw": float(raw),
        "null_lzc_norm": float(norm),
        "null_fraction": float(sequence.weights_sec[~keep].sum() / total) if total > 0 else np.nan,
    }
    return {**hard, **trajectory, **null}


def analyze_two_level_fixed_k(
    sequences: list[PeakSequence],
    subject_ids: list[str],
    algorithm: FixedAlgorithm,
    n_states: int,
    settings: ReadoutSettings,
    *,
    channel_names: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    templates, _, metadata = fit_two_level_templates(
        [sequence.maps for sequence in sequences],
        algorithm,
        n_states,
        channel_names=channel_names,
        random_state=random_state,
    )
    rows = []
    for subject_id, sequence in zip(subject_ids, sequences):
        model = backfit_templates(sequence.maps, templates)
        rows.append(
            {
                "subject_id": subject_id,
                "algorithm": algorithm,
                "level": "group_two_level",
                "K": n_states,
                **matched_readouts(sequence, model.labels, model.evidence, settings),
            }
        )
    return pd.DataFrame(rows), templates, metadata


def analyze_subject_native_fixed_k(
    sequences: list[PeakSequence],
    subject_ids: list[str],
    algorithm: FixedAlgorithm,
    n_states: int,
    settings: ReadoutSettings,
    *,
    channel_names: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for index, (subject_id, sequence) in enumerate(zip(subject_ids, sequences)):
        lengths = [int(np.sum(sequence.block_ids == block)) for block in np.unique(sequence.block_ids)]
        model = fit_fixed_model(
            sequence.maps,
            algorithm,
            n_states,
            channel_names=channel_names,
            random_state=random_state + index,
            lengths=lengths,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "algorithm": algorithm,
                "level": "subject_native",
                "K": n_states,
                **matched_readouts(sequence, model.labels, model.evidence, settings),
            }
        )
    return pd.DataFrame(rows)


def analyze_subject_adaptive(
    sequences: list[PeakSequence],
    subject_ids: list[str],
    algorithm: AdaptiveAlgorithm,
    settings: ReadoutSettings,
    *,
    random_state: int = 42,
    knn_fraction: float = 0.01,
) -> pd.DataFrame:
    rows = []
    for index, (subject_id, sequence) in enumerate(zip(subject_ids, sequences)):
        model = fit_adaptive_model(
            sequence.maps,
            algorithm,
            random_state=random_state + index,
            knn_fraction=knn_fraction,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "algorithm": algorithm,
                "level": "subject_adaptive",
                "K": model.templates.shape[0],
                **matched_readouts(sequence, model.labels, model.evidence, settings),
            }
        )
    return pd.DataFrame(rows)
