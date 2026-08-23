"""Matched hard-label and evidence-trajectory temporal descriptors."""

from __future__ import annotations

import numpy as np

from .complexity import boundary_aware_lzc
from .evidence import sharpen_evidence, weighted_quantile
from .temporal import contiguous_segments, merge_short_segments, true_run_durations


def hard_label_descriptors(
    labels: np.ndarray,
    weights_sec: np.ndarray,
    *,
    n_states: int | None = None,
    block_ids: np.ndarray | None = None,
    min_duration_ms: float = 30.0,
    prefix: str = "hard",
) -> dict[str, float]:
    """Classic duration, occurrence, coverage, and LZC descriptors."""
    labels = merge_short_segments(
        labels,
        weights_sec,
        min_duration_sec=float(min_duration_ms) / 1000.0,
        block_ids=block_ids,
    )
    weights = np.asarray(weights_sec, dtype=np.float64)
    segments = contiguous_segments(labels, weights, block_ids)
    durations = np.asarray([item[3] for item in segments], dtype=np.float64)
    total = float(weights.sum())
    states = int(n_states if n_states is not None else (labels.max() + 1 if labels.size else 0))
    raw_lzc, norm_lzc = boundary_aware_lzc(labels, block_ids)
    out: dict[str, float] = {
        f"{prefix}_global_mean_duration_ms": float(durations.mean() * 1000.0) if durations.size else np.nan,
        f"{prefix}_global_median_duration_ms": float(np.median(durations) * 1000.0)
        if durations.size
        else np.nan,
        f"{prefix}_global_occurrence_per_sec": float(durations.size / total) if total > 0 else np.nan,
        f"{prefix}_lzc_raw": float(raw_lzc),
        f"{prefix}_lzc_norm": float(norm_lzc),
    }
    for state in range(states):
        state_durations = np.asarray([item[3] for item in segments if item[0] == state], dtype=np.float64)
        tag = f"{prefix}_state{state + 1}"
        out[f"{tag}_mean_duration_ms"] = (
            float(state_durations.mean() * 1000.0) if state_durations.size else np.nan
        )
        out[f"{tag}_occurrence_per_sec"] = float(state_durations.size / total) if total > 0 else np.nan
        out[f"{tag}_coverage"] = float(weights[labels == state].sum() / total) if total > 0 else np.nan
    return out


def trajectory_descriptors(
    evidence: np.ndarray,
    weights_sec: np.ndarray,
    *,
    gamma: float,
    percentile: float,
    block_ids: np.ndarray | None = None,
    min_duration_ms: float = 30.0,
    prefix: str = "trajectory",
) -> dict[str, float]:
    """Duration and occurrence of statewise high-evidence episodes."""
    trajectory = sharpen_evidence(evidence, gamma)
    weights = np.asarray(weights_sec, dtype=np.float64)
    total = float(weights.sum())
    all_durations: list[float] = []
    active_count = np.zeros(trajectory.shape[0], dtype=np.int16)
    out: dict[str, float] = {}
    for state in range(trajectory.shape[1]):
        cutoff = weighted_quantile(trajectory[:, state], weights, percentile)
        active = trajectory[:, state] > cutoff
        active = merge_short_segments(
            active.astype(np.int32),
            weights,
            min_duration_sec=float(min_duration_ms) / 1000.0,
            block_ids=block_ids,
        ).astype(bool)
        durations = true_run_durations(active, weights, block_ids)
        active_count += active.astype(np.int16)
        all_durations.extend(durations.tolist())
        tag = f"{prefix}_state{state + 1}"
        out[f"{tag}_mean_duration_ms"] = float(durations.mean() * 1000.0) if durations.size else np.nan
        out[f"{tag}_occurrence_per_sec"] = float(durations.size / total) if total > 0 else np.nan
        out[f"{tag}_threshold"] = float(cutoff)
    durations = np.asarray(all_durations, dtype=np.float64)
    out[f"{prefix}_global_mean_duration_ms"] = float(durations.mean() * 1000.0) if durations.size else np.nan
    out[f"{prefix}_global_median_duration_ms"] = (
        float(np.median(durations) * 1000.0) if durations.size else np.nan
    )
    out[f"{prefix}_global_occurrence_per_sec"] = float(durations.size / total) if total > 0 else np.nan
    out[f"{prefix}_null_fraction"] = float(weights[active_count == 0].sum() / total) if total > 0 else np.nan
    out[f"{prefix}_overlap_fraction"] = (
        float(weights[active_count > 1].sum() / total) if total > 0 else np.nan
    )
    return out
