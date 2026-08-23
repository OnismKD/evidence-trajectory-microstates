"""Boundary-aware operations on peak-indexed state sequences."""

from __future__ import annotations

import numpy as np


def boundary_slices(length: int, block_ids: np.ndarray | None = None) -> list[tuple[int, int]]:
    if block_ids is None:
        return [(0, int(length))] if length else []
    block_ids = np.asarray(block_ids)
    if block_ids.shape != (int(length),):
        raise ValueError("block_ids must have one entry per time point")
    changes = np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1
    starts, stops = np.r_[0, changes], np.r_[changes, length]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def contiguous_segments(
    labels: np.ndarray,
    weights: np.ndarray,
    block_ids: np.ndarray | None = None,
) -> list[tuple[int, int, int, float]]:
    labels = np.asarray(labels, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if labels.shape != weights.shape:
        raise ValueError("labels and weights must have equal length")
    segments: list[tuple[int, int, int, float]] = []
    for block_start, block_stop in boundary_slices(labels.size, block_ids):
        start = block_start
        for idx in range(block_start + 1, block_stop):
            if labels[idx] != labels[idx - 1]:
                segments.append((int(labels[start]), start, idx, float(weights[start:idx].sum())))
                start = idx
        segments.append((int(labels[start]), start, block_stop, float(weights[start:block_stop].sum())))
    return segments


def true_run_durations(
    indicator: np.ndarray,
    weights: np.ndarray,
    block_ids: np.ndarray | None = None,
) -> np.ndarray:
    indicator = np.asarray(indicator, dtype=bool)
    as_labels = indicator.astype(np.int32)
    segments = contiguous_segments(as_labels, weights, block_ids)
    return np.asarray([duration for label, _, _, duration in segments if label == 1], dtype=np.float64)


def merge_short_segments(
    labels: np.ndarray,
    weights: np.ndarray,
    min_duration_sec: float = 0.03,
    block_ids: np.ndarray | None = None,
    passes: int = 2,
) -> np.ndarray:
    """Merge runs shorter than ``min_duration_sec`` within each block.

    This reproduces the analysis implementation: an interior short run is
    assigned to its preceding state; a leading run is assigned to the next
    state. The operation is deterministic and repeated for at most ``passes``.
    """
    out = np.asarray(labels, dtype=np.int32).copy()
    weights = np.asarray(weights, dtype=np.float64)
    if out.size == 0 or min_duration_sec <= 0:
        return out
    for _ in range(int(passes)):
        changed = False
        for block_start, block_stop in boundary_slices(out.size, block_ids):
            local = out[block_start:block_stop]
            local_weights = weights[block_start:block_stop]
            for _, start, stop, duration in contiguous_segments(local, local_weights):
                if duration >= min_duration_sec:
                    continue
                if start > 0:
                    fill = local[start - 1]
                elif stop < local.size:
                    fill = local[stop]
                else:
                    continue
                if np.any(local[start:stop] != fill):
                    local[start:stop] = fill
                    changed = True
            out[block_start:block_stop] = local
        if not changed:
            break
    return out
