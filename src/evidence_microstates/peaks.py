"""GFP peak extraction and midpoint interval weighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PeakSequence:
    """Time-ordered GFP-peak topographies for one recording."""

    maps: np.ndarray
    indices: np.ndarray
    weights_sec: np.ndarray
    block_ids: np.ndarray
    sfreq: float
    n_samples: int

    def __post_init__(self) -> None:
        n = int(np.asarray(self.indices).size)
        if np.asarray(self.maps).ndim != 2 or np.asarray(self.maps).shape[0] != n:
            raise ValueError("maps must have one row per peak")
        if np.asarray(self.weights_sec).shape != (n,) or np.asarray(self.block_ids).shape != (n,):
            raise ValueError("weights_sec and block_ids must have one value per peak")
        if n and (not np.all(np.asarray(self.weights_sec) > 0)):
            raise ValueError("Every peak must represent a positive interval")


def global_field_power(data: np.ndarray) -> np.ndarray:
    """Across-channel standard deviation at each sample."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("data must have shape (channels, samples)")
    return np.std(x, axis=0)


def midpoint_weights(peaks: np.ndarray, n_samples: int, sfreq: float) -> np.ndarray:
    """Assign each peak the interval bounded by adjacent temporal midpoints."""
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("peaks must be strictly increasing")
    if peaks[0] < 0 or peaks[-1] >= int(n_samples):
        raise ValueError("peak indices fall outside the recording")
    if peaks.size == 1:
        return np.asarray([float(n_samples) / float(sfreq)])
    edges = np.empty(peaks.size + 1, dtype=np.float64)
    edges[0], edges[-1] = 0.0, float(n_samples)
    edges[1:-1] = (peaks[:-1] + peaks[1:]) / 2.0
    return np.diff(edges) / float(sfreq)


def extract_gfp_peak_sequence(
    data: np.ndarray,
    sfreq: float,
    *,
    min_peak_distance_ms: float = 10.0,
    blocks: list[tuple[int, int]] | None = None,
) -> PeakSequence:
    """Extract all valid GFP peaks, preserving discontinuity boundaries.

    Blocks are half-open sample intervals. Midpoint intervals are computed
    independently within each block, so a peak never represents time across a
    recording discontinuity.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("data must have shape (channels, samples)")
    n_samples = x.shape[1]
    if blocks is None:
        blocks = [(0, n_samples)]
    distance = max(1, round(float(min_peak_distance_ms) * float(sfreq) / 1000.0))

    all_indices: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_blocks: list[np.ndarray] = []
    for block_id, (start, stop) in enumerate(blocks):
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= n_samples:
            raise ValueError(f"Invalid block {(start, stop)} for {n_samples} samples")
        gfp = global_field_power(x[:, start:stop])
        local, _ = find_peaks(gfp, distance=distance)
        local = local[np.isfinite(x[:, start + local]).all(axis=0)]
        if local.size == 0:
            continue
        all_indices.append(local + start)
        all_weights.append(midpoint_weights(local, stop - start, sfreq))
        all_blocks.append(np.full(local.size, block_id, dtype=np.int32))

    if not all_indices:
        return PeakSequence(
            maps=np.empty((0, x.shape[0]), dtype=np.float32),
            indices=np.empty(0, dtype=np.int64),
            weights_sec=np.empty(0, dtype=np.float64),
            block_ids=np.empty(0, dtype=np.int32),
            sfreq=float(sfreq),
            n_samples=n_samples,
        )
    indices = np.concatenate(all_indices).astype(np.int64)
    return PeakSequence(
        maps=x[:, indices].T.astype(np.float32),
        indices=indices,
        weights_sec=np.concatenate(all_weights).astype(np.float64),
        block_ids=np.concatenate(all_blocks).astype(np.int32),
        sfreq=float(sfreq),
        n_samples=n_samples,
    )
