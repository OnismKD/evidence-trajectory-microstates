"""Lempel-Ziv complexity and confidence-aware symbolic sequences."""

from __future__ import annotations

import math

import numpy as np

from .evidence import percentile_winner_labels, sharpen_evidence, weighted_quantile
from .temporal import boundary_slices


def lz76_complexity(sequence: np.ndarray) -> int:
    """LZ76 exhaustive-history complexity for an arbitrary symbolic sequence."""
    raw = np.asarray(sequence).tolist()
    if not raw:
        return 0
    code = {value: idx for idx, value in enumerate(dict.fromkeys(raw))}
    s = [code[value] for value in raw]
    n = len(s)
    if n == 1:
        return 1
    i, k, length, k_max, complexity = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[length + k - 1]:
            k += 1
            if length + k > n:
                complexity += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == length:
                complexity += 1
                length += k_max
                if length >= n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return int(complexity)


def normalized_lzc(sequence: np.ndarray) -> tuple[int, float]:
    seq = np.asarray(sequence)
    raw = lz76_complexity(seq)
    n = int(seq.size)
    alphabet = len(set(seq.tolist()))
    if n <= 1 or alphabet <= 1:
        return raw, 0.0
    return raw, float(raw * math.log(n) / (n * math.log(alphabet)))


def boundary_aware_lzc(sequence: np.ndarray, block_ids: np.ndarray | None = None) -> tuple[int, float]:
    seq = np.asarray(sequence)
    if block_ids is None:
        return normalized_lzc(seq)
    raw_total, norm_sum, n_total = 0, 0.0, 0
    for start, stop in boundary_slices(seq.size, block_ids):
        raw, norm = normalized_lzc(seq[start:stop])
        n = stop - start
        raw_total += raw
        norm_sum += norm * n
        n_total += n
    return int(raw_total), float(norm_sum / n_total) if n_total else 0.0


def percentile_max_null_sequence(
    evidence: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the percentile-rank winner and emit null below ``threshold``."""
    labels, ranks = percentile_winner_labels(evidence, weights)
    keep = np.max(ranks, axis=1) > float(threshold)
    sequence = labels.copy()
    sequence[~keep] = evidence.shape[1]
    return sequence, keep


def percentile_self_null_sequence(
    evidence: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a supplied label only when its statewise percentile is high."""
    _, ranks = percentile_winner_labels(evidence, weights)
    labels = np.asarray(labels, dtype=np.int32)
    keep = ranks[np.arange(labels.size), labels] > float(threshold)
    sequence = labels.copy()
    sequence[~keep] = evidence.shape[1]
    return sequence, keep


def evidence_quantile_self_null_sequence(
    evidence: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact configurable form used by the latest percentile-self sensitivity scan."""
    trajectory = sharpen_evidence(evidence, gamma)
    labels = np.asarray(labels, dtype=np.int32)
    cutoffs = np.asarray(
        [weighted_quantile(trajectory[:, state], weights, threshold) for state in range(trajectory.shape[1])]
    )
    keep = trajectory[np.arange(labels.size), labels] > cutoffs[labels]
    sequence = labels.copy()
    sequence[~keep] = trajectory.shape[1]
    return sequence, keep
