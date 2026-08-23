"""Matched template evidence and percentile transformations."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize_topographies(maps: np.ndarray) -> np.ndarray:
    """Spatially demean and unit-normalize rows."""
    x = np.asarray(maps, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("topographies must be a two-dimensional array")
    x = x - np.mean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > EPS)


def spatial_evidence(maps: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Polarity-invariant absolute spatial correlation to each template."""
    x = normalize_topographies(maps)
    t = normalize_topographies(templates)
    if np.any(np.linalg.norm(t, axis=1) <= EPS):
        raise ValueError("At least one template is spatially constant")
    return np.abs(x @ t.T)


def sharpen_evidence(evidence: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Convert non-negative similarities into row-normalized trajectories."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    e = np.maximum(np.asarray(evidence, dtype=np.float64), 0.0)
    powered = np.power(e, float(gamma))
    denom = powered.sum(axis=1, keepdims=True)
    out = np.divide(powered, denom, out=np.zeros_like(powered), where=denom > 0)
    zero = np.isclose(out.sum(axis=1), 0.0)
    if np.any(zero):
        out[zero] = 1.0 / out.shape[1]
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return float("nan")
    v, w = v[ok], w[ok]
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cumulative = np.cumsum(w)
    return float(np.interp(float(quantile) * cumulative[-1], cumulative, v, left=v[0], right=v[-1]))


def weighted_percentile_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted empirical percentile midranks in ``[0, 1]``."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros(v.shape, dtype=np.float64)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    order = idx[np.argsort(v[idx], kind="mergesort")]
    total = float(w[order].sum())
    start, cumulative = 0, 0.0
    while start < order.size:
        stop = start + 1
        while stop < order.size and v[order[stop]] == v[order[start]]:
            stop += 1
        mass = float(w[order[start:stop]].sum())
        out[order[start:stop]] = (cumulative + 0.5 * mass) / total
        cumulative += mass
        start = stop
    return out


def percentile_rank_matrix(evidence: np.ndarray, weights: np.ndarray) -> np.ndarray:
    e = np.asarray(evidence, dtype=np.float64)
    return np.column_stack([weighted_percentile_ranks(e[:, k], weights) for k in range(e.shape[1])])


def percentile_winner_labels(evidence: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = percentile_rank_matrix(evidence, weights)
    return np.argmax(ranks, axis=1).astype(np.int32), ranks
