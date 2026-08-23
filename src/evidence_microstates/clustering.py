"""Fixed-K subject-level and classic two-level topographic models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .evidence import EPS, normalize_topographies, spatial_evidence

FixedAlgorithm = Literal["kmeans", "aahc", "hmm"]


@dataclass
class FittedStateModel:
    templates: np.ndarray
    labels: np.ndarray
    evidence: np.ndarray
    algorithm: str
    metadata: dict[str, object]


def _first_principal_topography(maps: np.ndarray) -> np.ndarray:
    covariance = maps.T @ maps
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, np.argmax(values)]
    return vector / max(float(np.linalg.norm(vector)), EPS)


def fit_polarity_invariant_kmeans(
    maps: np.ndarray,
    n_states: int,
    *,
    n_init: int = 20,
    max_iter: int = 300,
    random_state: int = 42,
) -> FittedStateModel:
    """Modified K-means using absolute spatial correlation."""
    x = normalize_topographies(maps).astype(np.float32)
    if x.shape[0] < n_states:
        raise ValueError(f"Need at least {n_states} maps, received {x.shape[0]}")
    rng = np.random.default_rng(random_state)
    best: tuple[float, np.ndarray, np.ndarray, int, int] | None = None
    for init in range(int(n_init)):
        centers = x[rng.choice(x.shape[0], size=n_states, replace=False)].copy()
        labels = np.full(x.shape[0], -1, dtype=np.int32)
        for iteration in range(int(max_iter)):
            similarities = np.abs(x @ centers.T)
            new_labels = similarities.argmax(axis=1).astype(np.int32)
            if iteration and np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for state in range(n_states):
                members = x[labels == state]
                centers[state] = (
                    _first_principal_topography(members) if members.size else x[rng.integers(x.shape[0])]
                )
        score = float(np.mean(np.max(np.abs(x @ centers.T), axis=1) ** 2))
        candidate = (score, centers.copy(), labels.copy(), init, iteration + 1)
        if best is None or score > best[0]:
            best = candidate
    assert best is not None
    score, centers, _, best_init, iterations = best
    evidence = spatial_evidence(maps, centers)
    labels = evidence.argmax(axis=1).astype(np.int32)
    return FittedStateModel(
        templates=centers.astype(np.float32),
        labels=labels,
        evidence=evidence.astype(np.float32),
        algorithm="kmeans",
        metadata={"gev_proxy": score, "best_init": best_init, "n_iter": iterations},
    )


def _raw_mean_templates(maps: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    maps = np.asarray(maps, dtype=np.float64)
    normalized = normalize_topographies(maps)
    templates = np.zeros((n_states, maps.shape[1]), dtype=np.float64)
    for state in range(n_states):
        indices = np.flatnonzero(labels == state)
        if indices.size == 0:
            continue
        reference = normalized[indices[0]]
        signs = np.sign(normalized[indices] @ reference)
        signs[signs == 0] = 1.0
        templates[state] = np.mean(maps[indices] * signs[:, None], axis=0)
    return templates.astype(np.float32)


def fit_fixed_model(
    maps: np.ndarray,
    algorithm: FixedAlgorithm,
    n_states: int,
    *,
    channel_names: list[str] | None = None,
    random_state: int = 42,
    kmeans_n_init: int = 20,
    kmeans_max_iter: int = 300,
    hmm_n_init: int = 3,
    hmm_n_iter: int = 300,
    lengths: list[int] | None = None,
) -> FittedStateModel:
    """Fit one subject's fixed-K model.

    AAHC and HMM dependencies are imported lazily so the K-means quickstart
    remains lightweight.
    """
    if algorithm == "kmeans":
        return fit_polarity_invariant_kmeans(
            maps,
            n_states,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=random_state,
        )
    if algorithm == "aahc":
        try:
            import mne
            from pycrostates.cluster import AAHCluster
        except ImportError as exc:
            raise RuntimeError("AAHC requires `pip install -e '.[full]'`") from exc
        if channel_names is None:
            channel_names = [f"EEG{idx + 1}" for idx in range(maps.shape[1])]
        normalized = normalize_topographies(maps).T
        raw = mne.io.RawArray(
            normalized,
            mne.create_info(channel_names, sfreq=1.0, ch_types="eeg"),
            verbose="ERROR",
        )
        model = AAHCluster(n_clusters=n_states, normalize_input=False)
        model._ignore_polarity = True
        model.fit(raw, picks="eeg", verbose="ERROR")
        templates = model.cluster_centers_.astype(np.float32)
        evidence = spatial_evidence(maps, templates)
        labels = evidence.argmax(axis=1).astype(np.int32)
        return FittedStateModel(
            templates,
            labels,
            evidence.astype(np.float32),
            "aahc",
            {"GEV": float(model.GEV_)},
        )
    if algorithm == "hmm":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise RuntimeError("HMM requires `pip install -e '.[full]'`") from exc
        x = normalize_topographies(maps)
        anchors = np.argmax(np.abs(x), axis=1)
        signs = np.sign(x[np.arange(x.shape[0]), anchors])
        signs[signs == 0] = 1.0
        x = x * signs[:, None]
        lengths = lengths or [x.shape[0]]
        best: tuple[float, object] | None = None
        for init in range(int(hmm_n_init)):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=int(hmm_n_iter),
                random_state=random_state + init,
                min_covar=1e-3,
            )
            model.fit(x, lengths)
            likelihood = float(model.score(x, lengths))
            if best is None or likelihood > best[0]:
                best = (likelihood, model)
        assert best is not None
        likelihood, model = best
        labels = model.predict(x, lengths).astype(np.int32)
        posterior = model.predict_proba(x, lengths).astype(np.float32)
        templates = _raw_mean_templates(maps, labels, n_states)
        return FittedStateModel(
            templates,
            labels,
            posterior,
            "hmm",
            {"log_likelihood": likelihood, "polarity_mode": "maxabs"},
        )
    raise ValueError(f"Unknown fixed-K algorithm: {algorithm}")


def fit_two_level_templates(
    subject_maps: list[np.ndarray],
    algorithm: FixedAlgorithm,
    n_states: int,
    *,
    channel_names: list[str] | None = None,
    random_state: int = 42,
    subject_n_init: int = 20,
    group_n_init: int = 100,
) -> tuple[np.ndarray, list[FittedStateModel], dict[str, object]]:
    """Classic two-level model: subject templates, then group templates."""
    subject_models = [
        fit_fixed_model(
            maps,
            algorithm,
            n_states,
            channel_names=channel_names,
            random_state=random_state + index,
            kmeans_n_init=subject_n_init,
        )
        for index, maps in enumerate(subject_maps)
    ]
    first_level_templates = np.concatenate([model.templates for model in subject_models], axis=0)
    # HMM states are temporal at level one; their unordered topographies are
    # aligned with polarity-invariant K-means at level two.
    group_algorithm: FixedAlgorithm = "kmeans" if algorithm == "hmm" else algorithm
    group_model = fit_fixed_model(
        first_level_templates,
        group_algorithm,
        n_states,
        channel_names=channel_names,
        random_state=random_state,
        kmeans_n_init=group_n_init,
    )
    return (
        group_model.templates,
        subject_models,
        {
            "template_learning": "subject_templates_then_second_level_group_clustering",
            "first_level_algorithm": algorithm,
            "second_level_algorithm": group_algorithm,
            "n_subjects": len(subject_models),
        },
    )


def backfit_templates(maps: np.ndarray, templates: np.ndarray) -> FittedStateModel:
    evidence = spatial_evidence(maps, templates)
    labels = evidence.argmax(axis=1).astype(np.int32)
    return FittedStateModel(
        templates=np.asarray(templates, dtype=np.float32),
        labels=labels,
        evidence=evidence.astype(np.float32),
        algorithm="template_backfit",
        metadata={"comparison": "absolute_spatial_correlation"},
    )
