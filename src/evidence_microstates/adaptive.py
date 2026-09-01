"""Subject-adaptive Leiden and Infomap topographic state models."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .clustering import FittedStateModel
from .evidence import normalize_topographies, spatial_evidence

AdaptiveAlgorithm = Literal["leiden", "infomap"]
GraphWeightMode = Literal["gaussian_distance", "abs_cosine"]
AdaptiveEvidenceMode = Literal["template_similarity", "community_affinity"]


def build_polarity_invariant_knn_graph(
    maps: np.ndarray,
    *,
    knn_fraction: float = 0.01,
    knn_min: int | None = None,
    weight_mode: GraphWeightMode = "gaussian_distance",
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    x = normalize_topographies(maps).astype(np.float32)
    n = x.shape[0]
    if n < 2:
        raise ValueError("At least two peak maps are required")
    k = round(knn_fraction * n)
    if knn_min is not None:
        k = max(k, int(knn_min))
    k = min(max(k, 1), n - 1)
    tree = cKDTree(x)
    _, positive = tree.query(x, k=min(k + 1, n))
    _, negative = tree.query(-x, k=min(k + 1, n))
    rows: list[int] = []
    columns: list[int] = []
    similarities: list[float] = []
    for row in range(n):
        neighbors = np.unique(np.r_[np.atleast_1d(positive[row]), np.atleast_1d(negative[row])])
        neighbors = neighbors[neighbors != row]
        if neighbors.size > k:
            candidate_values = np.abs(x[row] @ x[neighbors].T)
            neighbors = neighbors[np.argsort(candidate_values)[-k:]]
        # Preserve the paper implementation's operation order. Recomputing the
        # selected edge weights matters at float32 precision for Infomap.
        values = np.abs(x[row] @ x[neighbors].T)
        rows.extend([row] * neighbors.size)
        columns.extend(neighbors.tolist())
        similarities.extend(values.tolist())
    similarities_array = np.asarray(similarities)
    if weight_mode == "abs_cosine":
        sigma = np.nan
        values = similarities_array
    elif weight_mode == "gaussian_distance":
        distances = 1.0 - similarities_array
        positive_distances = distances[distances > 0]
        sigma = float(np.median(positive_distances)) if positive_distances.size else 1.0
        values = np.exp(-(distances**2) / (2.0 * sigma**2 + 1e-12))
    else:  # pragma: no cover - guarded by the public type and CLI choices
        raise ValueError(f"Unknown graph weight mode: {weight_mode}")
    graph = sparse.coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph, {
        "k": float(k),
        "n_edges": float(graph.nnz // 2),
        "sigma": sigma,
        "weight_mode": weight_mode,
    }


def community_affinity_evidence(
    graph: sparse.csr_matrix, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-normalized affinity to each detected community.

    This is the adaptive evidence definition used for the paper's subject-level
    predictive analysis. It intentionally differs from similarity to a single
    community-average template: every neighboring peak contributes evidence to
    its community.
    """
    labels = np.asarray(labels, dtype=np.int32)
    states = np.unique(labels)
    evidence = np.zeros((labels.size, states.size), dtype=np.float64)
    csr = graph.tocsr()
    for output_state, state in enumerate(states):
        members = labels == state
        evidence[:, output_state] = np.asarray(csr[:, members].sum(axis=1)).ravel()
        evidence[members, output_state] -= csr[members][:, members].diagonal()
    evidence = np.maximum(evidence, 0.0) + 1e-12
    evidence /= evidence.sum(axis=1, keepdims=True)
    relabeled = np.searchsorted(states, labels).astype(np.int32)
    return evidence.astype(np.float32), relabeled


def _community_templates(
    maps: np.ndarray, graph: sparse.csr_matrix, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    states = np.unique(labels)
    relabeled = np.searchsorted(states, labels).astype(np.int32)
    normalized = normalize_topographies(maps)
    templates = np.zeros((states.size, maps.shape[1]), dtype=np.float64)
    for output_state, state in enumerate(states):
        indices = np.flatnonzero(labels == state)
        subgraph = graph[indices][:, indices]
        weights = np.asarray(subgraph.sum(axis=1)).ravel().astype(np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(indices.size)
        reference = normalized[indices[np.argmax(weights)]]
        signs = np.sign(normalized[indices] @ reference)
        signs[signs == 0] = 1.0
        templates[output_state] = np.average(maps[indices] * signs[:, None], axis=0, weights=weights)
    return templates.astype(np.float32), relabeled


def fit_adaptive_model(
    maps: np.ndarray,
    algorithm: AdaptiveAlgorithm,
    *,
    knn_fraction: float = 0.01,
    knn_min: int | None = None,
    resolution: float = 1.0,
    markov_time: float = 1.0,
    infomap_trials: int = 10,
    graph_weight_mode: GraphWeightMode = "gaussian_distance",
    evidence_mode: AdaptiveEvidenceMode = "template_similarity",
    random_state: int = 42,
) -> FittedStateModel:
    graph, metadata = build_polarity_invariant_knn_graph(
        maps,
        knn_fraction=knn_fraction,
        knn_min=knn_min,
        weight_mode=graph_weight_mode,
    )
    upper = sparse.triu(graph, k=1).tocoo()
    edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    if algorithm == "leiden":
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise RuntimeError("Leiden requires `pip install -e '.[full]'`") from exc
        network = ig.Graph(n=graph.shape[0], edges=edges, directed=False)
        network.es["weight"] = upper.data.tolist()
        partition = leidenalg.find_partition(
            network,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )
        labels = np.asarray(partition.membership, dtype=np.int32)
        metadata.update({"quality": float(partition.quality()), "resolution": float(resolution)})
    elif algorithm == "infomap":
        try:
            from infomap import Infomap
        except ImportError as exc:
            raise RuntimeError("Infomap requires `pip install -e '.[full]'`") from exc
        command = (
            f"--two-level --seed {int(random_state)} --silent "
            f"--markov-time {float(markov_time)} --num-trials {int(infomap_trials)}"
        )
        model = Infomap(command)
        for source, target, weight in zip(upper.row, upper.col, upper.data):
            model.add_link(int(source), int(target), float(weight))
        model.run()
        module_map: dict[int, int] = {}
        labels = np.zeros(graph.shape[0], dtype=np.int32)
        for node in model.nodes:
            module_map.setdefault(node.module_id, len(module_map))
            labels[node.node_id] = module_map[node.module_id]
        metadata.update({"codelength": float(model.codelength), "markov_time": float(markov_time)})
    else:
        raise ValueError(f"Unknown adaptive algorithm: {algorithm}")
    templates, relabeled = _community_templates(maps, graph, labels)
    if evidence_mode == "community_affinity":
        evidence, labels = community_affinity_evidence(graph, labels)
    elif evidence_mode == "template_similarity":
        labels = relabeled
        evidence = spatial_evidence(maps, templates)
    else:  # pragma: no cover - guarded by the public type
        raise ValueError(f"Unknown adaptive evidence mode: {evidence_mode}")
    metadata["n_states"] = int(templates.shape[0])
    metadata["template_source"] = "within_community_affinity_weighted_mean"
    metadata["evidence_mode"] = evidence_mode
    return FittedStateModel(templates, labels, evidence.astype(np.float32), algorithm, metadata)
