"""Frozen subject-level protocol used for the paper's predictive table.

The general package API favors fully nested model selection. This module is a
compatibility implementation for reproducing the published ds004504 analysis:
the evidence grid is screened on the complete cohort, followed by fixed-model
repeated cross-validation. Keeping it separate makes that estimand explicit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .adaptive import fit_adaptive_model
from .clustering import fit_fixed_model
from .complexity import boundary_aware_lzc
from .descriptors import hard_label_descriptors
from .evidence import sharpen_evidence
from .peaks import PeakSequence
from .temporal import merge_short_segments, true_run_durations


@dataclass(frozen=True)
class PaperProtocolSettings:
    """Parameters frozen from the run that generated PDF Table 3."""

    gamma_values: tuple[float, ...] = tuple(float(value) for value in range(1, 11))
    percentiles: tuple[float, ...] = tuple(value / 100.0 for value in range(20, 81, 5))
    min_duration_ms: float = 30.0
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 200
    hmm_n_init: int = 3
    hmm_n_iter: int = 100
    knn_fraction: float = 0.01
    leiden_resolution: float = 1.0
    infomap_markov_time: float = 1.0
    infomap_trials: int = 10
    random_state: int = 42
    cv_splits: int = 5
    cv_repeats: int = 20


def _gamma_tag(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _percentile_tag(value: float) -> str:
    return str(round(float(value) * 100.0))


def _paper_weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    """Match the weighted-quantile implementation used by the original run.

    The compatibility path intentionally retains NumPy's default ``argsort``
    behavior. The general package helper uses a stable sort, which is preferable
    for new work but can move a threshold when many evidence values are tied.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    if not cumulative.size or cumulative[-1] <= 0:
        return np.full(len(quantiles), np.nan, dtype=np.float64)
    return np.interp(
        np.asarray(quantiles, dtype=np.float64) * cumulative[-1],
        cumulative,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def paper_grid_readouts(
    sequence: PeakSequence,
    labels: np.ndarray,
    evidence: np.ndarray,
    settings: PaperProtocolSettings,
) -> dict[str, float]:
    """Compute the hard baseline and complete trajectory/null-LZC grid."""
    weights = np.asarray(sequence.weights_sec, dtype=np.float64)
    blocks = np.asarray(sequence.block_ids, dtype=np.int32)
    labels = merge_short_segments(
        labels,
        weights,
        settings.min_duration_ms / 1000.0,
        blocks,
    ).astype(np.int32)
    output = hard_label_descriptors(
        labels,
        weights,
        n_states=evidence.shape[1],
        block_ids=blocks,
        min_duration_ms=0.0,
        prefix="hard",
    )
    total_time = float(weights.sum())
    hard_labels = labels.astype(np.int64)
    for gamma in settings.gamma_values:
        trajectory = sharpen_evidence(evidence, gamma)
        gamma_tag = _gamma_tag(gamma)
        thresholds = np.vstack(
            [
                _paper_weighted_quantiles(trajectory[:, state], weights, settings.percentiles)
                for state in range(trajectory.shape[1])
            ]
        )
        for q_index, percentile in enumerate(settings.percentiles):
            percentile_tag = _percentile_tag(percentile)
            prefix = f"traj_g{gamma_tag}_p{percentile_tag}"
            high = trajectory > thresholds[:, q_index][None, :]
            counts = np.zeros(trajectory.shape[0], dtype=np.int16)
            durations: list[float] = []
            for state in range(trajectory.shape[1]):
                active = merge_short_segments(
                    high[:, state].astype(np.int32),
                    weights,
                    settings.min_duration_ms / 1000.0,
                    blocks,
                ).astype(bool)
                counts += active.astype(np.int16)
                durations.extend(true_run_durations(active, weights, blocks).tolist())
            duration_array = np.asarray(durations, dtype=np.float64)
            output[f"{prefix}_global_mean_duration_ms"] = (
                float(duration_array.mean() * 1000.0) if duration_array.size else np.nan
            )
            output[f"{prefix}_global_median_duration_ms"] = (
                float(np.median(duration_array) * 1000.0) if duration_array.size else np.nan
            )
            output[f"{prefix}_global_occurrence_per_sec"] = (
                float(duration_array.size / total_time) if total_time > 0 else np.nan
            )
            output[f"{prefix}_global_n_episodes"] = int(duration_array.size)
            output[f"{prefix}_null_fraction_any"] = (
                float(weights[counts == 0].sum() / total_time) if total_time > 0 else np.nan
            )

            any_high = high.any(axis=1)
            self_high = high[np.arange(hard_labels.size), hard_labels]
            for variant, keep in (("hardany", any_high), ("hardself", self_high)):
                null_sequence = hard_labels.copy()
                null_sequence[~keep] = trajectory.shape[1]
                raw, normalized = boundary_aware_lzc(null_sequence, blocks)
                output[f"{prefix}_{variant}_null_lzc_raw"] = int(raw)
                output[f"{prefix}_{variant}_null_lzc_norm"] = float(normalized)
                output[f"{prefix}_{variant}_null_fraction"] = (
                    float(weights[~keep].sum() / total_time) if total_time > 0 else np.nan
                )
    return output


def paper_subject_feature_rows(
    sequence: PeakSequence,
    metadata: dict[str, object],
    channel_names: list[str],
    settings: PaperProtocolSettings | None = None,
) -> list[dict[str, object]]:
    """Fit all eight subject-native state models for one participant."""
    settings = settings or PaperProtocolSettings()
    rows: list[dict[str, object]] = []
    common = {
        "dataset": str(metadata.get("dataset", "ds004504")),
        "subject_id": str(metadata["subject_id"]),
        "group": str(metadata.get("group", "")),
        "age": metadata.get("age", np.nan),
        "sex": metadata.get("sex", ""),
        "n_peaks": int(sequence.maps.shape[0]),
        "n_blocks": int(np.unique(sequence.block_ids).size),
        "total_time_sec": float(sequence.weights_sec.sum()),
        "readout_domain": "subject_full_gfp_peaks_peak_label_sequence",
    }
    lengths = [int(np.sum(sequence.block_ids == block)) for block in np.unique(sequence.block_ids)]
    for algorithm in ("kmeans", "aahc", "hmm"):
        for n_states in (4, 7):
            model = fit_fixed_model(
                sequence.maps,
                algorithm,
                n_states,
                channel_names=channel_names,
                random_state=settings.random_state,
                kmeans_n_init=settings.kmeans_n_init,
                kmeans_max_iter=settings.kmeans_max_iter,
                hmm_n_init=settings.hmm_n_init,
                hmm_n_iter=settings.hmm_n_iter,
                lengths=lengths,
            )
            rows.append(
                {
                    **common,
                    "method": f"subject_{algorithm}_K{n_states}_fullpeaks",
                    "algorithm": algorithm,
                    "level": "subject_native",
                    "K_fixed": n_states,
                    "K_found": n_states,
                    "trajectory_evidence": (
                        "hmm_posterior" if algorithm == "hmm" else "classic_abs_spatial_correlation"
                    ),
                    **model.metadata,
                    **paper_grid_readouts(sequence, model.labels, model.evidence, settings),
                }
            )
    for algorithm in ("leiden", "infomap"):
        model = fit_adaptive_model(
            sequence.maps,
            algorithm,
            knn_fraction=settings.knn_fraction,
            resolution=settings.leiden_resolution,
            markov_time=settings.infomap_markov_time,
            infomap_trials=settings.infomap_trials,
            graph_weight_mode="abs_cosine",
            evidence_mode="community_affinity",
            random_state=settings.random_state,
        )
        rows.append(
            {
                **common,
                "method": f"subject_{algorithm}_fullpeaks",
                "algorithm": algorithm,
                "level": "subject_adaptive",
                "K_fixed": -1,
                "K_found": int(model.evidence.shape[1]),
                "trajectory_evidence": "community_affinity",
                **model.metadata,
                **paper_grid_readouts(sequence, model.labels, model.evidence, settings),
            }
        )
    return rows


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    pooled = (
        (group_a.size - 1) * np.var(group_a, ddof=1)
        + (group_b.size - 1) * np.var(group_b, ddof=1)
    ) / (group_a.size + group_b.size - 2)
    return float((group_a.mean() - group_b.mean()) / math.sqrt(pooled)) if pooled > 0 else np.nan


def select_paper_features(features: pd.DataFrame) -> pd.DataFrame:
    """Select the full-cohort maximum-effect trajectory feature per model/family."""
    rows: list[dict[str, object]] = []
    model_columns = ["method", "algorithm", "level", "K_fixed"]
    for keys, model_frame in features.groupby(model_columns, dropna=False):
        candidates = {
            "global_duration": [
                "hard_global_mean_duration_ms",
                *[
                    column
                    for column in model_frame
                    if re.match(r"^traj_g.+_p\d+_global_mean_duration_ms$", column)
                ],
            ],
            "lzc": [
                "hard_lzc_norm",
                *[
                    column
                    for column in model_frame
                    if re.match(r"^traj_g.+_p\d+_(hardany|hardself)_null_lzc_norm$", column)
                ],
            ],
        }
        for family, columns in candidates.items():
            effects: list[dict[str, object]] = []
            for column in columns:
                values = pd.to_numeric(model_frame[column], errors="coerce")
                ad = values[model_frame["group"].astype(str).eq("AD")].dropna().to_numpy(float)
                hc = values[model_frame["group"].astype(str).eq("HC")].dropna().to_numpy(float)
                ad, hc = ad[np.isfinite(ad)], hc[np.isfinite(hc)]
                if ad.size < 3 or hc.size < 3 or np.nanstd(np.r_[ad, hc]) == 0:
                    continue
                effect = _cohens_d(ad, hc)
                p_value = float(stats.ttest_ind(ad, hc, equal_var=False).pvalue)
                effects.append(
                    {"feature": column, "effect": effect, "abs_effect": abs(effect), "p_value": p_value}
                )
            effect_frame = pd.DataFrame(effects)
            hard_name = "hard_global_mean_duration_ms" if family == "global_duration" else "hard_lzc_norm"
            hard = effect_frame.loc[effect_frame["feature"].eq(hard_name)].iloc[0]
            trajectory = effect_frame.loc[~effect_frame["feature"].eq(hard_name)].sort_values(
                ["abs_effect", "p_value"], ascending=[False, True]
            ).iloc[0]
            method, algorithm, level, k_fixed = keys
            rows.append(
                {
                    "dataset": "ds004504",
                    "method": method,
                    "algorithm": algorithm,
                    "level": level,
                    "K_fixed": int(k_fixed),
                    "K_found": round(pd.to_numeric(model_frame["K_found"]).median()),
                    "metric_family": family,
                    "hard_feature": hard_name,
                    "traj_feature": trajectory["feature"],
                    "hard_effect": float(hard["effect"]),
                    "traj_effect": float(trajectory["effect"]),
                    "hard_abs_effect": float(hard["abs_effect"]),
                    "traj_abs_effect": float(trajectory["abs_effect"]),
                    "hard_p": float(hard["p_value"]),
                    "traj_p": float(trajectory["p_value"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["algorithm", "K_fixed", "metric_family"]).reset_index(drop=True)


def build_paper_feature_matrix(features: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Create the matched 16 Hard, 16 Trajectory, and 32 Combined predictors."""
    metadata_columns = ["dataset", "subject_id", "group", "age", "sex"]
    output = features[metadata_columns].drop_duplicates("subject_id").sort_values("subject_id")
    for selected in selection.to_dict("records"):
        match = (
            features["method"].eq(selected["method"])
            & features["algorithm"].eq(selected["algorithm"])
            & features["level"].eq(selected["level"])
            & pd.to_numeric(features["K_fixed"]).astype(int).eq(int(selected["K_fixed"]))
        )
        part = features.loc[
            match,
            ["subject_id", selected["hard_feature"], selected["traj_feature"]],
        ].copy()
        model = (
            f"{selected['algorithm']}_K{int(selected['K_fixed'])}"
            if int(selected["K_fixed"]) > 0
            else str(selected["algorithm"])
        )
        label = f"{model}_{selected['metric_family']}"
        part = part.rename(
            columns={
                selected["hard_feature"]: f"hard__{label}",
                selected["traj_feature"]: f"traj__{label}",
            }
        )
        output = output.merge(part, on="subject_id", how="left")
    return output.reset_index(drop=True)


def paper_repeated_classification(
    feature_matrix: pd.DataFrame,
    settings: PaperProtocolSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed-C repeated CV analysis reported in PDF Table 3."""
    settings = settings or PaperProtocolSettings()
    frame = feature_matrix[feature_matrix["group"].isin(["AD", "HC"])].copy()
    y = frame["group"].map({"HC": 0, "AD": 1}).to_numpy(int)
    hard = [column for column in frame if column.startswith("hard__")]
    trajectory = [column for column in frame if column.startswith("traj__")]
    families = {"Hard": hard, "Trajectory": trajectory, "Combined": hard + trajectory}
    splitter = RepeatedStratifiedKFold(
        n_splits=settings.cv_splits,
        n_repeats=settings.cv_repeats,
        random_state=settings.random_state,
    )
    splits = list(splitter.split(np.zeros(y.size), y))
    summaries: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for family, columns in families.items():
        x = frame[columns].apply(pd.to_numeric, errors="coerce")
        x = x[[column for column in x if x[column].notna().any()]]
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=settings.random_state,
                    ),
                ),
            ]
        )
        scores = {"accuracy": [], "balanced_accuracy": [], "f1": [], "roc_auc": []}
        for fold, (train, test) in enumerate(splits):
            pipeline.fit(x.iloc[train], y[train])
            prediction = pipeline.predict(x.iloc[test])
            probability = pipeline.predict_proba(x.iloc[test])[:, 1]
            fold_score = {
                "accuracy": accuracy_score(y[test], prediction),
                "balanced_accuracy": balanced_accuracy_score(y[test], prediction),
                "f1": f1_score(y[test], prediction, zero_division=0),
                "roc_auc": roc_auc_score(y[test], probability),
            }
            for metric, value in fold_score.items():
                scores[metric].append(float(value))
            fold_rows.append({"fold": fold, "feature_family": family, **fold_score})
        row: dict[str, object] = {
            "dataset": "ds004504",
            "task": "AD_vs_HC",
            "feature_family": family,
            "n_subjects": int(y.size),
            "n_features": int(x.shape[1]),
        }
        for metric, values in scores.items():
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
        summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(fold_rows)
