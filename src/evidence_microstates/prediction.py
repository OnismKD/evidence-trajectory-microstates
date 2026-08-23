"""Leakage-controlled subject-level predictive validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NestedCVSettings:
    outer_splits: int = 5
    repeats: int = 10
    inner_splits: int = 3
    random_state: int = 42
    n_jobs: int = 1


def _wilcoxon_greater(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0 or np.allclose(differences, 0):
        return 1.0
    return float(stats.wilcoxon(differences, alternative="greater").pvalue)


def nested_classification(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target: str,
    settings: NestedCVSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired repeated nested CV for binary classification feature families."""
    settings = settings or NestedCVSettings()
    y = frame[target].to_numpy()
    outer = RepeatedStratifiedKFold(
        n_splits=settings.outer_splits,
        n_repeats=settings.repeats,
        random_state=settings.random_state,
    )
    fold_rows = []
    for fold, (train, test) in enumerate(outer.split(np.zeros(y.size), y)):
        inner = StratifiedKFold(
            n_splits=settings.inner_splits,
            shuffle=True,
            random_state=settings.random_state + fold,
        )
        for family, columns in feature_sets.items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            )
            search = GridSearchCV(
                pipeline,
                {"model__C": np.logspace(-4, 4, 9)},
                scoring="roc_auc",
                cv=inner,
                n_jobs=settings.n_jobs,
            )
            x = frame[columns].to_numpy(dtype=float)
            search.fit(x[train], y[train])
            probability = search.predict_proba(x[test])[:, 1]
            prediction = search.predict(x[test])
            fold_rows.append(
                {
                    "fold": fold,
                    "feature_family": family,
                    "balanced_accuracy": balanced_accuracy_score(y[test], prediction),
                    "roc_auc": roc_auc_score(y[test], probability),
                    "best_parameter": float(search.best_params_["model__C"]),
                }
            )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("feature_family")
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            n_folds=("fold", "size"),
        )
        .reset_index()
    )
    if "hard" in feature_sets and "trajectory" in feature_sets:
        pivot = folds.pivot(index="fold", columns="feature_family", values="roc_auc")
        p_value = _wilcoxon_greater((pivot["trajectory"] - pivot["hard"]).to_numpy())
        summary["trajectory_greater_than_hard_auc_p"] = p_value
    return summary, folds


def nested_regression(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target: str,
    settings: NestedCVSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired repeated nested CV for continuous outcomes such as age."""
    settings = settings or NestedCVSettings()
    y = frame[target].to_numpy(dtype=float)
    outer = RepeatedKFold(
        n_splits=settings.outer_splits,
        n_repeats=settings.repeats,
        random_state=settings.random_state,
    )
    fold_rows = []
    for fold, (train, test) in enumerate(outer.split(np.zeros(y.size))):
        inner = KFold(
            n_splits=settings.inner_splits,
            shuffle=True,
            random_state=settings.random_state + fold,
        )
        for family, columns in feature_sets.items():
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            )
            search = GridSearchCV(
                pipeline,
                {"model__alpha": np.logspace(-4, 4, 9)},
                scoring="neg_mean_absolute_error",
                cv=inner,
                n_jobs=settings.n_jobs,
            )
            x = frame[columns].to_numpy(dtype=float)
            search.fit(x[train], y[train])
            prediction = search.predict(x[test])
            fold_rows.append(
                {
                    "fold": fold,
                    "feature_family": family,
                    "r2": r2_score(y[test], prediction),
                    "mae": mean_absolute_error(y[test], prediction),
                    "rmse": mean_squared_error(y[test], prediction) ** 0.5,
                    "best_parameter": float(search.best_params_["model__alpha"]),
                }
            )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("feature_family")
        .agg(
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            n_folds=("fold", "size"),
        )
        .reset_index()
    )
    if "hard" in feature_sets and "trajectory" in feature_sets:
        pivot = folds.pivot(index="fold", columns="feature_family", values="mae")
        p_value = _wilcoxon_greater((pivot["hard"] - pivot["trajectory"]).to_numpy())
        summary["trajectory_lower_than_hard_mae_p"] = p_value
    return summary, folds
