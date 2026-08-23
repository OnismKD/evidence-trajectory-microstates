"""Effect-size summaries used for group and lifespan validation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Pooled-SD Cohen's d, signed as group A minus group B."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = math.sqrt(
        ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def group_effects(
    features: pd.DataFrame,
    columns: list[str],
    *,
    group_column: str = "group",
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    rows = []
    for column in columns:
        a = pd.to_numeric(features.loc[features[group_column] == group_a, column], errors="coerce").dropna()
        b = pd.to_numeric(features.loc[features[group_column] == group_b, column], errors="coerce").dropna()
        test = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "feature": column,
                "contrast": f"{group_a}-{group_b}",
                "effect": cohens_d(a.to_numpy(), b.to_numpy()),
                "p_value": float(test.pvalue),
                "n_a": int(a.size),
                "n_b": int(b.size),
            }
        )
    return pd.DataFrame(rows)


def age_associations(
    features: pd.DataFrame,
    columns: list[str],
    *,
    age_column: str = "age",
) -> pd.DataFrame:
    rows = []
    age = pd.to_numeric(features[age_column], errors="coerce")
    for column in columns:
        value = pd.to_numeric(features[column], errors="coerce")
        valid = age.notna() & value.notna()
        correlation = stats.pearsonr(age[valid], value[valid]) if valid.sum() >= 3 else None
        rows.append(
            {
                "feature": column,
                "effect": float(correlation.statistic) if correlation else np.nan,
                "p_value": float(correlation.pvalue) if correlation else np.nan,
                "n": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)
