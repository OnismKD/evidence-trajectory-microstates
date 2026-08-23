#!/usr/bin/env python
"""Compare Hard, Trajectory, and Combined subject-level feature families."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evidence_microstates.prediction import NestedCVSettings, nested_classification, nested_regression


def pivot_features(path: Path) -> pd.DataFrame:
    long = pd.read_csv(path)
    identifiers = [column for column in ["dataset", "subject_id", "group", "age", "sex"] if column in long]
    metadata = long[identifiers].drop_duplicates(subset="subject_id")
    # Subject-native state indices are not aligned across participants. Only
    # label-invariant global descriptors are therefore eligible for prediction.
    metric_columns = [
        column
        for column in long.columns
        if column
        in {
            "hard_global_mean_duration_ms",
            "hard_lzc_norm",
            "trajectory_global_mean_duration_ms",
            "null_lzc_norm",
        }
    ]
    wide = long.pivot_table(
        index="subject_id",
        columns=["algorithm", "K"],
        values=metric_columns,
        aggfunc="first",
    )
    wide.columns = ["__".join(map(str, column)) for column in wide.columns]
    return metadata.merge(wide.reset_index(), on="subject_id", how="inner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--task", choices=["ad_hc", "young_old", "age_regression"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    frame = pivot_features(args.features)
    hard = [column for column in frame if column.startswith("hard_")]
    trajectory = [column for column in frame if column.startswith(("trajectory_", "null_"))]
    feature_sets = {"hard": hard, "trajectory": trajectory, "combined": hard + trajectory}
    settings = NestedCVSettings(repeats=args.repeats, n_jobs=args.n_jobs)
    if args.task == "ad_hc":
        frame = frame[frame["group"].isin(["AD", "HC"])].copy()
        frame["target"] = (frame["group"] == "AD").astype(int)
        summary, folds = nested_classification(frame, feature_sets, "target", settings)
    elif args.task == "young_old":
        frame = frame[(frame["age"] < 35) | (frame["age"] > 60)].copy()
        frame["target"] = (frame["age"] > 60).astype(int)
        summary, folds = nested_classification(frame, feature_sets, "target", settings)
    else:
        frame = frame[frame["age"].notna()].copy()
        summary, folds = nested_regression(frame, feature_sets, "age", settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    folds.to_csv(args.output_dir / "fold_scores.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
