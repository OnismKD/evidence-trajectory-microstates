#!/usr/bin/env python
"""Reproduce the ds004504 subject-level classification rows in PDF Table 3."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from evidence_microstates.cache import load_peak_sequence
from evidence_microstates.datasets import load_yaml, resolve_path
from evidence_microstates.paper_protocol import (
    PaperProtocolSettings,
    build_paper_feature_matrix,
    paper_repeated_classification,
    paper_subject_feature_rows,
    select_paper_features,
)

ROOT = Path(__file__).resolve().parents[1]


def _fit_subject(payload: tuple[object, dict[str, object], list[str], PaperProtocolSettings]):
    sequence, metadata, channel_names, settings = payload
    return paper_subject_feature_rows(sequence, metadata, channel_names, settings)


def _write_subject_cache(frame: pd.DataFrame, path: Path) -> None:
    """Write one completed subject atomically so interrupted runs can resume."""
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_complete_subject_cache(path: Path, subject_id: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != 8 or set(frame["subject_id"].astype(str)) != {subject_id}:
        return None
    return frame


def compute_feature_grid(
    config_path: Path,
    output_dir: Path,
    n_jobs: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    cache_root = resolve_path(config.get("output_dir", "../outputs/ds004504"), config)
    manifest = pd.read_csv(cache_root / "manifest_snapshot.csv")
    settings = PaperProtocolSettings()
    subject_cache_dir = output_dir / "subjects"
    subject_cache_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    frames: list[pd.DataFrame] = []
    for metadata in manifest.to_dict("records"):
        subject_id = str(metadata["subject_id"])
        cache_path = subject_cache_dir / f"{subject_id}_features.csv"
        cached = None if force else _load_complete_subject_cache(cache_path, subject_id)
        if cached is not None:
            frames.append(cached)
            print(f"reused {subject_id}", flush=True)
            continue
        sequence, channel_names = load_peak_sequence(cache_root / "peaks" / f"{subject_id}_peaks.npz")
        payloads.append((sequence, metadata, channel_names, settings))

    if n_jobs == 1:
        for index, payload in enumerate(payloads, start=1):
            frame = pd.DataFrame(_fit_subject(payload))
            subject_id = str(frame["subject_id"].iloc[0])
            _write_subject_cache(frame, subject_cache_dir / f"{subject_id}_features.csv")
            frames.append(frame)
            print(f"fitted {index}/{len(payloads)} pending subjects: {subject_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_fit_subject, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), start=1):
                frame = pd.DataFrame(future.result())
                subject_id = str(frame["subject_id"].iloc[0])
                _write_subject_cache(frame, subject_cache_dir / f"{subject_id}_features.csv")
                frames.append(frame)
                print(f"fitted {index}/{len(payloads)} pending subjects: {subject_id}", flush=True)
    if not frames:
        raise RuntimeError("No complete subject feature rows were produced")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["subject_id", "algorithm", "K_fixed"])
        .reset_index(drop=True)
    )


def verify_against_pdf(summary: pd.DataFrame, selection: pd.DataFrame) -> None:
    expected_summary = pd.read_csv(ROOT / "references" / "paper_ds004504_classification.csv")
    metrics = [
        "accuracy_mean",
        "accuracy_sd",
        "balanced_accuracy_mean",
        "balanced_accuracy_sd",
        "f1_mean",
        "f1_sd",
        "roc_auc_mean",
        "roc_auc_sd",
    ]
    actual = summary.set_index("feature_family").loc[expected_summary["feature_family"]]
    expected = expected_summary.set_index("feature_family")
    if actual[["n_subjects", "n_features"]].astype(int).to_dict() != expected[
        ["n_subjects", "n_features"]
    ].astype(int).to_dict():
        raise RuntimeError("PDF verification failed: cohort size or feature count differs")
    if not actual[metrics].round(3).equals(expected[metrics].round(3)):
        difference = actual[metrics].round(3) - expected[metrics].round(3)
        raise RuntimeError(f"PDF verification failed at three-decimal precision:\n{difference}")

    expected_selection = pd.read_csv(ROOT / "references" / "paper_ds004504_selection.csv")
    keys = ["algorithm", "K_fixed", "metric_family"]
    selected = selection[keys + ["traj_feature"]].sort_values(keys).reset_index(drop=True)
    expected_selected = expected_selection.sort_values(keys).reset_index(drop=True)
    if not selected.equals(expected_selected):
        comparison = expected_selected.merge(
            selected,
            on=keys,
            how="outer",
            suffixes=("_expected", "_actual"),
        )
        raise RuntimeError(f"PDF verification failed: selected g,p features differ:\n{comparison}")
    print("\nPASS: ds004504 results match PDF Table 3 at the reported three-decimal precision.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ds004504.yaml")
    parser.add_argument(
        "--features-grid",
        type=Path,
        help="reuse a complete paper-grid feature CSV instead of refitting the eight state models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ds004504" / "paper_reproduction",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / "features_paper_grid.csv"
    if args.features_grid:
        features = pd.read_csv(args.features_grid, low_memory=False)
    elif grid_path.exists() and not args.force:
        features = pd.read_csv(grid_path, low_memory=False)
    else:
        features = compute_feature_grid(
            args.config,
            args.output_dir,
            max(1, int(args.n_jobs)),
            force=args.force,
        )
        features.to_csv(grid_path, index=False)

    selection = select_paper_features(features)
    selection.to_csv(args.output_dir / "selected_features.csv", index=False)
    matrix = build_paper_feature_matrix(features, selection)
    matrix.to_csv(args.output_dir / "classification_feature_matrix.csv", index=False)
    summary, folds = paper_repeated_classification(matrix)
    summary.to_csv(args.output_dir / "classification_summary.csv", index=False)
    folds.to_csv(args.output_dir / "classification_fold_scores.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if args.verify:
        verify_against_pdf(summary, selection)


if __name__ == "__main__":
    main()
