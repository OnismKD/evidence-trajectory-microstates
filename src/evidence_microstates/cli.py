"""Command-line entry points for extraction, modeling, and smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import load_peak_sequence, save_peak_sequence
from .datasets import build_manifest, load_yaml, read_eeg, resolve_path
from .peaks import PeakSequence, extract_gfp_peak_sequence
from .preprocessing import preprocess_eeg
from .workflow import (
    ReadoutSettings,
    analyze_subject_adaptive,
    analyze_subject_native_fixed_k,
    analyze_two_level_fixed_k,
)


def _settings(config: dict) -> ReadoutSettings:
    readout = config.get("readout", {})
    return ReadoutSettings(
        gamma=float(readout.get("gamma", 1.0)),
        duration_percentile=float(readout.get("duration_percentile", 0.50)),
        lzc_percentile=float(readout.get("lzc_percentile", 0.80)),
        min_duration_ms=float(readout.get("min_duration_ms", 30.0)),
        null_mode=str(readout.get("null_mode", "evidence_quantile_self")),
    )


def extract_command(config_path: str) -> None:
    config = load_yaml(config_path)
    manifest = build_manifest(config)
    output = resolve_path(config.get("output_dir", "../outputs/example"), config)
    peaks_dir = output / "peaks"
    peaks_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "manifest_snapshot.csv", index=False)
    pre = config.get("preprocessing", {})
    peak_config = config.get("peaks", {})
    reference_channels: list[str] | None = None
    for record in manifest.to_dict("records"):
        raw, channel_names = read_eeg(record["eeg_path"])
        if reference_channels is None:
            reference_channels = channel_names
        elif channel_names != reference_channels:
            raise ValueError(
                f"Channel mismatch for {record['subject_id']}; harmonize channels within a dataset before group modeling"
            )
        data, sfreq = preprocess_eeg(
            raw.get_data(),
            float(raw.info["sfreq"]),
            target_sfreq=pre.get("target_sfreq", 200.0),
            broadband=tuple(pre["broadband"]) if pre.get("broadband") else None,
            alpha_band=tuple(pre.get("alpha_band", [8.0, 13.0])),
            filter_order=int(pre.get("filter_order", 5)),
            detrend=bool(pre.get("detrend", True)),
            common_average_reference=bool(pre.get("common_average_reference", True)),
        )
        sequence = extract_gfp_peak_sequence(
            data,
            sfreq,
            min_peak_distance_ms=float(peak_config.get("min_distance_ms", 10.0)),
        )
        if sequence.maps.shape[0] < int(peak_config.get("minimum_count", 20)):
            raise RuntimeError(f"Too few GFP peaks for {record['subject_id']}: {sequence.maps.shape[0]}")
        save_peak_sequence(peaks_dir / f"{record['subject_id']}_peaks.npz", sequence, channel_names)
        print(f"extracted {record['subject_id']}: {sequence.maps.shape[0]} peaks", flush=True)
    with (output / "extraction_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({"config": config_path, "n_subjects": len(manifest)}, stream, indent=2)


def _load_cached_study(config: dict) -> tuple[pd.DataFrame, list[PeakSequence], list[str], list[str]]:
    output = resolve_path(config.get("output_dir", "../outputs/example"), config)
    manifest = pd.read_csv(output / "manifest_snapshot.csv")
    sequences: list[PeakSequence] = []
    channel_names: list[str] | None = None
    for subject_id in manifest["subject_id"].astype(str):
        sequence, names = load_peak_sequence(output / "peaks" / f"{subject_id}_peaks.npz")
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            raise ValueError(f"Cached channel mismatch for {subject_id}")
        sequences.append(sequence)
    return manifest, sequences, manifest["subject_id"].astype(str).tolist(), channel_names or []


def _attach_manifest(features: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata = [column for column in ["subject_id", "dataset", "group", "age", "sex"] if column in manifest]
    return features.merge(manifest[metadata], on="subject_id", how="left")


def analyze_command(config_path: str) -> None:
    config = load_yaml(config_path)
    output = resolve_path(config.get("output_dir", "../outputs/example"), config)
    output.mkdir(parents=True, exist_ok=True)
    manifest, sequences, subject_ids, channel_names = _load_cached_study(config)
    settings = _settings(config)
    analysis = config.get("analysis", {})
    seed = int(analysis.get("random_state", 42))
    group_frames = []
    subject_frames = []
    models = output / "models"
    models.mkdir(exist_ok=True)

    for algorithm in analysis.get("fixed_algorithms", ["kmeans"]):
        for n_states in analysis.get("fixed_k", [4, 7]):
            print(f"group two-level: {algorithm} K={n_states}", flush=True)
            frame, templates, metadata = analyze_two_level_fixed_k(
                sequences,
                subject_ids,
                algorithm,
                int(n_states),
                settings,
                channel_names=channel_names,
                random_state=seed,
            )
            group_frames.append(frame)
            np.save(models / f"group_{algorithm}_K{n_states}_templates.npy", templates)
            with (models / f"group_{algorithm}_K{n_states}_metadata.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(metadata, stream, indent=2)
            if analysis.get("subject_native", True):
                print(f"subject native: {algorithm} K={n_states}", flush=True)
                subject_frames.append(
                    analyze_subject_native_fixed_k(
                        sequences,
                        subject_ids,
                        algorithm,
                        int(n_states),
                        settings,
                        channel_names=channel_names,
                        random_state=seed,
                    )
                )
    for algorithm in analysis.get("adaptive_algorithms", []):
        print(f"subject adaptive: {algorithm}", flush=True)
        subject_frames.append(
            analyze_subject_adaptive(
                sequences,
                subject_ids,
                algorithm,
                settings,
                random_state=seed,
                knn_fraction=float(analysis.get("knn_fraction", 0.01)),
            )
        )
    if group_frames:
        _attach_manifest(pd.concat(group_frames, ignore_index=True), manifest).to_csv(
            output / "features_group_two_level.csv", index=False
        )
    if subject_frames:
        _attach_manifest(pd.concat(subject_frames, ignore_index=True), manifest).to_csv(
            output / "features_subject_level.csv", index=False
        )


def _synthetic_study(
    n_subjects: int, n_peaks: int, random_state: int
) -> tuple[list[PeakSequence], list[str]]:
    rng = np.random.default_rng(random_state)
    n_channels, n_states = 19, 4
    canonical = rng.normal(size=(n_states, n_channels))
    canonical -= canonical.mean(axis=1, keepdims=True)
    canonical /= np.linalg.norm(canonical, axis=1, keepdims=True)
    sequences, subject_ids = [], []
    for subject in range(n_subjects):
        labels = np.empty(n_peaks, dtype=np.int32)
        labels[0] = rng.integers(n_states)
        for index in range(1, n_peaks):
            labels[index] = labels[index - 1] if rng.random() < 0.72 else rng.integers(n_states)
        maps = canonical[labels] + rng.normal(scale=0.35, size=(n_peaks, n_channels))
        polarity = rng.choice([-1.0, 1.0], size=n_peaks)
        maps *= polarity[:, None]
        weights = rng.uniform(0.02, 0.08, size=n_peaks)
        sequences.append(
            PeakSequence(
                maps=maps.astype(np.float32),
                indices=np.arange(n_peaks, dtype=np.int64),
                weights_sec=weights,
                block_ids=np.zeros(n_peaks, dtype=np.int32),
                sfreq=200.0,
                n_samples=round(weights.sum() * 200),
            )
        )
        subject_ids.append(f"sub-{subject + 1:03d}")
    return sequences, subject_ids


def demo_command(output_dir: str, random_state: int = 42) -> None:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sequences, subject_ids = _synthetic_study(12, 300, random_state)
    features, templates, metadata = analyze_two_level_fixed_k(
        sequences,
        subject_ids,
        "kmeans",
        4,
        ReadoutSettings(),
        random_state=random_state,
    )
    features.to_csv(output / "synthetic_features.csv", index=False)
    np.save(output / "synthetic_group_templates.npy", templates)
    with (output / "synthetic_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(
        features[
            [
                "subject_id",
                "hard_global_mean_duration_ms",
                "trajectory_global_mean_duration_ms",
                "hard_lzc_norm",
                "null_lzc_norm",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote reproducibility smoke-test outputs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="evidence-microstates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run a synthetic two-level K-means smoke test")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--random-state", type=int, default=42)
    extract = subparsers.add_parser("extract", help="preprocess EEG and cache GFP-peak topographies")
    extract.add_argument("--config", required=True)
    analyze = subparsers.add_parser("analyze", help="fit state models and compute matched descriptors")
    analyze.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        demo_command(args.output_dir, args.random_state)
    elif args.command == "extract":
        extract_command(args.config)
    elif args.command == "analyze":
        analyze_command(args.config)


if __name__ == "__main__":
    main()
