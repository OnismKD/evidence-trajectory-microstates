"""Configuration-driven EEG loading with a ds004504 convenience adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_dir"] = str(path.parent)
    return config


def resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def ds004504_manifest(root: str | Path) -> pd.DataFrame:
    """Build the AD/HC manifest from the public BIDS participants table."""
    root = Path(root).resolve()
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants = participants[participants["Group"].isin(["A", "C"])].copy()
    participants["group"] = participants["Group"].map({"A": "AD", "C": "HC"})
    participants["subject_id"] = participants["participant_id"].astype(str)
    participants["age"] = pd.to_numeric(participants["Age"], errors="coerce")
    participants["sex"] = participants["Gender"].astype(str)
    paths = []
    for subject_id in participants["subject_id"]:
        matches = sorted((root / subject_id / "eeg").glob(f"{subject_id}_task-eyesclosed_eeg.*"))
        matches = [
            path for path in matches if path.suffix.lower() in {".set", ".edf", ".bdf", ".vhdr", ".fif"}
        ]
        paths.append(str(matches[0]) if matches else "")
    participants["eeg_path"] = paths
    participants["dataset"] = "ds004504"
    participants["include"] = participants["eeg_path"].str.len() > 0
    return participants[["subject_id", "dataset", "eeg_path", "group", "age", "sex", "include"]]


def build_manifest(config: dict[str, Any]) -> pd.DataFrame:
    dataset = config["dataset"]
    if dataset == "ds004504" and config.get("adapter") == "openneuro_ds004504":
        manifest = ds004504_manifest(resolve_path(config["dataset_root"], config))
    else:
        manifest_path = config.get("manifest_path")
        if not manifest_path:
            raise ValueError("Non-ds004504 datasets require manifest_path")
        manifest_file = resolve_path(manifest_path, config)
        manifest = pd.read_csv(manifest_file)
        if "dataset" not in manifest:
            manifest["dataset"] = dataset
        for index, path in manifest["eeg_path"].items():
            candidate = Path(str(path)).expanduser()
            if not candidate.is_absolute():
                candidate = manifest_file.parent / candidate
            manifest.loc[index, "eeg_path"] = str(candidate.resolve())
    required = {"subject_id", "dataset", "eeg_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    if "include" in manifest:
        manifest = manifest[manifest["include"].astype(bool)]
    return manifest.reset_index(drop=True)


def read_eeg(path: str | Path) -> tuple[Any, list[str]]:
    """Load a supported continuous EEG file and retain EEG channels only."""
    path = Path(path)
    suffix = path.suffix.lower()
    readers = {
        ".set": mne.io.read_raw_eeglab,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".fif": mne.io.read_raw_fif,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported EEG extension: {suffix}")
    raw = readers[suffix](path, preload=True, verbose="ERROR")
    picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        eog=False,
        ecg=False,
        emg=False,
        stim=False,
        exclude="bads",
    )
    raw.pick(picks)
    return raw, list(raw.ch_names)
