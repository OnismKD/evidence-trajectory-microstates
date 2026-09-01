#!/usr/bin/env python
"""Download the public ds004504 AD/HC eyes-closed example from OpenNeuro."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_subjects(value: str) -> list[int]:
    subjects: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", maxsplit=1))
            subjects.extend(range(start, stop + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/ds004504"))
    parser.add_argument("--subjects", default="1-65", help="comma-separated IDs or ranges")
    parser.add_argument("--tag", default="1.0.9")
    parser.add_argument(
        "--input",
        choices=["derivatives", "raw", "both"],
        default="derivatives",
        help="download the preprocessed derivatives used by the paper, raw EEG, or both",
    )
    args = parser.parse_args()
    try:
        import openneuro
    except ImportError as exc:
        raise SystemExit("Install the downloader with: pip install -e '.[download]'") from exc
    include = ["participants.tsv", "participants.json", "dataset_description.json"]
    for subject in parse_subjects(args.subjects):
        subject_id = f"sub-{subject:03d}"
        if args.input in {"raw", "both"}:
            include.append(f"{subject_id}/eeg/*task-eyesclosed*")
        if args.input in {"derivatives", "both"}:
            include.append(f"derivatives/{subject_id}/eeg/*task-eyesclosed*")
    openneuro.download(
        dataset="ds004504",
        tag=args.tag,
        target_dir=args.output,
        include=include,
    )


if __name__ == "__main__":
    main()
