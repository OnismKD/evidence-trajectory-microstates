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
    args = parser.parse_args()
    try:
        import openneuro
    except ImportError as exc:
        raise SystemExit("Install the downloader with: pip install -e '.[download]'") from exc
    include = [f"sub-{subject:03d}/eeg/*task-eyesclosed*" for subject in parse_subjects(args.subjects)]
    openneuro.download(
        dataset="ds004504",
        tag=args.tag,
        target_dir=args.output,
        include=include,
    )


if __name__ == "__main__":
    main()
