"""Portable peak-cache I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .peaks import PeakSequence


def save_peak_sequence(path: str | Path, sequence: PeakSequence, channel_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        peak_maps=sequence.maps,
        peak_indices=sequence.indices,
        peak_weights_sec=sequence.weights_sec,
        peak_block_id=sequence.block_ids,
        fs=np.asarray(sequence.sfreq),
        n_time=np.asarray(sequence.n_samples),
        channel_names=np.asarray(channel_names, dtype=str),
    )


def load_peak_sequence(path: str | Path) -> tuple[PeakSequence, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        sequence = PeakSequence(
            maps=cache["peak_maps"].astype(np.float32),
            indices=cache["peak_indices"].astype(np.int64),
            weights_sec=cache["peak_weights_sec"].astype(np.float64),
            block_ids=cache["peak_block_id"].astype(np.int32),
            sfreq=float(cache["fs"]),
            n_samples=int(cache["n_time"]),
        )
        channel_names = [str(value) for value in cache["channel_names"].tolist()]
    return sequence, channel_names
