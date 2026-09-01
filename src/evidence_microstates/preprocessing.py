"""Dataset-agnostic EEG preprocessing used before topographic modeling."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float, order: int) -> np.ndarray:
    if not 0 < low < high < sfreq / 2:
        raise ValueError(f"Invalid pass band ({low}, {high}) for sampling rate {sfreq}")
    sos = signal.butter(order, (low, high), btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def _bandpass_ba(data: np.ndarray, sfreq: float, low: float, high: float, order: int) -> np.ndarray:
    """Direct-form Butterworth filter retained for exact paper reproduction."""
    nyquist = 0.5 * float(sfreq)
    b, a = signal.butter(
        int(order),
        [float(low) / nyquist, float(high) / nyquist],
        btype="band",
    )
    return signal.filtfilt(b, a, data, axis=-1)


def preprocess_paper_ds004504_raw(
    raw: object,
    *,
    target_sfreq: float = 200.0,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    filter_order: int = 5,
    detrend: bool = True,
    common_average_reference: bool = True,
) -> tuple[np.ndarray, float]:
    """Reproduce the ds004504 preprocessing order used for the paper.

    Resampling is performed by MNE on the continuous ``Raw`` object with
    automatic padding. The alpha filter then uses SciPy's direct-form
    Butterworth coefficients and zero-phase ``filtfilt``. Keeping this profile
    explicit prevents a numerically similar modern filter implementation from
    silently changing GFP peak locations and downstream paper results.
    """
    work = raw.copy()
    if not np.isclose(float(work.info["sfreq"]), float(target_sfreq)):
        work.resample(float(target_sfreq), npad="auto", verbose="ERROR")
    x = work.get_data().astype(np.float64)
    fs = float(work.info["sfreq"])
    x = _bandpass_ba(x, fs, float(alpha_band[0]), float(alpha_band[1]), int(filter_order))
    if detrend:
        x = signal.detrend(x, axis=-1, type="linear")
    if common_average_reference:
        x = x - x.mean(axis=0, keepdims=True)
    return x.astype(np.float32), fs


def preprocess_eeg(
    data: np.ndarray,
    sfreq: float,
    *,
    target_sfreq: float | None = 200.0,
    broadband: tuple[float, float] | None = None,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    filter_order: int = 5,
    detrend: bool = True,
    common_average_reference: bool = True,
) -> tuple[np.ndarray, float]:
    """Preprocess a continuous EEG array of shape ``(channels, samples)``.

    ``broadband`` is optional because some public derivatives have already
    received the dataset's official broad-band preprocessing. The final
    state-modeling signal is always filtered to ``alpha_band``.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 10:
        raise ValueError(f"Expected a channels-by-samples EEG array, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("EEG contains NaN or infinite samples")

    fs = float(sfreq)
    if target_sfreq is not None and not np.isclose(fs, float(target_sfreq)):
        ratio = Fraction(float(target_sfreq) / fs).limit_denominator(1000)
        x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
        fs = float(target_sfreq)

    if broadband is not None:
        x = _bandpass(x, fs, float(broadband[0]), float(broadband[1]), filter_order)
    x = _bandpass(x, fs, float(alpha_band[0]), float(alpha_band[1]), filter_order)

    if detrend:
        x = signal.detrend(x, axis=-1, type="linear")
    if common_average_reference:
        x = x - x.mean(axis=0, keepdims=True)
    return x.astype(np.float32), fs
