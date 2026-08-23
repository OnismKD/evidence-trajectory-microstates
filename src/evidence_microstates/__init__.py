"""Evidence-aware readouts for EEG topographic state models."""

from .complexity import lz76_complexity, normalized_lzc
from .descriptors import hard_label_descriptors, trajectory_descriptors
from .evidence import sharpen_evidence, spatial_evidence
from .peaks import PeakSequence, extract_gfp_peak_sequence

__all__ = [
    "PeakSequence",
    "extract_gfp_peak_sequence",
    "hard_label_descriptors",
    "lz76_complexity",
    "normalized_lzc",
    "sharpen_evidence",
    "spatial_evidence",
    "trajectory_descriptors",
]

__version__ = "0.1.0"
