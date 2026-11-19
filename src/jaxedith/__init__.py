"""Photon-native aperture photometry and detector-based exposure planning."""

from ._version import __version__
from .calculator import PostProcessing, exposure_time, snr
from .public import aperture_rates

__all__ = [
    "PostProcessing",
    "__version__",
    "aperture_rates",
    "exposure_time",
    "snr",
]
