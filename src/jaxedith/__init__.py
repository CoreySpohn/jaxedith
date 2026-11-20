"""Photon-native detector statistics, S/N and exposure planning."""

from ._version import __version__
from .calculator import PostProcessing, exposure_time, snr

__all__ = [
    "PostProcessing",
    "__version__",
    "exposure_time",
    "snr",
]
