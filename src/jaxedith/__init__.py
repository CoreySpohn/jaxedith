"""Photon-native aperture photometry and detector-based exposure planning."""

from ._version import __version__
from .calculator import PostProcessing, exposure_time, snr
from .exozodi import AYOExozodi
from .public import (
    PreparedAperture,
    aperture_quadrature,
    aperture_rates,
    prepare_aperture,
)

__all__ = [
    "AYOExozodi",
    "PostProcessing",
    "PreparedAperture",
    "__version__",
    "aperture_quadrature",
    "aperture_rates",
    "exposure_time",
    "prepare_aperture",
    "snr",
]
