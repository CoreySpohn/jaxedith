"""Analytic aperture rates for exposure-time calculations."""

from ._version import __version__
from .analytic import (
    photon_counting_frame_time,
    planet_exposure_time,
    planet_snr,
    rates,
    scene_rates,
)

__all__ = [
    "__version__",
    "photon_counting_frame_time",
    "planet_exposure_time",
    "planet_snr",
    "rates",
    "scene_rates",
]
