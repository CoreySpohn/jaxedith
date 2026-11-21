"""Analytic aperture rates for exposure-time calculations."""

from ._version import __version__
from .analytic import exposure_time, photon_counting_frame_time, rates, snr

__all__ = ["__version__", "exposure_time", "photon_counting_frame_time", "rates", "snr"]
