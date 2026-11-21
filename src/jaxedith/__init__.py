"""Analytic aperture rates for exposure-time calculations."""

from ._version import __version__
from .analytic import photon_counting_frame_time, rates

__all__ = ["__version__", "photon_counting_frame_time", "rates"]
