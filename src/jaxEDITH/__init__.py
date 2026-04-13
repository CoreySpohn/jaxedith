"""JAX ETC -- public API.

This module provides the main user-facing functions for the JAX exposure time
calculator. It bridges :mod:`optixstuff` hardware objects (stored as
equinox modules) and astrophysical scene parameters to the pure-JAX count-rate
and solver functions.

Example:
-------
>>> from optixstuff import OpticalPath
>>> from jaxedith import calc_exptime, CONFIG
>>>
>>> # Build optical_path from optixstuff objects
>>> optical_path = OpticalPath(primary=..., coronagraph=..., detector=...)
>>> scene = ETCScene(F0=..., Fs_over_F0=..., Fp_over_Fs=..., ...)
>>>
>>> t_exp = calc_exptime(optical_path, scene, config=CONFIG)
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# ── Re-exports ────────────────────────────────────────────────────────────────

from .config import (
    AYO_CONFIG,
    CONFIG,
    EXOSIMS_CHARACTERIZATION_CONFIG,
    EXOSIMS_DETECTION_CONFIG,
    ETCConfig,
)
from .core import (
    calc_count_rates,
    calc_exptime,
    calc_exptime_spectrum,
    calc_snr,
)
from .observation import (
    calc_exptime_from_observation,
    calc_exptime_from_system,
    calc_snr_from_observation,
    calc_snr_from_system,
    observation_geometry,
)
from .scene import ETCScene

__all__ = [
    "AYO_CONFIG",
    "CONFIG",
    "EXOSIMS_CHARACTERIZATION_CONFIG",
    "EXOSIMS_DETECTION_CONFIG",
    "ETCConfig",
    "ETCScene",
    "__version__",
    "calc_count_rates",
    "calc_exptime",
    "calc_exptime_from_observation",
    "calc_exptime_from_system",
    "calc_exptime_spectrum",
    "calc_snr",
    "calc_snr_from_observation",
    "calc_snr_from_system",
    "observation_geometry",
]
