"""JAX ETC -- public API."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .components import (
    binary_background,
    detector_noise,
    exozodi_background,
    planet_signal,
    stellar_leakage,
    stellar_noise_floor,
    thermal_background,
    zodi_background,
)
from .public import (
    count_rates_ayo,
    count_rates_exosims_char,
    count_rates_exosims_det,
    count_rates_from_system_ayo,
    count_rates_from_system_exosims_char,
    count_rates_from_system_exosims_det,
    exptime_ayo,
    exptime_exosims_char,
    exptime_exosims_det,
    exptime_from_system_ayo,
    exptime_from_system_exosims_char,
    exptime_from_system_exosims_det,
    snr_ayo,
    snr_exosims_char,
    snr_exosims_det,
    snr_from_system_ayo,
    snr_from_system_exosims_char,
    snr_from_system_exosims_det,
)
from .scene import ETCScene
from .zodi import zodi_fn_ayo, zodi_fn_leinert

__all__ = [
    "ETCScene",
    "__version__",
    "binary_background",
    "count_rates_ayo",
    "count_rates_exosims_char",
    "count_rates_exosims_det",
    "count_rates_from_system_ayo",
    "count_rates_from_system_exosims_char",
    "count_rates_from_system_exosims_det",
    "detector_noise",
    "exozodi_background",
    "exptime_ayo",
    "exptime_exosims_char",
    "exptime_exosims_det",
    "exptime_from_system_ayo",
    "exptime_from_system_exosims_char",
    "exptime_from_system_exosims_det",
    "planet_signal",
    "snr_ayo",
    "snr_exosims_char",
    "snr_exosims_det",
    "snr_from_system_ayo",
    "snr_from_system_exosims_char",
    "snr_from_system_exosims_det",
    "stellar_leakage",
    "stellar_noise_floor",
    "thermal_background",
    "zodi_background",
    "zodi_fn_ayo",
    "zodi_fn_leinert",
]
