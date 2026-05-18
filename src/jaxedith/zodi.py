"""Zodi callable adapters.

Each ``zodi_fn_*`` matches the callable contract documented on the
``*_from_system_*`` wrappers:

    zodi_fn(observatory, exposure, star) -> Fzodi  # scalar

Implementations are thin adapters around orbix's
``zodi_fzodi_{ayo,leinert}`` helpers. Callers pass whichever they want
as a kwarg to the wrapper; the default on every wrapper is
``zodi_fn_ayo`` (preserves legacy ``zodi_mode='ayo'`` behavior without
a string argument).
"""

import jax.numpy as jnp
from orbix.observatory import zodi_fzodi_ayo, zodi_fzodi_leinert


def zodi_fn_ayo(observatory, exposure, star):
    """AYO default zodi: position-independent wavelength-only helper.

    ``observatory`` and ``star`` are accepted for interface conformance
    and ignored. Returns scalar ``Fzodi`` [arcsec^-2].
    """
    del observatory, star
    return zodi_fzodi_ayo(exposure.central_wavelength_nm)


def zodi_fn_leinert(observatory, exposure, star):
    """Leinert position-dependent zodi.

    Uses ``star.ra_deg`` / ``star.dec_deg`` and the first element of
    ``jnp.atleast_1d(exposure.start_time_jd)`` so the callable is safe
    whether the exposure carries a scalar or a vector of epochs.
    """
    ra_rad = jnp.deg2rad(star.ra_deg)
    dec_rad = jnp.deg2rad(star.dec_deg)
    jd = jnp.atleast_1d(exposure.start_time_jd)[0]
    mjd = jd - 2400000.5
    ecl_lat_deg = observatory.orbit.ecliptic_latitude_deg(mjd, ra_rad, dec_rad)
    # Leinert table input is the helio-ecliptic longitude difference
    # Delta_lambda_sun, not the 3D solar elongation. For ecliptic-plane
    # targets these coincide; for off-ecliptic targets only the longitude
    # difference is correct (see orbix.ObservatoryL2Halo docstrings).
    sol_lon_deg = observatory.orbit.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad)
    return zodi_fzodi_leinert(exposure.central_wavelength_nm, ecl_lat_deg, sol_lon_deg)
