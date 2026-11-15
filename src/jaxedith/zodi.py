"""Zodi callable adapters.

Each ``zodi_fn_*`` matches the callable contract documented on the
``*_from_system_*`` wrappers:

    zodi_fn(observatory, exposure, star, *, epoch) -> Fzodi  # scalar

Implementations are thin wrappers around the ``_fzodi_{ayo,leinert}``
helpers defined below, which in turn convert a ``zodi`` library AB magnitude
into the EXOSIMS/pyEDITH ``Fzodi = 10**(-0.4*mag)`` convention. Callers
pass whichever they want as a kwarg to the wrapper; the default on every
wrapper is ``zodi_fn_ayo`` (preserves legacy ``zodi_mode='ayo'``
behavior without a string argument).
"""

import jax
import jax.numpy as jnp
import zodi
from zodi import units

from jaxedith._types import ScalarArray


@jax.jit
def _fzodi_ayo(wavelength_nm):
    """AYO-default Fzodi = 10**(-0.4 * mag); arcsec^-2, dimensionless."""
    return units.mag_to_flux_ratio(zodi.ayo_default_ab_mag(jnp.asarray(wavelength_nm)))


@jax.jit
def _fzodi_leinert(wavelength_nm, ecliptic_lat_deg=0.0, solar_lon_deg=135.0):
    """Position-dependent Leinert Fzodi = 10**(-0.4 * mag); arcsec^-2."""
    mag = zodi.surface_brightness_ab_mag(
        jnp.asarray(solar_lon_deg),
        jnp.asarray(ecliptic_lat_deg),
        jnp.asarray(wavelength_nm),
        anchor="table19",
    )
    return units.mag_to_flux_ratio(mag)


def zodi_fn_ayo(observatory, exposure, star, *, epoch) -> ScalarArray:
    """AYO default local zodi: position-independent wavelength-only helper.

    This is the local zodiacal light, whose color is the Leinert (scattered
    sunlight) color. It is not an exozodi level: the exozodi takes the host
    star's own color, so pass ``Fexozodi`` computed from the host's flux, for
    example with ``zodi.exozodi_flux_ratio_band``.

    ``observatory``, ``star`` and the host-side ``epoch`` are accepted for
    interface conformance and ignored. Returns scalar ``Fzodi`` [arcsec^-2].
    """
    return _fzodi_ayo(exposure.central_wavelength_nm)


def zodi_fn_leinert(observatory, exposure, star, *, epoch) -> ScalarArray:
    """Leinert position-dependent zodi.

    Uses ``star.ra_deg`` / ``star.dec_deg`` and the first element of
    ``jnp.atleast_1d(exposure.start_time_d)`` in days from ``epoch``, so
    the callable accepts scalar or vector times. ``epoch`` is a host-side
    ``hwoutils.epoch.Epoch`` (the scene epoch for system wrappers). Its
    split date is reduced to one static float offset before traced arithmetic.
    """
    ra_rad = jnp.deg2rad(star.ra_deg)
    dec_rad = jnp.deg2rad(star.dec_deg)
    epoch_mjd = (epoch.jd1 - 2400000.5) + epoch.jd2
    time_d = jnp.atleast_1d(exposure.start_time_d)[0]
    mjd = time_d + epoch_mjd
    ecl_lat_deg = observatory.orbit.ecliptic_latitude_deg(mjd, ra_rad, dec_rad)
    # Leinert table input is the helio-ecliptic longitude difference
    # Delta_lambda_sun, not the 3D solar elongation. For ecliptic-plane
    # targets these coincide; for off-ecliptic targets only the longitude
    # difference is correct (see orbix.ObservatoryL2Halo docstrings).
    sol_lon_deg = observatory.orbit.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad)
    return _fzodi_leinert(exposure.central_wavelength_nm, ecl_lat_deg, sol_lon_deg)
