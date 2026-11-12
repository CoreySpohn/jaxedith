"""Tests for the jaxedith.zodi callable adapters."""

import jax.numpy as jnp
import pytest
from hwoutils import constants as const
from optixstuff import ExposureConfig
from orbix.observatory import Observatory, ObservatoryL2Halo
from skyscapes.scene import Star

from jaxedith.zodi import zodi_fn_ayo, zodi_fn_leinert


@pytest.fixture
def star():
    """Synthetic Solar-analog star at (RA=90, Dec=30) deg, 10 pc."""
    return Star(
        Ms_kg=const.Msun2kg,
        dist_pc=10.0,
        ra_deg=90.0,
        dec_deg=30.0,
        diameter_arcsec=0.0,
        luminosity_lsun=1.0,
        wavelengths_nm=jnp.array([400.0, 800.0]),
        times_jd=jnp.array([0.0, 1.0]),
        flux_density_jy=jnp.full((2, 2), 3631.0),
    )


@pytest.fixture
def observatory():
    """Observatory wrapping the default L2 halo orbit."""
    return Observatory(orbit=ObservatoryL2Halo.from_default())


@pytest.fixture
def exposure():
    """1 h, 500 nm exposure at MJD 60000."""
    return ExposureConfig(
        start_time_jd=jnp.asarray(2460000.5),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(20.0),
        position_angle_deg=jnp.asarray(0.0),
    )


def test_zodi_fn_ayo_matches_fzodi_helper(observatory, exposure, star):
    """zodi_fn_ayo yields Fzodi = 10**(-0.4*mag) from the zodi AYO magnitude."""
    from zodi import ayo_default_ab_mag

    fz_new = zodi_fn_ayo(observatory, exposure, star)
    mag = ayo_default_ab_mag(exposure.central_wavelength_nm)
    fz_ref = 10.0 ** (-0.4 * mag)
    assert jnp.allclose(fz_new, fz_ref)


def test_zodi_fn_leinert_matches_fzodi_helper(observatory, exposure, star):
    """zodi_fn_leinert matches _fzodi_leinert with helio-ecliptic geometry."""
    from jaxedith.zodi import _fzodi_leinert

    ra_rad = jnp.deg2rad(star.ra_deg)
    dec_rad = jnp.deg2rad(star.dec_deg)
    mjd = jnp.atleast_1d(exposure.start_time_jd)[0] - 2400000.5
    ecl_lat = observatory.orbit.ecliptic_latitude_deg(mjd, ra_rad, dec_rad)
    sol_lon = observatory.orbit.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad)

    fz_new = zodi_fn_leinert(observatory, exposure, star)
    fz_ref = _fzodi_leinert(exposure.central_wavelength_nm, ecl_lat, sol_lon)
    assert jnp.allclose(fz_new, fz_ref)


@pytest.mark.parametrize("wavelength_nm", [500.0, 910.0])
def test_zodi_fn_ayo_color_matches_leinert_color(observatory, star, wavelength_nm):
    """zodi_fn_ayo and zodi_fn_leinert scale with wavelength identically.

    The AYO default fixes the brightness at V and carries it with the
    Table 19 color; the Leinert path at fixed geometry uses the same
    color. Their ratios to the V-band value must match, which fails by
    (lambda / 550)**2 (2.7 at 910 nm) if the color is not converted to
    the AB (per unit frequency) dialect. Tolerance basis: shared tables,
    so only float rounding remains.
    """

    def exposure_at(lam):
        return ExposureConfig(
            start_time_jd=jnp.asarray(2460000.5),
            exposure_time_s=jnp.asarray(3600.0),
            central_wavelength_nm=jnp.asarray(lam),
            bin_width_nm=jnp.asarray(20.0),
            position_angle_deg=jnp.asarray(0.0),
        )

    ayo = zodi_fn_ayo(observatory, exposure_at(wavelength_nm), star) / zodi_fn_ayo(
        observatory, exposure_at(550.0), star
    )
    leinert = zodi_fn_leinert(
        observatory, exposure_at(wavelength_nm), star
    ) / zodi_fn_leinert(observatory, exposure_at(550.0), star)
    assert jnp.allclose(ayo, leinert, rtol=1e-5)
