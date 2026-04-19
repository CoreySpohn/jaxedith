"""Tests for the jaxedith.zodi callable adapters."""

import jax.numpy as jnp
import pytest
from optixstuff import Exposure
from orbix.observatory import Observatory, ObservatoryL2Halo
from skyscapes.scene import SpectrumStar

from hwoutils import constants as const
from jaxedith.zodi import zodi_fn_ayo, zodi_fn_leinert


@pytest.fixture
def star():
    return SpectrumStar(
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
    return Observatory(orbit=ObservatoryL2Halo.from_default())


@pytest.fixture
def exposure():
    return Exposure(
        start_time_jd=jnp.asarray(2460000.5),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(20.0),
        position_angle_deg=jnp.asarray(0.0),
    )


def test_zodi_fn_ayo_matches_orbix_helper(observatory, exposure, star):
    from orbix.observatory import zodi_fzodi_ayo

    fz_new = zodi_fn_ayo(observatory, exposure, star)
    fz_ref = zodi_fzodi_ayo(exposure.central_wavelength_nm)
    assert jnp.allclose(fz_new, fz_ref)


def test_zodi_fn_ayo_returns_scalar(observatory, exposure, star):
    fz = zodi_fn_ayo(observatory, exposure, star)
    assert fz.shape == ()


def test_zodi_fn_leinert_matches_orbix_helper(observatory, exposure, star):
    from orbix.observatory import zodi_fzodi_leinert

    ra_rad = jnp.deg2rad(star.ra_deg)
    dec_rad = jnp.deg2rad(star.dec_deg)
    mjd = jnp.atleast_1d(exposure.start_time_jd)[0] - 2400000.5
    ecl_lat = observatory.orbit.ecliptic_latitude(ra_rad, dec_rad)
    sol_lon = observatory.orbit.solar_longitude(mjd, ra_rad, dec_rad)

    fz_new = zodi_fn_leinert(observatory, exposure, star)
    fz_ref = zodi_fzodi_leinert(
        exposure.central_wavelength_nm, ecl_lat, sol_lon
    )
    assert jnp.allclose(fz_new, fz_ref)


def test_zodi_fn_leinert_returns_scalar(observatory, exposure, star):
    fz = zodi_fn_leinert(observatory, exposure, star)
    assert fz.shape == ()
