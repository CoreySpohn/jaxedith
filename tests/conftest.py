# ruff: noqa: E402
"""Shared fixtures for jaxedith tests."""

import importlib.util
import os

import jax

jax.config.update("jax_enable_x64", True)

from hypothesis import settings
from jaxtyping import install_import_hook

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

# Library checks (finite, non-negative rates; NaN-free exposure division
# intermediates) raise during tests. SNR solvers and overheads are not checked.
os.environ.setdefault("JAXEDITH_DEBUG_CHECKS", "1")

# Runtime shape and dtype checking of the jaxtyping annotations, tests only.
# The hook must be installed before jaxedith is first imported. Set
# JAXEDITH_TYPECHECK=0 to run without it.
if os.environ.get("JAXEDITH_TYPECHECK", "1") == "1":
    install_import_hook("jaxedith", "beartype.beartype")


import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from hwoutils.conversions import jy_to_photons_per_nm_per_m2
from hwoutils.epoch import Epoch
from hwoutils.grids import PixelGrid
from hwoutils.spectral import Band
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import Planet, System
from skyscapes.scene.star import AbstractStar
from yippy import Circular


class ABStar(AbstractStar):
    """Analytic flat-Fnu fixture, with the real AbstractStar photon contract."""

    flux_jy: object = 36.31
    luminosity_Lsun: float = 1.0

    def photon_flux_density(self, wavelength_nm, time_d, *, dist_pc):
        return jy_to_photons_per_nm_per_m2(self.flux_jy, wavelength_nm)


class ChromaticCoronagraph(ox.ScalarCoronagraph):
    """Synthetic linear chromatic throughput on a prepared scalar record."""

    def throughput(self, separation_lod, wavelength_nm=None, **kwargs):
        return 0.30 + 0.0006 * (jnp.asarray(wavelength_nm) - 550)


def make_system(star=None, *, chromatic=True, disk=None):
    waves = jnp.array([500.0, 600.0])
    contrast = 1e-10 * (1 - (0.004 if chromatic else 0) * (waves - 550))
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        physical_model=GridPhysicalModel(
            wavelengths_nm=waves,
            phase_angle_deg=jnp.array([0.0, 180.0]),
            contrast_grid=jnp.broadcast_to(contrast[None, :, None], (1, 2, 2)),
        ),
    )
    return System.from_classical(
        star=ABStar() if star is None else star,
        planets=(planet,),
        star_mass_Msun=1.0,
        planet_mass_Msun=3e-6,
        dist_pc=10.0,
        epoch=Epoch(2451545.0),
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([0.0]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        disk=disk,
    )


def make_path(*, chromatic=True, pixel_scale_lod=0.25, leakage=0.0, area_lod2=None):
    primary = ox.SimplePrimary(circumscribed_diameter_m=8.0, collecting_area_m2=40.0)
    cls = ChromaticCoronagraph if chromatic else ox.ScalarCoronagraph
    area = np.pi * 0.7**2 if area_lod2 is None else area_lod2
    coro = cls(
        grid=PixelGrid.focal(24, pixel_scale_lod, primary.frame(550)),
        throughput_fraction=0.3,
        aperture_area_lod2=area,
        stellar_leakage_density_lod2=leakage / area,
        sky_transmission=0.65,
        rule=Circular(0.7),
    )
    detector = ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(12, 12),
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=3e-4,
        clock_induced_charge_e_per_pixel_frame=0.005,
        read_noise_e_rms=0.2,
    )
    return ox.OpticalPath(primary, (ox.ConstantThroughput(0.4),), coro, detector)


@pytest.fixture(scope="module")
def example():
    from jaxedith import aperture_quadrature, aperture_rates

    system, path, band = make_system(), make_path(), Band(500, 600)
    quadrature = aperture_quadrature(system, path, band, zodi=None)
    rates = aperture_rates(
        system, path, band, time_d=0.0, planet_index=0, quadrature=quadrature, zodi=None
    )
    return system, path, band, quadrature, rates


def oracle(order=64):
    """Independent SI primitives and a 64-node Gauss-Legendre integral."""
    x, weights = np.polynomial.legendre.leggauss(order)
    wave = 550 + 50 * x
    flux = 36.31 * 1e-26 / (6.62607015e-34 * wave * 1e-9) * 1e-9
    planet = 40 * 0.4 * (0.3 + 0.0006 * (wave - 550)) * flux
    planet *= 1e-10 * (1 - 0.004 * (wave - 550))
    return (
        50 * weights @ planet,
        50 * weights @ (planet * (0.8 - 0.002 * (wave - 550)) * 0.9),
    )


from hwoutils.grids import Image
from optixstuff.coronagraph import AbstractTableCoronagraph


class TabulatedCoronagraph(AbstractTableCoronagraph):
    """Small physical transmission map with a separately calibrated record."""

    grid: PixelGrid
    sky_trans: Image
    psf_datacube: object
    record: object
    outer_working_angle_lod: float | None = None

    def curves(self, rule=Circular(0.7), *, leakage_rule="aperture_mean"):  # noqa: B008 - frozen rule
        if rule != self.record.rule:
            raise ValueError("unprepared rule")
        return self.record

    def stellar_intens(self, stellar_diam_lod):
        raise NotImplementedError("only transmission is needed by this fixture")

    def create_psfs(self, x_lod, y_lod):
        raise NotImplementedError("only transmission is needed by this fixture")
