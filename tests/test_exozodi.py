"""Absolute scalar-dialect anchors for analytic aperture exozodi."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
from conftest import make_path, make_system
from hwoutils import constants
from hwoutils.grids import PixelGrid, SkyOrientation
from hwoutils.spectral import Band
from skyscapes.disk import ExovistaDisk

import jaxedith as je
from jaxedith.primitives import exozodi_rate


@vv.case("aperture-photon-budget", "code-verification")
@vv.seam("jaxedith", "optixstuff")
def test_analytic_default_matches_scalar_electrons():
    """Fails on wrong zero point, colour, sr units, yield or missing default.

    Basis: old scalar AYO dialect with exact AB zero point. For the 0.002-nm
    band, centre-times-width error is (width/550)^2/12 < 1.2e-12 for 1/lambda;
    linear QE adds <2e-12. The 1e-9 relative bound includes rounding.
    """
    system, path, band = (
        make_system(),
        make_path(chromatic=False),
        Band(549.999, 550.001),
    )
    q = je.aperture_quadrature(system, path, band, zodi=None)
    rates = je.aperture_rates(system, path, band, time_d=0.0, planet_index=0, zodi=None)
    # AB definition in cgs converted to Jy; independent of library conversions.
    zero_jy = 10 ** (-0.4 * 48.60) * constants.m22cm2 / constants.J2erg / constants.Jy
    f0 = (
        zero_jy * constants.Jy / (constants.h * (550 * constants.nm2m)) * constants.nm2m
    )
    expected = exozodi_rate(
        f0,
        3 * 10 ** (-0.4 * 22),
        (550 * constants.nm2m) / 8 / constants.arcsec2rad,
        0.65,
        40.0,
        0.4 * 0.8 * 0.9,
        band.width_nm,
        1,
        np.pi * 0.7**2,
        10.0,
        0.1,
    )
    np.testing.assert_allclose(
        path.detector.electrons(rates, q).exozodi, expected, rtol=1e-9, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_separation_law_ignored_disk_and_disabled():
    """Fails if distance or angle is omitted, or default renders a disk.

    Basis: powers-of-two distances/separations give exactly r^-2; the same
    angular position at any PA agrees to 1e-12 for trig roundoff. JIT/vmap
    agrees at 1e-12, with no grid supplied and a deliberately off-grid disk.
    """
    grid = PixelGrid.regular((1, 1), 0.01, "arcsec", orientation=SkyOrientation(0.0))
    disk = ExovistaDisk(grid, jnp.array([500.0, 550.0, 600.0]), jnp.ones((3, 1, 1)))
    system, path, band = (
        make_system(disk=disk),
        make_path(chromatic=False),
        Band(549, 551),
    )
    q = je.aperture_quadrature(system, path, band, zodi=None)
    empty = eqx.tree_at(lambda s: s.disk, system, None)
    empty_q = je.aperture_quadrature(empty, path, band, zodi=None)
    np.testing.assert_array_equal(q.nodes_nm, empty_q.nodes_nm)
    np.testing.assert_array_equal(q.weights_nm, empty_q.weights_nm)
    disabled_q = je.aperture_quadrature(system, path, band, zodi=None, exozodi=None)
    np.testing.assert_array_equal(disabled_q.nodes_nm, empty_q.nodes_nm)
    scene_q = je.aperture_quadrature(system, path, band, zodi=None, exozodi="scene")
    assert scene_q.nodes_nm.size == 2 * empty_q.nodes_nm.size
    prepared = je.prepare_aperture(path, band, q)

    def read(distance, separation, angle=0.0, **kwargs):
        scene = eqx.tree_at(lambda s: s.dist_pc, system, distance)
        return je.aperture_rates(
            scene,
            path,
            band,
            time_d=0.0,
            separation_arcsec=separation,
            position_angle_deg=angle,
            zodi=None,
            prepared_aperture=prepared,
            **kwargs,
        ).exozodi

    base = read(10.0, 0.1)
    np.testing.assert_array_equal(read(20.0, 0.1), base / 4)
    np.testing.assert_array_equal(read(10.0, 0.2), base / 4)
    np.testing.assert_array_equal(read(20.0, 0.05), base)
    np.testing.assert_allclose(read(10.0, 0.1, 37.0), base, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(read(10.0, 0.1, disk_grid=grid), base)
    np.testing.assert_array_equal(
        je.aperture_rates(
            empty,
            path,
            band,
            time_d=0.0,
            separation_arcsec=0.1,
            zodi=None,
            prepared_aperture=prepared,
        ).exozodi,
        base,
    )
    np.testing.assert_array_equal(read(10.0, 0.0, exozodi=None), 0)
    distances, separations = jnp.array([10.0, 20.0, 10.0]), jnp.array([0.1, 0.1, 0.2])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(read))(distances, separations),
        jnp.stack([base, base / 4, base / 4]),
        rtol=1e-12,
        atol=0,
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_exozodi_knots_split_quadrature():
    """Fails if a supplied exozodi spectral kink is omitted from quadrature.

    Basis: a piecewise linear photon radiance integrates exactly with one
    node per segment; 1e-12 allows floating-point area arithmetic.
    """
    from conftest import TabulatedExozodi

    system, path, band = make_system(), make_path(chromatic=False), Band(500, 600)
    model = TabulatedExozodi(
        jnp.array([500.0, 537.0, 600.0]), jnp.array([1e10, 4e10, 2e10])
    )
    q = je.aperture_quadrature(
        system, path, band, quadrature_order=1, zodi=None, exozodi=model
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(q.weights_nm)[np.asarray(q.nodes_nm) < 537]), 37, rtol=1e-12
    )
    rates = je.aperture_rates(
        system,
        path,
        band,
        time_d=0.0,
        separation_arcsec=0.1,
        quadrature_order=1,
        zodi=None,
        exozodi=model,
    )
    expected = (
        40
        * 0.4
        * 0.65
        * np.pi
        * 0.7**2
        * ((550 * constants.nm2m) / 8) ** 2
        * (37 * 2.5e10 + 63 * 3e10)
    )
    np.testing.assert_allclose(rates.integrate(q).exozodi, expected, rtol=1e-12, atol=0)
