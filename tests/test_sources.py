"""Aperture rates read the scene's node rows: a rough star is integrated, not sampled.

The truths are band integrals computed independently of the operator that
reduces the tabulated star: Gauss-Legendre points on every cell between the
table knots, with the star and the planet table read through their own point
interpolants. The star is a synthetic line-rich table (80 absorption lines
0.15 to 0.50 nm wide, up to 90 percent deep, on 0.1 nm knots), so four nodes
per band cannot see the lines: a point sample is wrong by percent. The
instrument response is a cubic in wavelength, the highest degree a four-node
piece integrates exactly, so a wrong distribution of the source over the nodes
shows, which a flat response would not.
"""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.conversions import arcsec_to_rad, lambda_d_to_arcsec
from hwoutils.epoch import Epoch
from hwoutils.numerics import sum_roundoff
from hwoutils.testing import ulps
from numpy.polynomial.legendre import leggauss
from optixstuff import IdealDetector, Instrument, Quadrature, SimplePrimary
from optixstuff.optical_elements import AbstractUniformElement
from optixstuff.spectral import Band, Resolution
from skyscapes.disk.base import AbstractDisk
from skyscapes.physical_model import TimeSeriesPhysicalModel
from skyscapes.scene import Planet, Scene, Star
from test_rates import (
    AREA,
    AREA_M2,
    DENSITY,
    DIAMETER_M,
    DIST_PC,
    SKY,
    TAU,
    _curves,
)

import jaxedith

BANDS = (Band(500.0, 555.0), Band(735.0, 785.0))
KNOTS_NM = np.arange(400.0, 900.0001, 0.1)
PLANET_NM = np.arange(300.0, 900.0001, 1.0)
DISK_RATIO = 4e-9


class _CubicThroughput(AbstractUniformElement):
    """A throughput that is a cubic in wavelength, degree Q - 1 of a four-node piece."""

    def get_throughput(self, wavelength_nm):
        """Return the cubic at ``wavelength_nm``."""
        u = (jnp.asarray(wavelength_nm) - 650.0) / 200.0
        return 0.5 * (1.0 + 0.3 * u - 0.2 * u**2 + 0.1 * u**3)

    def spectral_breaks_nm(self):
        """Declare no breaks."""
        return np.zeros(0)


class _WideInstrument(Instrument):
    """An instrument that declares a response field, as a coronagraph camera does."""

    def spectral_field_lod(self, band):
        """Return the half-diagonal of a 128-pixel detector at 0.25 lambda/D."""
        return 22.6


class _StarRelativeDisk(AbstractDisk):
    """A disk whose radiance is a fixed multiple of the host star's flux density."""

    ratio: float
    flux_scale: float = 1.0

    def _photon_radiance_at(
        self,
        wavelength_nm,
        time_d,
        east_arcsec,
        north_arcsec,
        *,
        star,
        dist_pc,
        inclination_deg,
        position_angle_deg,
    ):
        """Return ``ratio`` times the star's flux density, wavelength axes first."""
        wave = jnp.asarray(wavelength_nm)
        sky = jnp.broadcast_shapes(jnp.shape(east_arcsec), jnp.shape(north_arcsec))
        flux = star.photon_flux_density(wave, time_d, dist_pc=dist_pc)
        return self.ratio * flux.reshape(wave.shape + (1,) * len(sky)) * jnp.ones(sky)


def _line_star(flux_scale=1.0):
    """A flat-continuum star with 80 narrow, deep lines, linearly interpolated.

    A second epoch at 100 days is 30 percent brighter in a slow wavelength
    ripple, so a read at the wrong time shows.
    """
    rng = np.random.default_rng(11)
    flux = np.full(KNOTS_NM.size, 3000.0)
    for centre, width, depth in zip(
        rng.uniform(400.0, 900.0, 80),
        rng.uniform(0.15, 0.50, 80),
        rng.uniform(0.3, 0.9, 80),
        strict=True,
    ):
        flux *= 1.0 - depth * np.exp(-0.5 * ((KNOTS_NM - centre) / width) ** 2)
    return Star(
        wavelengths_nm=jnp.asarray(KNOTS_NM),
        times_d=jnp.array([0.0, 100.0]),
        flux_density_jy=jnp.stack(
            [flux, flux * (1.0 + 0.3 * np.cos(KNOTS_NM / 37.0))], axis=1
        ),
        interpolation_method="linear",
        flux_scale=flux_scale,
    )


def _contrast(nm):
    """An albedo-like contrast table with a 50 nm absorption band in each band."""
    bump = lambda c: np.clip(1 - np.abs(nm - c) / 25.0, 0, None)  # noqa: E731
    return 3e-10 * (1 - 0.7 * bump(527.0) - 0.7 * bump(760.0))


def _scene(star=None, *, disk=False):
    model = TimeSeriesPhysicalModel(
        PLANET_NM, np.zeros(1), _contrast(PLANET_NM)[None, :, None]
    )
    return Scene.from_classical(
        _line_star() if star is None else star,
        (Planet(jnp.ones(1), model),),
        star_mass_Msun=1.0,
        planet_mass_Msun=0.0,
        dist_pc=DIST_PC,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.array([1.0]),
        e=0.0,
        W_rad=0.0,
        i_rad=0.5,
        w_rad=0.0,
        M0_rad=0.3,
        disk=_StarRelativeDisk(DISK_RATIO) if disk else None,
    )


def _instrument(cls=Instrument):
    return cls(
        SimplePrimary(circumscribed_diameter_m=DIAMETER_M, collecting_area_m2=AREA_M2),
        (_CubicThroughput(),),
        None,
        IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)),
    )


def _build(scene, instrument=None):
    """The documented curves-route build: ``field_lod=0``, instrument as a break."""
    instrument = _instrument() if instrument is None else instrument
    models = [m for m in (instrument, scene.local_zodi, scene.disk) if m is not None]
    quadrature = Quadrature.build(BANDS, *models, field_lod=0)
    return quadrature, Resolution.build(quadrature, *scene.spectral_tables())


def _cells(band, *grids, n=4):
    """Gauss points and weights, n per cell between the grid knots inside the band."""
    inner = [g[(g > band.low_nm) & (g < band.high_nm)] for g in grids]
    edges = np.union1d(np.concatenate([[band.low_nm, band.high_nm], *inner]), [])
    x, w = leggauss(n)
    length = np.diff(edges)[:, None]
    return (edges[:-1, None] + length * (x + 1) / 2).ravel(), (length * w / 2).ravel()


def _star_at(star, x):
    return np.asarray(
        star.photon_flux_density(jnp.asarray(x), jnp.zeros(()), dist_pc=DIST_PC)
    )


def _truth(scene, instrument, band, weight):
    """Band integral of ``weight(x)`` times the area, the response and the star."""
    x, w = _cells(band, KNOTS_NM, PLANET_NM)
    area = np.asarray(instrument.effective_area_m2(jnp.asarray(x)))
    return np.sum(w * area * _star_at(scene.star, x) * weight(x))


def _read(scene, instrument, times=(0.0,)):
    quadrature, resolution = _build(scene, instrument)
    return quadrature, jaxedith.scene_rates(
        _curves(),
        scene,
        instrument,
        quadrature,
        jnp.asarray(times),
        resolution=resolution,
    )


@vv.case("aperture-photon-budget", "code-verification", basis="floating-point")
def test_scene_rates_integrate_a_line_rich_star_and_an_absorbing_planet():
    """Fails if a rate samples the star at the nodes, or the planet's table through it.

    Truth: eight-fold Gauss points on every cell between the star and planet
    knots, the star through its own interpolant, the contrast table through
    ``numpy.interp``, the cubic response through the instrument, times the
    constant curve value at the band centre (and, for the exozodi, the
    aperture solid angle of the curves). Frozen tolerance 1e-4 relative; the
    rule is exact on each cell, so the measured error is floating point
    (6e-16). Measured on this fixture, a point-sampled star misses star
    leakage and exozodi by 1.4e-2 and 5.6e-2 (the two bands), a point-sampled
    contrast by 3.1e-2 and 2.3e-2, and both together miss the planet by
    7.4e-2 and 8.9e-2.
    """
    scene, instrument = _scene(disk=True), _instrument()
    quadrature, record = _read(scene, instrument)
    assert record.planet.shape == (2, 4, 1, 1)
    for index, band in enumerate(BANDS):
        lod_arcsec = float(lambda_d_to_arcsec(1.0, band.center_nm, DIAMETER_M))
        solid_angle = AREA * float(arcsec_to_rad(lod_arcsec)) ** 2
        planet = _truth(
            scene,
            instrument,
            band,
            lambda x: np.interp(x, PLANET_NM, _contrast(PLANET_NM)) * TAU,
        )
        leakage = _truth(scene, instrument, band, lambda x: DENSITY * AREA)
        disk = DISK_RATIO * SKY * solid_angle
        exozodi = _truth(scene, instrument, band, lambda x, disk=disk: disk + 0 * x)
        for name, truth in (
            ("planet", planet),
            ("star_leakage", leakage),
            ("exozodi", exozodi),
        ):
            got = quadrature.integrate(getattr(record, name)[:, :, 0, 0])[index]
            np.testing.assert_allclose(got, truth, rtol=1e-4, err_msg=name)


@vv.case("aperture-photon-budget", "code-verification", basis="exact-algebra")
def test_rates_reads_the_node_rows_and_not_the_scene_star():
    """Fails if ``rates`` evaluates a spectrum itself instead of reading ``sources``.

    The rows come from a scene whose star is scaled by 2 and the call is made
    with the unscaled scene: star leakage and the star-relative exozodi follow
    the rows, and so does the planet, which is the star row times the contrast
    argument. Each doubles exactly. Basis: products of float64 scalars, rtol 1e-12.
    """
    base, doubled = _scene(disk=True), _scene(_line_star(2.0), disk=True)
    instrument = _instrument()
    quadrature, resolution = _build(base, instrument)
    args = (_curves(), base, instrument, quadrature, 0.03, 0.04, 3e-10)

    def read(rows):
        return jaxedith.rates(
            *args[:-1],
            contrast=args[-1],
            time_d=0.0,
            sources=rows.sources(quadrature, resolution, 0.0),
        )

    once, twice = read(base), read(doubled)
    for name, factor in (("star_leakage", 2.0), ("exozodi", 2.0), ("planet", 2.0)):
        np.testing.assert_allclose(
            getattr(twice, name), factor * getattr(once, name), rtol=1e-12
        )


@vv.case("aperture-photon-budget", "code-verification", basis="exact-algebra")
def test_scene_rates_reads_the_sources_once_for_all_planets_and_times(monkeypatch):
    """Fails if the sources are recomputed inside the map over planets and times.

    ``Scene.sources`` is called exactly once, with the whole time vector and no
    batch dimension added by a map; a call inside the mapped read would see a
    single time. Exact counts.
    """
    seen = []
    original = Scene.sources

    def counting(self, quadrature, resolution, time_d, **kwargs):
        seen.append(jnp.shape(time_d))
        return original(self, quadrature, resolution, time_d, **kwargs)

    monkeypatch.setattr(Scene, "sources", counting)
    scene, instrument = _scene(), _instrument()
    _read(scene, instrument, times=(0.0, 10.0, 20.0))
    assert seen == [(3,)]


@vv.case("aperture-photon-budget", "code-verification", basis="exact-algebra")
def test_scene_rates_equal_the_per_time_rates_and_run_under_jit_and_grad():
    """Fails if a time of ``scene_rates`` reads another time's rows, or a trace breaks.

    Three times, a star that varies in time through its scale at the node rows:
    each time of the vmapped read equals ``rates`` fed ``sources`` of that time
    alone. Allow 16 rounding operations for the temporal interpolation,
    cubic throughput and aperture products, plus the pairwise-sum depths
    of the star and contrast table reductions on each path. The derivative
    with respect to the star's scale equals the rate because the planet
    signal is linear in that scale. That comparison allows eight products
    and the pairwise sum of eight band/node rates on both paths.
    """
    scene, instrument = _scene(disk=True), _instrument()
    quadrature, resolution = _build(scene, instrument)
    times = jnp.array([0.0, 40.0, 90.0])
    got = jax.jit(
        lambda t: jaxedith.scene_rates(
            _curves(),
            scene,
            instrument,
            quadrature,
            t,
            resolution=resolution,
        )
    )(times)
    sources = scene.sources(quadrature, resolution, times)
    east, north = scene.positions(times)
    rtol = float(
        ulps(16, 1.0)
        + 2 * (sum_roundoff(1.0, KNOTS_NM.size) + sum_roundoff(1.0, PLANET_NM.size))
    )
    for t in range(3):
        want = jaxedith.rates(
            _curves(),
            scene,
            instrument,
            quadrature,
            east[0, t],
            north[0, t],
            contrast=sources.contrast[:, :, 0, t],
            time_d=times[t],
            sources=scene.sources(quadrature, resolution, times[t]),
        )
        for name in ("planet", "star_leakage", "exozodi"):
            np.testing.assert_allclose(
                getattr(got, name)[:, :, 0, t], getattr(want, name), rtol=rtol
            )

    def total(scale):
        tuned = eqx.tree_at(lambda s: s.star.flux_scale, scene, scale)
        return jaxedith.scene_rates(
            _curves(),
            tuned,
            instrument,
            quadrature,
            times[:1],
            resolution=resolution,
        ).planet.sum()

    scale = jnp.asarray(1.0)
    gradient_rtol = float(
        ulps(8, scale) + 2 * sum_roundoff(scale, quadrature.nodes_nm.size)
    )
    np.testing.assert_allclose(jax.grad(total)(scale), total(scale), rtol=gradient_rtol)


@vv.case("aperture-photon-budget", "code-verification", basis="exact-algebra")
def test_the_curves_route_build_keeps_the_nodes_of_the_order():
    """Fails if the documented curves-route build lets the response add nodes.

    An instrument that declares a 22.6 lambda/D field, passed as a break
    model, with ``field_lod=0`` gives every band exactly the nodes of a build
    that has no field, here four each. The same build with the field left to
    the instrument must have more, which keeps this test from passing for a
    build that ignores the argument. Exact counts.
    """
    instrument = _instrument(_WideInstrument)
    flat = Quadrature.build(BANDS, order=4)
    curves = Quadrature.build(BANDS, instrument, order=4, field_lod=0)
    declared = Quadrature.build(BANDS, instrument, order=4, field_lod=None)
    assert curves.nodes_nm.shape == flat.nodes_nm.shape == (2, 4)
    np.testing.assert_array_equal(curves.nodes_nm, flat.nodes_nm)
    assert declared.nodes_nm.shape[1] > 4


@vv.case("aperture-photon-budget", "code-verification", basis="exact-algebra")
def test_rates_refuse_sources_of_another_layout():
    """Fails if ``rates`` accepts rows for another quadrature or several times.

    The rows must be those of one time on this quadrature; a mismatch would
    contract the wrong nodes silently. Exact: the call raises before any read.
    """
    scene, instrument = _scene(), _instrument()
    quadrature, resolution = _build(scene, instrument)
    other = Quadrature.build(BANDS[:1], instrument, field_lod=0)
    several = scene.sources(quadrature, resolution, jnp.array([0.0, 1.0]))
    wrong = scene.sources(other, Resolution.build(other, *scene.spectral_tables()), 0.0)
    for rows in (several, wrong):
        with pytest.raises(ValueError, match="sources hold star rows"):
            jaxedith.rates(
                _curves(),
                scene,
                instrument,
                quadrature,
                0.03,
                0.04,
                contrast=3e-10,
                time_d=0.0,
                sources=rows,
            )
