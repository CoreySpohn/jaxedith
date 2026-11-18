"""Absolute photon/electron anchors for the shared aperture reader."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from conftest import ABStar, FlatCompanions, make_path, make_system, oracle
from hwoutils import constants
from optixstuff import prepare_aperture
from optixstuff.grids import Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Quadrature, Transmission
from skyscapes.disk import GraterDisk
from skyscapes.scene import Star
from zodi import SurfaceBrightnessDust
from zodi.jax import AYOExozodi, PrecomputedZodi

from jaxedith import (
    ApertureRates,
    PostProcessing,
    aperture_rates,
    snr,
)


@vv.case("aperture-photon-budget", "code-verification")
@vv.seam("skyscapes+optixstuff", "jaxedith")
def test_worked_example(example):
    """Fails if photon integration or detector count variance changes units.

    Basis: order 4 versus 8 error <3e-11; 1e-9 allows integration rounding.
    Printed-digit checks use half a last displayed decimal place.
    """
    _, path, rates = example
    quad = rates.mean.quadrature
    photons, electrons = oracle()
    assert abs(oracle(4)[1] / oracle(8)[1] - 1) < 3e-11
    np.testing.assert_allclose(
        rates.mean.integrate().planet, photons, rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        path.detector.electrons(rates.mean).planet, electrons, rtol=1e-9, atol=0
    )
    assert photons == pytest.approx(0.0477819, abs=5e-8)
    assert electrons == pytest.approx(0.0346767, abs=5e-8)
    assert (36.31 * constants.Jy) / (
        constants.h * (550 * constants.nm2m)
    ) * constants.nm2m == pytest.approx(9.9634e5, abs=5)
    # A constant spectral background gives exactly 0.004 e/s after yield.
    mean = eqx.tree_at(
        lambda r: r.stellar_companions,
        rates.mean,
        jnp.full(quad.nodes_nm.shape, 0.004 / (100 * 0.8 * 0.9)),
    )
    rates = ApertureRates(mean, mean, jnp.array([12.0]), jnp.array([12.0]))
    plan = ox.FramePlan(100, 36)
    signal = electrons * 3600
    variance = signal + 0.004 * 3600 + 12 * (3e-4 * 3600 + 0.005 * 36 + 0.2**2 * 36)
    _, actual_variance = path.detector.mean_and_variance(
        path.detector.electrons(rates.mean), plan, 12
    )
    np.testing.assert_allclose(actual_variance, variance, rtol=1e-9, atol=0)
    actual = snr(rates, path.detector, plan, post_processing=PostProcessing(0, 0))
    np.testing.assert_allclose(actual, signal / np.sqrt(variance), rtol=1e-9, atol=0)
    assert signal == pytest.approx(124.84, abs=0.005)
    assert variance == pytest.approx(171.64, abs=0.005)
    assert actual[0] == pytest.approx(9.53, abs=0.005)


@vv.case("aperture-photon-budget", "code-verification")
def test_node_qe_cannot_be_replaced_by_band_mean(example):
    """Fails only if spectral QE is averaged before multiplying photon nodes.

    Basis: independent 64-node integral, 1e-9 quadrature bound; mutant >0.5%.
    """
    _, path, rates = example
    detector = eqx.tree_at(
        lambda d: d.survival_fraction, path.detector, jnp.asarray(1.0)
    )
    expected = oracle()[1] / 0.9
    np.testing.assert_allclose(
        detector.electrons(rates.mean).planet, expected, rtol=1e-9, atol=0
    )
    wrong = oracle()[0] * 0.8
    assert abs(wrong / expected - 1) > 0.005


@vv.case("aperture-photon-budget", "code-verification")
def test_survival_is_applied_once_after_reader(example):
    """Fails only if survival is absent, duplicated, or leaks into photons.

    Basis: constant QE eliminates spectral averaging; exact algebra at 1e-12.
    """
    system, path, rates = example
    q = rates.mean.quadrature
    detector = ox.Detector(
        pixel_scale_arcsec=0.01, shape=(1, 1), qe=0.8, survival_fraction=0.6
    )
    changed_path = eqx.tree_at(lambda p: p.detector, path, detector)
    changed = aperture_rates(
        system, changed_path, q, time_d=0.0, planet_index=0, zodi=None
    )
    np.testing.assert_array_equal(changed.mean.planet, rates.mean.planet)
    expected = rates.mean.integrate().planet * 0.8 * 0.6
    np.testing.assert_allclose(
        detector.electrons(changed.mean).planet, expected, rtol=1e-12, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_chromatic_bias(example):
    """Fails if the reader uses centre times width; 1e-9 quadrature basis."""
    _, path, rates = example
    centre = (
        40
        * 0.4
        * 0.3
        * (
            (36.31 * constants.Jy)
            / (constants.h * (550 * constants.nm2m))
            * constants.nm2m
        )
        * 1e-10
        * 100
        * 0.8
        * 0.9
    )
    expected = oracle()[1]
    assert 0.007 < abs(centre / expected - 1) < 0.012
    np.testing.assert_allclose(
        path.detector.electrons(rates.mean).planet, expected, rtol=1e-9, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_leakage_is_aperture_sum(example):
    """Fails a per-pixel/per-lod^2 substitution (16x at 0.25 lod).

    Basis: integral of 1/lambda is a logarithm; 1e-9 quadrature bound.
    """
    system, _, reference = example
    band = reference.mean.quadrature.bands[0]
    expected = (
        40 * 0.4 * 2e-10 * (36.31 * constants.Jy) / constants.h * np.log(600 / 500)
    )
    values = []
    for step in (0.25, 0.125):
        path = make_path(pixel_scale_lod=step, leakage=2e-10)
        q = Quadrature.build((band,), system, path, AYOExozodi())
        rates = aperture_rates(system, path, q, time_d=0.0, planet_index=0, zodi=None)
        values.append(rates.mean.integrate().star_leakage)
    np.testing.assert_allclose(values, expected, rtol=1e-9, atol=0)
    np.testing.assert_array_equal(values[0], values[1])


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
def test_disk_absolute_scale_and_centre_approximation():
    """Fails if disk is omitted, host-scaled twice, or loses sr conversion.

    Basis: calibrated face-on reference equals typed 22 mag law to 1e-10;
    footprint difference <5% for this smooth ring and 0.7 lod radius.
    """
    disk = GraterDisk(
        sma_AU=1.0,
        alpha_in=2.0,
        alpha_out=-2.0,
        ksi0_AU=0.05,
        gamma=2.0,
        beta=1.0,
        rmin_AU=0.1,
        rmax_AU=5.0,
        wavelengths_nm=jnp.array([500.0, 600.0]),
        g_HG_grid=jnp.zeros(2),
        Ag_grid=jnp.ones(2),
        amplitude=SurfaceBrightnessDust(level=3.0),
        n_slices_los=24,
    )
    star = ABStar(flux_jy=3636 * 10 ** (-0.4 * 4.83))
    system, path, band = (
        make_system(star, disk=disk),
        make_path(chromatic=False),
        Band(549.99, 550.01),
    )
    q = Quadrature.build((band,), system, path)
    kwargs = dict(
        time_d=0.0,
        separation_arcsec=0.1,
        position_angle_deg=90.0,
        zodi=None,
    )
    rates = aperture_rates(system, path, q, **kwargs)
    arcsec_rad = constants.arcsec2rad
    f0 = (
        (3636 * constants.Jy)
        / (constants.h * np.asarray(q.nodes_nm) * constants.nm2m)
        * constants.nm2m
    )
    radiance = 3 * 10 ** (-0.4 * 22) * f0 / arcsec_rad**2
    omega = np.pi * 0.7**2 * (550 * constants.nm2m / 8) ** 2
    expected = 40 * 0.4 * 0.65 * radiance * omega
    np.testing.assert_allclose(rates.mean.exozodi, expected, rtol=1e-10, atol=0)
    doubled = eqx.tree_at(lambda s: s.disk.amplitude.level, system, 6.0)
    np.testing.assert_allclose(
        aperture_rates(doubled, path, q, **kwargs).mean.exozodi,
        2 * expected,
        rtol=1e-10,
        atol=0,
    )
    empty = eqx.tree_at(lambda s: s.disk, system, None)
    np.testing.assert_array_equal(
        aperture_rates(empty, path, q, **kwargs).mean.exozodi, 0
    )
    radius = 0.7 * (550 * constants.nm2m) / 8 / arcsec_rad
    step = 2 * radius / 31
    grid = PixelGrid.regular(
        (31, 31),
        step,
        "arcsec",
        center=(15 - 0.1 / step, 15),
        orientation=SkyOrientation(0, 1),
    )
    x, y = grid.coordinates()
    mask = (np.asarray(x) - 0.1) ** 2 + np.asarray(y) ** 2 <= radius**2
    footprint = np.asarray(
        system.photon_radiance(jnp.asarray(550.0), 0.0, grid=grid).data
    )[mask].mean()
    centre = (
        3
        * 10 ** (-0.4 * 22)
        * (3636 * constants.Jy)
        / (constants.h * (550 * constants.nm2m))
        * constants.nm2m
        / arcsec_rad**2
    )
    assert abs(footprint / centre - 1) < 0.05
    from reference.legacy import etc_scene_from_system
    from reference.primitives import exozodi_rate

    legacy = etc_scene_from_system(
        system,
        550.0,
        time_d=0.0,
        separation_arcsec=0.1,
        position_angle_deg=90.0,
        zodi=None,
    )
    legacy_disk = exozodi_rate(
        legacy.F0,
        legacy.Fexozodi,
        (550 * constants.nm2m) / 8 / arcsec_rad,
        0.65,
        40.0,
        0.4,
        band.width_nm,
        1,
        np.pi * 0.7**2,
        legacy.dist_pc,
        legacy.sep_arcsec,
    )
    # Width is only 0.02 nm: the 1/lambda curvature bias is below 2e-10.
    np.testing.assert_allclose(
        rates.mean.integrate().exozodi, legacy_disk, rtol=2e-10, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_all_tabulated_breakpoints_and_multiple_bands():
    """Fails if star, planet, optics, zodi or QE knots are omitted.

    Basis: degree <=4 piecewise products, two nodes fail and four are exact
    to a 1e-12 floating-point bound after splitting every breakpoint.
    """
    waves = jnp.array([500.0, 523.0, 600.0])
    star = Star(
        wavelengths_nm=waves,
        times_d=jnp.array([0.0]),
        flux_density_jy=jnp.array([[30.0], [60.0], [20.0]]),
        interpolation_method="linear",
    )
    system, path = make_system(star), make_path(chromatic=False)
    model = system.planets[0].physical_model
    model = eqx.tree_at(
        lambda m: (m.wavelengths_nm, m.contrast_grid),
        model,
        (
            jnp.array([500.0, 531.0, 600.0]),
            jnp.broadcast_to(
                jnp.array([1.0, 3.0, 1.0])[None, :, None] * 1e-10, (1, 3, 2)
            ),
        ),
    )
    system = eqx.tree_at(lambda s: s.planets[0].physical_model, system, model)
    element = ox.TransmissiveElement(
        Transmission(jnp.array([500.0, 547.0, 600.0]), jnp.array([0.2, 0.8, 0.3]))
    )
    detector = ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(1, 1),
        qe_curve=(jnp.array([500.0, 569.0, 600.0]), jnp.array([0.1, 0.9, 0.2])),
    )
    path = eqx.tree_at(
        lambda p: (p.attenuating_elements, p.detector), path, ((element,), detector)
    )
    zodi = PrecomputedZodi(
        jnp.array([500.0, 581.0, 600.0]), jnp.array([1e10, 3e10, 2e10])
    )
    bands = (Band(500, 550), Band(550, 600))

    def read(sampling):
        return aperture_rates(
            system, path, sampling, time_d=0.0, planet_index=0, zodi=zodi
        ).mean

    photons = read(bands)
    q = photons.quadrature
    q8 = Quadrature.build(bands, system, path, zodi, AYOExozodi(), order=8)
    for knot in (523, 531, 547, 569, 581):
        # Nodes/weights reveal a split, not just greater unsplit order.
        i = 0 if knot < 550 else 1
        assert np.sum(
            np.asarray(q.weights("response")[i])[np.asarray(q.nodes_nm[i]) < knot]
        ) == pytest.approx(knot - bands[i].lo_nm, abs=1e-12)

    np.testing.assert_allclose(
        detector.electrons(photons).total(),
        detector.electrons(read(q8)).total(),
        rtol=1e-12,
        atol=0,
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_thermal_companions_and_absolute_local_zodi(example):
    """Fails if thermal gets optics twice or absolute sky gets host flux.

    Basis: independently typed Planck/sr arithmetic, 1e-12 rounding bound.
    """
    system, path, reference = example
    bands = reference.mean.quadrature.bands
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.array([2e10, 2e10]))

    kwargs = dict(
        time_d=0.0,
        planet_index=0,
        zodi=zodi,
        temperature_K=2000.0,
        eps_warm_T_cold=0.2,
        companion_flux_density=FlatCompanions(0.003),
    )
    rates = aperture_rates(system, path, bands, **kwargs)
    q = rates.mean.quadrature
    w = np.asarray(q.nodes_nm) * constants.nm2m
    omega = np.pi * 0.7**2 * (550 * constants.nm2m / 8) ** 2
    expected = (
        40
        * 0.2
        * omega
        * 2
        * constants.c
        / w**4
        / np.expm1(constants.h * constants.c / (w * constants.k_B * 2000))
        * constants.nm2m
    )
    np.testing.assert_allclose(rates.mean.thermal, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        rates.mean.stellar_companions,
        40 * 0.4 * 0.65 * np.pi * 0.7**2 * 0.003,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        rates.mean.local_zodi, 40 * 0.4 * 0.65 * 2e10 * omega, rtol=1e-12, atol=0
    )
    brighter = eqx.tree_at(lambda s: s.star.flux_jy, system, 3631.0)
    np.testing.assert_array_equal(
        aperture_rates(brighter, path, q, **kwargs).mean.local_zodi,
        rates.mean.local_zodi,
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_transmission_map_and_multiband_dispatch():
    """Fails if the sky term is not the annular mean of the aperture integral.

    Over the ring at separation s the aperture mean of x vanishes and that of
    y**2 is s**2/2 + radius**2/4 (equally spaced position angles average both
    exactly). The 0.3% tolerance covers finite native pixels in the quadratic
    field at a 0.25 lod pitch and linear interpolation between knots 0.05 lod
    apart (at most 0.05**2/8 * 0.1 of the field, below 1e-4 relative).
    """
    import jax
    from conftest import TabulatedCoronagraph
    from optixstuff.grids import Image, PixelGrid
    from optixstuff.performance import PerformanceCurves
    from skyscapes.scene import FlatStar

    system, path = (
        make_system(FlatStar(1e6), chromatic=False),
        make_path(chromatic=False),
    )
    grid = PixelGrid.focal(24, 0.25, path.coronagraph.frame)
    x, y = grid.coordinates()
    # Dense calibrated radii around the requested offset (1 lod at 550 nm,
    # 1.1 lod at 500 nm). The y**2 term distinguishes an aperture integral from
    # its centre value and the x term an annular mean from one position angle,
    # in both moments.
    record = path.coronagraph.curves()
    axis = jnp.linspace(0.8, 1.3, 11)

    record = PerformanceCurves(
        axis,
        record.frame,
        record.rule,
        **{
            name: jnp.broadcast_to(getattr(record, name), axis.shape)
            for name in record.interpolation
        },
        iwa_lod=record.iwa_lod,
        data_extent_lod=record.data_extent_lod,
    )
    table = TabulatedCoronagraph(
        grid,
        Image(
            0.5 + 0.02 * x + 0.1 * y**2, grid, (), Measure.intensive(), "transmission"
        ),
        record,
        bandpass=Band(490, 550),
    )
    other = make_path(chromatic=False, bandpass=Band(550, 620)).coronagraph
    other = eqx.tree_at(lambda c: c.sky_transmission, other, 0.8)
    multi = ox.MultiBandCoronagraph((table, other))
    path = eqx.tree_at(lambda p: p.coronagraph, path, multi)
    band = Band(500, 600)
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e10))
    q = Quadrature.build((band,), system, path, zodi, AYOExozodi())

    prepared = prepare_aperture(path, q)

    def read(separation):
        rates = aperture_rates(
            system,
            path,
            prepared,
            time_d=0.0,
            separation_arcsec=separation,
            zodi=zodi,
        )
        return jnp.stack(
            [rate.integrate().local_zodi for rate in (rates.mean, rates.variance)]
        )

    scale = (550 * constants.nm2m) / 8 / (constants.arcsec2rad)
    # At their shared edge (550 nm) the first band supplies the area.
    expected = (
        40
        * 0.4
        * 1e10
        * np.pi
        * 0.7**2
        * ((550 * constants.nm2m) / 8) ** 2
        * (
            50 * 0.5
            + 50 * 0.8
            + 0.1 * (0.7**2 / 4 + 1 / 2) * 550**2 * (1 / 500 - 1 / 550)
        )
    )
    np.testing.assert_allclose(read(scale), expected, rtol=0.003, atol=0)
    np.testing.assert_allclose(jax.jit(read)(scale), expected, rtol=0.003, atol=0)


@vv.case("aperture-photon-budget", "code-verification")
def test_disk_traced_sky_position():
    """Fails if scene disk sampling loses orientation or depends on host coordinates.

    The linear map is recovered to 1e-12 relative along the orbit. Its
    separation derivative uses a 1e-4 arcsec central step inside one native
    stencil. There is no truncation for this affine map; 1e-9 covers
    float64 subtraction and aperture arithmetic.
    """
    import jax
    from skyscapes.disk import ExovistaDisk
    from skyscapes.scene import FlatStar

    grid = PixelGrid.regular(
        (9, 9), 0.04, "arcsec", orientation=SkyOrientation(90.0, -1)
    )
    east, north = grid.orientation.detector_to_sky(*grid.coordinates())
    # Input cube is host-relative flux per native pixel, not per sr.
    contrast = 1e-8 * (1 + east + 2 * north)
    disk = ExovistaDisk(
        grid, jnp.array([500.0, 600.0]), jnp.stack([contrast, contrast])
    )
    system = make_system(FlatStar(1e6), chromatic=False, disk=disk)
    path, band = make_path(chromatic=False), Band(549.0, 551.0)
    q = Quadrature.build((band,), system, path)

    prepared = prepare_aperture(path, q)

    def read(scene, t):
        return (
            aperture_rates(
                scene,
                path,
                prepared,
                time_d=t,
                zodi=None,
                planet_index=0,
            )
            .mean.integrate()
            .exozodi[0]
        )

    def orbit(t):
        return read(system, t)

    times = jnp.array([0.0, 30.0, 60.0])
    positions = np.asarray(system.positions(times))[:, 0, :]
    omega = np.pi * 0.7**2 * (550 * constants.nm2m / 8) ** 2
    expected = (
        40
        * 0.4
        * 0.65
        * 1e6
        * 1e-8
        * (1 + positions[0] + 2 * positions[1])
        / (0.04 * constants.arcsec2rad) ** 2
        * omega
        * 2
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(orbit))(times), expected, rtol=1e-12, atol=0
    )

    def radial(separation):
        # At the reference epoch this circular, face-on orbit projects its
        # semimajor axis onto the sky without foreshortening.
        scene = eqx.tree_at(
            lambda s: s.dynamics.orbits.a_AU,
            system,
            jnp.atleast_1d(separation * system.dist_pc),
        )
        return read(scene, 0.0)

    centre, step = 0.09, 1e-4  # arcsec; avoid interpolation knots and cube edges
    derivative = jax.jit(jax.grad(radial))(centre)
    difference = (radial(centre + step) - radial(centre - step)) / (2 * step)
    assert jnp.isfinite(derivative)
    assert difference > 0
    np.testing.assert_allclose(derivative, difference, rtol=1e-9, atol=0)


@vv.case("aperture-photon-budget", "code-verification")
@vv.seam("optixstuff", "jaxedith")
def test_fixed_angle_uniform_leakage(fixed_angle):
    """Fails if a fixed lod radius replaces the fixed angular aperture.

    Basis: uniform density integrated over pi*theta^2 at test-side Gauss
    nodes; 1e-9 relative permits quadrature rounding. The table's primitive
    uniform-density calibration isolates the reader from PSF photometry.
    """
    system, path, prepared = fixed_angle
    q = prepared.quadrature
    from conftest import TabulatedCoronagraph

    calibration = make_path(chromatic=False, leakage=1e-9 * np.pi * 0.7**2)
    table = TabulatedCoronagraph(
        path.coronagraph.grid,
        path.coronagraph.sky_trans,
        calibration.coronagraph.curves(),
    )
    path = eqx.tree_at(lambda p: p.coronagraph, path, table)
    prepared = prepare_aperture(path, q)
    rates = aperture_rates(
        system,
        path,
        prepared,
        time_d=0.0,
        planet_index=0,
        zodi=None,
    )
    x, w = np.polynomial.legendre.leggauss(4)
    wave = np.array([525.0, 575.0])[:, None] + 25 * x
    radius = 0.7 * np.array([525.0, 575.0])[:, None] / wave
    expected = np.sum(
        25
        * w
        * 16
        * 1e6
        * 1e-9
        * np.pi
        * radius**2
        * (0.8 - 0.002 * (wave - 550))
        * 0.9,
        axis=1,
    )
    np.testing.assert_allclose(
        path.detector.electrons(rates.mean).star_leakage, expected, rtol=1e-9, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_planet_scaled_node_curve(fixed_angle):
    """Fails if either separation or radius uses reference instead of node.

    Basis: API seam equality at 1e-12 floating-point rounding; independent
    scaled-rule preparation and position conversion on the expected side.
    """
    from optixstuff.apertures import Circular

    system, path, prepared = fixed_angle
    q = prepared.quadrature
    rates = aperture_rates(
        system,
        path,
        prepared,
        time_d=0.0,
        planet_index=0,
        zodi=None,
    )
    w = float(q.nodes_nm[0, 0])
    radius = 0.7 * 525 / w
    coro = path.coronagraph.with_rules((Circular(radius),))
    separation = 1 / system.dist_pc / (w * constants.nm2m / 8 / (constants.arcsec2rad))
    expected = coro.curves(Circular(radius)).evaluate("throughput", separation)
    np.testing.assert_allclose(
        rates.mean.planet[0, 0] / 0.0016, expected, rtol=1e-12, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_monochromatic_reference_aperture(fixed_angle):
    """Fails if a monochromatic node is scaled away from its reference.

    Basis: exact array equality to the unscaled reference curve at the
    monochromatic wavelength.
    """
    system, path, _ = fixed_angle
    q = Quadrature.monochromatic(550.0)
    prepared = prepare_aperture(path, q)
    rates = aperture_rates(
        system,
        path,
        prepared,
        time_d=0.0,
        planet_index=0,
        zodi=None,
    )
    sep = 1 / system.dist_pc / path.primary.frame(550.0).lod_to_arcsec(550.0)
    phi = system.planet_flux_densities(jnp.asarray(550.0), jnp.array([0.0]))[0, 0]
    expected = 16 * path.coronagraph.curves().evaluate("throughput", sep) * phi
    np.testing.assert_array_equal(rates.mean.planet[0, 0], expected)
    # A finite circular aperture captures only part of the Gaussian point image.
    full_field = 16 * path.coronagraph.point_energy_fraction(sep, 0.0) * phi
    assert 0 < rates.mean.planet[0, 0] < full_field


@vv.case("aperture-photon-budget", "code-verification")
def test_prepared_aperture_jit(fixed_angle):
    """Fails if scaled-rule preparation occurs inside the traced reader.

    Basis: eager/JIT arithmetic equality within 1e-12 floating-point error.
    """
    system, path, prepared = fixed_angle

    def read(scene, aperture):
        return aperture_rates(
            scene,
            path,
            aperture,
            time_d=0.0,
            planet_index=0,
            zodi=None,
        )

    eager = read(system, prepared)
    compiled = eqx.filter_jit(read)(system, prepared)
    for actual, expected in zip(
        jax.tree.leaves(compiled), jax.tree.leaves(eager), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@vv.case("aperture-photon-budget", "code-verification")
@vv.seam("coronagraphoto", "jaxedith")
def test_finite_star_image_aperture_sum(finite_star_setup):
    """Exact-algebra seam: uniform finite-star image, rtol=1e-9, atol=0.

    A uniform native YIP makes pixel-box integration and fractional circular
    aperture summation commute. Real photon gains and the stellar diameter
    are shared, with independently computed detector circle-square weights.
    This isolates absolute diameter/area units from curved-image quadrature.
    """
    from coronagraphoto import photon_rates
    from optixstuff import Detector, Exposure
    from optixstuff.performance import _circle_square_weights

    system, path, prepared = finite_star_setup
    q = prepared.quadrature
    bands = q.bands
    system = eqx.tree_at(lambda s: s.star.fixed_diameter_arcsec, system, 0.002)
    scale = path.coronagraph.frame.lod_to_arcsec(550)
    pitch = 0.5 * scale
    path = eqx.tree_at(
        lambda p: p.detector, path, Detector(pixel_scale_arcsec=pitch, shape=(24, 24))
    )
    separation = 2 * scale
    rates = aperture_rates(
        system,
        path,
        prepared,
        time_d=0.0,
        separation_arcsec=separation,
        zodi=None,
    )
    images = photon_rates(
        eqx.tree_at(lambda s: s.disk, system, None),
        path,
        Exposure(time_d=0.0, roll_deg=0.0, quadrature=q),
    )
    yy, xx = jnp.indices((24, 24))
    for i, band in enumerate(bands):
        radius_pix = 0.7 * band.center_nm / 550 / 0.5
        weights = _circle_square_weights(
            yy - 11.5, xx - 11.5 - separation / pitch, radius_pix
        )
        expected = jnp.sum(images.star_leakage[i, 0] * weights)
        np.testing.assert_allclose(
            rates.mean.star_leakage[i, 0], expected, rtol=1e-9, atol=0
        )
