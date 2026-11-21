"""Analytic aperture rates: each term against the scalar formulas, and contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from hwoutils.conversions import arcsec_to_rad, lambda_d_to_arcsec
from hwoutils.epoch import Epoch
from optixstuff import (
    ConstantThroughput,
    IdealDetector,
    Instrument,
    PerformanceCurves,
    Quadrature,
    SampledProfile,
    SimplePrimary,
)
from optixstuff.apertures import Circular
from optixstuff.grids import Frame
from optixstuff.spectral import Band
from reference.primitives import (
    exozodi_rate,
    planet_rate,
    stellar_leakage_rate,
    thermal_rate,
    zodi_rate,
)
from skyscapes import AYOZodi
from skyscapes.disk import AYOExozodi
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import FlatStar, Planet, Scene

import jaxedith

DIAMETER_M = 6.0
AREA_M2 = 28.0
OPTICS = 0.4
FLUX = 1e6
BAND = Band(500.0, 600.0)
EAST, NORTH = 0.03, 0.04
DIST_PC = 10.0
CONTRAST = 3e-10
# Distinct mean and squared-weight moments.
TAU, TAU2 = 0.3, 0.22
AREA, AREA2 = 1.5, 1.1
DENSITY, DENSITY2 = 2e-11, 1.4e-11
SKY = 0.5
RTOL = 1e-9


@pytest.fixture(scope="module", autouse=True)
def _float64():
    """The reference formulas hold to float64 rounding; scope x64 to this module."""
    with jax.enable_x64():
        yield


def _frame():
    return Frame(550.0, DIAMETER_M, "circumscribed")


def _curves(frame=None, throughput=(TAU, TAU), extent=15.0):
    """Constant curves on a two-point axis; the throughput may vary with separation."""
    flat = lambda value: np.array([value, value])  # noqa: E731
    return PerformanceCurves(
        np.array([0.0, 20.0]),
        _frame() if frame is None else frame,
        Circular(0.7),
        np.array(throughput),
        flat(AREA),
        flat(1e-10),
        flat(DENSITY),
        flat(1e-9),
        1.0,
        extent,
        throughput_variance=flat(TAU2),
        core_area_variance_lod2=flat(AREA2),
        leakage_fraction_variance=flat(1e-10),
        leakage_density_variance_lod2=flat(DENSITY2),
    )


def _sky():
    return SampledProfile(
        np.array([0.0, 20.0]),
        np.array([SKY, SKY]),
        quantity="uniform_sky_response",
        frame=_frame(),
        interpolation="linear",
    )


def _scene(*, disk=True, zodi=True):
    wavelengths = jnp.array([400.0, 700.0])
    model = GridPhysicalModel(
        wavelengths,
        jnp.array([0.0, 180.0]),
        jnp.full((1, 2, 2), 1e-10),
    )
    return Scene.from_classical(
        FlatStar(flux_phot_per_nm_m2=FLUX),
        (Planet(jnp.ones(1), model),),
        star_mass_Msun=1.0,
        planet_mass_Msun=3e-6,
        dist_pc=DIST_PC,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.array([1.0]),
        e=0.0,
        W_rad=0.0,
        i_rad=0.0,
        w_rad=0.0,
        M0_rad=0.0,
        disk=AYOExozodi(level_zodis=3.0) if disk else None,
        local_zodi=AYOZodi() if zodi else None,
    )


def _instrument(diameter=DIAMETER_M, **kwargs):
    return Instrument(
        SimplePrimary(circumscribed_diameter_m=diameter, collecting_area_m2=AREA_M2),
        (ConstantThroughput(OPTICS),),
        None,
        IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)),
        **kwargs,
    )


def _quadrature(bands=(BAND,)):
    return Quadrature.build(bands, order=1)


def _read(curves=None, scene=None, instrument=None, quadrature=None, **kwargs):
    quadrature = _quadrature() if quadrature is None else quadrature
    arguments = dict(east_arcsec=EAST, north_arcsec=NORTH, contrast=CONTRAST)
    arguments.update(kwargs)
    return jaxedith.rates(
        _curves() if curves is None else curves,
        _scene() if scene is None else scene,
        _instrument() if instrument is None else instrument,
        quadrature,
        arguments.pop("east_arcsec"),
        arguments.pop("north_arcsec"),
        arguments.pop("contrast"),
        time_d=0.0,
        uniform_sky=_sky(),
    )


def _scalar_expectations():
    """Every term of the first band from the scalar formulas, in photons/s."""
    quadrature = _quadrature()
    node = quadrature.nodes_nm[0, 0]
    width = float(quadrature.weights("response")[0, 0])
    scene = _scene()
    lod_arcsec = float(lambda_d_to_arcsec(1.0, BAND.center_nm, DIAMETER_M))
    separation = np.hypot(EAST, NORTH)
    zodi_ratio = float(scene.local_zodi.photon_radiance(node, scene.sky_frame, 0.0))
    disk = float(scene.photon_radiance_at(node, 0.0, EAST, NORTH))
    sr = float(arcsec_to_rad(1.0)) ** 2
    out = {}
    for name, (tau, area, density) in {
        "mean": (TAU, AREA, DENSITY),
        "variance": (TAU2, AREA2, DENSITY2),
    }.items():
        common = (AREA_M2, OPTICS)
        out[name] = dict(
            planet=planet_rate(FLUX, 1.0, CONTRAST, *common, tau, width, 1),
            star_leakage=stellar_leakage_rate(
                FLUX, 1.0, *common, width, 1, area, density
            ),
            local_zodi=zodi_rate(
                1.0, zodi_ratio * sr, lod_arcsec, SKY, *common, width, 1, area
            ),
            exozodi=exozodi_rate(
                1.0,
                disk * sr * (DIST_PC * separation) ** 2,
                lod_arcsec,
                SKY,
                *common,
                width,
                1,
                area,
                DIST_PC,
                separation,
            ),
            n_pix=area * lod_arcsec**2 / 0.01**2,
        )
    return out


def test_every_term_matches_the_scalar_formulas():
    """Fails if a rate departs from its AYO or pyEDITH scalar product.

    Basis: products of float64 scalars on one node of a flat-spectrum scene
    with constant curves; rtol 1e-9, the bound the deleted suite used for the
    same algebra. The exozodi reference is built from the scene's own disk
    radiance, so it checks the aperture factors and not the disk model.
    """
    rates = _read()
    assert rates.carrier == "photon" and rates.spectral
    assert rates.planet.shape == (1, 1) and rates.n_pix.shape == (1,)
    # Detector yield is not applied: electrons of an ideal detector are the rates
    # integrated over the band, which the reference's bandwidth factor includes.
    electrons = ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)).electrons(rates)
    expected = _scalar_expectations()
    for statistic, record, electron in (
        ("mean", rates, electrons),
        ("variance", rates.variance, electrons.variance),
    ):
        for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
            np.testing.assert_allclose(
                electron_value := getattr(electron, name)[0],
                expected[statistic][name],
                rtol=RTOL,
                err_msg=f"{statistic} {name}",
            )
            assert np.isfinite(electron_value) and electron_value > 0
        np.testing.assert_allclose(
            record.n_pix[0], expected[statistic]["n_pix"], rtol=RTOL
        )
    assert float(jnp.max(jnp.abs(rates.stellar_companions))) == 0.0


@pytest.mark.parametrize("temperature_K", [290.0, 1500.0])
def test_thermal_matches_the_pyedith_scalar_formula(temperature_K):
    """Fails if the warm telescope's rate departs from pyEDITH calculate_CRbth.

    Basis: ``reference.primitives.thermal_rate`` (the pyEDITH product of Planck
    photon radiance, emissivity times cold transmission, collecting area
    without throughput, bandwidth and aperture solid angle) in float64 on the
    node of a one-node band, for the mean and the squared-weight records;
    rtol 1e-12 for the exp and a six-factor product. The system throughput
    and flux scale must not enter, so both are non-unity here.
    """
    eps = 0.638
    instrument = _instrument(
        warm_emission=ox.WarmEmission(temperature_K, eps), flux_scale=1.3
    )
    rates = _read(instrument=instrument)
    electrons = ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)).electrons(rates)
    node = float(_quadrature().nodes_nm[0, 0])
    width = float(_quadrature().weights("response")[0, 0])
    lod_arcsec = float(lambda_d_to_arcsec(1.0, BAND.center_nm, DIAMETER_M))
    lod_rad = float(arcsec_to_rad(lod_arcsec))
    for record, area in ((electrons, AREA), (electrons.variance, AREA2)):
        expected = thermal_rate(
            node, AREA_M2, width, temperature_K, lod_rad, eps, 1, 1, area
        )
        assert expected > 0
        np.testing.assert_allclose(record.thermal[0], expected, rtol=1e-12)
    others = _read()
    for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
        np.testing.assert_allclose(
            getattr(rates, name), 1.3 * getattr(others, name), rtol=1e-12
        )


def test_default_instrument_has_zero_thermal():
    """Fails if an instrument without a warm-emission description emits.

    Basis: exact zeros in the mean and the variance record, and in a scene read.
    """
    assert _instrument().warm_emission is None
    rates = _read()
    assert float(jnp.max(jnp.abs(rates.thermal))) == 0.0
    assert float(jnp.max(jnp.abs(rates.variance.thermal))) == 0.0
    scene_read = jaxedith.scene_rates(
        _curves(),
        _scene(),
        _instrument(),
        _quadrature(),
        jnp.array([0.0]),
        uniform_sky=_sky(),
    )
    assert float(jnp.max(jnp.abs(scene_read.thermal))) == 0.0


def test_scene_rates_carries_thermal_to_every_planet():
    """Fails if the vmapped scene read drops or reshapes the thermal term.

    Basis: the scene read at one planet and time equals the scalar read,
    exactly the same arithmetic, so rtol 1e-12.
    """
    instrument = _instrument(warm_emission=ox.WarmEmission(290.0, 0.638))
    scene = _scene()
    read = jaxedith.scene_rates(
        _curves(),
        scene,
        instrument,
        _quadrature(),
        jnp.array([0.0]),
        uniform_sky=_sky(),
    )
    east, north = scene.positions(jnp.array([0.0]))
    contrast = scene.contrasts(_quadrature().nodes_nm[0, 0], jnp.array([0.0]))
    scalar = _read(
        instrument=instrument,
        east_arcsec=east[0, 0],
        north_arcsec=north[0, 0],
        contrast=contrast[0, 0],
    )
    assert read.thermal.shape == (1, 1, 1, 1)
    assert float(read.thermal[0, 0, 0, 0]) > 0
    np.testing.assert_allclose(
        read.thermal[0, 0, 0, 0], scalar.thermal[0, 0], rtol=1e-12
    )


def test_missing_sources_are_zero_and_the_sky_is_required_with_them():
    """Fails if an absent disk or zodi contributes or a missing sky profile passes.

    Basis: exact zeros; the error is raised before any read.
    """
    bare = _scene(disk=False, zodi=False)
    rates = jaxedith.rates(
        _curves(),
        bare,
        _instrument(),
        _quadrature(),
        EAST,
        NORTH,
        CONTRAST,
        time_d=0.0,
        uniform_sky=None,
    )
    np.testing.assert_array_equal(rates.local_zodi, 0.0)
    np.testing.assert_array_equal(rates.exozodi, 0.0)
    assert float(rates.planet[0, 0]) > 0
    with pytest.raises(ValueError, match="uniform_sky"):
        jaxedith.rates(
            _curves(),
            _scene(),
            _instrument(),
            _quadrature(),
            EAST,
            NORTH,
            CONTRAST,
            time_d=0.0,
            uniform_sky=None,
        )


def test_separation_is_read_at_each_band_centre():
    """Fails if a band reads another band's lambda/D, or per-band curves mix.

    Basis: throughput equal to the separation in lambda/D; the planet rate over
    its prefactor must be the angle over the band-centre lambda/D (rtol 1e-12).
    """
    bands = (Band(500.0, 560.0), Band(640.0, 760.0))
    quadrature = _quadrature(bands)
    curves = _curves(throughput=(0.0, 20.0))
    rates = _read(curves=curves, quadrature=quadrature)
    separation = np.hypot(EAST, NORTH)
    prefactor = AREA_M2 * OPTICS * FLUX * CONTRAST
    centres = np.array([band.center_nm for band in bands])
    lod = lambda_d_to_arcsec(1.0, centres, DIAMETER_M)
    np.testing.assert_allclose(
        rates.planet[:, 0] / prefactor, separation / lod, rtol=1e-12
    )
    # One record per band equals the same record repeated.
    each = _read(curves=(curves, curves), quadrature=quadrature)
    np.testing.assert_array_equal(each.planet, rates.planet)
    with pytest.raises(ValueError, match="one per band"):
        _read(curves=(curves,), quadrature=quadrature)


def test_beyond_the_data_is_nan_and_a_diameter_mismatch_raises():
    """Fails if an out-of-data read looks valid or another telescope's curves pass.

    Basis: exact NaN beyond ``data_extent_lod``; a 5 percent diameter change is
    far outside the 1e-6 relative tolerance of the frame check.
    """
    short = _curves(extent=1.0)
    outside = _read(curves=short)
    for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
        assert np.isnan(getattr(outside, name)).all()
        assert np.isnan(getattr(outside.variance, name)).all()
    assert np.isnan(outside.n_pix).all()
    with pytest.raises(ValueError, match="differs from the curves"):
        _read(instrument=_instrument(diameter=DIAMETER_M * 1.05))


def test_transformations_are_finite_inside_the_data():
    """Fails if jit, vmap or grad of the read produces a non-finite value.

    Basis: finite outputs and a central-difference check of the planet rate in
    contrast (the rate is linear in it, so the step is exact to 1e-9).
    """
    curves, scene, instrument = _curves(), _scene(), _instrument()
    quadrature, sky = _quadrature(), _sky()

    def planet(east, north, contrast):
        rates = jaxedith.rates(
            curves,
            scene,
            instrument,
            quadrature,
            east,
            north,
            contrast,
            time_d=0.0,
            uniform_sky=sky,
        )
        return rates.planet[0, 0] + rates.exozodi[0, 0]

    easts = jnp.linspace(0.02, 0.1, 5)
    values = jax.jit(jax.vmap(planet, in_axes=(0, 0, None)))(easts, easts, CONTRAST)
    assert np.isfinite(values).all()
    grad = jax.grad(planet, argnums=(0, 1, 2))(EAST, NORTH, CONTRAST)
    assert all(np.isfinite(g) for g in grad)
    step = 1e-3 * CONTRAST
    difference = (
        planet(EAST, NORTH, CONTRAST + step) - planet(EAST, NORTH, CONTRAST - step)
    ) / (2 * step)
    np.testing.assert_allclose(grad[2], difference, rtol=1e-9)


def test_a_large_batch_of_positions_compiles_once():
    """Fails if mapping over many positions retraces or rebuilds the scene reads.

    Basis: a trace counter, exact. One compile serves two batches of 1e5
    positions of the same shape.
    """
    curves, scene, instrument = _curves(), _scene(), _instrument()
    quadrature, sky = _quadrature(), _sky()
    traces = []

    @jax.jit
    @jax.vmap
    def planets(east, north):
        traces.append(1)
        return jaxedith.rates(
            curves,
            scene,
            instrument,
            quadrature,
            east,
            north,
            CONTRAST,
            time_d=0.0,
            uniform_sky=sky,
        ).planet

    key = jax.random.PRNGKey(0)
    for k in jax.random.split(key, 2):
        east = jax.random.uniform(k, (100_000,), minval=0.02, maxval=0.15)
        result = planets(east, east)
        assert result.shape == (100_000, 1, 1) and np.isfinite(result).all()
    assert len(traces) == 1


def test_photon_counting_frame_time_follows_the_occupancy_rule():
    """Fails if the frame time is not 1 / (6.73 r) per band.

    Basis: exact algebra, rtol 1e-12, from electron rates with a known total and
    pixel count; a photon-rate record reaches the same value through the
    detector's yield.
    """
    detector = IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8))
    electrons = ox.Rates(
        (BAND, Band(600.0, 700.0)),
        "electron",
        planet=jnp.array([0.01, 0.02]),
        star_leakage=jnp.array([0.1, 0.2]),
        local_zodi=jnp.array([0.2, 0.1]),
        exozodi=jnp.array([0.3, 0.1]),
        n_pix=jnp.array([20.0, 40.0]),
    )
    expected = 1.0 / (6.73 * jnp.array([0.61 / 20.0, 0.42 / 40.0]))
    np.testing.assert_allclose(
        jaxedith.photon_counting_frame_time(electrons, detector), expected, rtol=1e-12
    )
    rates = _read()
    frame = jaxedith.photon_counting_frame_time(rates, detector)
    total = detector.electrons(rates).total()
    np.testing.assert_allclose(frame, rates.n_pix / (6.73 * total), rtol=1e-12)
    with pytest.raises(ValueError, match="n_pix"):
        jaxedith.photon_counting_frame_time(
            eqx.tree_at(lambda r: r.n_pix, rates, None, is_leaf=lambda x: x is None),
            detector,
        )


def test_scene_entry_points_equal_the_two_step_calls():
    """Fails if a scene entry point does any work of its own.

    Basis: exact equality with ``rates`` followed by the optixstuff function,
    for a unit background factor with no residual, for the AYO setting and for
    per-band times.
    """
    quadrature = _quadrature((BAND, Band(600.0, 700.0)))
    curves, scene, instrument = _curves(), _scene(), _instrument()
    common = (curves, scene, instrument, quadrature, EAST, NORTH, 1e-6)
    keywords = dict(time_d=0.0, uniform_sky=_sky())
    rates = jaxedith.rates(*common, **keywords)
    frame = jnp.array([100.0, 60.0])
    for post_processing in (
        ox.PostProcessing(background_variance_factor=1.0),
        ox.PostProcessing(0.1, 0.05, background_variance_factor=2.0),
    ):
        wanted = ox.exposure_time(
            rates,
            instrument.detector,
            jnp.array([5.0, 7.0]),
            frame_time_s=frame,
            post_processing=post_processing,
        )
        got = jaxedith.planet_exposure_time(
            *common,
            **keywords,
            target_snr=jnp.array([5.0, 7.0]),
            frame_time_s=frame,
            post_processing=post_processing,
        )
        assert bool(got.valid.all())
        np.testing.assert_array_equal(got.time_s, wanted.time_s)
        np.testing.assert_array_equal(got.valid, wanted.valid)
        np.testing.assert_array_equal(
            jaxedith.planet_snr(
                *common,
                **keywords,
                time_s=3600.0,
                frame_time_s=frame,
                post_processing=post_processing,
            ),
            ox.snr(
                rates,
                instrument.detector,
                3600.0,
                frame_time_s=frame,
                post_processing=post_processing,
            ),
        )


def _two_planet_scene():
    wavelengths = jnp.array([400.0, 700.0])
    angles = jnp.array([0.0, 180.0])
    models = (
        GridPhysicalModel(wavelengths, angles, jnp.full((1, 2, 2), 1e-10)),
        GridPhysicalModel(wavelengths, angles, jnp.full((1, 2, 2), 4e-10)),
    )
    return Scene.from_classical(
        FlatStar(flux_phot_per_nm_m2=FLUX),
        tuple(Planet(jnp.ones(1), model) for model in models),
        star_mass_Msun=1.0,
        planet_mass_Msun=jnp.array([3e-6, 1e-5]),
        dist_pc=DIST_PC,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.array([1.0, 2.0]),
        e=jnp.array([0.0, 0.1]),
        W_rad=0.0,
        i_rad=jnp.array([0.3, 0.6]),
        w_rad=0.0,
        M0_rad=jnp.array([0.0, 1.0]),
        disk=AYOExozodi(level_zodis=3.0),
        local_zodi=AYOZodi(),
    )


def test_scene_rates_equals_the_loop_over_rates_and_compiles_once():
    """Fails if scene_rates differs from the loop over rates.

    Basis: two planets, three times, two bands. Each field, pixel count and
    squared-weight field is compared with the Python loop over ``rates`` at the
    scene's own positions and contrasts to float rounding (measured 1.6e-16,
    one ulp; rtol 1e-14); one trace serves a second call with new times.
    """
    curves, scene, instrument = _curves(), _two_planet_scene(), _instrument()
    quadrature, sky = _quadrature((BAND, Band(600.0, 700.0))), _sky()
    times = jnp.array([0.0, 40.0, 90.0])
    traces = []

    @jax.jit
    def read(time_d):
        traces.append(1)
        return jaxedith.scene_rates(
            curves, scene, instrument, quadrature, time_d, uniform_sky=sky
        )

    got = read(times)
    assert got.planet.shape == (2, 1, 2, 3) and got.n_pix.shape == (2, 2, 3)
    assert got.variance.planet.shape == (2, 1, 2, 3)
    east, north = scene.positions(times)
    for k in range(2):
        for t in range(3):
            contrast = jnp.stack(
                [scene.contrasts(n, times)[k, t] for n in quadrature.nodes_nm[:, 0]]
            )[:, None]
            want = jaxedith.rates(
                curves,
                scene,
                instrument,
                quadrature,
                east[k, t],
                north[k, t],
                contrast,
                time_d=times[t],
                uniform_sky=sky,
            )
            for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
                for record, expected in ((got, want), (got.variance, want.variance)):
                    np.testing.assert_allclose(
                        getattr(record, name)[:, :, k, t],
                        getattr(expected, name),
                        rtol=1e-14,
                    )
            np.testing.assert_allclose(got.n_pix[:, k, t], want.n_pix, rtol=1e-14)
    read(times + 1.0)
    assert len(traces) == 1
    with pytest.raises(ValueError, match="shape"):
        jaxedith.scene_rates(
            curves, scene, instrument, quadrature, 0.0, uniform_sky=sky
        )
