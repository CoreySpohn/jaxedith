"""Analytic aperture rates: each term against the scalar formulas, and contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from hwoutils.constants import Rsun2m
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
from optixstuff.spectral import Band, Resolution
from reference.primitives import (
    exozodi_rate,
    planet_rate,
    stellar_leakage_rate,
    thermal_rate,
    zodi_rate,
)
from reference.scalar_case import ConstantDisk
from skyscapes import AYOZodi
from skyscapes.disk import AYOExozodi
from skyscapes.physical_model import BlackbodyPhysicalModel, GridPhysicalModel
from skyscapes.scene import BlackbodyStar, FlatStar, Planet, Scene

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


def _curves(frame=None, throughput=(TAU, TAU), extent=15.0, last=20.0, *, sky=SKY):
    """Constant curves on a two-point axis; the throughput may vary with separation."""
    flat = lambda value: np.array([value, value])  # noqa: E731
    frame = _frame() if frame is None else frame
    return PerformanceCurves(
        np.array([0.0, last]),
        frame,
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
        uniform_sky=None if sky is None else _sky(frame, sky),
    )


def _sky(frame, transmission=SKY):
    return SampledProfile(
        np.array([0.0, 20.0]),
        np.full(2, transmission),
        quantity="uniform_sky_response",
        frame=frame,
        interpolation="linear",
    )


def _scene(*, disk=True, zodi=True, diameter_arcsec=0.0):
    wavelengths = jnp.array([300.0, 900.0])
    model = GridPhysicalModel(
        wavelengths,
        jnp.array([0.0, 180.0]),
        jnp.full((1, 2, 2), 1e-10),
    )
    return Scene.from_classical(
        FlatStar(flux_phot_per_nm_m2=FLUX, fixed_diameter_arcsec=diameter_arcsec),
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


def _quadrature_for(band, order):
    return Quadrature.build((band,), order=order)


def _resolution(scene, quadrature):
    """Operators for the scene's tables on the quadrature, built on the host."""
    return Resolution.build(quadrature, *scene.spectral_tables())


def _sources(scene, quadrature, time_d=0.0):
    """The scene's node-effective sources at one time."""
    return scene.sources(quadrature, _resolution(scene, quadrature), time_d)


def _read(curves=None, scene=None, instrument=None, quadrature=None, **kwargs):
    quadrature = _quadrature() if quadrature is None else quadrature
    scene = _scene() if scene is None else scene
    arguments = dict(east_arcsec=EAST, north_arcsec=NORTH, contrast=CONTRAST)
    arguments.update(kwargs)
    return jaxedith.rates(
        _curves() if curves is None else curves,
        scene,
        _instrument() if instrument is None else instrument,
        quadrature,
        arguments.pop("east_arcsec"),
        arguments.pop("north_arcsec"),
        contrast=arguments.pop("contrast"),
        time_d=0.0,
        sources=_sources(scene, quadrature),
    )


def _scalar_expectations():
    """Every term of the first band from the scalar formulas, in photons/s."""
    quadrature = _quadrature()
    node = quadrature.nodes_nm[0, 0]
    width = float(quadrature.weights()[0, 0])
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


def _scaled(scene, **scales):
    """The scene with ``flux_scale`` set on the named sources."""
    where = dict(
        star=lambda s: s.star.flux_scale,
        zodi=lambda s: s.local_zodi.flux_scale,
        disk=lambda s: s.disk.flux_scale,
    )
    for name, value in scales.items():
        scene = eqx.tree_at(where[name], scene, jnp.asarray(value))
    return scene


def test_each_flux_scale_scales_its_own_terms_exactly():
    """Fails if a source's scale is applied twice, or reaches another source.

    Basis: the star scale multiplies the planet and the leakage, the zodi scale
    the local zodi, the disk scale the exozodi, each by exactly its factor
    (rtol 1e-12, products of float64 scalars) and nothing else.
    """
    base = _read()
    terms = ("planet", "star_leakage", "local_zodi", "exozodi")
    for source, hit, factor in (
        ("star", ("planet", "star_leakage"), 2.0),
        ("zodi", ("local_zodi",), 3.0),
        ("disk", ("exozodi",), 5.0),
    ):
        rates = _read(scene=_scaled(_scene(), **{source: factor}))
        for name in terms:
            want = factor if name in hit else 1.0
            np.testing.assert_allclose(
                getattr(rates, name),
                want * getattr(base, name),
                rtol=1e-12,
                err_msg=f"{source} scale on {name}",
            )


def test_a_disk_hook_implementer_sees_the_scale_applied_once():
    """Fails if ``ConstantDisk`` or the base method bypasses the private hook.

    ``ConstantDisk`` implements only ``_photon_radiance_at``; the public method
    must return its radiance times ``flux_scale``, exactly.
    """
    args = (jnp.array([550.0]), 0.0, jnp.array(0.1), jnp.array(0.2))
    kwargs = dict(
        star=FlatStar(flux_phot_per_nm_m2=1.0),
        dist_pc=10.0,
        inclination_deg=0.0,
        position_angle_deg=0.0,
    )
    plain = ConstantDisk(7.0).photon_radiance_at(*args, **kwargs)
    scaled = ConstantDisk(7.0, flux_scale=3.0).photon_radiance_at(*args, **kwargs)
    np.testing.assert_array_equal(plain, 7.0)
    np.testing.assert_array_equal(scaled, 21.0)


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
    width = float(_quadrature().weights()[0, 0])
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
        resolution=_resolution(_scene(), _quadrature()),
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
        resolution=_resolution(scene, _quadrature()),
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


@pytest.mark.parametrize(
    "leaf, planet_factor, leakage_factor",
    [("star", 1.0, 2.0), ("planet", 2.0, 1.0)],
)
def test_thermal_planet_scale_is_its_own_in_the_scene_read(
    leaf, planet_factor, leakage_factor
):
    """Fails if a star's ``flux_scale`` reaches a thermal planet in a scene read.

    A 300 K blackbody planet's own flux does not depend on the host, so a star
    scale of two doubles only the leakage and a planet scale of two doubles
    only the planet, each by exactly its factor (rtol 1e-12, float64).
    """
    scene = eqx.tree_at(
        lambda s: (s.star, s.planets[0].physical_model),
        _scene(),
        (
            BlackbodyStar(5772.0, Rsun2m),
            BlackbodyPhysicalModel(teff_K=jnp.array([300.0])),
        ),
    )
    where = {
        "star": lambda s: s.star.flux_scale,
        "planet": lambda s: s.planets[0].flux_scale,
    }[leaf]

    def read(scene):
        return jaxedith.scene_rates(
            _curves(),
            scene,
            _instrument(),
            _quadrature(),
            jnp.array([0.0]),
            resolution=_resolution(scene, _quadrature()),
        )

    base = read(scene)
    scaled = read(eqx.tree_at(where, scene, jnp.full_like(where(scene), 2.0)))
    np.testing.assert_allclose(scaled.planet, planet_factor * base.planet, rtol=1e-12)
    np.testing.assert_allclose(
        scaled.star_leakage, leakage_factor * base.star_leakage, rtol=1e-12
    )


def test_missing_sources_are_zero_and_the_sky_is_required_with_them():
    """Fails if an absent disk or zodi contributes or a missing sky profile passes.

    Basis: exact zeros; the error is raised before any read.
    """
    bare = _scene(disk=False, zodi=False)
    rates = jaxedith.rates(
        _curves(sky=None),
        bare,
        _instrument(),
        _quadrature(),
        EAST,
        NORTH,
        contrast=CONTRAST,
        time_d=0.0,
        sources=_sources(bare, _quadrature()),
    )
    np.testing.assert_array_equal(rates.local_zodi, 0.0)
    np.testing.assert_array_equal(rates.exozodi, 0.0)
    assert float(rates.planet[0, 0]) > 0
    scene = _scene(disk=False)
    quadrature = _quadrature((BAND, Band(735.0, 785.0)))
    curves = (_curves(), _curves(sky=None))
    with pytest.raises(ValueError, match=r"band 1 \(760\.0 nm\).*no uniform_sky"):
        jaxedith.rates(
            curves,
            scene,
            _instrument(),
            quadrature,
            EAST,
            NORTH,
            contrast=CONTRAST,
            time_d=0.0,
            sources=_sources(scene, quadrature),
        )


def test_extended_sources_use_each_bands_sky_transmission():
    """Hand radiance products to float64 roundoff catch reusing the first profile."""
    quadrature = _quadrature((BAND, Band(735.0, 785.0)))
    scene, instrument = _scene(), _instrument()
    transmissions = (0.2, 0.8)
    curves = tuple(_curves(sky=sky) for sky in transmissions)
    actual = _read(
        curves=curves, scene=scene, instrument=instrument, quadrature=quadrature
    )
    for w, (band, sky) in enumerate(zip(quadrature.bands, transmissions, strict=True)):
        node = quadrature.nodes_nm[w, 0]
        lod_arcsec = lambda_d_to_arcsec(1.0, band.center_nm, DIAMETER_M)
        solid_angle = AREA * arcsec_to_rad(lod_arcsec) ** 2
        prefactor = AREA_M2 * OPTICS * sky * solid_angle
        zodi = scene.local_zodi.photon_radiance(node, scene.sky_frame, 0.0)
        disk = scene.photon_radiance_at(node, 0.0, EAST, NORTH)
        np.testing.assert_allclose(
            actual.local_zodi[w, 0], prefactor * zodi, rtol=1e-12
        )
        np.testing.assert_allclose(actual.exozodi[w, 0], prefactor * disk, rtol=1e-12)


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


def test_beyond_the_data_is_zero_and_unsupported_and_a_diameter_mismatch_raises():
    """Fails if an out-of-data read looks supported or another telescope's curves pass.

    Basis: exact zero in every field and ``supported`` False once the aperture
    (separation plus the rule's radius) leaves ``data_extent_lod``, and the
    flag turning the exposure time into NaN with ``valid`` False; a read whose
    aperture ends exactly inside the extent is supported. A 5 percent diameter
    change is far outside the 1e-6 relative tolerance of the frame check.
    """
    short = _curves(extent=1.0)
    outside = _read(curves=short)
    assert not np.any(outside.supported) and not np.any(outside.variance.supported)
    for name in ("planet", "star_leakage", "local_zodi", "exozodi", "thermal"):
        np.testing.assert_array_equal(getattr(outside, name), 0.0)
        np.testing.assert_array_equal(getattr(outside.variance, name), 0.0)
    np.testing.assert_array_equal(outside.n_pix, 0.0)
    result = ox.exposure_time(
        outside,
        _instrument().detector,
        5.0,
        frame_time_s=1.0,
        post_processing=ox.PostProcessing(background_variance_factor=1.0),
    )
    assert np.isnan(result.time_s).all() and not np.any(result.valid)
    # The aperture reaches the extent exactly at separation + radius = extent.
    lod = float(lambda_d_to_arcsec(1.0, BAND.center_nm, DIAMETER_M))
    separation = np.hypot(EAST, NORTH) / lod
    edge = _curves(extent=separation + 0.7)
    assert np.all(_read(curves=edge).supported)
    assert not np.any(_read(curves=_curves(extent=separation + 0.7 - 1e-6)).supported)
    with pytest.raises(ValueError, match="differs from the curves"):
        _read(instrument=_instrument(diameter=DIAMETER_M * 1.05))


@pytest.mark.parametrize("offset", [-0.01, 0.0, 0.01])
def test_a_read_past_the_last_sample_is_unsupported_inside_the_extent(offset):
    """Fails if samples that end inside the array extent are read as data beyond them.

    Basis: the last sample sits ``offset`` lambda/D from the planet while the
    extent (15) leaves the aperture room; past the sample every field is an exact
    zero, ``supported`` False and the exposure time NaN, and at or inside it the
    values equal those of curves that run on (rtol 1e-12: constant curves refit on a
    shorter axis differ by float64 rounding only).
    """
    lod = float(lambda_d_to_arcsec(1.0, BAND.center_nm, DIAMETER_M))
    separation = np.hypot(EAST, NORTH) / lod
    short = _curves(last=separation + offset)
    got = _read(curves=short)
    detector = _instrument().detector
    time = ox.exposure_time(
        got,
        detector,
        5.0,
        frame_time_s=1.0,
        post_processing=ox.PostProcessing(background_variance_factor=1.0),
    ).time_s
    if offset < 0:
        assert not np.any(got.supported) and not np.any(got.variance.supported)
        for name in ("planet", "star_leakage", "local_zodi", "exozodi", "thermal"):
            np.testing.assert_array_equal(getattr(got, name), 0.0)
        np.testing.assert_array_equal(got.n_pix, 0.0)
        assert np.isnan(time).all()
    else:
        whole = _read(curves=_curves())
        assert np.all(got.supported)
        for name in ("planet", "star_leakage", "local_zodi", "exozodi", "thermal"):
            np.testing.assert_allclose(
                getattr(got, name), getattr(whole, name), rtol=1e-12
            )
        assert np.isfinite(time).all()


def test_scene_rates_flag_the_unsupported_planets_and_have_no_nan_gradient():
    """Fails if ``scene_rates`` drops the flag or the zeroed read poisons a gradient.

    Basis: ``supported`` has shape (W, K, T) and is False for every planet beyond
    the extent, whose fields are exact zeros; the gradient of the summed planet
    rate in the planet position is exactly zero there, not NaN.
    """
    short = _curves(extent=1.0)
    record = jaxedith.scene_rates(
        short,
        _scene(),
        _instrument(),
        _quadrature(),
        jnp.array([0.0]),
        resolution=_resolution(_scene(), _quadrature()),
    )
    assert record.supported.shape == (1, 1, 1)
    assert not np.any(record.supported)
    np.testing.assert_array_equal(record.planet, 0.0)

    def total(east):
        return _read(curves=short, east_arcsec=east).planet.sum()

    assert float(jax.grad(total)(EAST)) == 0.0


def test_transformations_are_finite_inside_the_data():
    """Fails if jit, vmap or grad of the read produces a non-finite value.

    Basis: finite outputs and a central-difference check of the planet rate in
    contrast (the rate is linear in it, so the step is exact to 1e-9).
    """
    curves, scene, instrument = _curves(), _scene(), _instrument()
    quadrature = _quadrature()
    sources = _sources(scene, quadrature)

    def planet(east, north, contrast):
        rates = jaxedith.rates(
            curves,
            scene,
            instrument,
            quadrature,
            east,
            north,
            contrast=contrast,
            time_d=0.0,
            sources=sources,
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
    quadrature = _quadrature()
    sources = _sources(scene, quadrature)
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
            contrast=CONTRAST,
            time_d=0.0,
            sources=sources,
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
    keywords = dict(time_d=0.0, sources=_sources(scene, quadrature))
    rates = jaxedith.rates(*common[:-1], contrast=common[-1], **keywords)
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
    wavelengths = jnp.array([300.0, 900.0])
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
    quadrature = _quadrature((BAND, Band(600.0, 700.0)))
    times = jnp.array([0.0, 40.0, 90.0])
    resolution = _resolution(scene, quadrature)
    sources = scene.sources(quadrature, resolution, times)
    traces = []

    @jax.jit
    def read(time_d):
        traces.append(1)
        return jaxedith.scene_rates(
            curves,
            scene,
            instrument,
            quadrature,
            time_d,
            resolution=resolution,
        )

    got = read(times)
    assert got.planet.shape == (2, 1, 2, 3) and got.n_pix.shape == (2, 2, 3)
    assert got.variance.planet.shape == (2, 1, 2, 3)
    east, north = scene.positions(times)
    for k in range(2):
        for t in range(3):
            at_time = scene.sources(quadrature, resolution, times[t])
            want = jaxedith.rates(
                curves,
                scene,
                instrument,
                quadrature,
                east[k, t],
                north[k, t],
                contrast=sources.contrast[:, :, k, t],
                time_d=times[t],
                sources=at_time,
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
            curves,
            scene,
            instrument,
            quadrature,
            0.0,
            resolution=resolution,
        )


DIAMETERS_LOD = np.array([0.0, 0.05, 0.2])
DENSITY_ROWS = np.array([1.0, 3.0, 10.0])


def _diameter_curves(diameters=DIAMETERS_LOD, rows=DENSITY_ROWS):
    """Curves with one flat leakage profile per tabulated stellar diameter."""
    flat = lambda value: np.array([value, value])  # noqa: E731
    over = lambda value: np.outer(rows, flat(value))  # noqa: E731
    return PerformanceCurves(
        np.array([0.0, 20.0]),
        _frame(),
        Circular(0.7),
        flat(TAU),
        flat(AREA),
        over(1e-10),
        over(DENSITY),
        over(1e-9),
        1.0,
        15.0,
        stellar_diameters_lod=diameters,
        throughput_variance=flat(TAU2),
        core_area_variance_lod2=flat(AREA2),
        leakage_fraction_variance=over(1e-10),
        leakage_density_variance_lod2=over(DENSITY2),
        uniform_sky=_sky(_frame()),
    )


def _node_lod_arcsec():
    return float(
        _frame().lod_to_arcsec(float(_quadrature().nodes_nm[0, 0])),
    )


def _star_leakage(curves, diameter_lod):
    """The leakage of a star of ``diameter_lod`` (lambda/D at the node)."""
    scene = _scene(diameter_arcsec=diameter_lod * _node_lod_arcsec())
    return _read(curves=curves, scene=scene)


@pytest.mark.parametrize("diameter_lod", [0.0, 0.03, 0.05, 0.12])
def test_leakage_is_read_at_the_stars_diameter_in_lambda_over_d(diameter_lod):
    """Fails if the star's size is dropped, or its units are not lambda/D at the node.

    Basis: the leakage rate is linear in the density the record returns, so the
    ratio to a point star's rate equals the record's own read at that diameter
    (rtol 1e-9, float64 products). The record's rows differ by 3 and 10 times,
    so a diameter ignored or read in arcseconds changes the ratio by far more.
    """
    curves = _diameter_curves()
    base = _star_leakage(curves, 0.0)
    rates = _star_leakage(curves, diameter_lod)
    separation = np.hypot(EAST, NORTH) / _node_lod_arcsec()
    for record, rate, point in (
        (curves, rates, base),
        (curves, rates.variance, base.variance),
    ):
        name = (
            "leakage_density_variance_lod2"
            if rate is rates.variance
            else ("leakage_density_lod2")
        )
        want = record.evaluate(name, separation, diameter_lod) / record.evaluate(
            name, separation, 0.0
        )
        np.testing.assert_allclose(
            rate.star_leakage / point.star_leakage, want, rtol=RTOL
        )
    if diameter_lod > 0:
        assert float(rates.star_leakage[0, 0]) > float(base.star_leakage[0, 0])


def test_leakage_gradient_in_the_star_diameter():
    """Fails if the star's size cannot be differentiated through the rates.

    Basis: the autodiff derivative of the leakage rate in the stellar diameter
    against a central finite difference (rtol 1e-6, float64 step 1e-5 lambda/D);
    it is finite and nonzero.
    """
    curves = _diameter_curves()
    lod = _node_lod_arcsec()

    def leakage(diameter_lod):
        scene = _scene(diameter_arcsec=diameter_lod * lod)
        return _read(curves=curves, scene=scene).star_leakage[0, 0]

    at = 0.07
    grad = float(jax.grad(leakage)(at))
    step = 1e-5
    fd = float(leakage(at + step) - leakage(at - step)) / (2 * step)
    assert np.isfinite(grad) and grad != 0.0
    np.testing.assert_allclose(grad, fd, rtol=1e-6)


def test_a_one_profile_record_serves_only_its_own_star():
    """Fails if a record measured for one star is read for another without notice."""
    own = 0.05
    curves = _diameter_curves(np.array([own]), np.array([1.0]))
    rates = _star_leakage(curves, own)
    assert np.isfinite(float(rates.star_leakage[0, 0]))
    with pytest.raises(ValueError, match="one profile"):
        _star_leakage(curves, 0.3)


def test_a_diameter_record_refuses_a_star_beyond_its_table():
    """Fails if a star larger than the tabulated range is read by extrapolation."""
    with pytest.raises(ValueError, match="outside the tabulated range"):
        _star_leakage(_diameter_curves(), 0.5)


def test_the_diameter_is_read_at_each_node_not_the_reference_or_band_centre():
    """Fails if the star's size is read at the wrong wavelength, in the wrong unit
    or with the ratio inverted.

    The record's rows are exp(3 d), so the log is linear in diameter and the
    not-a-knot spline of the log is exact between knots. The ratio of a 6 mas
    star's rate to a point star's is then exp(3 d) with d = theta / (lambda / D)
    at each of three nodes of a 700 to 900 nm band, a band away from the frame's
    550 nm reference. The truth uses hwoutils constants alone. Tolerance 1e-6:
    the spline abscissa of the zero knot is the float32 epsilon, which leaves
    3.6e-7 measured. A read at the reference wavelength changes the ratio by
    up to 43 percent, at the band centre by 6.8 percent, and with the ratio
    inverted or arcsec taken as lambda/D by far more.
    """
    from hwoutils.constants import rad2arcsec

    a = 3.0
    diameters = np.linspace(0.0, 0.5, 6)
    curves = _diameter_curves(diameters, np.exp(a * diameters))
    quadrature = _quadrature_for(Band(700.0, 900.0), order=3)
    theta = 0.006

    def read(arcsec):
        scene = _scene(diameter_arcsec=arcsec)
        return _read(curves=curves, scene=scene, quadrature=quadrature)

    ratio = np.asarray(read(theta).star_leakage[0]) / np.asarray(
        read(0.0).star_leakage[0]
    )
    nodes = np.asarray(quadrature.nodes_nm[0])
    lod_arcsec = nodes * 1e-9 / DIAMETER_M * rad2arcsec
    np.testing.assert_allclose(ratio, np.exp(a * theta / lod_arcsec), rtol=1e-6)
    assert np.ptp(ratio) > 0.1


def test_zero_star_thermal_planet_flux():
    """Truth: source photon density times unit collecting gain and curve throughput.

    Floating-point basis: rtol=1e-12, atol=0 for the x64 hand product;
    dividing by the exactly zero host before multiplying gives NaN.
    """
    scene = _scene(disk=False, zodi=False)
    scene = eqx.tree_at(
        lambda s: s.star, scene, FlatStar(flux_phot_per_nm_m2=1.0, flux_scale=0.0)
    )
    scene = eqx.tree_at(
        lambda s: s.planets,
        scene,
        (Planet(jnp.ones(1), BlackbodyPhysicalModel(jnp.array([300.0]))),),
    )
    instrument = Instrument(
        SimplePrimary(circumscribed_diameter_m=DIAMETER_M, collecting_area_m2=1.0),
        (ConstantThroughput(1.0),),
        None,
        IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)),
    )
    quadrature = _quadrature()
    times = jnp.array([0.0])
    resolution = _resolution(scene, quadrature)
    sources = scene.sources(quadrature, resolution, times)
    record = jaxedith.scene_rates(
        _curves(throughput=(1.0, 1.0), sky=None),
        scene,
        instrument,
        quadrature,
        times,
        resolution=resolution,
    )
    assert np.isfinite(record.planet).all()
    assert record.supported.all()
    np.testing.assert_allclose(record.planet, sources.planets, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        record.variance.planet, sources.planets * TAU2, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("kwargs", [{}, {"contrast": CONTRAST, "planet": 3.0}])
def test_rates_requires_one_planet_input(kwargs):
    """Truth: one radiometric input; exact-algebra host refusal, no tolerance."""
    scene, quadrature = _scene(), _quadrature()
    for read in (jaxedith.rates, eqx.filter_jit(jaxedith.rates)):
        with pytest.raises(ValueError, match=r"contrast.*planet|planet.*contrast"):
            read(
                _curves(),
                scene,
                _instrument(),
                quadrature,
                EAST,
                NORTH,
                time_d=0.0,
                sources=_sources(scene, quadrature),
                **kwargs,
            )
