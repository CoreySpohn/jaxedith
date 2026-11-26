"""Analytic aperture rates: each term against the scalar formulas, and contracts."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from hwoutils.constants import Rsun2m, rad2arcsec
from hwoutils.conversions import arcsec_to_rad, lambda_d_to_arcsec
from hwoutils.epoch import Epoch
from hwoutils.testing import precision_pair, ulps
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

    Basis: 16 rounding operations cover the constant fits, angular conversion,
    products and one-node integration on both paths. The exozodi reference
    uses the scene's disk radiance and independently checks the aperture factors.
    """
    rates = _read()
    assert rates.carrier == "photon" and rates.spectral
    assert rates.planet.shape == (1, 1) and rates.n_pix.shape == (1,)
    # Detector yield is not applied: electrons of an ideal detector are the rates
    # integrated over the band, which the reference's bandwidth factor includes.
    electrons = ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)).electrons(rates)
    expected = _scalar_expectations()
    rtol = float(ulps(16, 1.0))
    for statistic, record, electron in (
        ("mean", rates, electrons),
        ("variance", rates.variance, electrons.variance),
    ):
        for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
            np.testing.assert_allclose(
                electron_value := getattr(electron, name)[0],
                expected[statistic][name],
                rtol=rtol,
                err_msg=f"{statistic} {name}",
            )
            assert np.isfinite(electron_value) and electron_value > 0
        np.testing.assert_allclose(
            record.n_pix[0], expected[statistic]["n_pix"], rtol=rtol
        )
    assert float(jnp.max(jnp.abs(rates.stellar_companions))) == 0.0


@pytest.mark.parametrize("temperature_K", [290.0, 1500.0])
def test_thermal_matches_the_pyedith_scalar_formula(temperature_K):
    """Fails if the warm telescope's rate departs from pyEDITH calculate_CRbth.

    Basis: ``reference.primitives.thermal_rate`` (the pyEDITH product of Planck
    photon radiance, emissivity times cold transmission, collecting area
    without throughput, bandwidth and aperture solid angle) on the node of a
    one-node band, for the mean and squared-weight records. Both paths share
    the Planck factors. Allow 12 rounding operations for the angular conversion,
    area, emissivity and band products across the two paths, and the same
    allowance for rate rescaling. The system throughput and flux scale must
    not enter, so both are non-unity here.
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
    rtol = float(ulps(12, 1.0))
    for record, area in ((electrons, AREA), (electrons.variance, AREA2)):
        expected = thermal_rate(
            node, AREA_M2, width, temperature_K, lod_rad, eps, 1, 1, area
        )
        assert expected > 0
        np.testing.assert_allclose(record.thermal[0], expected, rtol=rtol)
    others = _read()
    for name in ("planet", "star_leakage", "local_zodi", "exozodi"):
        np.testing.assert_allclose(
            getattr(rates, name), 1.3 * getattr(others, name), rtol=rtol
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
    """Hand radiance products catch reusing the first profile.

    Basis: 12 rounding operations for the angular conversion, area and
    radiance products on both paths, including the input casts.
    """
    quadrature = _quadrature((BAND, Band(735.0, 785.0)))
    scene, instrument = _scene(), _instrument()
    transmissions = (0.2, 0.8)
    curves = tuple(_curves(sky=sky) for sky in transmissions)
    actual = _read(
        curves=curves, scene=scene, instrument=instrument, quadrature=quadrature
    )
    rtol = float(ulps(12, 1.0))
    for w, (band, sky) in enumerate(zip(quadrature.bands, transmissions, strict=True)):
        node = quadrature.nodes_nm[w, 0]
        lod_arcsec = lambda_d_to_arcsec(1.0, band.center_nm, DIAMETER_M)
        solid_angle = AREA * arcsec_to_rad(lod_arcsec) ** 2
        prefactor = AREA_M2 * OPTICS * sky * solid_angle
        zodi = scene.local_zodi.photon_radiance(node, scene.sky_frame, 0.0)
        disk = scene.photon_radiance_at(node, 0.0, EAST, NORTH)
        np.testing.assert_allclose(actual.local_zodi[w, 0], prefactor * zodi, rtol=rtol)
        np.testing.assert_allclose(actual.exozodi[w, 0], prefactor * disk, rtol=rtol)


@pytest.mark.parametrize("disk,zodi", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("sky_start", [2.0, 4.0000004], ids=["overlap", "disjoint"])
def test_sky_profile_support_is_required_only_for_extended_sources(
    disk, zodi, sky_start
):
    """Fails if sky coverage gates bare scenes or slack bridges a gap in coverage.

    Basis: [2, 4] lies inside the aperture domain, while [4.0000004, 6]
    starts just beyond a curve ending at 4 lambda/D. A disjoint profile
    cannot support an extended-source read even within roundoff of both
    endpoints. Unsupported rates are exact zeros.
    """
    sky_end = 4.0 if sky_start == 2.0 else 6.0
    profile = SampledProfile(
        [sky_start, sky_end],
        [0.8, 0.4],
        quantity="uniform_sky_response",
        frame=_frame(),
        interpolation="linear",
    )
    last = 20.0 if sky_start == 2.0 else 4.0
    curves = eqx.tree_at(lambda c: c.uniform_sky, _curves(last=last), profile)
    scene, instrument, quadrature = (
        _scene(disk=disk, zodi=zodi),
        _instrument(),
        _quadrature(),
    )
    sources = _sources(scene, quadrature)
    lod = _frame().lod_to_arcsec(BAND.center_nm)

    def read(east):
        return jaxedith.rates(
            curves,
            scene,
            instrument,
            quadrature,
            east,
            0.0,
            contrast=CONTRAST,
            time_d=0.0,
            sources=sources,
        )

    for evaluate in (read, eqx.filter_jit(read)):
        for separation in (1.0, 2.0, 4.0, 7.0):
            result = evaluate(jnp.asarray(separation * lod))
            supported = separation <= last and (
                (not disk and not zodi) or sky_start <= separation <= sky_end
            )
            np.testing.assert_array_equal(result.supported, [supported])
            np.testing.assert_array_equal(result.variance.supported, [supported])
            if supported:
                assert np.all(result.planet > 0)
                if disk:
                    assert np.all(result.exozodi > 0)
                if zodi:
                    assert np.all(result.local_zodi > 0)
            else:
                for record in (result, result.variance):
                    for name in record.component_names():
                        np.testing.assert_array_equal(getattr(record, name), 0.0)
                    np.testing.assert_array_equal(record.n_pix, 0.0)


def test_separation_is_read_at_each_band_centre():
    """Fails if a band reads another band's lambda/D, or per-band curves mix.

    Basis: throughput equal to the separation in lambda/D; the planet rate over
    its prefactor must be the angle over the band-centre lambda/D. Allow
    12 rounding operations for hypot, unit conversion, the linear curve
    read, the planet-rate product and division by the prefactor.
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
        rates.planet[:, 0] / prefactor, separation / lod, rtol=float(ulps(12, 1.0))
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
    # The extent and radius round to the same float32 value before subtraction.
    scene, instrument, quadrature = (
        _scene(disk=False, zodi=False),
        _instrument(),
        _quadrature(),
    )
    sources = _sources(scene, quadrature)

    def on_axis(curves, east):
        return jaxedith.rates(
            curves,
            scene,
            instrument,
            quadrature,
            east,
            0.0,
            contrast=CONTRAST,
            time_d=0.0,
            sources=sources,
        )

    for extent, supported in ((0.7, True), (0.7 - 1e-8, False)):
        curves = _curves(extent=extent)
        for read in (on_axis, eqx.filter_jit(on_axis)):
            np.testing.assert_array_equal(
                read(curves, jnp.asarray(0.0)).supported, [supported]
            )
    with pytest.raises(ValueError, match="differs from the curves"):
        _read(instrument=_instrument(diameter=DIAMETER_M * 1.05))


@pytest.mark.parametrize("boundary", [3.0, 10.0])
@pytest.mark.parametrize("limit", ["last_sample", "aperture_extent"])
def test_curve_and_aperture_support_at_float32_neighbors(boundary, limit):
    """Fails if the support flags diverge near either radial upper bound.

    Basis: symmetric float32 neighbors at 1, 2 and 4 representable steps
    and the relative offsets 2**-22, 2**-21 and 2**-20 bracket contact.
    The two-operation radial allowance covers hypot and angular conversion. Outside
    support, rates are exact zero and exposure time is NaN. Eight rounding
    operations cover the constant profile reads and final rate products.
    """
    curves = (
        _curves(last=boundary)
        if limit == "last_sample"
        else _curves(extent=boundary + 0.7)
    )
    scene, instrument, quadrature = _scene(), _instrument(), _quadrature()
    sources = _sources(scene, quadrature)
    lod = _frame().lod_to_arcsec(BAND.center_nm)
    step = float(np.spacing(np.float32(boundary)))
    accepted = [0, step, 2 * step, boundary * 2**-22]
    rejected = [4 * step, boundary * 2**-21, boundary * 2**-20]
    samples = sorted(
        {
            (
                float(np.float32(boundary + sign * offset)),
                sign < 0 or offset in accepted,
            )
            for offset in (*accepted, *rejected)
            for sign in (-1, 1)
        }
    )
    east = jnp.asarray([separation * lod for separation, _ in samples])
    supported = np.array([expected for _, expected in samples])

    def read(east, record=curves):
        return jaxedith.rates(
            record,
            scene,
            instrument,
            quadrature,
            east,
            0.0,
            contrast=CONTRAST,
            time_d=0.0,
            sources=sources,
        )

    whole_curves = _curves()
    comparisons = [
        read(position, whole_curves) if expected else None
        for position, expected in zip(east, supported, strict=True)
    ]
    # Reuse one compilation across all the positions near this boundary.
    for evaluate in (read, eqx.filter_jit(read)):
        for position, expected, whole in zip(east, supported, comparisons, strict=True):
            got = evaluate(position)
            time = ox.exposure_time(
                got,
                instrument.detector,
                5.0,
                frame_time_s=1.0,
                post_processing=ox.PostProcessing(background_variance_factor=1.0),
            ).time_s
            np.testing.assert_array_equal(got.supported, [expected])
            np.testing.assert_array_equal(got.variance.supported, [expected])
            if whole is not None:
                for name in got.component_names():
                    np.testing.assert_allclose(
                        getattr(got, name),
                        getattr(whole, name),
                        rtol=float(ulps(8, 1.0)),
                        atol=0,
                    )
                assert np.isfinite(time).all()
            else:
                for record in (got, got.variance):
                    for name in record.component_names():
                        np.testing.assert_array_equal(getattr(record, name), 0.0)
                    np.testing.assert_array_equal(record.n_pix, 0.0)
                assert np.isnan(time).all()


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


@pytest.mark.float64_only(reason="Finite-difference gradient reference")
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

    Basis: eight rounding operations cover the additions and three operations
    in the occupancy formula. A photon-rate record reaches the same value
    through the detector's yield.
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
    rtol = float(ulps(8, 1.0))
    np.testing.assert_allclose(
        jaxedith.photon_counting_frame_time(electrons, detector), expected, rtol=rtol
    )
    rates = _read()
    frame = jaxedith.photon_counting_frame_time(rates, detector)
    total = detector.electrons(rates).total()
    np.testing.assert_allclose(frame, rates.n_pix / (6.73 * total), rtol=rtol)
    with pytest.raises(ValueError, match="n_pix"):
        jaxedith.photon_counting_frame_time(
            eqx.tree_at(lambda r: r.n_pix, rates, None, is_leaf=lambda x: x is None),
            detector,
        )


def test_photon_counting_frame_time_integrates_and_aligns_pixel_counts_by_band():
    """The two bandwidths and pixel counts act along the leading band axis.

    Basis: eight rounding operations cover the one-node integration and the
    occupancy formula on host float64 products, including the input casts.
    """
    quadrature = _quadrature((BAND, Band(600.0, 650.0)))
    detector = IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8))
    per_nm = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]])
    pixel_counts = np.array([20.0, 40.0])
    rates = ox.Rates(
        quadrature, "electron", planet=per_nm[:, None, :], n_pix=pixel_counts
    )
    expected = pixel_counts[:, None] / (6.73 * np.array([[100.0], [50.0]]) * per_nm)
    for record in (rates, rates.integrate()):
        got = jaxedith.photon_counting_frame_time(record, detector)
        np.testing.assert_allclose(
            got, expected, rtol=float(ulps(8, jnp.ones((), got.dtype)))
        )


@pytest.fixture
def planet_exposure_inputs():
    """The two-band observation shared by the one-planet exposure tests."""
    quadrature = _quadrature((BAND, Band(600.0, 700.0)))
    scene = _scene()
    common = (_curves(), scene, _instrument(), quadrature, EAST, NORTH)
    keywords = dict(time_d=0.0, sources=_sources(scene, quadrature))
    return common, keywords, jnp.array([5.0, 7.0]), jnp.array([100.0, 60.0])


@vv.case("etc-inversion", "code-verification", basis="spike", srq="exposure-time")
@pytest.mark.float64_only(reason="precision_pair needs x64")
@pytest.mark.parametrize(
    ("quantity", "bound"),
    [
        ("time_s", 1.4e-6),
        ("planet", 4.0e-9),
        ("star_leakage", 9.0e-7),
        ("background", 1.2e-6),
    ],
)
def test_planet_exposure_precision(planet_exposure_inputs, quantity, bound):
    """Fails if narrowing the ETC inputs loses accuracy in time or count rates.

    A flat star at 10 pc, a 6 m primary and a planet at 0.05 arcsec with
    contrast 1e-6 or 3e-10. The 500-600 and 600-700 nm bands use S/N 5 and 7
    and frames of 100 and 60 s, with zero residuals and unit background factor.
    Maximum errors relative to each band's float64 value were 4.51e-7 for
    time, 1.30e-9 for planet, 2.97e-7 for leakage and 3.93e-7 for total
    background. Bounds are three times the unrounded maxima, rounded up to
    two significant figures: 1.4e-6, 4.0e-9, 9.0e-7 and 1.2e-6, respectively.
    """
    common, keywords, target_snr, frame = planet_exposure_inputs
    exposure = dict(
        target_snr=target_snr,
        frame_time_s=frame,
        post_processing=ox.PostProcessing(background_variance_factor=1.0),
    )

    def read(common, keywords, exposure, contrast):
        if quantity == "time_s":
            value = jaxedith.planet_exposure_time(
                *common, contrast, **keywords, **exposure
            ).time_s
        else:
            rates = jaxedith.rates(*common, contrast=contrast, **keywords).integrate()
            if quantity == "background":
                value = (
                    rates.star_leakage
                    + rates.local_zodi
                    + rates.exozodi
                    + rates.thermal
                    + rates.stellar_companions
                )
            else:
                value = getattr(rates, quantity)
        assert value.dtype == contrast.dtype
        return value

    for contrast in (1e-6, CONTRAST):
        args64 = (common, keywords, exposure, jnp.asarray(contrast))
        precision_pair(read, args64, bound=bound, scale=0.0)


def test_scene_entry_points_equal_the_two_step_calls(planet_exposure_inputs):
    """Fails if a scene entry point does any work of its own.

    Basis: exact equality with ``rates`` followed by the optixstuff function,
    for a unit background factor with no residual, for the AYO setting and for
    per-band times.
    """
    common, keywords, target_snr, frame = planet_exposure_inputs
    instrument = common[2]
    rates = jaxedith.rates(*common, contrast=1e-6, **keywords)
    for post_processing in (
        ox.PostProcessing(background_variance_factor=1.0),
        ox.PostProcessing(0.1, 0.05, background_variance_factor=2.0),
    ):
        wanted = ox.exposure_time(
            rates,
            instrument.detector,
            target_snr,
            frame_time_s=frame,
            post_processing=post_processing,
        )
        got = jaxedith.planet_exposure_time(
            *common,
            contrast=1e-6,
            **keywords,
            target_snr=target_snr,
            frame_time_s=frame,
            post_processing=post_processing,
        )
        assert bool(got.valid.all())
        np.testing.assert_array_equal(got.time_s, wanted.time_s)
        np.testing.assert_array_equal(got.valid, wanted.valid)
        np.testing.assert_array_equal(
            jaxedith.planet_snr(
                *common,
                contrast=1e-6,
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
    scene's own positions and contrasts. Allow 16 rounding operations for
    the scalar products and fits. The zodi terms add the projected-orbit
    and inverse-square arithmetic (24 operations) and three magnitude
    operations at the sensitivity of the AB exponential, 0.4 ln(10) per
    magnitude. One trace serves a second call with new times.
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
    rtol = float(ulps(16, 1.0))
    mag_scale = scene.local_zodi.surface_brightness_mag_arcsec2
    zodi_ops = 24 + int(np.ceil(3 * 0.4 * np.log(10.0) * mag_scale))
    zodi_rtol = float(ulps(zodi_ops, 1.0))
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
                        rtol=zodi_rtol if name in ("local_zodi", "exozodi") else rtol,
                        err_msg=name,
                    )
            np.testing.assert_allclose(got.n_pix[:, k, t], want.n_pix, rtol=rtol)
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


def _band_lod_arcsec():
    return _frame().lod_to_arcsec(BAND.center_nm)


def _star_leakage(curves, diameter_lod):
    """The leakage of a star of ``diameter_lod`` (lambda/D at the band centre)."""
    scene = _scene(diameter_arcsec=diameter_lod * _band_lod_arcsec())
    return _read(curves=curves, scene=scene)


@pytest.mark.parametrize("diameter_lod", [0.0, 0.03, 0.05, 0.12])
def test_leakage_is_read_at_the_stars_diameter_in_lambda_over_d(diameter_lod):
    """Fails if the star's size is dropped, or its units are not band-centre lambda/D.

    Basis: the leakage rate is linear in the density the record returns, so the
    ratio to a point star's rate equals the host curve read at that diameter.
    Six rounding operations cover the rate products and ratio. The log-density
    spline adds nine operations for its Horner weights and three-term dot
    product, at the scale of the largest absolute log-density. Exponentiation
    converts absolute error there to relative error in the rate. The rows
    differ by factors of 3 and 10, well above either lane's bound.
    """
    curves = _diameter_curves()
    base = _star_leakage(curves, 0.0)
    rates = _star_leakage(curves, diameter_lod)
    separation = np.hypot(EAST, NORTH) / _band_lod_arcsec()
    for name, rate, point in (
        ("leakage_density_lod2", rates, base),
        ("leakage_density_variance_lod2", rates.variance, base.variance),
    ):
        want = curves.evaluate(name, separation, diameter_lod) / curves.evaluate(
            name, separation, 0.0
        )
        log_scale = np.max(np.abs(np.log(getattr(curves, name))))
        rtol = ulps(6, 1.0) + ulps(9, log_scale)
        np.testing.assert_allclose(
            rate.star_leakage / point.star_leakage, want, rtol=float(rtol)
        )
    if diameter_lod > 0:
        assert float(rates.star_leakage[0, 0]) > float(base.star_leakage[0, 0])


@pytest.mark.float64_only(reason="Finite-difference gradient reference")
def test_leakage_gradient_in_the_star_diameter():
    """Fails if the star's size cannot be differentiated through the rates.

    Basis: the autodiff derivative of the leakage rate in the stellar diameter
    against a central finite difference (rtol 1e-6, float64 step 1e-5 lambda/D);
    it is finite and nonzero.
    """
    curves = _diameter_curves()
    lod = _band_lod_arcsec()

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


def test_leakage_reads_the_stellar_diameter_at_the_observing_band_centre():
    """All nodes see the same star size in the yield-code aperture.

    The 495 nm observing band centre is 0.9 times the 550 nm frame reference.
    At that scale the star is exactly 0.25 lambda/D, a knot of the linear
    density table. The hand rate is the incident photon density times
    effective area, aperture area and tabulated leakage density. Allow six
    rounding operations for the products and four more at the log-density
    scale for conversion of the diameter and a read within rounding of the
    knot. Exponentiation maps that absolute log error to relative rate error.
    """
    diameters = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
    curves = _diameter_curves(diameters, 1 + 3 * diameters)
    band = Band(470.0, 520.0)
    quadrature = Quadrature.build((band,), breaks_nm=[490.0], order=2)
    diameter_lod = 0.25
    diameter_arcsec = diameter_lod * (495e-9 / DIAMETER_M) * rad2arcsec
    scene = _scene(diameter_arcsec=diameter_arcsec, disk=False, zodi=False)
    result = _read(curves=curves, scene=scene, quadrature=quadrature)
    for rate, area, density in (
        (result, AREA, DENSITY),
        (result.variance, AREA2, DENSITY2),
    ):
        expected = FLUX * AREA_M2 * OPTICS * area * density * (1 + 3 * diameter_lod)
        log_scale = abs(np.log(density))
        rtol = ulps(6, 1.0) + ulps(4, log_scale)
        np.testing.assert_allclose(rate.star_leakage, expected, rtol=float(rtol))


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
