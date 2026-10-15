"""Stellar diameter propagation through scalar and system ETC surfaces."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from coronalyze import PPConfig
from hwoutils.constants import Msun2kg
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.orbit import KeplerianOrbit
from skyscapes.physical_model import GridPhysicalModel
from skyscapes.scene import FlatStar, Planet, Star, System

from jaxedith import public
from jaxedith.intermediates import stellar_leakage, stellar_noise_floor
from jaxedith.scene import ETCScene

VARIANTS = ("ayo", "exosims_det", "exosims_char")
OPERATIONS = ("count_rates", "exptime", "snr")
DIAMETERS_MAS = jnp.array([0.0, 0.5, 1.0, 2.0, 4.0])
SCENE_KWARGS = dict(
    F0=1.34e8,
    Fs_over_F0=0.005311,
    Fp_over_Fs=1e-8,
    Fzodi=1e-14,
    Fexozodi=1e-14,
)


class DiameterCoronagraph(eqx.Module):
    """Scalar-only test stand-in with diameter-dependent leakage."""

    def throughput(self, separation_lod, wavelength_nm, *, time_s=0.0):
        return 0.3

    def core_area(self, separation_lod, wavelength_nm, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(
        self, separation_lod, wavelength_nm, *, time_s=0.0, stellar_diam_lod=0.0
    ):
        base = 1e-8 / (1.0 + jnp.asarray(separation_lod) ** 2)
        return base * (1.0 + (stellar_diam_lod / 0.01) ** 2)

    def occulter_transmission(self, separation_lod, wavelength_nm, *, time_s=0.0):
        return 1.0


@pytest.fixture
def optical_path():
    return ox.OpticalPath(
        primary=ox.SimplePrimary(diameter_m=7.877, obscuration=0.0),
        attenuating_elements=(ox.ConstantThroughput(throughput=0.5),),
        coronagraph=DiameterCoronagraph(),
        detector=ox.Detector(pixel_scale_arcsec=0.01, shape=(10, 10)),
    )


def _scalar_result(optical_path, scene, variant, operation):
    fn = getattr(public, f"{operation}_{variant}")
    tail = {"count_rates": (), "exptime": (7.0,), "snr": (3600.0,)}[operation]
    ppfact = 1e8 if variant == "ayo" else 1e-8
    return fn(optical_path, scene, 500.0, 5.0, 20.0, *tail, ppfact=ppfact)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_explicit_zero_is_bit_identical_to_default(optical_path, variant, operation):
    default = ETCScene(**SCENE_KWARGS)
    explicit = ETCScene(**SCENE_KWARGS, stellar_diam_mas=0.0)
    fn = getattr(public, f"{operation}_{variant}")
    tail = {"count_rates": (), "exptime": (7.0,), "snr": (3600.0,)}[operation]
    old = fn(optical_path, default, 500.0, 5.0, 20.0, *tail)
    np.testing.assert_array_equal(
        fn(optical_path, explicit, 500.0, 5.0, 20.0, *tail), old
    )
    assert np.all(np.isfinite(old))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("wavelength_nm,diameter_m", [(500.0, 7.877), (750.0, 6.0)])
def test_converted_diameter_reaches_coronagraph(
    optical_path, monkeypatch, variant, wavelength_nm, diameter_m
):
    received = []
    original = DiameterCoronagraph.core_mean_intensity

    def record(self, separation_lod, wavelength_nm, *, stellar_diam_lod=0.0):
        received.append(stellar_diam_lod)
        return original(
            self, separation_lod, wavelength_nm, stellar_diam_lod=stellar_diam_lod
        )

    monkeypatch.setattr(DiameterCoronagraph, "core_mean_intensity", record)
    optical_path = eqx.tree_at(lambda p: p.primary.diameter_m, optical_path, diameter_m)
    scene = ETCScene(**SCENE_KWARGS, stellar_diam_mas=1.0)
    getattr(public, f"count_rates_{variant}")(
        optical_path, scene, wavelength_nm, 5.0, 20.0
    )
    expected = 1 / (1000 * wavelength_nm * 1e-9 / diameter_m * 206264.80624709636)
    assert len(received) == (2 if variant == "ayo" else 1)
    np.testing.assert_allclose(received, expected, rtol=1e-6)
    if wavelength_nm == 500.0:
        np.testing.assert_allclose(received, 1 / 13.09, rtol=1e-3)


@pytest.mark.parametrize("term", [stellar_leakage, stellar_noise_floor])
def test_intermediate_uses_diameter_for_stellar_intensity(optical_path, term):
    args = (optical_path, 500.0, 5.0, 20.0, 1.34e8, 0.005311)
    point = term(*args)
    extended = term(*args, stellar_diam_lod=0.02)
    np.testing.assert_allclose(extended, 5.0 * point, rtol=1e-6)


def test_ayo_exptime_strictly_increases_with_diameter(optical_path):
    old = _scalar_result(optical_path, ETCScene(**SCENE_KWARGS), "ayo", "exptime")
    times = [
        _scalar_result(
            optical_path,
            ETCScene(**SCENE_KWARGS, stellar_diam_mas=d),
            "ayo",
            "exptime",
        )
        for d in DIAMETERS_MAS
    ]
    np.testing.assert_array_equal(times[0], old)
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) > 0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_diameter_jit_and_vmap_match_eager(optical_path, variant, operation):
    def evaluate(diameter_mas):
        return jnp.asarray(
            _scalar_result(
                optical_path,
                ETCScene(**SCENE_KWARGS, stellar_diam_mas=diameter_mas),
                variant,
                operation,
            )
        )

    expected = jnp.stack([evaluate(d) for d in DIAMETERS_MAS])
    compiled = jax.jit(evaluate)
    np.testing.assert_allclose(
        jnp.stack([compiled(d) for d in DIAMETERS_MAS]), expected, rtol=1e-6
    )
    np.testing.assert_allclose(jax.vmap(evaluate)(DIAMETERS_MAS), expected, rtol=1e-6)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(DIAMETERS_MAS), expected, rtol=1e-6
    )
    assert np.all(np.isfinite(expected))
    assert np.any(np.asarray(expected[0]) != np.asarray(expected[-1]))


@pytest.fixture
def system():
    star = Star(
        Ms_kg=Msun2kg,
        dist_pc=10.0,
        diameter_arcsec=0.0,
        wavelengths_nm=jnp.array([400.0, 800.0]),
        times_jd=jnp.array([2460000.5]),
        flux_density_jy=jnp.full((2, 1), 3631.0),
        interpolation_method="linear",
    )
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([0.0]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        t0_d=jnp.array([0.0]),
    )
    planet = Planet(
        Rp_Rearth=jnp.array([1.0]),
        Mp_Mearth=jnp.array([1.0]),
        orbit=orbit,
        physical_model=GridPhysicalModel(
            wavelengths_nm=jnp.array([400.0, 800.0]),
            phase_angle_deg=jnp.array([0.0, 180.0]),
            contrast_grid=jnp.full((1, 2, 2), 1e-8),
        ),
    )
    return System(
        star=star,
        planets=(planet,),
        trig_solver=get_grid_solver(level="scalar", E=False, trig=True, jit=True),
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_from_system_propagates_star_diameter(optical_path, system, variant, operation):
    fn = getattr(public, f"{operation}_from_system_{variant}")
    exposure = ox.ExposureConfig(
        start_time_jd=jnp.array([2460000.5, 2460001.0]),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(20.0),
        position_angle_deg=jnp.asarray(0.0),
    )
    observatory = SimpleNamespace(
        temperature_K=270.0,
        stability_fact=1.0,
        overhead_multi=1.0,
        overhead_fixed_s=0.0,
    )
    ppconfig = PPConfig(ppfact=1e8 if variant == "ayo" else 1e-8)
    tail = {"count_rates": (), "exptime": (7.0,), "snr": (3600.0,)}[operation]

    def evaluate(s):
        return jnp.asarray(
            fn(
                s,
                optical_path,
                observatory,
                exposure,
                ppconfig,
                *tail,
                zodi_fn=lambda *_: 0.0,
            )
        )

    point = evaluate(system)
    extended_system = eqx.tree_at(lambda s: s.star.diameter_arcsec, system, 0.002)
    extended = evaluate(extended_system)
    assert np.all(np.isfinite(extended))
    if operation == "count_rates":
        np.testing.assert_array_equal(extended[0], point[0])
        assert np.all(extended[1:] > point[1:])
        expected_gain = (
            1 + (2 / (1000 * 500e-9 / 7.877 * 206264.80624709636) / 0.01) ** 2
        )
        np.testing.assert_allclose(extended[2] / point[2], expected_gain, rtol=1e-6)
    elif operation == "exptime":
        assert np.all(extended > point)
    else:
        assert np.all(extended < point)

    flat_star = FlatStar(
        Ms_kg=system.star.Ms_kg,
        dist_pc=system.star.dist_pc,
        flux_phot_per_nm_m2=system.star.spec_flux_density(500.0, 2460000.5),
    )
    without_diameter = eqx.tree_at(lambda s: s.star, system, flat_star)
    np.testing.assert_array_equal(evaluate(without_diameter), point)
