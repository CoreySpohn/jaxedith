"""Parity tests for Layer 2 components.

Each Layer 2 function is a pass-through wrapper over a Layer 1
count-rate function. These tests assert that, for a canonical
``OpticalPath + ETCScene`` fixture, each Layer 2 output equals the
matching decomposed output from ``count_rates_ayo``.

Because both paths call the identical Layer 1 function with identical
scalar arguments, outputs must be bitwise identical -- no rtol.
"""

import jax.numpy as jnp
import pytest
from yippy.datasets import fetch_coronagraph

import optixstuff as ox
from hwoutils.constants import nm2m, rad2arcsec
from jaxedith import ETCScene, components
from jaxedith.count_rates import (
    count_rate_binary,
    count_rate_detector,
    count_rate_exozodi,
    count_rate_stellar_leakage,
    count_rate_thermal,
    count_rate_zodi,
    noise_floor_stellar,
    photon_counting_time,
)
from jaxedith.public import count_rates_ayo

# These match the default n_channels and temp_K that optical_path carries
# (OpticalPath.n_channels=1.0, npix_multiplier=1.0) and the temp default.
# The Layer 1 calls below use these directly for parity with Layer 2.
N_CHANNELS = 1.0
TEMP_K = 270.0


@pytest.fixture(scope="module")
def yip_path():
    """Download the default AAVC coronagraph YIP via pooch (cached)."""
    return fetch_coronagraph()


@pytest.fixture(scope="module")
def optical_path(yip_path):
    """Canonical OpticalPath from the AAVC integration benchmark."""
    primary = ox.SimplePrimary(diameter_m=6.0, obscuration=0.14)
    coronagraph = ox.YippyCoronagraph(yip_path)
    detector = ox.Detector(
        pixel_scale=0.010,
        shape=(100, 100),
        quantum_efficiency=0.9,
        dark_current_rate=3e-5,
        read_noise_electrons=0.0,
        cic_rate=1.3e-3,
        frame_time=1000.0,
        read_time=1000.0,
        dqe=1.0,
    )
    optics_filter = ox.ConstantThroughputElement(throughput=0.5, name="optics")
    return ox.OpticalPath(
        primary=primary,
        coronagraph=coronagraph,
        attenuating_elements=(optics_filter,),
        detector=detector,
    )


@pytest.fixture(scope="module")
def etc_scene():
    """Canonical ETCScene: sun-like star at 10 pc with Earth-like planet."""
    return ETCScene(
        F0=1.34e8,
        Fs_over_F0=0.005311,
        Fp_over_Fs=1e-10,
        Fzodi=3.5e-10,
        Fexozodi=7.15e-9,
        dist_pc=10.0,
        sep_arcsec=0.1,
        Fbinary=0.0,
    )


@pytest.fixture(scope="module")
def observation():
    """Observation scalars matching the integration-test benchmark."""
    return {
        "wavelength_nm": 500.0,
        "separation_lod": 5.0,
        "dlambda_nm": 100.0,
        "snr": 7.0,
    }


@pytest.fixture(scope="module")
def reference_count_rates(optical_path, etc_scene, observation):
    """(Cp, Cb, Cnf_rate) from count_rates_ayo on canonical inputs."""
    return count_rates_ayo(
        optical_path,
        etc_scene,
        observation["wavelength_nm"],
        observation["separation_lod"],
        observation["dlambda_nm"],
        temp_K=TEMP_K,
    )


def test_components_module_imports():
    """Smoke test: the components module can be imported."""
    assert components is not None


def test_reference_fixture_returns_three_tuple(reference_count_rates):
    """Sanity: the reference call returns (Cp, Cb, Cnf_rate)."""
    Cp, Cb, Cnf_rate = reference_count_rates
    assert float(Cp) > 0.0
    assert float(Cb) > 0.0


def test_planet_signal_parity(
    optical_path, etc_scene, observation, reference_count_rates
):
    """planet_signal output must equal Cp from count_rates_ayo."""
    Cp_ref, _, _ = reference_count_rates
    Cp_layer2 = components.planet_signal(
        optical_path,
        wavelength_nm=observation["wavelength_nm"],
        separation_lod=observation["separation_lod"],
        dlambda_nm=observation["dlambda_nm"],
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        Fp_over_Fs=etc_scene.Fp_over_Fs,
        n_channels=N_CHANNELS,
    )
    assert float(Cp_layer2) == float(Cp_ref)


def test_stellar_leakage_parity(optical_path, etc_scene, observation):
    """stellar_leakage must equal the decomposed CRbs call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    coro = optical_path.coronagraph
    primary = optical_path.primary

    CRbs_ref = count_rate_stellar_leakage(
        etc_scene.F0,
        etc_scene.Fs_over_F0,
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        coro.core_area(sep, wl),
        coro.core_mean_intensity(sep, wl),
    )

    CRbs_layer2 = components.stellar_leakage(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        n_channels=N_CHANNELS,
    )
    assert float(CRbs_layer2) == float(CRbs_ref)


def test_zodi_background_parity(optical_path, etc_scene, observation):
    """zodi_background must equal the decomposed CRbz call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    coro = optical_path.coronagraph
    primary = optical_path.primary
    lod_rad = (wl * nm2m) / primary.diameter_m
    lod_arcsec = lod_rad * rad2arcsec

    CRbz_ref = count_rate_zodi(
        etc_scene.F0,
        etc_scene.Fzodi,
        lod_arcsec,
        coro.occulter_transmission(sep, wl),
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        coro.core_area(sep, wl),
    )

    CRbz_layer2 = components.zodi_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fzodi=etc_scene.Fzodi,
        n_channels=N_CHANNELS,
    )
    assert float(CRbz_layer2) == float(CRbz_ref)


def test_exozodi_background_parity(optical_path, etc_scene, observation):
    """exozodi_background must equal the decomposed CRbez call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    coro = optical_path.coronagraph
    primary = optical_path.primary
    lod_rad = (wl * nm2m) / primary.diameter_m
    lod_arcsec = lod_rad * rad2arcsec

    CRbez_ref = count_rate_exozodi(
        etc_scene.F0,
        etc_scene.Fexozodi,
        lod_arcsec,
        coro.occulter_transmission(sep, wl),
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        coro.core_area(sep, wl),
        etc_scene.dist_pc,
        etc_scene.sep_arcsec,
    )

    CRbez_layer2 = components.exozodi_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fexozodi=etc_scene.Fexozodi,
        dist_pc=etc_scene.dist_pc,
        sep_arcsec=etc_scene.sep_arcsec,
        n_channels=N_CHANNELS,
    )
    assert float(CRbez_layer2) == float(CRbez_ref)


def test_binary_background_parity(optical_path, etc_scene, observation):
    """binary_background must equal the decomposed CRbbin call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    coro = optical_path.coronagraph
    primary = optical_path.primary

    CRbbin_ref = count_rate_binary(
        etc_scene.F0,
        etc_scene.Fbinary,
        coro.occulter_transmission(sep, wl),
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        coro.core_area(sep, wl),
    )

    CRbbin_layer2 = components.binary_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fbinary=etc_scene.Fbinary,
        n_channels=N_CHANNELS,
    )
    assert float(CRbbin_layer2) == float(CRbbin_ref)


def test_thermal_background_parity(optical_path, etc_scene, observation):
    """thermal_background must equal the decomposed CRbth call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    primary = optical_path.primary
    coro = optical_path.coronagraph
    detector = optical_path.detector
    lod_rad = (wl * nm2m) / primary.diameter_m
    eps_warm_T_cold = 0.0

    CRbth_ref = count_rate_thermal(
        wl,
        primary.area_m2,
        dl,
        TEMP_K,
        lod_rad,
        eps_warm_T_cold,
        detector.quantum_efficiency,
        detector.dqe,
        coro.core_area(sep, wl),
    )

    CRbth_layer2 = components.thermal_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        temp_K=TEMP_K,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    assert float(CRbth_layer2) == float(CRbth_ref)


def test_detector_noise_parity(optical_path, etc_scene, observation):
    """detector_noise must equal the decomposed CRbd call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    coro = optical_path.coronagraph
    detector = optical_path.detector

    core_area_lod2 = coro.core_area(sep, wl)
    lod_arcsec = (wl * nm2m) / optical_path.primary.diameter_m * rad2arcsec
    det_pixscale_lod = detector.pixel_scale / lod_arcsec
    n_pix = (
        core_area_lod2
        / (det_pixscale_lod ** 2)
        * optical_path.n_channels
        * 1.0
    )

    Cp_ref = components.planet_signal(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fs_over_F0=etc_scene.Fs_over_F0,
        Fp_over_Fs=etc_scene.Fp_over_Fs, n_channels=N_CHANNELS,
    )
    CRbs_ref = components.stellar_leakage(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fs_over_F0=etc_scene.Fs_over_F0,
        n_channels=N_CHANNELS,
    )
    CRbz_ref = components.zodi_background(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fzodi=etc_scene.Fzodi, n_channels=N_CHANNELS,
    )
    CRbez_ref = components.exozodi_background(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fexozodi=etc_scene.Fexozodi, dist_pc=etc_scene.dist_pc,
        sep_arcsec=etc_scene.sep_arcsec, n_channels=N_CHANNELS,
    )
    total_photon_cr = Cp_ref + CRbs_ref + CRbz_ref + CRbez_ref
    t_photon = photon_counting_time(jnp.maximum(total_photon_cr, 1e-30), n_pix)

    CRbd_ref = count_rate_detector(
        n_pix,
        detector.dark_current_rate,
        detector.read_noise_electrons,
        detector.read_time,
        detector.cic_rate,
        t_photon,
    )

    CRbd_layer2 = components.detector_noise(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        total_photon_rate=total_photon_cr,
        npix_multiplier=1.0,
    )
    assert float(CRbd_layer2) == float(CRbd_ref)


def test_stellar_noise_floor_parity(optical_path, etc_scene, observation):
    """stellar_noise_floor must equal the decomposed CRnf_star_rate call."""
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]
    ppfact = 1.0  # default ppfact

    coro = optical_path.coronagraph
    primary = optical_path.primary
    pixscale_lod = coro.pixel_scale_lod
    noisefloor_value = coro.core_mean_intensity(sep, wl) / (
        ppfact * pixscale_lod ** 2
    )

    CRnf_star_ref = noise_floor_stellar(
        etc_scene.F0,
        etc_scene.Fs_over_F0,
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        noisefloor_value,
        coro.core_area(sep, wl),
    )

    CRnf_star_layer2 = components.stellar_noise_floor(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        n_channels=N_CHANNELS,
        ppfact=ppfact,
    )
    assert float(CRnf_star_layer2) == float(CRnf_star_ref)


def test_layer2_reexported_from_package():
    """Layer 2 functions are accessible as jaxedith.planet_signal, etc."""
    import jaxedith

    expected = [
        "planet_signal",
        "stellar_leakage",
        "zodi_background",
        "exozodi_background",
        "binary_background",
        "thermal_background",
        "detector_noise",
        "stellar_noise_floor",
    ]
    for name in expected:
        assert hasattr(jaxedith, name), f"jaxedith.{name} missing from public API"


def test_layer2_end_to_end_reconstructs_count_rates_ayo(
    optical_path, etc_scene, observation, reference_count_rates
):
    """Composing Layer 2 components reproduces count_rates_ayo outputs.

    Exercises the full Layer 2 surface composed the same way public.py composes
    Layer 1. The resulting Cp and Cb must match count_rates_ayo's
    corresponding values bitwise.
    """
    import jaxedith
    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    Cp = jaxedith.planet_signal(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fs_over_F0=etc_scene.Fs_over_F0,
        Fp_over_Fs=etc_scene.Fp_over_Fs, n_channels=N_CHANNELS,
    )
    CRbs = jaxedith.stellar_leakage(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fs_over_F0=etc_scene.Fs_over_F0,
        n_channels=N_CHANNELS,
    )
    CRbz = jaxedith.zodi_background(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fzodi=etc_scene.Fzodi, n_channels=N_CHANNELS,
    )
    CRbez = jaxedith.exozodi_background(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fexozodi=etc_scene.Fexozodi, dist_pc=etc_scene.dist_pc,
        sep_arcsec=etc_scene.sep_arcsec, n_channels=N_CHANNELS,
    )
    CRbbin = jaxedith.binary_background(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        F0=etc_scene.F0, Fbinary=etc_scene.Fbinary,
        n_channels=N_CHANNELS,
    )
    CRbth = jaxedith.thermal_background(
        optical_path, wavelength_nm=wl, separation_lod=sep, dlambda_nm=dl,
        temp_K=TEMP_K,
    )
    total_photon_cr = Cp + CRbs + CRbz + CRbez
    CRbd = jaxedith.detector_noise(
        optical_path, wavelength_nm=wl, separation_lod=sep,
        total_photon_rate=total_photon_cr, npix_multiplier=1.0,
    )
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    Cp_ref, Cb_ref, _ = reference_count_rates
    assert float(Cp) == float(Cp_ref)
    assert float(Cb) == float(Cb_ref)
