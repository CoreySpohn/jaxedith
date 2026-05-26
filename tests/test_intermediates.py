"""Parity tests for Layer 2 intermediates.

Each Layer 2 function is a pass-through wrapper over a Layer 1
primitive. These tests assert that, for a canonical
``OpticalPath + ETCScene`` fixture, each Layer 2 output equals the
matching decomposed output from ``count_rates_ayo``.

Because both paths call the identical Layer 1 function with identical
scalar arguments, outputs must be bitwise identical -- no rtol.
"""

import jax.numpy as jnp
import optixstuff as ox
import pytest
from coronagraphoto.datasets import fetch_coronagraph
from hwoutils.constants import nm2m, rad2arcsec

from jaxedith import ETCScene, intermediates
from jaxedith.primitives import (
    binary_rate,
    detector_noise_rate,
    exozodi_rate,
    noise_floor_stellar,
    photon_counting_time,
    stellar_leakage_rate,
    thermal_rate,
    zodi_rate,
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
        pixel_scale_arcsec=0.010,
        shape=(100, 100),
        quantum_efficiency=0.9,
        dark_current_rate_e_per_s=3e-5,
        read_noise_e=0.0,
        clock_induced_charge_rate_e_per_frame=1.3e-3,
        frame_time_s=1000.0,
        read_time_s=1000.0,
        dqe=1.0,
    )
    optics_filter = ox.ConstantThroughput(throughput=0.5, name="optics")
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


def test_intermediates_module_imports():
    """Smoke test: the intermediates module can be imported."""
    assert intermediates is not None


def test_reference_fixture_returns_three_tuple(reference_count_rates):
    """Sanity: the reference call returns (Cp, Cb, Cnf_rate)."""
    Cp, Cb, _Cnf_rate = reference_count_rates
    assert float(Cp) > 0.0
    assert float(Cb) > 0.0


def test_planet_signal_parity(
    optical_path, etc_scene, observation, reference_count_rates
):
    """planet_signal output must equal Cp from count_rates_ayo."""
    Cp_ref, _, _ = reference_count_rates
    Cp_layer2 = intermediates.planet_signal(
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

    CRbs_ref = stellar_leakage_rate(
        etc_scene.F0,
        etc_scene.Fs_over_F0,
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        coro.core_area(sep, wl),
        coro.core_mean_intensity(sep, wl),
    )

    CRbs_layer2 = intermediates.stellar_leakage(
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

    CRbz_ref = zodi_rate(
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

    CRbz_layer2 = intermediates.zodi_background(
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

    CRbez_ref = exozodi_rate(
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

    CRbez_layer2 = intermediates.exozodi_background(
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

    CRbbin_ref = binary_rate(
        etc_scene.F0,
        etc_scene.Fbinary,
        coro.occulter_transmission(sep, wl),
        primary.area_m2,
        optical_path.system_throughput(wl),
        dl,
        N_CHANNELS,
        coro.core_area(sep, wl),
    )

    CRbbin_layer2 = intermediates.binary_background(
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

    CRbth_ref = thermal_rate(
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

    CRbth_layer2 = intermediates.thermal_background(
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
    det_pixscale_lod = detector.pixel_scale_arcsec / lod_arcsec
    n_pix = core_area_lod2 / (det_pixscale_lod**2) * optical_path.n_channels * 1.0

    Cp_ref = intermediates.planet_signal(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        Fp_over_Fs=etc_scene.Fp_over_Fs,
        n_channels=N_CHANNELS,
    )
    CRbs_ref = intermediates.stellar_leakage(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        n_channels=N_CHANNELS,
    )
    CRbz_ref = intermediates.zodi_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fzodi=etc_scene.Fzodi,
        n_channels=N_CHANNELS,
    )
    CRbez_ref = intermediates.exozodi_background(
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
    total_photon_cr = Cp_ref + CRbs_ref + CRbz_ref + CRbez_ref
    t_photon = photon_counting_time(jnp.maximum(total_photon_cr, 1e-30), n_pix)

    CRbd_ref = detector_noise_rate(
        n_pix,
        detector.dark_current_rate_e_per_s,
        detector.read_noise_e,
        detector.read_time_s,
        detector.clock_induced_charge_rate_e_per_frame,
        t_photon,
    )

    CRbd_layer2 = intermediates.detector_noise(
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
    noisefloor_value = coro.core_mean_intensity(sep, wl) / (ppfact * pixscale_lod**2)

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

    CRnf_star_layer2 = intermediates.stellar_noise_floor(
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
    """Composing Layer 2 intermediates reproduces count_rates_ayo outputs.

    Exercises the full Layer 2 surface composed the same way public.py composes
    Layer 1. The resulting Cp and Cb must match count_rates_ayo's
    corresponding values bitwise.
    """
    import jaxedith

    wl = observation["wavelength_nm"]
    sep = observation["separation_lod"]
    dl = observation["dlambda_nm"]

    Cp = jaxedith.planet_signal(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        Fp_over_Fs=etc_scene.Fp_over_Fs,
        n_channels=N_CHANNELS,
    )
    CRbs = jaxedith.stellar_leakage(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        n_channels=N_CHANNELS,
    )
    CRbz = jaxedith.zodi_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fzodi=etc_scene.Fzodi,
        n_channels=N_CHANNELS,
    )
    CRbez = jaxedith.exozodi_background(
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
    CRbbin = jaxedith.binary_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fbinary=etc_scene.Fbinary,
        n_channels=N_CHANNELS,
    )
    CRbth = jaxedith.thermal_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        temp_K=TEMP_K,
    )
    total_photon_cr = Cp + CRbs + CRbz + CRbez
    CRbd = jaxedith.detector_noise(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        total_photon_rate=total_photon_cr,
        npix_multiplier=1.0,
    )
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    Cp_ref, Cb_ref, _ = reference_count_rates
    assert float(Cp) == float(Cp_ref)
    assert float(Cb) == float(Cb_ref)
