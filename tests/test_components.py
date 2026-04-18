"""Parity tests for Layer 2 components.

Each Layer 2 function is a pass-through wrapper over a Layer 1
count-rate function. These tests assert that, for a canonical
``OpticalPath + ETCScene`` fixture, each Layer 2 output equals the
matching decomposed output from ``core._compute_count_rates``.

Because both paths call the identical Layer 1 function with identical
scalar arguments, outputs must be bitwise identical -- no rtol.
"""

import pytest
from yippy.datasets import fetch_coronagraph

import optixstuff as ox
from hwoutils.constants import nm2m, rad2arcsec
from jaxedith import ETCScene, components
from jaxedith.config import CONFIG
from jaxedith.core import _compute_count_rates
from jaxedith.count_rates import count_rate_stellar_leakage, count_rate_zodi


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
        n_channels=2.0,
        Fbinary=0.0,
        temp_K=270.0,
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
    """(Cp, Cb, Cnf, Csp) from core._compute_count_rates on canonical inputs."""
    return _compute_count_rates(
        optical_path,
        etc_scene,
        observation["wavelength_nm"],
        observation["separation_lod"],
        observation["dlambda_nm"],
        observation["snr"],
        CONFIG,
    )


def test_components_module_imports():
    """Smoke test: the components module can be imported."""
    assert components is not None


def test_reference_fixture_returns_four_tuple(reference_count_rates):
    """Sanity: the reference call returns (Cp, Cb, Cnf, Csp)."""
    Cp, Cb, Cnf, Csp = reference_count_rates
    assert float(Cp) > 0.0
    assert float(Cb) > 0.0


def test_planet_signal_parity(
    optical_path, etc_scene, observation, reference_count_rates
):
    """planet_signal output must equal Cp from _compute_count_rates."""
    Cp_ref, _, _, _ = reference_count_rates
    Cp_layer2 = components.planet_signal(
        optical_path,
        wavelength_nm=observation["wavelength_nm"],
        separation_lod=observation["separation_lod"],
        dlambda_nm=observation["dlambda_nm"],
        F0=etc_scene.F0,
        Fs_over_F0=etc_scene.Fs_over_F0,
        Fp_over_Fs=etc_scene.Fp_over_Fs,
        n_channels=etc_scene.n_channels,
    )
    assert float(Cp_layer2) == float(Cp_ref)


def test_stellar_leakage_parity(optical_path, etc_scene, observation):
    """stellar_leakage must equal the decomposed CRbs call in core."""
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
        etc_scene.n_channels,
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
        n_channels=etc_scene.n_channels,
    )
    assert float(CRbs_layer2) == float(CRbs_ref)


def test_zodi_background_parity(optical_path, etc_scene, observation):
    """zodi_background must equal the decomposed CRbz call in core."""
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
        etc_scene.n_channels,
        coro.core_area(sep, wl),
    )

    CRbz_layer2 = components.zodi_background(
        optical_path,
        wavelength_nm=wl,
        separation_lod=sep,
        dlambda_nm=dl,
        F0=etc_scene.F0,
        Fzodi=etc_scene.Fzodi,
        n_channels=etc_scene.n_channels,
    )
    assert float(CRbz_layer2) == float(CRbz_ref)
