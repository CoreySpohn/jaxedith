"""Parity tests for the variant-explicit Layer 3 publics (Plan 4).

Each test compares the new scalar public against the old variant-string API
driven by the matching preset config. Numerical equality is required --
Plan 4 is a pure refactor.
"""

import jax.numpy as jnp
import pytest
from yippy.datasets import fetch_coronagraph

import optixstuff as ox
from jaxedith import (
    AYO_CONFIG,
    CONFIG,
    EXOSIMS_CHARACTERIZATION_CONFIG,
    EXOSIMS_DETECTION_CONFIG,
    ETCScene,
    calc_count_rates,
    calc_exptime,
    calc_snr,
    count_rates_ayo,
    count_rates_exosims_char,
    count_rates_exosims_det,
    exptime_ayo,
    exptime_exosims_char,
    exptime_exosims_det,
    snr_ayo,
    snr_exosims_char,
    snr_exosims_det,
)


WL_NM = 500.0
SEP_LOD = 3.0
DLAMBDA_NM = 20.0
SNR = 7.0
T_OBS = 3600.0


@pytest.fixture(scope="module")
def yip_path():
    return fetch_coronagraph()


@pytest.fixture(scope="module")
def optical_path(yip_path):
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


@pytest.fixture
def scene():
    return ETCScene(
        F0=9400.0,
        Fs_over_F0=1.0,
        Fp_over_Fs=1e-10,
        Fzodi=23.0,
        Fexozodi=22.0,
        dist_pc=10.0,
        sep_arcsec=0.1,
        Fbinary=0.0,
        n_channels=1.0,
        ez_ppf=30.0,
        temp_K=270.0,
    )


def test_count_rates_ayo_matches_jaxedith_config(optical_path, scene):
    Cp_new, Cb_new, Cnf_rate_new = count_rates_ayo(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        temp_K=scene.temp_K, ez_ppf=scene.ez_ppf, ppfact=CONFIG.ppfact,
    )
    Cp_old, Cb_old, Cnf_old, _Csp = calc_count_rates(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, config=CONFIG,
    )
    assert jnp.allclose(Cp_new, Cp_old)
    assert jnp.allclose(Cb_new, Cb_old)
    assert jnp.allclose(Cnf_rate_new, Cnf_old)


def test_exptime_ayo_matches_jaxedith_config(optical_path, scene):
    t_new = exptime_ayo(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR,
        temp_K=scene.temp_K, ez_ppf=scene.ez_ppf,
        ppfact=CONFIG.ppfact, bg_multiplier=CONFIG.bg_multiplier,
        overhead_multi=CONFIG.overhead_multi,
        overhead_fixed_s=CONFIG.overhead_fixed_s,
        n_rolls=CONFIG.n_rolls,
    )
    t_old = calc_exptime(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR, config=CONFIG,
    )
    assert jnp.allclose(t_new, t_old)


def test_snr_ayo_matches_jaxedith_config(optical_path, scene):
    snr_new = snr_ayo(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, T_OBS,
        temp_K=scene.temp_K, ez_ppf=scene.ez_ppf,
        ppfact=CONFIG.ppfact, bg_multiplier=CONFIG.bg_multiplier,
        overhead_multi=CONFIG.overhead_multi,
        overhead_fixed_s=CONFIG.overhead_fixed_s,
        n_rolls=CONFIG.n_rolls,
    )
    snr_old = calc_snr(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, T_OBS, config=CONFIG,
    )
    assert jnp.allclose(snr_new, snr_old)


def test_exptime_ayo_matches_ayo_preset(optical_path, scene):
    t_new = exptime_ayo(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR,
        temp_K=scene.temp_K, ez_ppf=jnp.inf,
        ppfact=AYO_CONFIG.ppfact, bg_multiplier=AYO_CONFIG.bg_multiplier,
        overhead_multi=AYO_CONFIG.overhead_multi,
        overhead_fixed_s=AYO_CONFIG.overhead_fixed_s,
        n_rolls=AYO_CONFIG.n_rolls,
    )
    t_old = calc_exptime(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR, config=AYO_CONFIG,
    )
    assert jnp.allclose(t_new, t_old)


def test_count_rates_exosims_det_matches_preset(optical_path, scene):
    Cp_new, Cb_new, Csp_new = count_rates_exosims_det(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        temp_K=scene.temp_K,
        ppfact=EXOSIMS_DETECTION_CONFIG.ppfact,
        stability_fact=EXOSIMS_DETECTION_CONFIG.stability_fact,
    )
    Cp_old, Cb_old, _Cnf, Csp_old = calc_count_rates(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        config=EXOSIMS_DETECTION_CONFIG,
    )
    assert jnp.allclose(Cp_new, Cp_old)
    assert jnp.allclose(Cb_new, Cb_old)
    assert jnp.allclose(Csp_new, Csp_old)


def test_exptime_exosims_det_matches_preset(optical_path, scene):
    t_new = exptime_exosims_det(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR,
        temp_K=scene.temp_K,
        ppfact=EXOSIMS_DETECTION_CONFIG.ppfact,
        stability_fact=EXOSIMS_DETECTION_CONFIG.stability_fact,
        overhead_multi=EXOSIMS_DETECTION_CONFIG.overhead_multi,
        overhead_fixed_s=EXOSIMS_DETECTION_CONFIG.overhead_fixed_s,
        n_rolls=EXOSIMS_DETECTION_CONFIG.n_rolls,
    )
    t_old = calc_exptime(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR,
        config=EXOSIMS_DETECTION_CONFIG,
    )
    assert jnp.allclose(t_new, t_old)


def test_snr_exosims_det_matches_preset(optical_path, scene):
    snr_new = snr_exosims_det(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, T_OBS,
        temp_K=scene.temp_K,
        ppfact=EXOSIMS_DETECTION_CONFIG.ppfact,
        stability_fact=EXOSIMS_DETECTION_CONFIG.stability_fact,
        overhead_multi=EXOSIMS_DETECTION_CONFIG.overhead_multi,
        overhead_fixed_s=EXOSIMS_DETECTION_CONFIG.overhead_fixed_s,
        n_rolls=EXOSIMS_DETECTION_CONFIG.n_rolls,
    )
    snr_old = calc_snr(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, T_OBS,
        config=EXOSIMS_DETECTION_CONFIG,
    )
    assert jnp.allclose(snr_new, snr_old)


def test_count_rates_exosims_char_matches_preset(optical_path, scene):
    Cp_new, Cb_new, Csp_new = count_rates_exosims_char(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        temp_K=scene.temp_K,
        ppfact=EXOSIMS_CHARACTERIZATION_CONFIG.ppfact,
        stability_fact=EXOSIMS_CHARACTERIZATION_CONFIG.stability_fact,
    )
    Cp_old, Cb_old, _Cnf, Csp_old = calc_count_rates(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        config=EXOSIMS_CHARACTERIZATION_CONFIG,
    )
    assert jnp.allclose(Cp_new, Cp_old)
    assert jnp.allclose(Cb_new, Cb_old)
    assert jnp.allclose(Csp_new, Csp_old)


def test_exptime_exosims_char_matches_preset(optical_path, scene):
    t_new = exptime_exosims_char(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR,
        temp_K=scene.temp_K,
        ppfact=EXOSIMS_CHARACTERIZATION_CONFIG.ppfact,
        stability_fact=EXOSIMS_CHARACTERIZATION_CONFIG.stability_fact,
        overhead_multi=EXOSIMS_CHARACTERIZATION_CONFIG.overhead_multi,
        overhead_fixed_s=EXOSIMS_CHARACTERIZATION_CONFIG.overhead_fixed_s,
        n_rolls=EXOSIMS_CHARACTERIZATION_CONFIG.n_rolls,
    )
    t_old = calc_exptime(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, SNR,
        config=EXOSIMS_CHARACTERIZATION_CONFIG,
    )
    assert jnp.allclose(t_new, t_old)


def test_snr_exosims_char_matches_preset(optical_path, scene):
    snr_new = snr_exosims_char(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, T_OBS,
        temp_K=scene.temp_K,
        ppfact=EXOSIMS_CHARACTERIZATION_CONFIG.ppfact,
        stability_fact=EXOSIMS_CHARACTERIZATION_CONFIG.stability_fact,
        overhead_multi=EXOSIMS_CHARACTERIZATION_CONFIG.overhead_multi,
        overhead_fixed_s=EXOSIMS_CHARACTERIZATION_CONFIG.overhead_fixed_s,
        n_rolls=EXOSIMS_CHARACTERIZATION_CONFIG.n_rolls,
    )
    snr_old = calc_snr(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM, T_OBS,
        config=EXOSIMS_CHARACTERIZATION_CONFIG,
    )
    assert jnp.allclose(snr_new, snr_old)


def test_count_rates_exosims_det_scales_with_ppfact_stability(optical_path, scene):
    """Csp must scale linearly with both ppfact and stability_fact."""
    _Cp, _Cb, Csp_baseline = count_rates_exosims_det(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        temp_K=scene.temp_K, ppfact=1.0, stability_fact=1.0,
    )
    _Cp, _Cb, Csp_scaled = count_rates_exosims_det(
        optical_path, scene, WL_NM, SEP_LOD, DLAMBDA_NM,
        temp_K=scene.temp_K, ppfact=0.5, stability_fact=0.8,
    )
    # Csp = CRbs * ppfact * stability_fact, so scaling is 0.5 * 0.8 = 0.4.
    assert jnp.allclose(Csp_scaled, 0.4 * Csp_baseline)
