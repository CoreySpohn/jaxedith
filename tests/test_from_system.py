"""Tests for the *_from_system_* wrappers (Plan 5).

Three axes verified per wrapper:

1. Shape: return is always ``(K, T)``.
2. Parity: element ``[k, t]`` of the vmapped result equals the scalar
   public called on the same ``(k, t)`` scene.
3. Zodi swap: passing ``zodi_fn_leinert`` changes ``Fzodi`` as expected
   (smoke test; exact value tested in ``test_zodi.py``).
"""

import jax.numpy as jnp
import pytest
from hwoutils import constants as const
from optixstuff import Exposure
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.observatory import Observatory, ObservatoryL2Halo
from orbix.system.orbit import KeplerianOrbit
from skyscapes.atmosphere import GridAtmosphere
from skyscapes.scene import Planet, SpectrumStar, System
from yippy.datasets import fetch_coronagraph

import optixstuff as ox
from coronalyze import PPConfig
from jaxedith import (
    ETCScene,
    count_rates_from_system_ayo,
    exptime_ayo,
    exptime_from_system_ayo,
    snr_ayo,
    snr_from_system_ayo,
    zodi_fn_ayo,
    zodi_fn_leinert,
)

SNR = 7.0
T_OBS = 3600.0
ARCSEC_PER_RAD = 206264.80624709636


@pytest.fixture(scope="session")
def yip_path():
    return fetch_coronagraph()


@pytest.fixture(scope="session")
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
def system():
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    # times_jd covers the exposure epochs (2460000.5, 2460001.0)
    star = SpectrumStar(
        Ms_kg=const.Msun2kg,
        dist_pc=10.0,
        ra_deg=90.0,
        dec_deg=30.0,
        diameter_arcsec=0.0,
        luminosity_lsun=1.0,
        wavelengths_nm=jnp.array([400.0, 800.0]),
        times_jd=jnp.array([2459999.0, 2460002.0]),
        flux_density_jy=jnp.full((2, 2), 3631.0),
    )
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0, 1.5]),
        e=jnp.array([0.0, 0.0]),
        W_rad=jnp.array([0.0, 0.0]),
        i_rad=jnp.array([0.0, 0.0]),
        w_rad=jnp.array([0.0, 0.0]),
        M0_rad=jnp.array([0.0, 0.0]),
        t0_d=jnp.array([0.0, 0.0]),
    )
    atmosphere = GridAtmosphere(
        Rp_Rearth=jnp.array([1.0, 1.0]),
        wavelengths_nm=jnp.array([400.0, 800.0]),
        phase_angle_deg=jnp.array([0.0, 180.0]),
        contrast_grid=jnp.full((2, 2, 2), 1e-10),
    )
    return System(
        star=star,
        planets=(Planet(orbit=orbit, atmosphere=atmosphere),),
        trig_solver=solver,
    )


@pytest.fixture
def observatory():
    return Observatory(orbit=ObservatoryL2Halo.from_default())


@pytest.fixture
def ppconfig():
    return PPConfig(ppfact=1.0, n_rolls=1, ez_ppf=jnp.inf)


@pytest.fixture
def exposure():
    return Exposure(
        start_time_jd=jnp.array([2460000.5, 2460001.0]),
        exposure_time_s=jnp.asarray(3600.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(20.0),
        position_angle_deg=jnp.asarray(0.0),
    )


def _expected_sep_lod(alpha_arcsec, wavelength_nm, diameter_m):
    lod_rad = wavelength_nm * 1e-9 / diameter_m
    return alpha_arcsec / (lod_rad * ARCSEC_PER_RAD)


def test_count_rates_from_system_ayo_shape(
    optical_path, system, observatory, exposure, ppconfig
):
    Cp, Cb, Cnf_rate = count_rates_from_system_ayo(
        system, optical_path, observatory, exposure, ppconfig,
        zodi_fn=zodi_fn_ayo,
    )
    K = system.n_planets
    T = jnp.atleast_1d(exposure.start_time_jd).shape[0]
    assert Cp.shape == (K, T)
    assert Cb.shape == (K, T)
    assert Cnf_rate.shape == (K, T)


def test_exptime_from_system_ayo_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    t_new = exptime_from_system_ayo(
        system, optical_path, observatory, exposure, ppconfig, SNR,
        zodi_fn=zodi_fn_ayo,
    )
    K = system.n_planets
    T = jnp.atleast_1d(exposure.start_time_jd).shape[0]
    assert t_new.shape == (K, T)

    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)

    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(
        alpha[0, 0], wl, optical_path.primary.diameter_m
    )

    scene = ETCScene(
        F0=F0,
        Fs_over_F0=1.0,
        Fp_over_Fs=contrasts[0, 0],
        Fzodi=Fzodi,
        Fexozodi=0.0,
        dist_pc=system.star.dist_pc,
        sep_arcsec=alpha[0, 0],
        Fbinary=0.0,
    )
    t_scalar = exptime_ayo(
        optical_path, scene, wl, sep_lod, dlambda, SNR,
        temp_K=observatory.temperature_K,
        ez_ppf=ppconfig.ez_ppf,
        ppfact=ppconfig.ppfact,
        overhead_multi=observatory.overhead_multi,
        overhead_fixed_s=observatory.overhead_fixed_s,
        n_rolls=ppconfig.n_rolls,
    )
    assert jnp.allclose(t_new[0, 0], t_scalar)


def test_snr_from_system_ayo_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    snr_new = snr_from_system_ayo(
        system, optical_path, observatory, exposure, ppconfig, T_OBS,
        zodi_fn=zodi_fn_ayo,
    )
    K = system.n_planets
    T = jnp.atleast_1d(exposure.start_time_jd).shape[0]
    assert snr_new.shape == (K, T)

    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)
    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(alpha[0, 0], wl, optical_path.primary.diameter_m)

    scene = ETCScene(
        F0=F0,
        Fs_over_F0=1.0,
        Fp_over_Fs=contrasts[0, 0],
        Fzodi=Fzodi,
        Fexozodi=0.0,
        dist_pc=system.star.dist_pc,
        sep_arcsec=alpha[0, 0],
        Fbinary=0.0,
    )
    snr_scalar = snr_ayo(
        optical_path, scene, wl, sep_lod, dlambda, T_OBS,
        temp_K=observatory.temperature_K,
        ez_ppf=ppconfig.ez_ppf,
        ppfact=ppconfig.ppfact,
        overhead_multi=observatory.overhead_multi,
        overhead_fixed_s=observatory.overhead_fixed_s,
        n_rolls=ppconfig.n_rolls,
    )
    assert jnp.allclose(snr_new[0, 0], snr_scalar)


def test_exptime_from_system_ayo_respects_zodi_swap(
    optical_path, system, observatory, exposure, ppconfig
):
    t_ayo = exptime_from_system_ayo(
        system, optical_path, observatory, exposure, ppconfig, SNR,
        zodi_fn=zodi_fn_ayo,
    )
    t_leinert = exptime_from_system_ayo(
        system, optical_path, observatory, exposure, ppconfig, SNR,
        zodi_fn=zodi_fn_leinert,
    )
    # Must differ somewhere -- Leinert is position-dependent, AYO isn't
    assert not jnp.allclose(t_ayo, t_leinert)


def test_count_rates_from_system_ayo_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    from jaxedith import count_rates_ayo

    Cp_new, Cb_new, Cnf_new = count_rates_from_system_ayo(
        system, optical_path, observatory, exposure, ppconfig,
        zodi_fn=zodi_fn_ayo,
    )
    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)
    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(alpha[0, 0], wl, optical_path.primary.diameter_m)

    scene = ETCScene(
        F0=F0,
        Fs_over_F0=1.0,
        Fp_over_Fs=contrasts[0, 0],
        Fzodi=Fzodi,
        Fexozodi=0.0,
        dist_pc=system.star.dist_pc,
        sep_arcsec=alpha[0, 0],
        Fbinary=0.0,
    )
    Cp_scalar, Cb_scalar, Cnf_scalar = count_rates_ayo(
        optical_path, scene, wl, sep_lod, dlambda,
        temp_K=observatory.temperature_K,
        ez_ppf=ppconfig.ez_ppf,
        ppfact=ppconfig.ppfact,
    )
    assert jnp.allclose(Cp_new[0, 0], Cp_scalar)
    assert jnp.allclose(Cb_new[0, 0], Cb_scalar)
    assert jnp.allclose(Cnf_new[0, 0], Cnf_scalar)


from jaxedith import (
    count_rates_exosims_char,
    count_rates_exosims_det,
    count_rates_from_system_exosims_char,
    count_rates_from_system_exosims_det,
    exptime_exosims_char,
    exptime_exosims_det,
    exptime_from_system_exosims_char,
    exptime_from_system_exosims_det,
    snr_exosims_char,
    snr_exosims_det,
    snr_from_system_exosims_char,
    snr_from_system_exosims_det,
)


def _scalar_scene(system, sep_arcsec, Fp_over_Fs, F0, Fzodi):
    return ETCScene(
        F0=F0,
        Fs_over_F0=1.0,
        Fp_over_Fs=Fp_over_Fs,
        Fzodi=Fzodi,
        Fexozodi=0.0,
        dist_pc=system.star.dist_pc,
        sep_arcsec=sep_arcsec,
        Fbinary=0.0,
    )


def test_count_rates_from_system_exosims_det_shape(
    optical_path, system, observatory, exposure, ppconfig
):
    Cp, Cb, Csp = count_rates_from_system_exosims_det(
        system, optical_path, observatory, exposure, ppconfig,
        zodi_fn=zodi_fn_ayo,
    )
    K = system.n_planets
    T = jnp.atleast_1d(exposure.start_time_jd).shape[0]
    assert Cp.shape == (K, T)
    assert Cb.shape == (K, T)
    assert Csp.shape == (K, T)


def test_exptime_from_system_exosims_det_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    t_new = exptime_from_system_exosims_det(
        system, optical_path, observatory, exposure, ppconfig, SNR,
        zodi_fn=zodi_fn_ayo,
    )
    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)
    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(alpha[0, 0], wl, optical_path.primary.diameter_m)

    scene = _scalar_scene(system, alpha[0, 0], contrasts[0, 0], F0, Fzodi)
    t_scalar = exptime_exosims_det(
        optical_path, scene, wl, sep_lod, dlambda, SNR,
        temp_K=observatory.temperature_K,
        ppfact=ppconfig.ppfact,
        stability_fact=observatory.stability_fact,
        overhead_multi=observatory.overhead_multi,
        overhead_fixed_s=observatory.overhead_fixed_s,
        n_rolls=ppconfig.n_rolls,
    )
    assert jnp.allclose(t_new[0, 0], t_scalar)


def test_snr_from_system_exosims_det_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    snr_new = snr_from_system_exosims_det(
        system, optical_path, observatory, exposure, ppconfig, T_OBS,
        zodi_fn=zodi_fn_ayo,
    )
    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)
    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(alpha[0, 0], wl, optical_path.primary.diameter_m)

    scene = _scalar_scene(system, alpha[0, 0], contrasts[0, 0], F0, Fzodi)
    snr_scalar = snr_exosims_det(
        optical_path, scene, wl, sep_lod, dlambda, T_OBS,
        temp_K=observatory.temperature_K,
        ppfact=ppconfig.ppfact,
        stability_fact=observatory.stability_fact,
        overhead_multi=observatory.overhead_multi,
        overhead_fixed_s=observatory.overhead_fixed_s,
        n_rolls=ppconfig.n_rolls,
    )
    assert jnp.allclose(snr_new[0, 0], snr_scalar)


def test_exptime_from_system_exosims_char_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    t_new = exptime_from_system_exosims_char(
        system, optical_path, observatory, exposure, ppconfig, SNR,
        zodi_fn=zodi_fn_ayo,
    )
    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)
    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(alpha[0, 0], wl, optical_path.primary.diameter_m)

    scene = _scalar_scene(system, alpha[0, 0], contrasts[0, 0], F0, Fzodi)
    t_scalar = exptime_exosims_char(
        optical_path, scene, wl, sep_lod, dlambda, SNR,
        temp_K=observatory.temperature_K,
        ppfact=ppconfig.ppfact,
        stability_fact=observatory.stability_fact,
        overhead_multi=observatory.overhead_multi,
        overhead_fixed_s=observatory.overhead_fixed_s,
        n_rolls=ppconfig.n_rolls,
    )
    assert jnp.allclose(t_new[0, 0], t_scalar)


def test_snr_from_system_exosims_char_matches_scalar(
    optical_path, system, observatory, exposure, ppconfig
):
    snr_new = snr_from_system_exosims_char(
        system, optical_path, observatory, exposure, ppconfig, T_OBS,
        zodi_fn=zodi_fn_ayo,
    )
    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    t_jd = jnp.atleast_1d(exposure.start_time_jd)
    Fzodi = zodi_fn_ayo(observatory, exposure, system.star)
    alpha, _ = system.alpha_dMag(t_jd)
    contrasts = system.contrasts(jnp.atleast_1d(wl), t_jd)
    F0 = system.star.spec_flux_density(wl, t_jd[0])
    sep_lod = _expected_sep_lod(alpha[0, 0], wl, optical_path.primary.diameter_m)

    scene = _scalar_scene(system, alpha[0, 0], contrasts[0, 0], F0, Fzodi)
    snr_scalar = snr_exosims_char(
        optical_path, scene, wl, sep_lod, dlambda, T_OBS,
        temp_K=observatory.temperature_K,
        ppfact=ppconfig.ppfact,
        stability_fact=observatory.stability_fact,
        overhead_multi=observatory.overhead_multi,
        overhead_fixed_s=observatory.overhead_fixed_s,
        n_rolls=ppconfig.n_rolls,
    )
    assert jnp.allclose(snr_new[0, 0], snr_scalar)
