"""Layer 2: Structured count-rate components.

Each function accepts an ``optixstuff.OpticalPath`` and astrophysical /
observation scalars, unpacks the optical path, and calls the corresponding
pure Layer 1 function in :mod:`jaxedith.count_rates`.

Layer 2 turns the 8-to-12-argument pure-float call sites in
:mod:`jaxedith.count_rates` into structured calls that take an
``OpticalPath`` + a few scalars, without introducing heavy scene objects.

Each Layer 2 wrapper mirrors a single invocation inside
``jaxedith.core._compute_count_rates``; parity is tested in
``tests/test_components.py``.
"""

import jax.numpy as jnp
from hwoutils.constants import nm2m, rad2arcsec

from jaxedith.count_rates import (
    count_rate_binary,
    count_rate_detector,
    count_rate_exozodi,
    count_rate_planet,
    count_rate_stellar_leakage,
    count_rate_thermal,
    count_rate_zodi,
    noise_floor_stellar,
    photon_counting_time,
)


def _lod_rad(optical_path, wavelength_nm):
    """lambda/D in radians for the optical path's primary diameter."""
    return (wavelength_nm * nm2m) / optical_path.primary.diameter_m


def _lod_arcsec(optical_path, wavelength_nm):
    """lambda/D in arcsec for the optical path's primary diameter."""
    return _lod_rad(optical_path, wavelength_nm) * rad2arcsec


def planet_signal(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    F0,
    Fs_over_F0,
    Fp_over_Fs,
    n_channels=1.0,
):
    """Planet signal count rate Cp [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_planet`.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lam/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        F0: Flux zero point [ph/s/m^2/nm].
        Fs_over_F0: Stellar flux ratio (dimensionless).
        Fp_over_Fs: Planet-to-star contrast (dimensionless).
        n_channels: Number of spectral channels.
    """
    return count_rate_planet(
        F0,
        Fs_over_F0,
        Fp_over_Fs,
        optical_path.primary.area_m2,
        optical_path.system_throughput(wavelength_nm),
        optical_path.coronagraph.throughput(separation_lod, wavelength_nm),
        dlambda_nm,
        n_channels,
    )


def stellar_leakage(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    F0,
    Fs_over_F0,
    n_channels=1.0,
):
    """Stellar leakage count rate CRbs [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_stellar_leakage`.
    """
    coro = optical_path.coronagraph
    return count_rate_stellar_leakage(
        F0,
        Fs_over_F0,
        optical_path.primary.area_m2,
        optical_path.system_throughput(wavelength_nm),
        dlambda_nm,
        n_channels,
        coro.core_area(separation_lod, wavelength_nm),
        coro.core_mean_intensity(separation_lod, wavelength_nm),
    )


def zodi_background(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    F0,
    Fzodi,
    n_channels=1.0,
):
    """Local zodiacal light count rate CRbz [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_zodi`.
    """
    coro = optical_path.coronagraph
    return count_rate_zodi(
        F0,
        Fzodi,
        _lod_arcsec(optical_path, wavelength_nm),
        coro.occulter_transmission(separation_lod, wavelength_nm),
        optical_path.primary.area_m2,
        optical_path.system_throughput(wavelength_nm),
        dlambda_nm,
        n_channels,
        coro.core_area(separation_lod, wavelength_nm),
    )


def exozodi_background(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    F0,
    Fexozodi,
    dist_pc,
    sep_arcsec,
    n_channels=1.0,
):
    """Exozodiacal light count rate CRbez [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_exozodi`.
    """
    coro = optical_path.coronagraph
    return count_rate_exozodi(
        F0,
        Fexozodi,
        _lod_arcsec(optical_path, wavelength_nm),
        coro.occulter_transmission(separation_lod, wavelength_nm),
        optical_path.primary.area_m2,
        optical_path.system_throughput(wavelength_nm),
        dlambda_nm,
        n_channels,
        coro.core_area(separation_lod, wavelength_nm),
        dist_pc,
        sep_arcsec,
    )


def binary_background(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    F0,
    Fbinary,
    n_channels=1.0,
):
    """Binary / neighbor stray light count rate CRbbin [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_binary`.
    """
    coro = optical_path.coronagraph
    return count_rate_binary(
        F0,
        Fbinary,
        coro.occulter_transmission(separation_lod, wavelength_nm),
        optical_path.primary.area_m2,
        optical_path.system_throughput(wavelength_nm),
        dlambda_nm,
        n_channels,
        coro.core_area(separation_lod, wavelength_nm),
    )


def thermal_background(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    temp_K,
    eps_warm_T_cold=0.0,
):
    """Thermal background count rate CRbth [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_thermal`.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lam/D (used for core aperture).
        dlambda_nm: Bandwidth [nm].
        temp_K: Telescope mirror temperature [K].
        eps_warm_T_cold: Warm-optics emissivity times cold transmission.
            Defaults to 0.0 (no thermal background).
    """
    coro = optical_path.coronagraph
    detector = optical_path.detector
    return count_rate_thermal(
        wavelength_nm,
        optical_path.primary.area_m2,
        dlambda_nm,
        temp_K,
        _lod_rad(optical_path, wavelength_nm),
        eps_warm_T_cold,
        detector.quantum_efficiency,
        detector.dqe,
        coro.core_area(separation_lod, wavelength_nm),
    )


def detector_noise(
    optical_path,
    wavelength_nm,
    separation_lod,
    total_photon_rate,
    npix_multiplier=1.0,
):
    """Detector noise count rate CRbd [e/s].

    Wraps :func:`jaxedith.count_rates.count_rate_detector`.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        wavelength_nm: Observation wavelength [nm] (passed to coronagraph
            core-area lookup).
        separation_lod: Planet separation in lam/D (passed to coronagraph
            core-area lookup).
        total_photon_rate: Total photon count rate Cp + CRbs + CRbz + CRbez
            [e/s], used by ``photon_counting_time`` to derive t_photon.
        npix_multiplier: Aperture pixel-count correction (ETC config).
    """
    coro = optical_path.coronagraph
    detector = optical_path.detector
    core_area_lod2 = coro.core_area(separation_lod, wavelength_nm)
    pixscale_lod = coro.pixel_scale_lod
    n_pix = core_area_lod2 / (pixscale_lod ** 2) * npix_multiplier

    t_photon = photon_counting_time(
        jnp.maximum(total_photon_rate, 1e-30),
        n_pix,
    )
    return count_rate_detector(
        n_pix,
        detector.dark_current_rate,
        detector.read_noise_electrons,
        detector.read_time,
        detector.cic_rate,
        t_photon,
    )


def stellar_noise_floor(
    optical_path,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    F0,
    Fs_over_F0,
    n_channels=1.0,
    ppfact=1.0,
):
    """Stellar noise floor *rate* CRnf_star_rate [e/s].

    Wraps :func:`jaxedith.count_rates.noise_floor_stellar`. The
    ``noisefloor_value`` is derived as
    ``core_mean_intensity(sep, wl) / ppfact`` to match
    ``core._compute_count_rates``.
    """
    coro = optical_path.coronagraph
    noisefloor_value = coro.core_mean_intensity(separation_lod, wavelength_nm) / ppfact
    return noise_floor_stellar(
        F0,
        Fs_over_F0,
        optical_path.primary.area_m2,
        optical_path.system_throughput(wavelength_nm),
        dlambda_nm,
        n_channels,
        noisefloor_value,
        coro.core_area(separation_lod, wavelength_nm),
    )
