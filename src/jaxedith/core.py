"""JAX ETC -- core orchestration functions.

This module contains the main user-facing functions for the JAX exposure time
calculator. It bridges :mod:`optixstuff` hardware objects and astrophysical
scene parameters to the pure-JAX count-rate and solver functions.

All public functions accept ``optixstuff.OpticalPath`` and ``ETCScene``
eqx.Modules directly, making the entire pipeline JIT-able end-to-end via
``eqx.filter_jit``.
"""

import jax
import jax.numpy as jnp
from hwoutils.constants import nm2m, rad2arcsec

from .config import CONFIG
from .count_rates import (
    count_rate_binary,
    count_rate_detector,
    count_rate_exozodi,
    count_rate_planet,
    count_rate_stellar_leakage,
    count_rate_thermal,
    count_rate_zodi,
    noise_floor_exozodi,
    noise_floor_stellar,
    noise_floor_total,
    photon_counting_time,
    speckle_residual,
)
from .solver import (
    solve_exptime_ayo,
    solve_exptime_exosims_char,
    solve_exptime_exosims_det,
    solve_snr_ayo,
    solve_snr_exosims_char,
    solve_snr_exosims_det,
)

# -- Trace-time dispatch ------------------------------------------------------


def _dispatch_exptime(Cp, Cb, Cnf, Csp, snr, config):
    """Select and call the correct solve_exptime variant at trace time."""
    kwargs = dict(
        overhead_multi=config.overhead_multi,
        overhead_fixed=config.overhead_fixed_s,
        n_rolls=config.n_rolls,
    )
    if config.variant in ("ayo", "jaxedith"):
        return solve_exptime_ayo(
            Cp,
            Cb,
            Cnf,
            snr,
            bg_multiplier=config.bg_multiplier,
            **kwargs,
        )
    elif config.variant == "exosims_det":
        return solve_exptime_exosims_det(Cp, Cb, Csp, snr, **kwargs)
    elif config.variant == "exosims_char":
        return solve_exptime_exosims_char(Cp, Cb, Csp, snr, **kwargs)
    else:
        raise ValueError(f"Unknown config variant: {config.variant!r}")


def _dispatch_snr(Cp, Cb, Cnf_rate, Csp, t_obs, config):
    """Select and call the correct solve_snr variant at trace time."""
    kwargs = dict(
        overhead_multi=config.overhead_multi,
        overhead_fixed=config.overhead_fixed_s,
        n_rolls=config.n_rolls,
    )
    if config.variant in ("ayo", "jaxedith"):
        return solve_snr_ayo(
            Cp,
            Cb,
            Cnf_rate,
            t_obs,
            bg_multiplier=config.bg_multiplier,
            **kwargs,
        )
    elif config.variant == "exosims_det":
        return solve_snr_exosims_det(Cp, Cb, Csp, t_obs, **kwargs)
    elif config.variant == "exosims_char":
        return solve_snr_exosims_char(Cp, Cb, Csp, t_obs, **kwargs)
    else:
        raise ValueError(f"Unknown config variant: {config.variant!r}")


# -- Count rate orchestration --------------------------------------------------


def _compute_count_rates(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    snr,
    config,
    eps_warm_T_cold=0.0,
):
    """Compute all count rates from an OpticalPath and scene.

    This function reads directly from the ``OpticalPath`` pytree
    (primary, coronagraph, detector) so the full computation is
    JIT-traceable.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        scene: :class:`ETCScene` instance.
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lam/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        snr: Target SNR (or 1.0 for SNR-solve mode).
        config: :class:`ETCConfig` instance.
        eps_warm_T_cold: Warm-optics emissivity times cold transmission.
            Defaults to 0.0 (no thermal background from optics).

    Returns:
        Tuple of (Cp, Cb, Cnf, Csp) -- all in [e/s].
    """
    coro = optical_path.coronagraph
    primary = optical_path.primary
    detector = optical_path.detector

    # Telescope
    area_m2 = primary.area_m2
    throughput = optical_path.system_throughput(wavelength_nm)

    # Coronagraph performance (interpax splines, fully JIT-safe)
    core_throughput = coro.throughput(separation_lod, wavelength_nm)
    core_area_lod2 = coro.core_area(separation_lod, wavelength_nm)
    sky_trans = coro.occulter_transmission(separation_lod, wavelength_nm)
    core_mean_intensity_val = coro.core_mean_intensity(separation_lod, wavelength_nm)
    noisefloor_value = core_mean_intensity_val / config.ppfact

    # Angular scales
    lam_m = wavelength_nm * nm2m
    lod_rad = lam_m / primary.diameter_m
    lod_arcsec = lod_rad * rad2arcsec

    # Detector-derived quantities
    pixscale_lod = coro.pixel_scale_lod
    n_pix = core_area_lod2 / (pixscale_lod**2) * config.npix_multiplier

    # Planet signal
    Cp = count_rate_planet(
        scene.F0,
        scene.Fs_over_F0,
        scene.Fp_over_Fs,
        area_m2,
        throughput,
        core_throughput,
        dlambda_nm,
        scene.n_channels,
    )

    # Stellar leakage
    CRbs = count_rate_stellar_leakage(
        scene.F0,
        scene.Fs_over_F0,
        area_m2,
        throughput,
        dlambda_nm,
        scene.n_channels,
        core_area_lod2,
        core_mean_intensity_val,
    )

    # Zodiacal light
    CRbz = count_rate_zodi(
        scene.F0,
        scene.Fzodi,
        lod_arcsec,
        sky_trans,
        area_m2,
        throughput,
        dlambda_nm,
        scene.n_channels,
        core_area_lod2,
    )

    # Exozodiacal light
    CRbez = count_rate_exozodi(
        scene.F0,
        scene.Fexozodi,
        lod_arcsec,
        sky_trans,
        area_m2,
        throughput,
        dlambda_nm,
        scene.n_channels,
        core_area_lod2,
        scene.dist_pc,
        scene.sep_arcsec,
    )

    # Binary
    CRbbin = count_rate_binary(
        scene.F0,
        scene.Fbinary,
        sky_trans,
        area_m2,
        throughput,
        dlambda_nm,
        scene.n_channels,
        core_area_lod2,
    )

    # Thermal
    CRbth = count_rate_thermal(
        wavelength_nm,
        area_m2,
        dlambda_nm,
        scene.temp_K,
        lod_rad,
        eps_warm_T_cold,
        detector.quantum_efficiency,
        detector.dqe,
        core_area_lod2,
    )

    # Detector noise
    total_photon_cr = Cp + CRbs + CRbz + CRbez
    t_photon = photon_counting_time(jnp.maximum(total_photon_cr, 1e-30), n_pix)
    CRbd = count_rate_detector(
        n_pix,
        detector.dark_current_rate,
        detector.read_noise_electrons,
        detector.read_time,
        detector.cic_rate,
        t_photon,
    )

    # Total background
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    # Noise floor rates (AYO/jaxedith path)
    CRnf_star_rate = noise_floor_stellar(
        scene.F0,
        scene.Fs_over_F0,
        area_m2,
        throughput,
        dlambda_nm,
        scene.n_channels,
        noisefloor_value,
        core_area_lod2,
    )
    CRnf_ez_rate = noise_floor_exozodi(CRbez, scene.ez_ppf)
    Cnf_rate = noise_floor_total(
        CRnf_star_rate, CRnf_ez_rate, config.include_exozodi_noise_floor
    )
    # Preserve the old solver API for Task 1: AYO exptime expects Cnf
    # with snr baked in. Task 2 changes this when equations.py lands.
    Cnf = snr * Cnf_rate

    # Speckle residual (EXOSIMS path)
    Csp = speckle_residual(CRbs, config.ppfact, config.stability_fact)

    return Cp, Cb, Cnf, Csp


# -- Public functions ----------------------------------------------------------


def calc_exptime(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    snr,
    config=None,
    eps_warm_T_cold=0.0,
):
    """Calculate exposure time for a single wavelength and separation.

    This is the main entry point for the JAX ETC. All inputs are
    ``eqx.Module`` pytrees, so the full computation is JIT-traceable
    via ``eqx.filter_jit``.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module containing
            primary, coronagraph, attenuating elements, and detector.
        scene: :class:`ETCScene` instance with astrophysical parameters.
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lam/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        snr: Target signal-to-noise ratio.
        config: :class:`ETCConfig` instance. Defaults to ``CONFIG``.
        eps_warm_T_cold: Warm-optics emissivity times cold transmission.
            Defaults to 0.0 (no thermal background from optics).

    Returns:
        Exposure time in seconds.
    """
    if config is None:
        config = CONFIG

    Cp, Cb, Cnf, Csp = _compute_count_rates(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        snr,
        config,
        eps_warm_T_cold,
    )
    return _dispatch_exptime(Cp, Cb, Cnf, Csp, snr, config)


def calc_snr(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    t_obs,
    config=None,
    eps_warm_T_cold=0.0,
):
    """Calculate achieved SNR for a given observation time.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        scene: :class:`ETCScene` instance with astrophysical parameters.
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lam/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        t_obs: Total observation time [s].
        config: :class:`ETCConfig` instance. Defaults to ``CONFIG``.
        eps_warm_T_cold: Warm-optics emissivity times cold transmission.
            Defaults to 0.0.

    Returns:
        Achieved signal-to-noise ratio.
    """
    if config is None:
        config = CONFIG

    # For SNR-solve mode, compute noise floor rates with SNR=1
    Cp, Cb, Cnf, Csp = _compute_count_rates(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        1.0,
        config,
        eps_warm_T_cold,
    )
    return _dispatch_snr(Cp, Cb, Cnf, Csp, t_obs, config)


def calc_count_rates(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    snr=7.0,
    config=None,
    eps_warm_T_cold=0.0,
):
    """Compute all count rates without solving for exposure time.

    Useful for debugging and comparing individual count rates against
    jaxedith/AYO/EXOSIMS values.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        scene: :class:`ETCScene` instance.
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lam/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        snr: SNR used for noise floor calculations.
        config: :class:`ETCConfig` instance. Defaults to ``CONFIG``.
        eps_warm_T_cold: Warm-optics emissivity times cold transmission.
            Defaults to 0.0.

    Returns:
        Tuple of (Cp, Cb, Cnf, Csp).
    """
    if config is None:
        config = CONFIG

    return _compute_count_rates(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        snr,
        config,
        eps_warm_T_cold,
    )


def calc_exptime_spectrum(
    optical_path,
    scene,
    wavelength_array_nm,
    separation_lod,
    dlambda_array_nm,
    snr_array,
    config=None,
    eps_warm_T_cold=0.0,
):
    """Vectorized exposure time calculation over wavelength bins.

    Uses ``jax.vmap`` to compute exposure times independently for each
    wavelength bin. All inputs are eqx.Module pytrees, so this function
    is JIT-traceable via ``eqx.filter_jit``.

    Args:
        optical_path: ``optixstuff.OpticalPath`` eqx.Module.
        scene: :class:`ETCScene` instance. ``F0`` and ``Fs_over_F0`` should
            be 1D arrays of the same length as ``wavelength_array_nm``.
        wavelength_array_nm: Array of wavelengths [nm].
        separation_lod: Scalar separation in lam/D (same for all bins).
        dlambda_array_nm: Array of bandwidths [nm], one per wavelength bin.
        snr_array: Array of target SNRs, one per wavelength bin.
        config: :class:`ETCConfig` instance. Defaults to ``CONFIG``.
        eps_warm_T_cold: Warm-optics emissivity times cold transmission.
            Defaults to 0.0.

    Returns:
        Array of exposure times in seconds.

    Raises:
        ValueError: If array inputs have mismatched shapes or are not 1D.
    """
    if config is None:
        config = CONFIG

    # Convert to arrays and validate shapes (runs before tracing)
    wavelength_array_nm = jnp.asarray(wavelength_array_nm)
    dlambda_array_nm = jnp.asarray(dlambda_array_nm)
    snr_array = jnp.asarray(snr_array)

    if wavelength_array_nm.ndim != 1:
        raise ValueError(
            f"wavelength_array_nm must be 1D, got shape {wavelength_array_nm.shape}"
        )

    n_bins = wavelength_array_nm.shape[0]

    if dlambda_array_nm.shape != (n_bins,):
        raise ValueError(
            f"dlambda_array_nm shape {dlambda_array_nm.shape} does not match "
            f"wavelength_array_nm length {n_bins}"
        )

    if snr_array.shape != (n_bins,):
        raise ValueError(
            f"snr_array shape {snr_array.shape} does not match "
            f"wavelength_array_nm length {n_bins}"
        )

    # vmap over wavelength axis -- scene parameters that vary must be 1D arrays
    return jax.vmap(
        lambda wl, dl, s_snr: calc_exptime(
            optical_path,
            scene,
            wl,
            separation_lod,
            dl,
            s_snr,
            config,
            eps_warm_T_cold,
        )
    )(wavelength_array_nm, dlambda_array_nm, snr_array)
