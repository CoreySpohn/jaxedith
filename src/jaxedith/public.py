"""Layer 3: variant-explicit scalar public API for jaxedith.

Each public accepts an ``optixstuff.OpticalPath`` and an ``ETCScene`` plus
observation geometry and scalar kwargs. Post-processing, thermal, and
overhead knobs that used to live on ``ETCConfig`` are plain keyword
arguments with defaults. Variant is baked into the function name --
there is no runtime dispatch.

Private ``_count_rates_{ayo,exosims}`` helpers assemble the rate triple
from Layer 2 components; the public ``*_ayo`` / ``*_exosims_*`` functions
call them and dispatch to the matching ``equations.py`` equation.
"""

import jax.numpy as jnp

from jaxedith.components import (
    binary_background,
    detector_noise,
    exozodi_background,
    planet_signal,
    stellar_leakage,
    stellar_noise_floor,
    thermal_background,
    zodi_background,
)
from jaxedith.count_rates import noise_floor_exozodi, speckle_residual
from jaxedith.equations import (
    exptime_from_rates_ayo,
    exptime_from_rates_exosims_char,
    exptime_from_rates_exosims_det,
    snr_from_rates_ayo,
    snr_from_rates_exosims_char,
    snr_from_rates_exosims_det,
)


def _count_rates_ayo(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    *,
    temp_K=270.0,
    ez_ppf=jnp.inf,
    ppfact=1.0,
    eps_warm_T_cold=0.0,
):
    """Assemble the AYO rate triple ``(Cp, Cb, Cnf_rate)``.

    All Layer 2 components are called with ``n_channels`` and
    ``npix_multiplier`` read from ``optical_path`` (see Plan 2).

    The noise floor is returned in rate form (Plan 3): multiply by
    ``snr`` inside the equation, not here. Star and exozodi floors are
    combined in quadrature, matching ``noise_floor_total`` with
    ``include_ez=True``; setting ``ez_ppf=jnp.inf`` zeroes the exozodi
    contribution and reproduces the ``include_ez=False`` branch.
    """
    n_channels = optical_path.n_channels
    npix_multiplier = optical_path.npix_multiplier

    Cp = planet_signal(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fs_over_F0, scene.Fp_over_Fs,
        n_channels=n_channels,
    )
    CRbs = stellar_leakage(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fs_over_F0,
        n_channels=n_channels,
    )
    CRbz = zodi_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fzodi,
        n_channels=n_channels,
    )
    CRbez = exozodi_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fexozodi, scene.dist_pc, scene.sep_arcsec,
        n_channels=n_channels,
    )
    CRbbin = binary_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fbinary,
        n_channels=n_channels,
    )
    CRbth = thermal_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        temp_K, eps_warm_T_cold=eps_warm_T_cold,
    )
    total_photon_cr = Cp + CRbs + CRbz + CRbez
    CRbd = detector_noise(
        optical_path, wavelength_nm, separation_lod,
        total_photon_cr, npix_multiplier=npix_multiplier,
    )
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    CRnf_star_rate = stellar_noise_floor(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fs_over_F0,
        n_channels=n_channels, ppfact=ppfact,
    )
    CRnf_ez_rate = noise_floor_exozodi(CRbez, ez_ppf)
    Cnf_rate = jnp.sqrt(CRnf_star_rate ** 2 + CRnf_ez_rate ** 2)

    return Cp, Cb, Cnf_rate


def count_rates_ayo(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    *,
    temp_K=270.0,
    ez_ppf=jnp.inf,
    ppfact=1.0,
    eps_warm_T_cold=0.0,
):
    """AYO rate triple ``(Cp, Cb, Cnf_rate)``.

    ``Cnf_rate`` is the noise-floor *rate* (no SNR factor); equations
    multiply by SNR internally.
    """
    return _count_rates_ayo(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ez_ppf=ez_ppf, ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
    )


def exptime_ayo(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    snr,
    *,
    temp_K=270.0,
    ez_ppf=jnp.inf,
    ppfact=1.0,
    bg_multiplier=2.0,
    overhead_multi=1.0,
    overhead_fixed_s=0.0,
    n_rolls=1,
    eps_warm_T_cold=0.0,
):
    """AYO / jaxedith exposure time [s]."""
    Cp, Cb, Cnf_rate = _count_rates_ayo(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ez_ppf=ez_ppf, ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return exptime_from_rates_ayo(
        Cp, Cb, Cnf_rate, snr,
        bg_multiplier=bg_multiplier,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )


def snr_ayo(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    t_obs,
    *,
    temp_K=270.0,
    ez_ppf=jnp.inf,
    ppfact=1.0,
    bg_multiplier=2.0,
    overhead_multi=1.0,
    overhead_fixed_s=0.0,
    n_rolls=1,
    eps_warm_T_cold=0.0,
):
    """AYO / jaxedith achieved SNR for a fixed ``t_obs`` [s]."""
    Cp, Cb, Cnf_rate = _count_rates_ayo(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ez_ppf=ez_ppf, ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return snr_from_rates_ayo(
        Cp, Cb, Cnf_rate, t_obs,
        bg_multiplier=bg_multiplier,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )


def _count_rates_exosims(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    eps_warm_T_cold=0.0,
):
    """Assemble the EXOSIMS rate triple ``(Cp, Cb, Csp)``.

    Shared by both detection and characterization. Differs from the AYO
    helper in the noise term: EXOSIMS uses a speckle residual
    ``Csp = speckle_residual(CRbs, ppfact, stability_fact)`` instead of
    the data-driven noise floor.
    """
    n_channels = optical_path.n_channels
    npix_multiplier = optical_path.npix_multiplier

    Cp = planet_signal(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fs_over_F0, scene.Fp_over_Fs,
        n_channels=n_channels,
    )
    CRbs = stellar_leakage(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fs_over_F0,
        n_channels=n_channels,
    )
    CRbz = zodi_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fzodi,
        n_channels=n_channels,
    )
    CRbez = exozodi_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fexozodi, scene.dist_pc, scene.sep_arcsec,
        n_channels=n_channels,
    )
    CRbbin = binary_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        scene.F0, scene.Fbinary,
        n_channels=n_channels,
    )
    CRbth = thermal_background(
        optical_path, wavelength_nm, separation_lod, dlambda_nm,
        temp_K, eps_warm_T_cold=eps_warm_T_cold,
    )
    total_photon_cr = Cp + CRbs + CRbz + CRbez
    CRbd = detector_noise(
        optical_path, wavelength_nm, separation_lod,
        total_photon_cr, npix_multiplier=npix_multiplier,
    )
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    Csp = speckle_residual(CRbs, ppfact, stability_fact)
    return Cp, Cb, Csp


def count_rates_exosims_det(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    eps_warm_T_cold=0.0,
):
    """EXOSIMS detection rate triple ``(Cp, Cb, Csp)``."""
    return _count_rates_exosims(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ppfact=ppfact, stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )


def exptime_exosims_det(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    snr,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    overhead_multi=1.0,
    overhead_fixed_s=0.0,
    n_rolls=1,
    eps_warm_T_cold=0.0,
):
    """EXOSIMS detection exposure time [s]."""
    Cp, Cb, Csp = _count_rates_exosims(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ppfact=ppfact, stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return exptime_from_rates_exosims_det(
        Cp, Cb, Csp, snr,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )


def snr_exosims_det(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    t_obs,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    overhead_multi=1.0,
    overhead_fixed_s=0.0,
    n_rolls=1,
    eps_warm_T_cold=0.0,
):
    """EXOSIMS detection achieved SNR for a fixed ``t_obs`` [s]."""
    Cp, Cb, Csp = _count_rates_exosims(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ppfact=ppfact, stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return snr_from_rates_exosims_det(
        Cp, Cb, Csp, t_obs,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )


def count_rates_exosims_char(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    eps_warm_T_cold=0.0,
):
    """EXOSIMS characterization rate triple ``(Cp, Cb, Csp)``.

    Same rates as :func:`count_rates_exosims_det`; the equations differ.
    """
    return _count_rates_exosims(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ppfact=ppfact, stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )


def exptime_exosims_char(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    snr,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    overhead_multi=1.0,
    overhead_fixed_s=0.0,
    n_rolls=1,
    eps_warm_T_cold=0.0,
):
    """EXOSIMS characterization exposure time [s]."""
    Cp, Cb, Csp = _count_rates_exosims(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ppfact=ppfact, stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return exptime_from_rates_exosims_char(
        Cp, Cb, Csp, snr,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )


def snr_exosims_char(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    t_obs,
    *,
    temp_K=270.0,
    ppfact=1.0,
    stability_fact=1.0,
    overhead_multi=1.0,
    overhead_fixed_s=0.0,
    n_rolls=1,
    eps_warm_T_cold=0.0,
):
    """EXOSIMS characterization achieved SNR for a fixed ``t_obs`` [s]."""
    Cp, Cb, Csp = _count_rates_exosims(
        optical_path, scene, wavelength_nm, separation_lod, dlambda_nm,
        temp_K=temp_K, ppfact=ppfact, stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return snr_from_rates_exosims_char(
        Cp, Cb, Csp, t_obs,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )
