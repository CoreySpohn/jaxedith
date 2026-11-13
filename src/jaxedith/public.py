"""Layer 3: variant-explicit public API for jaxedith.

Two surfaces, both defined in this file:

* Scalar publics ``{count_rates,exptime,snr}_{ayo,exosims_det,exosims_char}``
  accept an ``optixstuff.OpticalPath`` and an ``ETCScene`` plus observation
  geometry and scalar kwargs. Post-processing, thermal, and overhead knobs
  are plain keyword arguments with defaults. Variant is baked into the
  function name. AYO additionally accepts a static cadence selector;
  the default remains the legacy photon-counting equation.
* System-level ``*_from_system_{ayo,exosims_det,exosims_char}`` wrappers
  accept a ``skyscapes.System`` + ``OpticalPath`` + ``Observatory`` +
  ``ExposureConfig`` + ``PPConfig``, extract per-(planet, epoch) astrophysics,
  and ``jax.vmap`` the matching scalar public over ``(K, T)``.

``ppfact`` means opposite things in the two variants. In the AYO publics
it divides: the stellar noise floor is the stellar leakage over ``ppfact``
(pyEDITH's post-processing factor, about 30). In the EXOSIMS publics it
multiplies: the speckle residual is the stellar leakage times ``ppfact``
(EXOSIMS' ``ppFact``, at most 1). Both default to 1, the raw leakage.

Private ``_count_rates_{ayo,exosims}`` helpers assemble the rate triple
from the Layer 2 ``intermediates``; the scalar publics call them and
dispatch to the matching ``etc.py`` equation. The from_system wrappers
share the ``_extract_per_kt`` / ``_sep_lod_from_arcsec`` /
``_vmap_over_kt`` plumbing.

All count rates are electrons per second. The intermediate adapters apply
detector QE exactly once; optical-path attenuating elements must not also
include detector QE. Scene fluxes and the from_system extraction remain in
photon units until passed to those adapters.
"""

import jax
import jax.numpy as jnp
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy

from jaxedith._checks import check_finite_nonneg
from jaxedith._types import PerKT, PerT, Scalar
from jaxedith.etc import (
    _variance_components,
    detector_variance_counts,
    exptime_from_rates_ayo,
    exptime_from_rates_exosims_char,
    exptime_from_rates_exosims_det,
    exptime_min_duration,
    exptime_whole_frames,
    snr_fixed_frame,
    snr_from_rates_ayo,
    snr_from_rates_exosims_char,
    snr_from_rates_exosims_det,
)
from jaxedith.intermediates import (
    _detector_n_pix,
    _lod_arcsec,
    binary_background,
    detector_noise,
    exozodi_background,
    planet_signal,
    stellar_leakage,
    stellar_noise_floor,
    thermal_background,
    zodi_background,
)
from jaxedith.primitives import noise_floor_exozodi, speckle_residual
from jaxedith.scene import ETCScene
from jaxedith.zodi import zodi_fn_ayo


def _check_cadence(cadence):
    """Validate the static cadence selector before tracing its arithmetic."""
    if cadence not in ("fixed_frame", "photon_counting_ayo"):
        raise ValueError("cadence must be 'fixed_frame' or 'photon_counting_ayo'")


def _fixed_frame_parameters(optical_path, wavelength_nm, separation_lod):
    """Map detector hardware and aperture geometry to finite-frame inputs."""
    detector = optical_path.detector
    return dict(
        n_pix=_detector_n_pix(
            optical_path, wavelength_nm, separation_lod, optical_path.npix_multiplier
        ),
        dark=detector.dark_current_rate_e_per_s,
        read_noise=detector.read_noise_e,
        cic=detector.clock_induced_charge_rate_e_per_frame,
        frame_time=detector.frame_time_s,
    )


def count_variance_components_ayo(
    optical_path,
    scene,
    wavelength_nm,
    separation_lod,
    dlambda_nm,
    t,
    *,
    cadence="photon_counting_ayo",
    bg_multiplier=2.0,
    ppfact=1.0,
    temp_K=270.0,
    ez_ppf=jnp.inf,
    eps_warm_T_cold=0.0,
):
    """Return planet, background, detector, systematic and total count variances.

    ``t`` is science integration time in seconds, with no overheads.
    Fixed-frame background is photon-only: leakage, zodi, exozodi, binary
    and thermal. Its variance is ``bg_multiplier * Cb * t``; detector
    variance is paid separately once per frame, including a partial frame.
    The fixed-frame floor is exactly ``(t * leakage / ppfact)**2``, zero
    for infinite ``ppfact``; ``ez_ppf`` applies only to the legacy cadence.
    The legacy cadence retains its rate-based detector variance, including
    the background multiplier on that term. All components are electrons^2.
    ``cadence`` is static under JIT; numeric options can be batched.
    """
    Cp, Cb, Cnf_rate, CRbd = _count_rates_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ez_ppf=ez_ppf,
        ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
        cadence=cadence,
        _separate_detector=True,
    )
    if cadence == "fixed_frame":
        detector = detector_variance_counts(
            t, **_fixed_frame_parameters(optical_path, wavelength_nm, separation_lod)
        )
    else:
        detector = bg_multiplier * CRbd * t
    return _variance_components(Cp, Cb, Cnf_rate, t, detector, bg_multiplier)


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
    cadence="photon_counting_ayo",
    _separate_detector=False,
):
    """Assemble the AYO rate triple ``(Cp, Cb, Cnf_rate)``.

    All Layer 2 intermediates are called with ``n_channels`` and
    ``npix_multiplier`` read from ``optical_path`` (see Plan 2).

    The noise floor is returned in rate form: multiply by ``snr`` inside
    the equation, not here. Star and exozodi floors are combined in
    quadrature; setting ``ez_ppf=jnp.inf`` zeroes the exozodi
    contribution and reduces the floor to the stellar term alone.
    ``ppfact`` divides here: the stellar floor is the leakage / ``ppfact``.
    """
    _check_cadence(cadence)
    n_channels = optical_path.n_channels
    npix_multiplier = optical_path.npix_multiplier
    stellar_diam_lod = scene.stellar_diam_mas / (
        1000.0 * _lod_arcsec(optical_path, wavelength_nm)
    )

    Cp = planet_signal(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fs_over_F0,
        scene.Fp_over_Fs,
        n_channels=n_channels,
    )
    CRbs = stellar_leakage(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fs_over_F0,
        n_channels=n_channels,
        stellar_diam_lod=stellar_diam_lod,
    )
    CRbz = zodi_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fzodi,
        n_channels=n_channels,
    )
    CRbez = exozodi_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fexozodi,
        scene.dist_pc,
        scene.sep_arcsec,
        n_channels=n_channels,
    )
    CRbbin = binary_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fbinary,
        n_channels=n_channels,
    )
    CRbth = thermal_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    CRbd = 0.0
    if cadence == "photon_counting_ayo":
        total_photon_cr = Cp + CRbs + CRbz + CRbez
        CRbd = detector_noise(
            optical_path,
            wavelength_nm,
            separation_lod,
            total_photon_cr,
            npix_multiplier=npix_multiplier,
        )
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    if cadence == "fixed_frame":
        Cnf_rate = jnp.where(jnp.isinf(ppfact), 0.0, CRbs / ppfact)
    else:
        CRnf_star_rate = stellar_noise_floor(
            optical_path,
            wavelength_nm,
            separation_lod,
            dlambda_nm,
            scene.F0,
            scene.Fs_over_F0,
            n_channels=n_channels,
            ppfact=ppfact,
            stellar_diam_lod=stellar_diam_lod,
        )
        CRnf_ez_rate = noise_floor_exozodi(CRbez, ez_ppf)
        Cnf_rate = jnp.sqrt(CRnf_star_rate**2 + CRnf_ez_rate**2)

    if _separate_detector:
        return (
            check_finite_nonneg(Cp, "Cp"),
            check_finite_nonneg(CRbs + CRbz + CRbez + CRbbin + CRbth, "Cb_photon"),
            check_finite_nonneg(Cnf_rate, "Cnf_rate"),
            check_finite_nonneg(CRbd, "CRbd"),
        )

    return (
        check_finite_nonneg(Cp, "Cp"),
        check_finite_nonneg(Cb, "Cb"),
        check_finite_nonneg(Cnf_rate, "Cnf_rate"),
    )


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
    cadence="photon_counting_ayo",
):
    """AYO rate triple ``(Cp, Cb, Cnf_rate)``.

    ``Cnf_rate`` is the noise-floor *rate* (no SNR factor); equations
    multiply by SNR internally. With ``cadence="fixed_frame"``, ``Cb``
    contains photon backgrounds only and ``Cnf_rate = leakage / ppfact``;
    use :func:`count_variance_components_ayo` for detector count variance.
    """
    return _count_rates_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ez_ppf=ez_ppf,
        ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
        cadence=cadence,
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
    cadence="photon_counting_ayo",
    planning="whole_frames",
    n_max=4096,
):
    """AYO / jaxedith exposure time [s], including configured overheads.

    The default cadence preserves the rate-based photon-counting equation.
    ``cadence="fixed_frame"`` uses ``planning="whole_frames"`` by default;
    ``planning="min_duration"`` permits a partial final frame. Both search
    up to static ``n_max`` frames per roll and return infinity if infeasible.
    Cadence and planning must be static under JIT.
    """
    Cp, Cb, Cnf_rate = _count_rates_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ez_ppf=ez_ppf,
        ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
        cadence=cadence,
    )
    if cadence == "fixed_frame":
        if planning not in ("whole_frames", "min_duration"):
            raise ValueError("planning must be 'whole_frames' or 'min_duration'")
        solver = (
            exptime_whole_frames if planning == "whole_frames" else exptime_min_duration
        )
        t_science = solver(
            snr,
            Cp,
            Cb,
            Cnf_rate,
            **_fixed_frame_parameters(optical_path, wavelength_nm, separation_lod),
            bg_multiplier=bg_multiplier,
            n_max=n_max,
        )
        return (t_science * overhead_multi + overhead_fixed_s) * n_rolls
    return exptime_from_rates_ayo(
        Cp,
        Cb,
        Cnf_rate,
        snr,
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
    cadence="photon_counting_ayo",
):
    """AYO achieved SNR after removing overheads from ``t_obs`` [s].

    ``cadence="fixed_frame"`` uses finite-frame detector variance separately
    from photon backgrounds and the stellar floor. The default cadence
    retains the legacy rate-based equation. Cadence must be static under JIT.
    """
    Cp, Cb, Cnf_rate = _count_rates_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ez_ppf=ez_ppf,
        ppfact=ppfact,
        eps_warm_T_cold=eps_warm_T_cold,
        cadence=cadence,
    )
    if cadence == "fixed_frame":
        t_science = (
            jnp.maximum(t_obs / n_rolls - overhead_fixed_s, 0.0) / overhead_multi
        )
        return snr_fixed_frame(
            Cp,
            Cb,
            Cnf_rate,
            t_science,
            **_fixed_frame_parameters(optical_path, wavelength_nm, separation_lod),
            bg_multiplier=bg_multiplier,
        )
    return snr_from_rates_ayo(
        Cp,
        Cb,
        Cnf_rate,
        t_obs,
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
    the data-driven noise floor. ``ppfact`` multiplies here, the opposite
    of the AYO path where it divides.
    """
    n_channels = optical_path.n_channels
    npix_multiplier = optical_path.npix_multiplier
    stellar_diam_lod = scene.stellar_diam_mas / (
        1000.0 * _lod_arcsec(optical_path, wavelength_nm)
    )

    Cp = planet_signal(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fs_over_F0,
        scene.Fp_over_Fs,
        n_channels=n_channels,
    )
    CRbs = stellar_leakage(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fs_over_F0,
        n_channels=n_channels,
        stellar_diam_lod=stellar_diam_lod,
    )
    CRbz = zodi_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fzodi,
        n_channels=n_channels,
    )
    CRbez = exozodi_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fexozodi,
        scene.dist_pc,
        scene.sep_arcsec,
        n_channels=n_channels,
    )
    CRbbin = binary_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        scene.F0,
        scene.Fbinary,
        n_channels=n_channels,
    )
    CRbth = thermal_background(
        optical_path,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    total_photon_cr = Cp + CRbs + CRbz + CRbez
    CRbd = detector_noise(
        optical_path,
        wavelength_nm,
        separation_lod,
        total_photon_cr,
        npix_multiplier=npix_multiplier,
    )
    Cb = CRbs + CRbz + CRbez + CRbbin + CRbth + CRbd

    Csp = speckle_residual(CRbs, ppfact, stability_fact)
    return (
        check_finite_nonneg(Cp, "Cp"),
        check_finite_nonneg(Cb, "Cb"),
        check_finite_nonneg(Csp, "Csp"),
    )


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
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ppfact=ppfact,
        stability_fact=stability_fact,
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
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ppfact=ppfact,
        stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return exptime_from_rates_exosims_det(
        Cp,
        Cb,
        Csp,
        snr,
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
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ppfact=ppfact,
        stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return snr_from_rates_exosims_det(
        Cp,
        Cb,
        Csp,
        t_obs,
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
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ppfact=ppfact,
        stability_fact=stability_fact,
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
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ppfact=ppfact,
        stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return exptime_from_rates_exosims_char(
        Cp,
        Cb,
        Csp,
        snr,
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
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        temp_K=temp_K,
        ppfact=ppfact,
        stability_fact=stability_fact,
        eps_warm_T_cold=eps_warm_T_cold,
    )
    return snr_from_rates_exosims_char(
        Cp,
        Cb,
        Csp,
        t_obs,
        overhead_multi=overhead_multi,
        overhead_fixed=overhead_fixed_s,
        n_rolls=n_rolls,
    )


# ---------------------------------------------------------------------------
# AYO from-system wrappers (Plan 5, Task 2)
# ---------------------------------------------------------------------------

ARCSEC_PER_RAD = 206264.80624709636


def _sep_lod_from_arcsec(sep_arcsec, wavelength_nm, diameter_m):
    """Convert angular separation to lambda/D units."""
    lod_rad = wavelength_nm * 1e-9 / diameter_m
    return sep_arcsec / (lod_rad * ARCSEC_PER_RAD)


def _stellar_diam_mas_from_system(system):
    """Read stellar angular diameter in mas for every from-system wrapper.

    Convert ``system.star.diameter_arcsec`` to milliarcseconds; star models
    without that attribute use 0 (a point star).
    """
    return 1000.0 * getattr(system.star, "diameter_arcsec", 0.0)


def _extract_per_kt(
    system, exposure
) -> tuple[Scalar, Scalar, PerT, PerKT, PerKT, Scalar, PerT]:
    """Pull the per-(K, T) astrophysics needed by every from_system call.

    ``F0`` is the flux zero point (a magnitude-0 source), NOT the star flux:
    the absolute sky terms (zodi, exozodi) reference it, and the star flux
    enters via ``Fs_over_F0``. This is the single source of the F0 convention
    for every from_system wrapper; folding the star flux into F0 (with
    Fs_over_F0=1) silently scaled zodi/exozodi by 10^(-0.4 m_star).

    Returns:
        wl: scalar central wavelength [nm].
        dlambda: scalar bandwidth [nm].
        time_jd: shape ``(T,)`` epoch array.
        contrasts: shape ``(K, T)`` planet-to-star flux ratio.
        alpha: shape ``(K, T)`` projected separation [arcsec].
        F0_zp: scalar flux zero point [ph/s/m^2/nm] at ``wl``.
        Fs_over_F0_t: shape ``(T,)`` star-to-zeropoint flux ratio per epoch.
    """
    wl = exposure.central_wavelength_nm
    dlambda = exposure.bin_width_nm
    time_jd = jnp.atleast_1d(exposure.start_time_jd)

    contrasts = system.contrasts(jnp.atleast_1d(wl), time_jd)
    alpha, _dMag = system.alpha_dMag(time_jd)
    F0_t = jax.vmap(lambda t: system.star.spec_flux_density(wl, t))(time_jd)
    F0_zp = jy_to_photons_per_nm_per_m2(mag_to_flux_jy(0.0), wl)
    return wl, dlambda, time_jd, contrasts, alpha, F0_zp, F0_t / F0_zp


def _vmap_over_kt(per_element):
    """Wrap a scalar per-(planet, epoch) callable into a ``(K, T)`` function.

    Input shapes expected on the returned function:
        contrasts:    ``(K, T)``
        alpha:        ``(K, T)``
        sep_lod:      ``(K, T)``
        Fs_over_F0_t: ``(T,)``

    Output: whatever ``per_element`` returns, batched as ``(K, T)`` (or
    ``tuple[(K, T), ...]`` if it returns a tuple).
    """
    return jax.vmap(
        jax.vmap(per_element, in_axes=(0, 0, 0, 0)),
        in_axes=(0, 0, 0, None),
    )


def count_variance_components(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    t,
    *,
    cadence="photon_counting_ayo",
    bg_multiplier=2.0,
    ppfact=None,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> dict[str, PerKT]:
    """Return the AYO variance budget for each planet and epoch.

    Inputs match :func:`snr_from_system_ayo`; ``t`` is total observation
    time in seconds. Overheads and rolls are removed before computing each
    per-roll science budget, as for that SNR wrapper. The result has keys
    ``planet``, ``background``, ``detector``, ``systematic`` and ``total``,
    each with shape ``(K, T)`` and units electrons^2.
    ``ppfact=None`` uses ``ppconfig.ppfact``; infinity disables the stellar
    floor. ``bg_multiplier`` (k) scales photon backgrounds on the fixed-frame
    path. See :func:`count_variance_components_ayo` for cadence conventions.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    t_science = (
        jnp.maximum(t / ppconfig.n_rolls - observatory.overhead_fixed_s, 0.0)
        / observatory.overhead_multi
    )

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=system.star.dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return count_variance_components_ayo(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            t_science,
            cadence=cadence,
            bg_multiplier=bg_multiplier,
            ppfact=ppconfig.ppfact if ppfact is None else ppfact,
            temp_K=observatory.temperature_K,
            ez_ppf=ppconfig.ez_ppf,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def count_rates_from_system_ayo(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
    cadence="photon_counting_ayo",
    ppfact=None,
) -> tuple[PerKT, PerKT, PerKT]:
    """AYO rate triple ``(Cp, Cb, Cnf_rate)`` with shape ``(K, T)`` each.

    ``cadence="fixed_frame"`` returns photon-only ``Cb``; detector noise
    requires :func:`count_variance_components`. ``ppfact=None`` reads the
    post-processing config; an explicit value overrides it.

    TODO(skyscapes): replace the ``Fexozodi=0.0`` default with
    ``system.disk.fexozodi_at(...)`` once
    ``skyscapes.AbstractDisk.fexozodi_at`` lands.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return count_rates_ayo(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            temp_K=observatory.temperature_K,
            ez_ppf=ppconfig.ez_ppf,
            ppfact=ppconfig.ppfact if ppfact is None else ppfact,
            eps_warm_T_cold=eps_warm_T_cold,
            cadence=cadence,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def exptime_from_system_ayo(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    snr,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    bg_multiplier=2.0,
    eps_warm_T_cold=0.0,
    cadence="photon_counting_ayo",
    ppfact=None,
    planning="whole_frames",
    n_max=4096,
) -> PerKT:
    """AYO exposure time with shape ``(K, T)``.

    ``cadence="fixed_frame"`` defaults to whole-frame planning; use
    ``planning="min_duration"`` to allow a partial final frame. ``n_max``
    bounds the search per roll. ``ppfact=None`` reads ``ppconfig.ppfact``;
    infinity disables the stellar floor. Cadence, planning and n_max must
    be static under JIT; numeric options can be vmapped.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``
    once ``AbstractDisk.fexozodi_at`` lands.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return exptime_ayo(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            snr,
            temp_K=observatory.temperature_K,
            ez_ppf=ppconfig.ez_ppf,
            ppfact=ppconfig.ppfact if ppfact is None else ppfact,
            bg_multiplier=bg_multiplier,
            overhead_multi=observatory.overhead_multi,
            overhead_fixed_s=observatory.overhead_fixed_s,
            n_rolls=ppconfig.n_rolls,
            eps_warm_T_cold=eps_warm_T_cold,
            cadence=cadence,
            planning=planning,
            n_max=n_max,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def snr_from_system_ayo(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    t_obs,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    bg_multiplier=2.0,
    eps_warm_T_cold=0.0,
    cadence="photon_counting_ayo",
    ppfact=None,
) -> PerKT:
    """AYO achieved SNR with shape ``(K, T)`` for a fixed ``t_obs`` [s].

    ``cadence="fixed_frame"`` separates photon, detector and systematic
    variance. ``bg_multiplier`` scales only photon backgrounds on this path;
    ``ppfact=None`` reads ``ppconfig.ppfact`` and infinity disables the floor.
    The default cadence retains the legacy equation and config defaults.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``
    once ``AbstractDisk.fexozodi_at`` lands.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return snr_ayo(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            t_obs,
            temp_K=observatory.temperature_K,
            ez_ppf=ppconfig.ez_ppf,
            ppfact=ppconfig.ppfact if ppfact is None else ppfact,
            bg_multiplier=bg_multiplier,
            overhead_multi=observatory.overhead_multi,
            overhead_fixed_s=observatory.overhead_fixed_s,
            n_rolls=ppconfig.n_rolls,
            eps_warm_T_cold=eps_warm_T_cold,
            cadence=cadence,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def count_rates_from_system_exosims_det(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> tuple[PerKT, PerKT, PerKT]:
    """EXOSIMS detection rate triple ``(Cp, Cb, Csp)`` with shape ``(K, T)``.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``
    once ``AbstractDisk.fexozodi_at`` lands.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return count_rates_exosims_det(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            temp_K=observatory.temperature_K,
            ppfact=ppconfig.ppfact,
            stability_fact=observatory.stability_fact,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def exptime_from_system_exosims_det(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    snr,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> PerKT:
    """EXOSIMS detection exposure time with shape ``(K, T)``.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return exptime_exosims_det(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            snr,
            temp_K=observatory.temperature_K,
            ppfact=ppconfig.ppfact,
            stability_fact=observatory.stability_fact,
            overhead_multi=observatory.overhead_multi,
            overhead_fixed_s=observatory.overhead_fixed_s,
            n_rolls=ppconfig.n_rolls,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def snr_from_system_exosims_det(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    t_obs,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> PerKT:
    """EXOSIMS detection achieved SNR with shape ``(K, T)``.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return snr_exosims_det(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            t_obs,
            temp_K=observatory.temperature_K,
            ppfact=ppconfig.ppfact,
            stability_fact=observatory.stability_fact,
            overhead_multi=observatory.overhead_multi,
            overhead_fixed_s=observatory.overhead_fixed_s,
            n_rolls=ppconfig.n_rolls,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def count_rates_from_system_exosims_char(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> tuple[PerKT, PerKT, PerKT]:
    """EXOSIMS characterization rate triple ``(Cp, Cb, Csp)`` with shape ``(K, T)``.

    Same rates as the detection variant; the equations differ at the
    exptime/snr layer.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return count_rates_exosims_char(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            temp_K=observatory.temperature_K,
            ppfact=ppconfig.ppfact,
            stability_fact=observatory.stability_fact,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def exptime_from_system_exosims_char(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    snr,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> PerKT:
    """EXOSIMS characterization exposure time with shape ``(K, T)``.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return exptime_exosims_char(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            snr,
            temp_K=observatory.temperature_K,
            ppfact=ppconfig.ppfact,
            stability_fact=observatory.stability_fact,
            overhead_multi=observatory.overhead_multi,
            overhead_fixed_s=observatory.overhead_fixed_s,
            n_rolls=ppconfig.n_rolls,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)


def snr_from_system_exosims_char(
    system,
    optical_path,
    observatory,
    exposure,
    ppconfig,
    t_obs,
    *,
    zodi_fn=zodi_fn_ayo,
    Fexozodi=0.0,
    eps_warm_T_cold=0.0,
) -> PerKT:
    """EXOSIMS characterization achieved SNR with shape ``(K, T)``.

    TODO(skyscapes): Fexozodi default auto-sourced from ``system.disk``.
    """
    wl, dlambda, _t_jd, contrasts, alpha, F0_zp, Fs_over_F0_t = _extract_per_kt(
        system, exposure
    )
    Fzodi = zodi_fn(observatory, exposure, system.star)
    sep_lod = _sep_lod_from_arcsec(alpha, wl, optical_path.primary.diameter_m)
    dist_pc = system.star.dist_pc

    def per_element(Fp_over_Fs, sep_arcsec, sep_lod_kt, Fs_over_F0):
        scene = ETCScene(
            F0=F0_zp,
            Fs_over_F0=Fs_over_F0,
            Fp_over_Fs=Fp_over_Fs,
            Fzodi=Fzodi,
            Fexozodi=Fexozodi,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec,
            Fbinary=0.0,
            stellar_diam_mas=_stellar_diam_mas_from_system(system),
        )
        return snr_exosims_char(
            optical_path,
            scene,
            wl,
            sep_lod_kt,
            dlambda,
            t_obs,
            temp_K=observatory.temperature_K,
            ppfact=ppconfig.ppfact,
            stability_fact=observatory.stability_fact,
            overhead_multi=observatory.overhead_multi,
            overhead_fixed_s=observatory.overhead_fixed_s,
            n_rolls=ppconfig.n_rolls,
            eps_warm_T_cold=eps_warm_T_cold,
        )

    return _vmap_over_kt(per_element)(contrasts, alpha, sep_lod, Fs_over_F0_t)
