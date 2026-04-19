"""Pure JAX count rate functions for the exposure time calculator.

All functions are pure JAX -- no astropy units, no side effects. Units are
enforced at the API boundary (the public-facing functions in ``__init__.py``).

All inputs are raw floats in consistent units:

- Fluxes: photons / s / m^2 / nm
- Areas: m^2
- Wavelengths: nm
- Angles: lam/D (dimensionless) unless otherwise noted
- Count rates: electrons / s (output)
"""

import jax.numpy as jnp
from hwoutils.constants import c, h, k_B, nm2m

# -- Planet signal -------------------------------------------------------------


def count_rate_planet(
    F0,
    Fs_over_F0,
    Fp_over_Fs,
    area_m2,
    throughput,
    core_throughput,
    dlambda_nm,
    n_channels,
):
    """Planet signal count rate Cp [e/s].

    This is a direct product of the flux chain. Matches
    ``pyEDITH.exposure_time_calculator.calculate_CRp``.

    Args:
        F0: flux zero point [ph/s/m^2/nm].
        Fs_over_F0: stellar flux ratio [dimensionless].
        Fp_over_Fs: planet-star contrast [dimensionless].
        area_m2: telescope collecting area [m^2].
        throughput: total throughput (optics x QE x contamination).
        core_throughput: coronagraph photometric aperture throughput.
        dlambda_nm: bandwidth [nm].
        n_channels: number of spectral channels.
    """
    return (
        F0
        * Fs_over_F0
        * Fp_over_Fs
        * area_m2
        * throughput
        * core_throughput
        * dlambda_nm
        * n_channels
    )


# -- Stellar leakage ----------------------------------------------------------


def count_rate_stellar_leakage(
    F0,
    Fs_over_F0,
    area_m2,
    throughput,
    dlambda_nm,
    n_channels,
    core_area_lod2,
    core_mean_intensity,
):
    r"""Stellar leakage count rate CRbs [e/s].

    Uses ``core_mean_intensity`` from ``EqxCoronagraph.core_mean_intensity(sep)``,
    which is the azimuthal average of the 2D stellar intensity map in
    :math:`(\lambda/D)^{-2}` units. Multiplying by ``core_area_lod2`` in
    :math:`(\lambda/D)^2` gives the dimensionless aperture-integrated
    stellar leakage.

    .. note::

        The original pyEDITH uses a per-pixel 2D lookup (``Istar / pixscale^2``).
        In the JAX backend we use the 1D curve because yippy computes
        ``core_mean_intensity`` by azimuthally averaging the 2D ``Istar`` map
        (via ``radial_profile``), making the two paths numerically identical.

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRbs``.

    Args:
        F0: flux zero point [ph/s/m^2/nm].
        Fs_over_F0: stellar flux ratio [dimensionless].
        area_m2: telescope collecting area [m^2].
        throughput: total throughput (optics x QE x contamination).
        dlambda_nm: bandwidth [nm].
        n_channels: number of spectral channels.
        core_area_lod2: solid angle of photometric aperture [(lam/D)^2].
        core_mean_intensity: mean intensity in (lam/D)^-2.
    """
    intensity_factor = core_mean_intensity * core_area_lod2
    return (
        F0
        * Fs_over_F0
        * intensity_factor
        * area_m2
        * throughput
        * dlambda_nm
        * n_channels
    )


# -- Zodiacal light -----------------------------------------------------------


def count_rate_zodi(
    F0,
    Fzodi,
    lod_arcsec,
    sky_trans,
    area_m2,
    throughput,
    dlambda_nm,
    n_channels,
    core_area_lod2,
):
    """Local zodiacal light count rate Cbz [e/s].

    ``Fzodi`` is the zodiacal flux ratio in arcsec^-2; multiplying by
    ``(lam/D)^2_arcsec`` and ``core_area_lod2`` gives the total flux in the
    photometric aperture.

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRbz``.

    Args:
        F0: flux zero point [ph/s/m^2/nm].
        Fzodi: zodiacal surface brightness ratio [arcsec^-2].
        lod_arcsec: lam/D in arcseconds.
        sky_trans: coronagraph sky (occulter) transmission [dimensionless].
        area_m2: telescope collecting area [m^2].
        throughput: total throughput.
        dlambda_nm: bandwidth [nm].
        n_channels: number of spectral channels.
        core_area_lod2: solid angle of photometric aperture [(lam/D)^2].
    """
    return (
        F0
        * Fzodi
        * (lod_arcsec**2)
        * sky_trans
        * core_area_lod2
        * area_m2
        * throughput
        * dlambda_nm
        * n_channels
    )


# -- Exozodiacal light --------------------------------------------------------


def count_rate_exozodi(
    F0,
    Fexozodi,
    lod_arcsec,
    sky_trans,
    area_m2,
    throughput,
    dlambda_nm,
    n_channels,
    core_area_lod2,
    dist_pc,
    sep_arcsec,
):
    """Exozodiacal light count rate Cbez [e/s].

    The exozodi surface brightness is defined at 1 AU and scales as
    ``1 / (dist_pc x sep_arcsec)^2`` to the planet separation in AU.
    (Since 1 AU at 1 pc = 1 arcsec, ``dist_pc x sep_arcsec`` gives the
    physical separation in AU.)

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRbez``.

    Args:
        F0: flux zero point [ph/s/m^2/nm].
        Fexozodi: exozodi surface brightness ratio at 1 AU [arcsec^-2].
        lod_arcsec: lam/D in arcseconds.
        sky_trans: coronagraph sky (occulter) transmission [dimensionless].
        area_m2: telescope collecting area [m^2].
        throughput: total throughput.
        dlambda_nm: bandwidth [nm].
        n_channels: number of spectral channels.
        core_area_lod2: solid angle of photometric aperture [(lam/D)^2].
        dist_pc: distance to star [pc].
        sep_arcsec: angular separation [arcsec].
    """
    return (
        F0
        * Fexozodi
        * (lod_arcsec**2)
        * sky_trans
        * core_area_lod2
        * area_m2
        * throughput
        * dlambda_nm
        * n_channels
        / (dist_pc * sep_arcsec) ** 2
    )


# -- Detector noise -----------------------------------------------------------


def count_rate_detector(
    n_pix,
    dark_current,
    read_noise,
    t_read,
    cic,
    t_photon,
):
    """Detector noise count rate Cbd [e/s].

    Uses the variance trick from pyEDITH: ``RN_variance = read_noise^2``
    but keeps the same units as ``read_noise`` alone (i.e. RN x RN.value).

    ``t_photon`` is the inverse of the photon arrival rate per pixel, used
    for the CIC timing term.

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRbd``.

    Args:
        n_pix: number of detector pixels in photometric aperture.
        dark_current: dark current rate [e/pix/s].
        read_noise: read noise [e/pix/read].
        t_read: time per read [s].
        cic: clock-induced charge [e/pix/frame].
        t_photon: photon counting time [s/frame].
    """
    rn_variance = read_noise * read_noise
    return n_pix * (dark_current + rn_variance / t_read + cic / t_photon)


# -- Thermal background -------------------------------------------------------


def count_rate_thermal(
    wavelength_nm,
    area_m2,
    dlambda_nm,
    temp_K,
    lod_rad,
    eps_warm_T_cold,
    QE,
    dQE,
    core_area_lod2,
):
    """Thermal background count rate Cbth [e/s].

    Evaluates the Planck function at the given wavelength and temperature,
    converts to photon spectral radiance, and integrates over the aperture.

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRbth``.

    Args:
        wavelength_nm: observation wavelength [nm].
        area_m2: telescope collecting area [m^2].
        dlambda_nm: bandwidth [nm].
        temp_K: telescope mirror temperature [K].
        lod_rad: lam/D in radians.
        eps_warm_T_cold: effective emissivity x cold transmission [dimensionless].
        QE: quantum efficiency [e/photon].
        dQE: QE degradation factor [dimensionless].
        core_area_lod2: solid angle of photometric aperture [(lam/D)^2].
    """
    lam_m = wavelength_nm * nm2m  # nm -> m
    photon_energy = h * c / lam_m  # J per photon

    # Planck function in energy spectral radiance [W/m^2/sr/m]
    exponent = h * c / (lam_m * k_B * temp_K)
    planck_energy = (2 * h * c**2 / lam_m**5) / (jnp.exp(exponent) - 1)

    # Convert to photon spectral radiance [ph/s/m^2/sr/m]
    planck_photons = planck_energy / photon_energy

    # Area is already in m^2; convert nm -> m for bandwidth
    # The solid angle in steradians = lod_rad^2 x core_area_lod2
    return (
        planck_photons
        * eps_warm_T_cold
        * (QE + dQE)
        * area_m2
        * dlambda_nm
        * nm2m
        * (lod_rad**2)
        * core_area_lod2
    )


# -- Binary / neighbor stray light --------------------------------------------


def count_rate_binary(
    F0,
    Fbinary,
    sky_trans,
    area_m2,
    throughput,
    dlambda_nm,
    n_channels,
    core_area_lod2,
):
    """Count rate from neighboring / binary stars Cbbin [e/s].

    Currently ``Fbinary`` defaults to 0 in pyEDITH, so this term is
    typically zero. Included for completeness.

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRbbin``.

    Args:
        F0: flux zero point [ph/s/m^2/nm].
        Fbinary: binary flux ratio [dimensionless].
        sky_trans: coronagraph sky transmission [dimensionless].
        area_m2: telescope collecting area [m^2].
        throughput: total throughput.
        dlambda_nm: bandwidth [nm].
        n_channels: number of spectral channels.
        core_area_lod2: solid angle of photometric aperture [(lam/D)^2].
    """
    return (
        F0
        * Fbinary
        * sky_trans
        * core_area_lod2
        * area_m2
        * throughput
        * dlambda_nm
        * n_channels
    )


# -- Noise floors -------------------------------------------------------------


def noise_floor_stellar(
    F0,
    Fs_over_F0,
    area_m2,
    throughput,
    dlambda_nm,
    n_channels,
    snr,
    noisefloor_value,
    core_area_lod2,
):
    """Stellar noise floor CRnf_star [e/s].

    In the 1D curve approach, we use
    ``noisefloor_value x core_area_lod2`` which is equivalent to
    pyEDITH's ``noisefloor[x,y] / pixscale^2 x omega_lod[x,y]``.

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRnf``.

    Args:
        F0: flux zero point [ph/s/m^2/nm].
        Fs_over_F0: stellar flux ratio [dimensionless].
        area_m2: telescope collecting area [m^2].
        throughput: total throughput.
        dlambda_nm: bandwidth [nm].
        n_channels: number of spectral channels.
        snr: target SNR (or 1.0 for SNR-solve mode).
        noisefloor_value: noise floor level [dimensionless].
        core_area_lod2: solid angle of photometric aperture [(lam/D)^2].
    """
    return (
        snr
        * F0
        * Fs_over_F0
        * noisefloor_value
        * core_area_lod2
        * area_m2
        * throughput
        * dlambda_nm
        * n_channels
    )


def noise_floor_exozodi(CRbez, snr, ez_ppf):
    """Exozodi noise floor CRnf_ez [e/s].

    Matches ``pyEDITH.exposure_time_calculator.calculate_CRnf_ez``.
    """
    return snr * CRbez / ez_ppf


def noise_floor_total(CRnf_star, CRnf_ez, include_ez):
    """Combined noise floor CRnf [e/s].

    When ``include_ez`` is True, combines stellar and exozodi noise floors
    in quadrature. Otherwise, uses stellar noise floor only.
    """
    return jnp.where(
        include_ez,
        jnp.sqrt(CRnf_star**2 + CRnf_ez**2),
        CRnf_star,
    )


# -- Speckle residual (EXOSIMS path) ------------------------------------------


def speckle_residual(C_sr, ppfact, stability_fact):
    """EXOSIMS-style speckle residual Csp [e/s].

    ``C_sr`` is the raw stellar residual count rate (e.g. from
    core_contrast x core_throughput x stellar flux).
    """
    return C_sr * ppfact * stability_fact


# -- Photon counting time (helper) --------------------------------------------


def photon_counting_time(det_CR, det_npix):
    """Average time to detect one photon per pixel [s].

    Used by pyEDITH to compute the CIC timing in ``count_rate_detector``.
    The magic constant 6.73 comes from the original pyEDITH code, I think from
    the Roman EMCCD.

    Args:
        det_CR: Total detector count rate [e/s] (summed over all sources).
        det_npix: Number of detector pixels.

    Returns:
        t_photon [s]: inverse of (6.73 x counts_per_second_per_pixel).
    """
    counts_per_pix_per_s = det_CR / det_npix
    # Magic constant 6.73 corresponds to Roman/HWO EMCCD characteristics.
    PHOTON_COUNTING_FACTOR = 6.73
    return 1.0 / (PHOTON_COUNTING_FACTOR * counts_per_pix_per_s)
