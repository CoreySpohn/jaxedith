"""Observation convenience functions -- bridge orbix observatory with ETC.

Provides :func:`calc_exptime_from_observation` and
:func:`calc_snr_from_observation` which compute zodiacal light from
observatory geometry and delegate to the core JAX ETC solvers.
"""

from __future__ import annotations

import jax.numpy as jnp
from orbix.observatory import (
    ObservatoryL2Halo,
    zodi_fzodi_ayo,
    zodi_fzodi_leinert,
)

from jaxedith.public import exptime_ayo, snr_ayo
from jaxedith.scene import ETCScene

# -- Public API ----------------------------------------------------------------


def calc_exptime_from_observation(
    optical_path,
    observatory: ObservatoryL2Halo,
    mjd: float,
    ra_rad: float,
    dec_rad: float,
    wavelength_nm: float,
    separation_lod: float,
    dlambda_nm: float,
    snr: float,
    *,
    F0: float,
    Fs_over_F0: float,
    Fp_over_Fs: float,
    dist_pc: float = 10.0,
    sep_arcsec: float = 0.1,
    Fexozodi: float = 0.0,
    n_channels: float = 1.0,
    temp_K: float = 270.0,
    zodi_mode: str = "ayo",
):
    """End-to-end: observation parameters -> exposure time.

    Computes zodiacal light from observatory geometry and calls
    :func:`jaxedith.exptime_ayo`.

    Args:
        optical_path: ``optixstuff.OpticalPath`` equinox module.
        observatory: orbix ``ObservatoryL2Halo`` instance.
        mjd: Observation time (Modified Julian Date).
        ra_rad: Target right ascension in radians (J2000).
        dec_rad: Target declination in radians (J2000).
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lambda/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        snr: Target signal-to-noise ratio.
        F0: Flux zero point [ph/s/m^2/nm].
        Fs_over_F0: Stellar flux ratio (dimensionless).
        Fp_over_Fs: Planet-to-star contrast (dimensionless).
        dist_pc: Distance to star [pc].
        sep_arcsec: Angular separation [arcsec].
        Fexozodi: Exozodiacal surface brightness [arcsec^-2].
        n_channels: Number of spectral channels.
        temp_K: Telescope temperature [K].
        zodi_mode: ``"ayo"`` for position-independent AYO default, or
            ``"leinert"`` for full Leinert position-dependent model.

    Returns:
        Exposure time in seconds.
    """
    Fzodi = _compute_fzodi(observatory, mjd, ra_rad, dec_rad, wavelength_nm, zodi_mode)

    scene = ETCScene(
        F0=F0,
        Fs_over_F0=Fs_over_F0,
        Fp_over_Fs=Fp_over_Fs,
        Fzodi=Fzodi,
        Fexozodi=Fexozodi,
        dist_pc=dist_pc,
        sep_arcsec=sep_arcsec,
        n_channels=n_channels,
        temp_K=temp_K,
    )

    return exptime_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        snr,
        temp_K=temp_K,
        ez_ppf=scene.ez_ppf,
    )


def calc_snr_from_observation(
    optical_path,
    observatory: ObservatoryL2Halo,
    mjd: float,
    ra_rad: float,
    dec_rad: float,
    wavelength_nm: float,
    separation_lod: float,
    dlambda_nm: float,
    t_obs: float,
    *,
    F0: float,
    Fs_over_F0: float,
    Fp_over_Fs: float,
    dist_pc: float = 10.0,
    sep_arcsec: float = 0.1,
    Fexozodi: float = 0.0,
    n_channels: float = 1.0,
    temp_K: float = 270.0,
    zodi_mode: str = "ayo",
):
    """End-to-end: observation parameters -> achieved SNR.

    Same as :func:`calc_exptime_from_observation` but solves for SNR
    given a fixed observation time.

    Args:
        optical_path: ``optixstuff.OpticalPath`` equinox module.
        observatory: orbix ``ObservatoryL2Halo`` instance.
        mjd: Observation time (Modified Julian Date).
        ra_rad: Target right ascension in radians (J2000).
        dec_rad: Target declination in radians (J2000).
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lambda/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        t_obs: Total observation time [s].
        F0: Flux zero point [ph/s/m^2/nm].
        Fs_over_F0: Stellar flux ratio (dimensionless).
        Fp_over_Fs: Planet-to-star contrast (dimensionless).
        dist_pc: Distance to star [pc].
        sep_arcsec: Angular separation [arcsec].
        Fexozodi: Exozodiacal surface brightness [arcsec^-2].
        n_channels: Number of spectral channels.
        temp_K: Telescope temperature [K].
        zodi_mode: ``"ayo"`` or ``"leinert"``.

    Returns:
        Achieved signal-to-noise ratio.
    """
    Fzodi = _compute_fzodi(observatory, mjd, ra_rad, dec_rad, wavelength_nm, zodi_mode)

    scene = ETCScene(
        F0=F0,
        Fs_over_F0=Fs_over_F0,
        Fp_over_Fs=Fp_over_Fs,
        Fzodi=Fzodi,
        Fexozodi=Fexozodi,
        dist_pc=dist_pc,
        sep_arcsec=sep_arcsec,
        n_channels=n_channels,
        temp_K=temp_K,
    )

    return snr_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        t_obs,
        temp_K=temp_K,
        ez_ppf=scene.ez_ppf,
    )


def observation_geometry(
    observatory: ObservatoryL2Halo,
    mjd: float,
    ra_rad: float,
    dec_rad: float,
):
    """Compute observing geometry for a target at a given time.

    Useful for understanding the zodiacal light contribution and
    keepout status without running the full ETC.

    Args:
        observatory: orbix ``ObservatoryL2Halo`` instance.
        mjd: Observation time (Modified Julian Date).
        ra_rad: Target right ascension in radians (J2000).
        dec_rad: Target declination in radians (J2000).

    Returns:
        Dictionary with keys:
            - ``sun_angle_deg``: Sun-target angle in degrees.
            - ``ecliptic_lat_deg``: Ecliptic latitude in degrees.
            - ``solar_lon_deg``: Solar longitude in degrees.
            - ``telescope_pos_au``: Heliocentric ecliptic position (3-vector, AU).
    """
    sun_angle_rad = observatory.sun_angle(mjd, ra_rad, dec_rad)
    ecl_lat_deg = observatory.ecliptic_latitude(ra_rad, dec_rad)  # already degrees
    sol_lon_deg = observatory.solar_longitude(mjd, ra_rad, dec_rad)  # already degrees
    pos = observatory.position_ecliptic(mjd)

    return {
        "sun_angle_deg": jnp.degrees(sun_angle_rad),
        "ecliptic_lat_deg": ecl_lat_deg,
        "solar_lon_deg": sol_lon_deg,
        "telescope_pos_au": pos,
    }


# -- Internal helpers ----------------------------------------------------------


def _compute_fzodi(observatory, mjd, ra_rad, dec_rad, wavelength_nm, mode):
    """Compute Fzodi using the specified mode."""
    if mode == "ayo":
        return zodi_fzodi_ayo(wavelength_nm)
    elif mode == "leinert":
        ecl_lat_deg = observatory.ecliptic_latitude(ra_rad, dec_rad)  # already degrees
        sol_lon_deg = observatory.solar_longitude(
            mjd, ra_rad, dec_rad
        )  # already degrees
        return zodi_fzodi_leinert(wavelength_nm, ecl_lat_deg, sol_lon_deg)
    else:
        raise ValueError(f"Unknown zodi_mode: {mode!r}. Use 'ayo' or 'leinert'.")


def _system_to_etc_scene(
    system,
    planet_index: int,
    wavelength_nm: float,
    time_jd: float,
    Fzodi: float,
    Fexozodi: float = 0.0,
    n_channels: float = 1.0,
    temp_K: float = 270.0,
):
    """Extract ETCScene from a ``skyscapes.scene.System``.

    Computes star flux, planet contrast, and angular separation at the
    given wavelength and time, then wraps them as an ``ETCScene``.

    Star flux is returned in ph/s/m^2/nm. Contrast and separation come
    from ``System.contrasts`` and ``System.alpha_dMag``, which return
    shape ``(K, T)``; we index ``[planet_index, 0]`` to pull out the
    scalar we want.
    """
    # Star flux at observation wavelength/time (ph/s/m^2/nm) -- scalar-in, scalar-out.
    F0 = system.star.spec_flux_density(wavelength_nm, time_jd)

    # (K, 1) -> scalar at [planet_index, 0]
    contrasts = system.contrasts(
        jnp.atleast_1d(wavelength_nm), jnp.atleast_1d(time_jd)
    )
    Fp_over_Fs = contrasts[planet_index, 0]

    alpha, _dMag = system.alpha_dMag(jnp.atleast_1d(time_jd))
    sep_arcsec = alpha[planet_index, 0]

    return ETCScene(
        F0=F0,
        Fs_over_F0=1.0,
        Fp_over_Fs=Fp_over_Fs,
        Fzodi=Fzodi,
        Fexozodi=Fexozodi,
        dist_pc=system.star.dist_pc,
        sep_arcsec=sep_arcsec,
        n_channels=n_channels,
        temp_K=temp_K,
    )


# -- System-based API ---------------------------------------------------------


def calc_exptime_from_system(
    optical_path,
    system,
    planet_index: int,
    observatory: ObservatoryL2Halo,
    mjd: float,
    wavelength_nm: float,
    separation_lod: float,
    dlambda_nm: float,
    snr: float,
    *,
    Fexozodi: float = 0.0,
    n_channels: float = 1.0,
    temp_K: float = 270.0,
    zodi_mode: str = "ayo",
):
    """Compute exposure time from a skyscapes.scene.System.

    Automatically extracts star flux, planet contrast, and separation
    from the System at the given time, and computes zodi from the
    observatory geometry.

    Args:
        optical_path: ``optixstuff.OpticalPath`` equinox module.
        system: ``skyscapes.scene.System`` with star + planets.
        planet_index: Index of the target planet in the System.
        observatory: orbix ``ObservatoryL2Halo`` instance.
        mjd: Observation time (Modified Julian Date).
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lambda/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        snr: Target signal-to-noise ratio.
        Fexozodi: Exozodiacal surface brightness [arcsec^-2].
        n_channels: Number of spectral channels.
        temp_K: Telescope temperature [K].
        zodi_mode: ``"ayo"`` or ``"leinert"``.

    Returns:
        Exposure time in seconds.
    """
    time_jd = mjd + 2400000.5

    # RA/Dec from star for zodi computation
    ra_rad = jnp.deg2rad(system.star.ra_deg)
    dec_rad = jnp.deg2rad(system.star.dec_deg)

    Fzodi = _compute_fzodi(observatory, mjd, ra_rad, dec_rad, wavelength_nm, zodi_mode)

    scene = _system_to_etc_scene(
        system,
        planet_index,
        wavelength_nm,
        time_jd,
        Fzodi=Fzodi,
        Fexozodi=Fexozodi,
        n_channels=n_channels,
        temp_K=temp_K,
    )

    return exptime_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        snr,
        temp_K=temp_K,
        ez_ppf=scene.ez_ppf,
    )


def calc_snr_from_system(
    optical_path,
    system,
    planet_index: int,
    observatory: ObservatoryL2Halo,
    mjd: float,
    wavelength_nm: float,
    separation_lod: float,
    dlambda_nm: float,
    t_obs: float,
    *,
    Fexozodi: float = 0.0,
    n_channels: float = 1.0,
    temp_K: float = 270.0,
    zodi_mode: str = "ayo",
):
    """Compute achieved SNR from a skyscapes.scene.System.

    Same as :func:`calc_exptime_from_system` but solves for SNR
    given a fixed observation time.

    Args:
        optical_path: ``optixstuff.OpticalPath`` equinox module.
        system: ``skyscapes.scene.System`` with star + planets.
        planet_index: Index of the target planet in the System.
        observatory: orbix ``ObservatoryL2Halo`` instance.
        mjd: Observation time (Modified Julian Date).
        wavelength_nm: Observation wavelength [nm].
        separation_lod: Planet separation in lambda/D.
        dlambda_nm: Bandwidth per spectral element [nm].
        t_obs: Total observation time [s].
        Fexozodi: Exozodiacal surface brightness [arcsec^-2].
        n_channels: Number of spectral channels.
        temp_K: Telescope temperature [K].
        zodi_mode: ``"ayo"`` or ``"leinert"``.

    Returns:
        Achieved signal-to-noise ratio.
    """
    time_jd = mjd + 2400000.5

    ra_rad = jnp.deg2rad(system.star.ra_deg)
    dec_rad = jnp.deg2rad(system.star.dec_deg)

    Fzodi = _compute_fzodi(observatory, mjd, ra_rad, dec_rad, wavelength_nm, zodi_mode)

    scene = _system_to_etc_scene(
        system,
        planet_index,
        wavelength_nm,
        time_jd,
        Fzodi=Fzodi,
        Fexozodi=Fexozodi,
        n_channels=n_channels,
        temp_K=temp_K,
    )

    return snr_ayo(
        optical_path,
        scene,
        wavelength_nm,
        separation_lod,
        dlambda_nm,
        t_obs,
        temp_K=temp_K,
        ez_ppf=scene.ez_ppf,
    )
