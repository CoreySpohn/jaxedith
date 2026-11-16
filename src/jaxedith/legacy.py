"""Scene and convention adapters for the scalar comparison dialects.

Rate arithmetic lives in primitives.py and the AYO/EXOSIMS equations in
etc.py. Their plain-number inputs and electron units remain unchanged.
These adapters never supply an alternative optical-path rate derivation.
"""

import jax.numpy as jnp
from hwoutils.constants import arcsec2rad
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy
from optixstuff import FramePlan, frames_for_time

from jaxedith.calculator import PostProcessing
from jaxedith.primitives import detector_noise_rate, photon_counting_time
from jaxedith.public import _DEFAULT_ZODI, _disk_radiance, _position
from jaxedith.scene import ETCScene


def post_processing_from_pyedith(ppfact) -> PostProcessing:
    """Convert pyEDITH's divisor to a speckle residual fraction; disk is zero."""
    return PostProcessing(1 / jnp.asarray(ppfact), 0.0)


def post_processing_from_exosims(ppFact) -> PostProcessing:
    """Read EXOSIMS ppFact as the residual fraction itself; disk is zero."""
    return PostProcessing(ppFact, 0.0)


def frame_plan_photon_counting(count_rate_per_pixel, *, time_s) -> FramePlan:
    """Use a frame time giving mean 1/6.73 electrons per pixel per frame.

    count_rate_per_pixel is strictly positive electrons/pixel/s. Whole
    frames are counted solely by optixstuff.frames_for_time.
    """
    frame_time_s = photon_counting_time(count_rate_per_pixel, 1.0)
    return frames_for_time(time_s, frame_time_s)


def legacy_pyedith_noise_rate(
    count_rate_per_pixel,
    *,
    n_pix,
    dark_current_e_per_pixel_s,
    read_noise_e_rms,
    read_frame_time_s,
    clock_induced_charge_e_per_pixel_frame,
):
    """Historical variance rate using separate read and occupancy CIC times.

    This is the legacy detector_noise_rate, in its original numerical
    convention, with the photon-counting time supplied by the 6.73 rule.
    It is not the shared detector's whole-frame count variance.
    """
    return detector_noise_rate(
        n_pix,
        dark_current_e_per_pixel_s,
        read_noise_e_rms,
        read_frame_time_s,
        clock_induced_charge_e_per_pixel_frame,
        photon_counting_time(count_rate_per_pixel, 1.0),
    )


def etc_scene_from_system(
    system,
    wavelength_nm,
    *,
    time_d,
    planet_index=None,
    separation_arcsec=None,
    position_angle_deg=0.0,
    zodi=_DEFAULT_ZODI,
    ecliptic_latitude_deg=0.0,
    solar_longitude_deg=135.0,
    disk_grid=None,
    companion_flux_density=None,
) -> ETCScene:
    """Produce monochromatic F0-relative ratios solely for legacy comparison.

    F0 is the AB zero-magnitude photon density. Fzodi is per arcsec^2.
    Fexozodi is the local disk ratio times projected radius in AU squared,
    so the legacy inverse-square primitive reconstructs the sampled local
    brightness. This effective 1-AU value does not assume a disk profile.
    Companion input uses the reader's per-reference-lod^2 photon convention.
    """
    east, north, ephemeris = _position(
        system, time_d, planet_index, separation_arcsec, position_angle_deg
    )
    sep = jnp.hypot(east, north)
    wave = jnp.asarray(wavelength_nm)
    zero = jy_to_photons_per_nm_per_m2(mag_to_flux_jy(0.0), wave)
    star = system.star.photon_flux_density(wave, time_d, dist_pc=system.dist_pc)
    planet = (
        0.0
        if planet_index is None
        else system.planet_flux_densities(
            wave, jnp.atleast_1d(time_d), ephemeris=ephemeris
        )[planet_index, 0]
    )
    local = (
        0.0
        if zodi is None
        else zodi.photon_radiance(
            wave,
            ecliptic_latitude_deg=ecliptic_latitude_deg,
            solar_longitude_deg=solar_longitude_deg,
        )
    )
    disk = _disk_radiance(system, wave, time_d, east, north, disk_grid)
    diameter = getattr(system.star, "diameter_arcsec", 0.0)
    if callable(diameter):
        diameter = diameter(dist_pc=system.dist_pc)
    return ETCScene(
        F0=zero,
        Fs_over_F0=star / zero,
        Fp_over_Fs=planet / star,
        Fzodi=local * arcsec2rad**2 / zero,
        Fexozodi=disk * arcsec2rad**2 / zero * (system.dist_pc * sep) ** 2,
        dist_pc=system.dist_pc,
        sep_arcsec=sep,
        stellar_diam_mas=1000 * diameter,
        Fbinary=0.0
        if companion_flux_density is None
        else companion_flux_density(wave, time_d) / zero,
    )
