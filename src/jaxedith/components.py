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

from hwoutils.constants import nm2m, rad2arcsec

from jaxedith.count_rates import (
    count_rate_planet,
    count_rate_stellar_leakage,
    count_rate_zodi,
)


def _lod_arcsec(optical_path, wavelength_nm):
    """lambda/D in arcsec for the optical path's primary diameter."""
    lod_rad = (wavelength_nm * nm2m) / optical_path.primary.diameter_m
    return lod_rad * rad2arcsec


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
