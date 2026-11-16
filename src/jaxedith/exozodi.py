"""Analytic exozodiacal radiance in the scalar AYO calculator dialect."""

import equinox as eqx
import jax.numpy as jnp
from hwoutils.constants import arcsec2rad
from hwoutils.conversions import jy_to_photons_per_nm_per_m2, mag_to_flux_jy

from ._types import FloatOut, Num


class AYOExozodi(eqx.Module):
    """Scalar AYO exozodi with flat AB colour and inverse-square illumination.

    At 1 AU, Fexozodi = level_zodis * 10**(-0.4 * brightness), per arcsec^2,
    where brightness is surface_brightness_mag_arcsec2 for one zodi.
    Its denominator is the AB magnitude-zero flux density F0(lambda), from
    mag_to_flux_jy(0), not the host-star flux. Radiance is F0 * Fexozodi
    divided by arcsec2rad**2 and separation_au**2. There is no host colour,
    luminosity, or inclination correction. The analytic AB zero point has
    no table knots; photon colour is 1/lambda (constant F_nu).

    One 22-mag/arcsec^2 zodi is 10^-8.8 relative to a magnitude-zero source.
    The alternative approximately 10^-6.87 ratio uses a solar absolute-V
    magnitude 4.83 source at 10 pc as denominator. Face-on and inclined
    definitions also differ by a factor 0.49; no such conversion is applied
    here. These dialects must not be substituted without conversion.

    separation_au is the projected physical separation dist_pc * sep_arcsec,
    not the orbital radius. The scalar law is singular at zero separation.
    """

    level_zodis: Num = 3.0
    surface_brightness_mag_arcsec2: Num = 22.0

    def photon_radiance(self, wavelength_nm: Num, *, separation_au: Num) -> FloatOut:
        """Return ph s^-1 m^-2 nm^-1 sr^-1 at each wavelength and separation."""
        zero_point = jy_to_photons_per_nm_per_m2(
            mag_to_flux_jy(0.0), jnp.asarray(wavelength_nm)
        )
        brightness = self.level_zodis * 10.0 ** (
            -0.4 * self.surface_brightness_mag_arcsec2
        )
        return zero_point * brightness / arcsec2rad**2 / jnp.asarray(separation_au) ** 2
