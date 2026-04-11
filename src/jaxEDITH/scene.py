"""Astrophysical scene dataclass for the JAX ETC."""

import equinox as eqx
import jax.numpy as jnp


class ETCScene(eqx.Module):
    """Astrophysical scene parameters for the JAX ETC.

    All values are raw floats in consistent units. No astropy quantities.

    Attributes:
        F0: Flux zero point [ph/s/m^2/nm].
        Fs_over_F0: Stellar-to-zeropoint flux ratio — ``10^(-0.4 x m_star)``.
        Fp_over_Fs: Planet-to-star contrast — ``10^(-0.4 x dMag)``.
        Fzodi: Local zodiacal surface brightness ratio [arcsec⁻²].
        Fexozodi: Exozodiacal surface brightness ratio at 1 AU [arcsec⁻²].
        dist_pc: Distance to the star [pc].
        sep_arcsec: Angular separation of the planet [arcsec].
        n_channels: Number of spectral channels.
        Fbinary: Flux ratio from binary/neighbor stars [dimensionless].
        ez_ppf: Exozodi post-processing factor.
        temp_K: Telescope mirror temperature [K].
    """

    F0: float
    Fs_over_F0: float
    Fp_over_Fs: float
    Fzodi: float = 0.0
    Fexozodi: float = 0.0
    dist_pc: float = 10.0
    sep_arcsec: float = 0.1
    n_channels: float = 1.0
    Fbinary: float = 0.0
    ez_ppf: float = jnp.inf
    temp_K: float = 270.0
