"""Legacy monochromatic scene record for scalar comparisons."""

import equinox as eqx

from jaxedith._types import Num


class ETCScene(eqx.Module):
    """Monochromatic astrophysical inputs to the comparison equations.

    All values are raw floats in consistent units. No astropy quantities,
    no instrument or post-processing knobs. Produce this record with
    ``reference.legacy.etc_scene_from_system`` for scalar comparisons.

    Attributes:
        F0: Flux zero point [ph/s/m^2/nm].
        Fs_over_F0: Stellar-to-zeropoint flux ratio -- ``10^(-0.4 x m_star)``.
        Fp_over_Fs: Planet-to-star contrast -- ``10^(-0.4 x dMag)``.
        Fzodi: Local zodiacal surface brightness ratio [arcsec^-2].
        Fexozodi: Exozodiacal surface brightness ratio at 1 AU [arcsec^-2].
        dist_pc: Distance to the star [pc].
        sep_arcsec: Angular separation of the planet [arcsec].
        Fbinary: Flux ratio from binary/neighbor stars [dimensionless].
        stellar_diam_mas: Stellar angular diameter [milliarcseconds]; 0 = point star.
    """

    F0: Num
    Fs_over_F0: Num
    Fp_over_Fs: Num
    Fzodi: Num = 0.0
    Fexozodi: Num = 0.0
    dist_pc: Num = 10.0
    sep_arcsec: Num = 0.1
    Fbinary: Num = 0.0
    stellar_diam_mas: Num = 0.0
