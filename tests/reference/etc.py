"""Legacy scalar AYO and EXOSIMS equations for cross-code comparison."""

import jax.numpy as jnp

from reference.types import FloatOut, Num


def _safe_divide(numerator: Num, denominator: Num) -> FloatOut:
    """Divide with NaN-gradient-safe guard for non-positive denominators.

    When ``denominator <= 0`` the result is ``jnp.inf``, but we substitute
    ``1.0`` before dividing so that ``jax.grad`` never sees ``x / 0``.
    """
    safe_denom = jnp.where(denominator > 0, denominator, 1.0)
    result = numerator / safe_denom
    return jnp.where(denominator > 0, result, jnp.inf)


# =============================================================================
# AYO / jaxedith
# =============================================================================


def exptime_from_rates_ayo(
    Cp: Num,
    Cb: Num,
    Cnf_rate: Num,
    snr: Num,
    bg_multiplier: Num = 2.0,
) -> FloatOut:
    r"""Solve for exposure time -- AYO/jaxedith equation.

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_p + m \cdot C_b}
            {C_p^2 - (\text{SNR} \cdot C_{nf,rate})^2}

    where *m* is the background multiplier (2 for ADI).

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Cnf_rate: Noise floor *rate* (SNR=1) [e/s].
        snr: Target signal-to-noise ratio.
        bg_multiplier: Background multiplier (2.0 for ADI, default).

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    Cnf = snr * Cnf_rate
    numerator = snr**2 * (Cp + bg_multiplier * Cb)
    denominator = Cp**2 - Cnf**2
    return _safe_divide(numerator, denominator)


def snr_from_rates_ayo(
    Cp: Num,
    Cb: Num,
    Cnf_rate: Num,
    t_obs: Num,
    bg_multiplier: Num = 2.0,
) -> FloatOut:
    r"""Solve for achieved SNR -- AYO/jaxedith equation.

    Inverse of :func:`exptime_from_rates_ayo`:

    .. math::
        \text{SNR} = \sqrt{\frac{t_{eff} \cdot C_p^2}
            {1 + t_{eff} \cdot C_{nf,rate}^2}}

    where ``t_eff = t_obs / (C_p + m C_b)``.

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Cnf_rate: Noise floor *rate* (SNR=1) [e/s].
        t_obs: Total observation time [s].
        bg_multiplier: Background multiplier (2.0 for ADI).

    Returns:
        Achieved SNR; 0.0 for non-physical results.
    """
    t_eff = jnp.maximum(t_obs, 0.0)
    rate_sum = jnp.maximum(Cp + bg_multiplier * Cb, 1e-30)
    t_eff = t_eff / rate_sum

    snr = jnp.sqrt(t_eff * Cp**2 / jnp.maximum(1.0 + t_eff * Cnf_rate**2, 1e-30))
    return jnp.where(jnp.isfinite(snr) & (snr > 0), snr, 0.0)


# =============================================================================
# EXOSIMS Detection
# =============================================================================


def exptime_from_rates_exosims_det(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    snr: Num,
) -> FloatOut:
    r"""Solve for exposure time -- EXOSIMS detection equation.

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_b}{C_p^2 - (\text{SNR} \cdot C_{sp})^2}

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        snr: Target signal-to-noise ratio.

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    numerator = snr**2 * Cb
    denominator = Cp**2 - (snr * Csp) ** 2
    return _safe_divide(numerator, denominator)


def snr_from_rates_exosims_det(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    t_obs: Num,
) -> FloatOut:
    r"""Solve for achieved SNR -- EXOSIMS detection equation.

    .. math::
        \text{SNR} = \sqrt{\frac{t_{eff} \cdot C_p^2}{C_b + t_{eff} \cdot C_{sp}^2}}

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        t_obs: Total observation time [s].

    Returns:
        Achieved SNR; 0.0 for non-physical results.
    """
    t_eff = jnp.maximum(t_obs, 0.0)

    snr = jnp.sqrt(t_eff * Cp**2 / jnp.maximum(Cb + t_eff * Csp**2, 1e-30))
    return jnp.where(jnp.isfinite(snr) & (snr > 0), snr, 0.0)


# =============================================================================
# EXOSIMS Characterization
# =============================================================================


def exptime_from_rates_exosims_char(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    snr: Num,
) -> FloatOut:
    r"""Solve for exposure time -- EXOSIMS characterization equation.

    Same as EXOSIMS detection but with :math:`C_p` added to :math:`C_b`:

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_b + C_p}
            {C_p^2 - (\text{SNR} \cdot C_{sp})^2}

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        snr: Target signal-to-noise ratio.

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    numerator = snr**2 * (Cb + Cp)
    denominator = Cp**2 - (snr * Csp) ** 2
    return _safe_divide(numerator, denominator)


def snr_from_rates_exosims_char(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    t_obs: Num,
) -> FloatOut:
    r"""Solve for achieved SNR -- EXOSIMS characterization equation.

    .. math::
        \text{SNR} = \sqrt{\frac{t_{eff} \cdot C_p^2}
            {(C_b + C_p) + t_{eff} \cdot C_{sp}^2}}

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        t_obs: Total observation time [s].

    Returns:
        Achieved SNR; 0.0 for non-physical results.
    """
    t_eff = jnp.maximum(t_obs, 0.0)

    Cb_eff = Cb + Cp
    snr = jnp.sqrt(t_eff * Cp**2 / jnp.maximum(Cb_eff + t_eff * Csp**2, 1e-30))
    return jnp.where(jnp.isfinite(snr) & (snr > 0), snr, 0.0)
