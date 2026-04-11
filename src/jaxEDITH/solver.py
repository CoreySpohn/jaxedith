"""ETC solver functions: separate implementations per equation variant.

Each solver function contains a single clear equation — no ``jnp.where``
branching. The orchestrator in ``__init__.py`` selects which solver to call
at trace time based on the config variant string, so JIT compiles only the
chosen code path.
"""

import jax.numpy as jnp

# ── Shared overhead / guard helpers ───────────────────────────────────────────


def _apply_overheads(t_science, overhead_multi, overhead_fixed, n_rolls):
    """Apply multiplicative + fixed overheads and telescope rolls."""
    t_total = t_science * overhead_multi + overhead_fixed
    t_total = jnp.where(t_total > 0, t_total, jnp.inf)
    return t_total * n_rolls


def _safe_divide(numerator, denominator):
    """Divide with NaN-gradient-safe guard for non-positive denominators.

    When ``denominator <= 0`` the result is ``jnp.inf``, but we substitute
    ``1.0`` before dividing so that ``jax.grad`` never sees ``x / 0``.
    This avoids the classic ``jnp.where`` NaN-gradient trap.
    """
    safe_denom = jnp.where(denominator > 0, denominator, 1.0)
    result = numerator / safe_denom
    return jnp.where(denominator > 0, result, jnp.inf)


# ═══════════════════════════════════════════════════════════════════════════════
# AYO / jaxEDITH
# ═══════════════════════════════════════════════════════════════════════════════


def solve_exptime_ayo(
    Cp,
    Cb,
    Cnf,
    snr,
    bg_multiplier=2.0,
    overhead_multi=1.0,
    overhead_fixed=0.0,
    n_rolls=1,
):
    r"""Solve for exposure time — AYO/jaxEDITH equation.

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_p + m \cdot C_b}{C_p^2 - C_{nf}^2}

    where *m* is the background multiplier (2 for ADI).

    ``Cnf`` is the *total* noise floor, already including the SNR factor:
    ``Cnf = SNR * noise_floor_rate``.

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Cnf: Noise floor count rate [e/s] (includes SNR factor).
        snr: Target signal-to-noise ratio.
        bg_multiplier: Background multiplier (2.0 for ADI, default).
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    numerator = snr**2 * (Cp + bg_multiplier * Cb)
    denominator = Cp**2 - Cnf**2
    t_science = _safe_divide(numerator, denominator)
    return _apply_overheads(t_science, overhead_multi, overhead_fixed, n_rolls)


def solve_snr_ayo(
    Cp,
    Cb,
    Cnf_rate,
    t_obs,
    bg_multiplier=2.0,
    overhead_multi=1.0,
    overhead_fixed=0.0,
    n_rolls=1,
):
    r"""Solve for achieved SNR — AYO/jaxEDITH equation.

    Inverse of :func:`solve_exptime_ayo`. Derived from

    .. math::
        t_{eff} = \frac{\text{SNR}^2}{C_p^2 - (\text{SNR} \cdot C_{nf,rate})^2}

    giving

    .. math::
        \text{SNR} = \sqrt{\frac{t_{eff} \cdot C_p^2}{1 + t_{eff} \cdot C_{nf,rate}^2}}

    where ``t_eff = t_per_roll / (overhead_multi * rate_sum)``.

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Cnf_rate: Noise floor *rate* [e/s] (with SNR=1).
        t_obs: Total observation time [s].
        bg_multiplier: Background multiplier (2.0 for ADI).
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Achieved SNR; 0.0 for non-physical results.
    """
    t_per_roll = jnp.maximum(t_obs / n_rolls - overhead_fixed, 0.0)
    rate_sum = jnp.maximum(Cp + bg_multiplier * Cb, 1e-30)
    t_eff = t_per_roll / (overhead_multi * rate_sum)

    snr = jnp.sqrt(t_eff * Cp**2 / jnp.maximum(1.0 + t_eff * Cnf_rate**2, 1e-30))
    return jnp.where(jnp.isfinite(snr) & (snr > 0), snr, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# EXOSIMS Detection
# ═══════════════════════════════════════════════════════════════════════════════


def solve_exptime_exosims_det(
    Cp,
    Cb,
    Csp,
    snr,
    overhead_multi=1.0,
    overhead_fixed=0.0,
    n_rolls=1,
):
    r"""Solve for exposure time — EXOSIMS detection equation.

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_b}{C_p^2 - (\text{SNR} \cdot C_{sp})^2}

    Compared to AYO:
    - No planet shot noise in numerator
    - No ADI factor on background
    - Speckle residual instead of noise floor

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        snr: Target signal-to-noise ratio.
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    numerator = snr**2 * Cb
    denominator = Cp**2 - (snr * Csp) ** 2
    t_science = _safe_divide(numerator, denominator)
    return _apply_overheads(t_science, overhead_multi, overhead_fixed, n_rolls)


def solve_snr_exosims_det(
    Cp,
    Cb,
    Csp,
    t_obs,
    overhead_multi=1.0,
    overhead_fixed=0.0,
    n_rolls=1,
):
    r"""Solve for achieved SNR — EXOSIMS detection equation.

    From ``t = SNR² * Cb / (Cp² - (SNR·Csp)²)`` we get:

    .. math::
        \text{SNR} = \sqrt{\frac{t_{eff} \cdot C_p^2}{C_b + t_{eff} \cdot C_{sp}^2}}

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        t_obs: Total observation time [s].
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Achieved SNR; 0.0 for non-physical results.
    """
    t_per_roll = jnp.maximum(t_obs / n_rolls - overhead_fixed, 0.0)
    t_eff = t_per_roll / overhead_multi

    snr = jnp.sqrt(t_eff * Cp**2 / jnp.maximum(Cb + t_eff * Csp**2, 1e-30))
    return jnp.where(jnp.isfinite(snr) & (snr > 0), snr, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# EXOSIMS Characterization
# ═══════════════════════════════════════════════════════════════════════════════


def solve_exptime_exosims_char(
    Cp,
    Cb,
    Csp,
    snr,
    overhead_multi=1.0,
    overhead_fixed=0.0,
    n_rolls=1,
):
    r"""Solve for exposure time — EXOSIMS characterization equation.

    Same as EXOSIMS detection but with :math:`C_p` added to :math:`C_b`:

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_b + C_p}{C_p^2 - (\text{SNR} \cdot C_{sp})^2}

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        snr: Target signal-to-noise ratio.
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    numerator = snr**2 * (Cb + Cp)
    denominator = Cp**2 - (snr * Csp) ** 2
    t_science = _safe_divide(numerator, denominator)
    return _apply_overheads(t_science, overhead_multi, overhead_fixed, n_rolls)


def solve_snr_exosims_char(
    Cp,
    Cb,
    Csp,
    t_obs,
    overhead_multi=1.0,
    overhead_fixed=0.0,
    n_rolls=1,
):
    r"""Solve for achieved SNR — EXOSIMS characterization equation.

    Same as detection but with :math:`C_p` added to :math:`C_b`:

    .. math::
        \text{SNR} = \sqrt{
            \frac{t_{eff} \cdot C_p^2}{(C_b + C_p) + t_{eff} \cdot C_{sp}^2}
        }

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Csp: Speckle residual count rate [e/s].
        t_obs: Total observation time [s].
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Achieved SNR; 0.0 for non-physical results.
    """
    t_per_roll = jnp.maximum(t_obs / n_rolls - overhead_fixed, 0.0)
    t_eff = t_per_roll / overhead_multi

    Cb_eff = Cb + Cp
    snr = jnp.sqrt(t_eff * Cp**2 / jnp.maximum(Cb_eff + t_eff * Csp**2, 1e-30))
    return jnp.where(jnp.isfinite(snr) & (snr > 0), snr, 0.0)
