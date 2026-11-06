"""Closed-form exposure-time and SNR equations.

Layer 3 of the jaxedith pipeline: ``primitives -> intermediates -> etc
-> public``. Six variant-explicit equations: three exptime-solve, three
snr-solve. Each accepts count *rates* (no SNR baked in) and returns
either an exposure time or an achieved SNR. SNR multiplication on the
noise term happens inside the equation, not at the caller.

Naming convention: ``{exptime,snr}_from_rates_{ayo,exosims_det,exosims_char}``.
The ``_from_rates_`` suffix marks this module as closed-form math; the
rate triple is computed upstream in ``intermediates`` + ``public``.

The finite-exposure fixed-frame functions additionally take detector
hardware parameters. Their variance is in counts, with separate photon,
detector and systematic terms. Exposure planners search bounded frame
counts, either at whole-frame boundaries or within each frame interval.

With ``JAXEDITH_DEBUG_CHECKS=1``, exposure solvers check their division
numerator and denominator for NaNs before applying the below-floor fallback.
SNR solvers and overhead arguments are outside the scope of these checks.
"""

import jax.numpy as jnp

from jaxedith._checks import check_not_nan
from jaxedith._types import FloatOut, Num


def frame_count(t: Num, frame_time: Num) -> FloatOut:
    """Count reads, including a partial final frame; zero time needs no reads.

    Durations and frame times are seconds, with ``t >= 0`` and
    ``frame_time > 0``. Ratios within relative tolerance 1e-12 of a positive
    integer are treated as exact multiples. Every positive sub-frame duration
    incurs one read. The derivative is zero, including at frame boundaries
    (JAX's convention for rounding). Use float64 to resolve the tolerance.
    """
    ratio = jnp.asarray(t) / frame_time
    nearest = jnp.round(ratio)
    exact = (nearest > 0) & (jnp.abs(ratio - nearest) <= 1e-12 * nearest)
    frames = jnp.maximum(jnp.ceil(jnp.where(exact, nearest, ratio)), 1.0)
    return jnp.where(t > 0, frames, 0.0)


def detector_variance_counts(
    t: Num,
    n_pix: Num,
    dark: Num,
    read_noise: Num,
    cic: Num,
    frame_time: Num,
) -> FloatOut:
    """Return aperture dark + read + CIC variance in squared electrons.

    ``dark`` is electrons/pixel/s, ``read_noise`` is RMS electrons/pixel/read,
    and ``cic`` is electrons/pixel/frame. Dark accumulates for ``t`` seconds;
    read variance and CIC are paid once per frame, even for a partial frame.
    This is the independent-noise CCD equation (Howell, Handbook of CCD
    Astronomy, 2nd ed., 2006), with an additional Poisson CIC term per read.
    Inputs must be non-negative and ``frame_time`` strictly positive.
    """
    return n_pix * (dark * t + (read_noise**2 + cic) * frame_count(t, frame_time))


def _variance_components(Cp, Cb, Cnf_rate, t, detector, bg_multiplier):
    """Assemble independent count variances and the correlated floor."""
    planet = Cp * t
    background = bg_multiplier * Cb * t
    systematic = (Cnf_rate * t) ** 2
    return dict(
        planet=planet,
        background=background,
        detector=detector,
        systematic=systematic,
        total=planet + background + detector + systematic,
    )


def _snr_from_count_variance(signal, total):
    """Guard the zero-duration signal/variance ratio before division."""
    return jnp.where(total > 0, signal / jnp.sqrt(jnp.where(total > 0, total, 1)), 0.0)


def snr_fixed_frame(
    Cp: Num,
    Cb: Num,
    Cnf_rate: Num,
    t: Num,
    n_pix: Num,
    dark: Num,
    read_noise: Num,
    cic: Num,
    frame_time: Num,
    *,
    bg_multiplier: Num = 2.0,
) -> FloatOut:
    """Return finite-exposure SNR with fixed frames and zero SNR at ``t=0``.

    ``Cp`` and ``Cb`` are planet and photon-background electron rates;
    ``Cnf_rate`` is leakage / ppfact (zero disables the systematic floor).
    Only photon backgrounds receive ``bg_multiplier``. Detector parameters
    follow :func:`detector_variance_counts`. No overheads are included.
    """
    detector = detector_variance_counts(t, n_pix, dark, read_noise, cic, frame_time)
    components = _variance_components(Cp, Cb, Cnf_rate, t, detector, bg_multiplier)
    return _snr_from_count_variance(components["planet"], components["total"])


def _frame_grid(n_max, *values):
    """Give frame search its own leading axis, preserving broadcast batches."""
    if n_max < 1:
        raise ValueError("n_max must be a positive static integer")
    shape = jnp.broadcast_shapes(*(jnp.shape(value) for value in values))
    return jnp.arange(1, n_max + 1).reshape((n_max,) + (1,) * len(shape))


def exptime_whole_frames(
    target_snr: Num,
    Cp: Num,
    Cb: Num,
    Cnf_rate: Num,
    n_pix: Num,
    dark: Num,
    read_noise: Num,
    cic: Num,
    frame_time: Num,
    *,
    bg_multiplier: Num = 2.0,
    n_max: int = 4096,
) -> FloatOut:
    """Find the smallest whole-frame duration achieving ``target_snr``.

    Search integer frame counts 1 through the static bound ``n_max``.
    Return infinity if no duration within the bound reaches the target,
    including targets at or above the floor ceiling ``Cp / Cnf_rate``.
    A non-positive target needs zero time. Rates and detector parameters
    follow :func:`snr_fixed_frame`; outputs contain no overheads.
    """
    n = _frame_grid(
        n_max,
        target_snr,
        Cp,
        Cb,
        Cnf_rate,
        n_pix,
        dark,
        read_noise,
        cic,
        frame_time,
        bg_multiplier,
    )
    t = n * frame_time
    achieved = snr_fixed_frame(
        Cp,
        Cb,
        Cnf_rate,
        t,
        n_pix,
        dark,
        read_noise,
        cic,
        frame_time,
        bg_multiplier=bg_multiplier,
    )
    feasible = (achieved >= target_snr) & (Cp > target_snr * Cnf_rate)
    result = jnp.min(jnp.where(feasible, t, jnp.inf), axis=0)
    return jnp.where(target_snr <= 0, 0.0, result)


def exptime_min_duration(
    target_snr: Num,
    Cp: Num,
    Cb: Num,
    Cnf_rate: Num,
    n_pix: Num,
    dark: Num,
    read_noise: Num,
    cic: Num,
    frame_time: Num,
    *,
    bg_multiplier: Num = 2.0,
    n_max: int = 4096,
) -> FloatOut:
    """Find the first feasible duration, allowing a partial final frame.

    Solve ``a*t**2 - b*t - c = 0`` in every interval
    ``((n-1)*frame_time, n*frame_time]`` for ascending integer ``n`` through
    static ``n_max``. Here ``a=Cp**2-(target_snr*Cnf_rate)**2``,
    ``b=target_snr**2*(Cp+bg_multiplier*Cb+n_pix*dark)``, and
    ``c=target_snr**2*n_pix*(read_noise**2+cic)*n``. SNR drops at each new
    read, so a continuous bisection across frame boundaries is inappropriate.
    Return infinity when no interval is feasible, zero for non-positive
    targets. Other parameters follow :func:`exptime_whole_frames`.
    """
    n = _frame_grid(
        n_max,
        target_snr,
        Cp,
        Cb,
        Cnf_rate,
        n_pix,
        dark,
        read_noise,
        cic,
        frame_time,
        bg_multiplier,
    )
    a = Cp**2 - (target_snr * Cnf_rate) ** 2
    b = target_snr**2 * (Cp + bg_multiplier * Cb + n_pix * dark)
    c = target_snr**2 * n_pix * (read_noise**2 + cic) * n
    safe_a = jnp.where(a > 0, a, 1.0)
    root = (b + jnp.sqrt(b**2 + 4 * safe_a * c)) / (2 * safe_a)
    # Allow for rounding in the quadratic and in fused versus unfused SNR
    # evaluation. Clamping preserves a feasible exact frame boundary.
    root = root * (1 + 8 * jnp.finfo(root.dtype).eps)
    candidate = jnp.minimum(root, n * frame_time)
    achieved = snr_fixed_frame(
        Cp,
        Cb,
        Cnf_rate,
        candidate,
        n_pix,
        dark,
        read_noise,
        cic,
        frame_time,
        bg_multiplier=bg_multiplier,
    )
    feasible = (a > 0) & (candidate > (n - 1) * frame_time)
    feasible = feasible & (achieved >= target_snr)
    result = jnp.min(jnp.where(feasible, candidate, jnp.inf), axis=0)
    return jnp.where(target_snr <= 0, 0.0, result)


# -- Shared overhead / guard helpers ------------------------------------------


def _apply_overheads(
    t_science: Num, overhead_multi: Num, overhead_fixed: Num, n_rolls: Num
) -> FloatOut:
    """Apply multiplicative + fixed overheads and telescope rolls."""
    t_total = t_science * overhead_multi + overhead_fixed
    t_total = jnp.where(t_total > 0, t_total, jnp.inf)
    return t_total * n_rolls


def _safe_divide(numerator: Num, denominator: Num) -> FloatOut:
    """Divide with NaN-gradient-safe guard for non-positive denominators.

    When ``denominator <= 0`` the result is ``jnp.inf``, but we substitute
    ``1.0`` before dividing so that ``jax.grad`` never sees ``x / 0``.
    """
    numerator = check_not_nan(numerator, "numerator")
    denominator = check_not_nan(denominator, "denominator")
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
    overhead_multi: Num = 1.0,
    overhead_fixed: Num = 0.0,
    n_rolls: Num = 1,
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
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.

    Returns:
        Total exposure time in seconds; ``jnp.inf`` if planet is below floor.
    """
    Cnf = snr * Cnf_rate
    numerator = snr**2 * (Cp + bg_multiplier * Cb)
    denominator = Cp**2 - Cnf**2
    t_science = _safe_divide(numerator, denominator)
    return _apply_overheads(t_science, overhead_multi, overhead_fixed, n_rolls)


def snr_from_rates_ayo(
    Cp: Num,
    Cb: Num,
    Cnf_rate: Num,
    t_obs: Num,
    bg_multiplier: Num = 2.0,
    overhead_multi: Num = 1.0,
    overhead_fixed: Num = 0.0,
    n_rolls: Num = 1,
) -> FloatOut:
    r"""Solve for achieved SNR -- AYO/jaxedith equation.

    Inverse of :func:`exptime_from_rates_ayo`:

    .. math::
        \text{SNR} = \sqrt{\frac{t_{eff} \cdot C_p^2}
            {1 + t_{eff} \cdot C_{nf,rate}^2}}

    where ``t_eff = t_per_roll / (overhead_multi * rate_sum)``.

    Args:
        Cp: Planet signal count rate [e/s].
        Cb: Total background count rate [e/s].
        Cnf_rate: Noise floor *rate* (SNR=1) [e/s].
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


# =============================================================================
# EXOSIMS Detection
# =============================================================================


def exptime_from_rates_exosims_det(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    snr: Num,
    overhead_multi: Num = 1.0,
    overhead_fixed: Num = 0.0,
    n_rolls: Num = 1,
) -> FloatOut:
    r"""Solve for exposure time -- EXOSIMS detection equation.

    .. math::
        t = \text{SNR}^2 \cdot \frac{C_b}{C_p^2 - (\text{SNR} \cdot C_{sp})^2}

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


def snr_from_rates_exosims_det(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    t_obs: Num,
    overhead_multi: Num = 1.0,
    overhead_fixed: Num = 0.0,
    n_rolls: Num = 1,
) -> FloatOut:
    r"""Solve for achieved SNR -- EXOSIMS detection equation.

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


# =============================================================================
# EXOSIMS Characterization
# =============================================================================


def exptime_from_rates_exosims_char(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    snr: Num,
    overhead_multi: Num = 1.0,
    overhead_fixed: Num = 0.0,
    n_rolls: Num = 1,
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


def snr_from_rates_exosims_char(
    Cp: Num,
    Cb: Num,
    Csp: Num,
    t_obs: Num,
    overhead_multi: Num = 1.0,
    overhead_fixed: Num = 0.0,
    n_rolls: Num = 1,
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
