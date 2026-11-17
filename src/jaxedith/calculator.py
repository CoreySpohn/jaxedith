"""Aperture S/N and whole-frame exposure planning through detector statistics."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.spectral import Quadrature
from optixstuff import FramePlan, frames_for_time
from optixstuff.detector import AbstractDetector

from jaxedith._checks import check_not_nan
from jaxedith._types import FloatOut, Num
from jaxedith.public import ApertureRates


class PostProcessing(eqx.Module):
    """Residual fractions in [0, 1]; zero means perfect subtraction."""

    speckle_residual_fraction: Num = eqx.field(converter=jnp.asarray)
    exozodi_residual_fraction: Num = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Reject concrete nonfinite or out-of-range fractions."""
        for value in (self.speckle_residual_fraction, self.exozodi_residual_fraction):
            if not isinstance(value, jax.core.Tracer) and not bool(
                jnp.all(jnp.isfinite(value) & (value >= 0) & (value <= 1))
            ):
                raise ValueError("residual fractions must be finite and in [0, 1]")


def _floor_sq(electrons, post_processing):
    """Squared residual electron rate, with independent speckle and disk terms."""
    return (post_processing.speckle_residual_fraction * electrons.star_leakage) ** 2 + (
        post_processing.exozodi_residual_fraction * electrons.exozodi
    ) ** 2


def _count_statistics(rates, detector, quadrature, plan):
    """Return mean source electrons/s and independent count variance."""
    electrons = detector.electrons(rates.mean, quadrature)
    variance_rates = detector.electrons(rates.variance, quadrature)
    _, variance = detector.mean_and_variance(
        electrons,
        plan,
        rates.n_pix,
        variance_rates=variance_rates,
        n_pix_variance=rates.n_pix_variance,
    )
    return electrons, variance


def snr(
    rates: ApertureRates,
    detector: AbstractDetector,
    quadrature: Quadrature,
    plan: FramePlan,
    *,
    post_processing: PostProcessing,
) -> FloatOut:
    """Return S/N per band, shape (W,), for aperture photon integrands.

    Signal is planet electrons times live time. The detector supplies all
    independent count variance, including source shot noise, dark, CIC and
    read noise. Residual speckles and disk light add squared count terms.
    Zero live time gives zero S/N. Infinite plans return the asymptotic
    ceiling (infinity for a positive signal without a systematic floor).
    """
    finite = jnp.isfinite(plan.n_frames)
    safe_plan = FramePlan(plan.frame_time_s, jnp.where(finite, plan.n_frames, 0))
    electrons, variance = _count_statistics(rates, detector, quadrature, safe_plan)
    floor_sq = _floor_sq(electrons, post_processing)
    variance = variance + floor_sq * safe_plan.live_time_s**2
    signal = electrons.planet * safe_plan.live_time_s
    # Only an exact zero is replaced; a NaN variance or floor propagates.
    result = signal / jnp.sqrt(jnp.where(variance == 0, 1, variance))
    ceiling = electrons.planet / jnp.sqrt(jnp.where(floor_sq == 0, 1, floor_sq))
    ceiling = jnp.where(floor_sq == 0, jnp.inf, ceiling)
    ceiling = jnp.where(electrons.planet > 0, ceiling, 0)
    return jnp.where(finite, result, ceiling)


def exposure_time(
    target_snr: Num,
    rates: ApertureRates,
    detector: AbstractDetector,
    quadrature: Quadrature,
    *,
    frame_time_s: Num,
    post_processing: PostProcessing,
) -> FramePlan:
    """Return the smallest whole-frame plan reaching each band's target S/N.

    Let p be planet electrons/s, b the detector variance for one frame
    divided by its live seconds, and f2 the sum of squared residual rates.
    Then S/N^2 = p^2 t / (b + f2 t), so the continuous solution is
    t = target_snr^2 b / (p^2 - target_snr^2 f2). Read and CIC noise
    are included exactly in b because every frame is fully collected.
    A positive target with p <= target_snr sqrt(f2) is infeasible: return
    a FramePlan with infinite frame count/live time. Nonpositive targets
    require zero frames. Only optixstuff.frames_for_time rounds frames.
    """
    one_frame = frames_for_time(frame_time_s, frame_time_s)
    electrons, variance = _count_statistics(rates, detector, quadrature, one_frame)
    b = variance / frame_time_s
    floor_sq = _floor_sq(electrons, post_processing)
    denominator = check_not_nan(
        electrons.planet**2 - target_snr**2 * floor_sq, "exposure denominator"
    )
    numerator = check_not_nan(target_snr**2 * b, "exposure numerator")
    # Feasibility compares rates, not a difference of squares, so a target
    # exactly at the speckle ceiling is infeasible despite rounding.
    feasible = (electrons.planet > target_snr * jnp.sqrt(floor_sq)) & (denominator > 0)
    time_s = numerator / jnp.where(feasible, denominator, 1)
    time_s = jnp.where(feasible, time_s, jnp.inf)
    return frames_for_time(jnp.where(target_snr <= 0, 0, time_s), frame_time_s)
