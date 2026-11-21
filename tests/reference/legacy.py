"""Convention adapters for the scalar comparison dialects.

Rate arithmetic lives in primitives.py and the AYO/EXOSIMS equations in
etc.py. Their plain-number inputs and electron units remain unchanged.
These adapters never supply an alternative optical-path rate derivation.
"""

import jax.numpy as jnp
from optixstuff import FramePlan, PostProcessing, frames_for_time

from reference.primitives import detector_noise_rate, photon_counting_time


def post_processing_from_pyedith(ppfact) -> PostProcessing:
    """Convert pyEDITH's divisor to a speckle residual fraction.

    The disk residual is zero and the background variance is counted twice, as
    pyEDITH's ``CRb_multiplier`` does.
    """
    return PostProcessing(1 / jnp.asarray(ppfact), 0.0, background_variance_factor=2.0)


def post_processing_from_exosims(ppFact) -> PostProcessing:
    """Read EXOSIMS ppFact as the residual fraction itself; disk is zero."""
    return PostProcessing(ppFact, 0.0, background_variance_factor=1.0)


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
