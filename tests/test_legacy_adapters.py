"""Cadence and noise conventions of the scalar comparison dialects."""

import ineedvalidation as vv
import numpy as np
import optixstuff as ox
from reference.legacy import frame_plan_photon_counting, legacy_pyedith_noise_rate


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
def test_legacy_photon_counting_cadence_and_noise_arithmetic():
    """Fails if the 6.73 rule or the historical noise arithmetic drifts.

    Basis: the 6.73 rule and historical noise arithmetic use 1e-12 rounding.
    """
    plan = frame_plan_photon_counting(0.02, time_s=100.0)
    np.testing.assert_allclose(plan.frame_time_s * 0.02, 1 / 6.73, rtol=1e-12)
    assert plan.n_frames == ox.frames_for_time(100.0, 1 / (6.73 * 0.02)).n_frames
    noise = legacy_pyedith_noise_rate(
        0.02,
        n_pix=12,
        dark_current_e_per_pixel_s=0.0003,
        read_noise_e_rms=0.2,
        read_frame_time_s=100.0,
        clock_induced_charge_e_per_pixel_frame=0.005,
    )
    np.testing.assert_allclose(
        noise, 12 * (0.0003 + 0.2**2 / 100 + 0.005 * 6.73 * 0.02), rtol=1e-12
    )
