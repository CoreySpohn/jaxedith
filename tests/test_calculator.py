"""Detector-owned variance and analytic exposure planning."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from reference.legacy import post_processing_from_exosims, post_processing_from_pyedith

from jaxedith import PostProcessing, exposure_time, snr


@vv.case("aperture-photon-budget", "code-verification")
def test_post_processing_count_variance(example):
    """Fails if residual fractions multiply variance or omit live-time squared.

    Basis: exact algebra with 1e-12 relative float64 rounding allowance.
    """
    detector, rates = example
    q = rates.quadrature
    rates = eqx.tree_at(
        lambda r: (r.star_leakage, r.exozodi),
        rates,
        (jnp.full(q.nodes_nm.shape, 0.001), jnp.full(q.nodes_nm.shape, 0.002)),
    )
    e = detector.electrons(rates)
    plan = ox.FramePlan(100.0, 36)
    _, base = detector.mean_and_variance(e, plan)
    for f, g in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.2)):
        actual = snr(
            rates,
            detector,
            plan,
            post_processing=PostProcessing(f, g),
        )
        expected = base + (f * e.star_leakage * 3600) ** 2 + (g * e.exozodi * 3600) ** 2
        np.testing.assert_allclose(
            (e.planet * 3600 / actual) ** 2, expected, rtol=1e-12, atol=0
        )
    a, b = post_processing_from_pyedith(10), post_processing_from_exosims(0.1)
    np.testing.assert_array_equal(
        a.speckle_residual_fraction, b.speckle_residual_fraction
    )
    for bad in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PostProcessing(bad, 0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            PostProcessing(0, bad)


@vv.case("aperture-photon-budget", "code-verification")
def test_exposure_inverts_snr_and_rejects_floor(example):
    """Fails if the planner misses whole-frame boundaries or floor infeasibility.

    Basis: exact inverse algebra; integer frame equality uses upstream rounding.
    """
    detector, rates = example
    q = rates.quadrature
    rates = eqx.tree_at(
        lambda r: r.star_leakage, rates, jnp.full(q.nodes_nm.shape, 1e-4)
    )
    pp = PostProcessing(0.1, 0.2)
    counts = jnp.array([1.0, 7.0, 36.0, 100.0])
    target = jax.vmap(
        lambda n: snr(
            rates,
            detector,
            ox.FramePlan(100.0, n),
            post_processing=pp,
        )[0]
    )(counts)

    def solve(t):
        return exposure_time(t, rates, detector, frame_time_s=100.0, post_processing=pp)

    plans = jax.jit(jax.vmap(solve))(target)
    np.testing.assert_array_equal(plans.n_frames[:, 0], counts)
    plan = solve(9.5)
    assert snr(rates, detector, plan, post_processing=pp)[0] >= 9.5
    previous = ox.frames_for_time(plan.live_time_s - 100.0, 100.0)
    assert snr(rates, detector, previous, post_processing=pp)[0] < 9.5
    e = detector.electrons(rates)
    ceiling = e.planet / (0.1 * e.star_leakage)
    assert jnp.isinf(solve(ceiling * 1.001).live_time_s).all()
    assert (solve(0.0).n_frames == 0).all()
    # A target exactly at the speckle ceiling is infeasible despite rounding,
    # and a NaN background reaches S/N as NaN instead of a valid-looking value.
    exact = snr(
        rates,
        detector,
        ox.FramePlan(100.0, jnp.inf),
        post_processing=pp,
    )
    assert jnp.isinf(solve(exact[0]).live_time_s).all()
    unknown = eqx.tree_at(
        lambda r: r.variance.local_zodi, rates, jnp.full(q.nodes_nm.shape, jnp.nan)
    )
    plan = ox.FramePlan(100.0, 36.0)
    assert jnp.isnan(snr(unknown, detector, plan, post_processing=pp)).all()
