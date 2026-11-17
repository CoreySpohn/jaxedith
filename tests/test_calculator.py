"""Detector-owned variance and analytic exposure planning."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from reference.legacy import post_processing_from_exosims, post_processing_from_pyedith

from jaxedith import (
    PostProcessing,
    aperture_rates,
    exposure_time,
    snr,
)


@vv.case("aperture-photon-budget", "code-verification")
def test_post_processing_count_variance(example):
    """Fails if residual fractions multiply variance or omit live-time squared.

    Basis: exact algebra with 1e-12 relative float64 rounding allowance.
    """
    _, path, _, q, rates = example
    rates = eqx.tree_at(
        lambda r: (r.mean.star_leakage, r.mean.exozodi),
        rates,
        (jnp.full(q.nodes_nm.shape, 0.001), jnp.full(q.nodes_nm.shape, 0.002)),
    )
    e = path.detector.electrons(rates.mean, q)
    plan = ox.FramePlan(100.0, 36)
    _, base = path.detector.mean_and_variance(
        e,
        plan,
        rates.n_pix,
        variance_rates=path.detector.electrons(rates.variance, q),
        n_pix_variance=rates.n_pix_variance,
    )
    for f, g in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.2)):
        actual = snr(
            rates,
            path.detector,
            q,
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
    _, path, _, q, rates = example
    rates = eqx.tree_at(
        lambda r: r.mean.star_leakage, rates, jnp.full(q.nodes_nm.shape, 1e-4)
    )
    pp = PostProcessing(0.1, 0.2)
    counts = jnp.array([1.0, 7.0, 36.0, 100.0])
    target = jax.vmap(
        lambda n: snr(
            rates,
            path.detector,
            q,
            ox.FramePlan(100.0, n),
            post_processing=pp,
        )[0]
    )(counts)

    def solve(t):
        return exposure_time(
            t, rates, path.detector, q, frame_time_s=100.0, post_processing=pp
        )

    plans = jax.jit(jax.vmap(solve))(target)
    np.testing.assert_array_equal(plans.n_frames[:, 0], counts)
    plan = solve(9.5)
    assert snr(rates, path.detector, q, plan, post_processing=pp)[0] >= 9.5
    previous = ox.frames_for_time(plan.live_time_s - 100.0, 100.0)
    assert snr(rates, path.detector, q, previous, post_processing=pp)[0] < 9.5
    e = path.detector.electrons(rates.mean, q)
    ceiling = e.planet / (0.1 * e.star_leakage)
    assert jnp.isinf(solve(ceiling * 1.001).live_time_s).all()
    assert (solve(0.0).n_frames == 0).all()
    # A target exactly at the speckle ceiling is infeasible despite rounding,
    # and a NaN background reaches S/N as NaN instead of a valid-looking value.
    exact = snr(
        rates,
        path.detector,
        q,
        ox.FramePlan(100.0, jnp.inf),
        post_processing=pp,
    )
    assert jnp.isinf(solve(exact[0]).live_time_s).all()
    unknown = eqx.tree_at(
        lambda r: r.variance.local_zodi, rates, jnp.full(q.nodes_nm.shape, jnp.nan)
    )
    plan = ox.FramePlan(100.0, 36.0)
    assert jnp.isnan(snr(unknown, path.detector, q, plan, post_processing=pp)).all()


@vv.case("aperture-photon-budget", "code-verification")
def test_reader_jit_vmap_and_scene_gradient(example):
    """Fails if JAX transformations lose scene dependence or use wrong derivatives.

    Basis: 1e-12 eager/JIT algebra; central difference h=1e-4 truncation
    is O(h^2), allowing 1e-7 relative for a smooth flux-dependent S/N.
    """
    system, path, band, q, _ = example
    pp = PostProcessing(0.0, 0.0)

    def evaluate(scale, time):
        scene = eqx.tree_at(lambda s: s.star.flux_jy, system, 36.31 * scale)
        rates = aperture_rates(
            scene, path, band, time_d=time, planet_index=0, quadrature=q, zodi=None
        )
        return snr(rates, path.detector, q, ox.FramePlan(100, 36), post_processing=pp)[
            0
        ]

    np.testing.assert_allclose(
        jax.jit(evaluate)(1.0, 0.0), evaluate(1.0, 0.0), rtol=1e-12, atol=0
    )
    times = jnp.array([0.0, 30.0, 60.0])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate, in_axes=(None, 0)))(1.0, times),
        jnp.stack([evaluate(1.0, t) for t in times]),
        rtol=1e-12,
        atol=0,
    )

    def planet_rate(index):
        rates = aperture_rates(
            system, path, band, time_d=0.0, planet_index=index, quadrature=q, zodi=None
        )
        return rates.mean.planet

    # A traced out-of-range index gives NaN, not the last planet's rates.
    by_index = jax.vmap(planet_rate)(jnp.array([0, system.n_planets]))
    assert jnp.isfinite(by_index[0]).all()
    assert jnp.isnan(by_index[1]).all()
    step = 1e-4
    finite_difference = (evaluate(1 + step, 0.0) - evaluate(1 - step, 0.0)) / (2 * step)
    np.testing.assert_allclose(
        jax.grad(evaluate)(1.0, 0.0), finite_difference, rtol=1e-7, atol=0
    )


@vv.case("aperture-photon-budget", "code-verification")
def test_matched_snr_agrees_with_weighted_poisson_draws():
    """Compare calculated S/N with independent weighted Poisson pixel sums.

    Estimate signal and noise from 8192 realizations. Four asymptotic
    standard errors of their ratio use the third and fourth cumulants of
    independent weighted Poisson counts.
    """
    from conftest import TabulatedCoronagraph, make_path, make_system
    from hwoutils.grids import Image, PixelGrid
    from hwoutils.spectral import Band, Quadrature
    from optixstuff.apertures import Matched
    from optixstuff.performance import performance_curves
    from skyscapes.scene import FlatStar

    path = make_path(chromatic=False)
    frame = path.primary.frame(550.0)
    grid = PixelGrid.focal(3, 1.0, frame)
    psf = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 64
    star = np.full(grid.shape, 1e-12)
    rule = Matched()
    record = performance_curves(
        lambda separation: Image(jnp.array(psf), grid, (), "per_pixel", "fraction"),
        Image(jnp.array(star), grid, (), "per_pixel", "fraction"),
        rule,
        separations_lod=[0.5],
    )
    table = TabulatedCoronagraph(
        grid,
        Image(jnp.ones(grid.shape), grid, (), "per_unit_area", "transmission"),
        None,
        record,
    )
    scale = frame.lod_to_arcsec(550.0)
    detector = ox.Detector(
        pixel_scale_arcsec=scale, shape=grid.shape, dark_current_e_per_pixel_s=3.0
    )
    path = eqx.tree_at(lambda p: (p.coronagraph, p.detector), path, (table, detector))
    system = make_system(FlatStar(1e11), chromatic=False)
    system = eqx.tree_at(lambda s: s.dist_pc, system, 1 / (0.5 * scale))
    band = Band(549.5, 550.5)
    quad = Quadrature.build((band,), None, 1)
    rates = aperture_rates(
        system,
        path,
        band,
        time_d=0.0,
        planet_index=0,
        quadrature=quad,
        rule=rule,
        zodi=None,
        exozodi=None,
    )
    calculated = snr(
        rates,
        detector,
        quad,
        ox.FramePlan(1.0, 1),
        post_processing=PostProcessing(0, 0),
    )[0]

    # Pixel means for a one-second, one-nm observation with unity QE.
    weights = psf / psf.max()
    star_incident = 1e11 * 40 * 0.4
    planet = star_incident * 1e-10 * psf
    pixel_means = planet + star_incident * star + 3.0
    signal = np.sum(weights * planet)
    variance = np.sum(weights**2 * pixel_means)
    draws = 8192
    pixels = np.random.default_rng(0).poisson(pixel_means, (draws, *grid.shape))
    counts = np.sum(weights * pixels, axis=(1, 2))
    background = np.sum(weights * (star_incident * star + 3.0))
    empirical = (counts.mean() - background) / counts.std(ddof=1)
    # Delta method for the sample mean divided by the sample SD after
    # background subtraction: their covariance comes from the third cumulant.
    expected_snr = signal / np.sqrt(variance)
    skew = np.sum(weights**3 * pixel_means) / variance**1.5
    excess = np.sum(weights**4 * pixel_means) / variance**2
    se = np.sqrt((1 + expected_snr**2 * (2 + excess) / 4 - expected_snr * skew) / draws)
    assert abs(calculated - empirical) < 4 * se
    # The former noise-equivalent area assigned to dark, plus mean planet
    # and stellar shot noise, is outside that same confidence interval.
    old_variance = np.sum(weights * (planet + star_incident * star)) + (
        3.0 * weights.sum() ** 2 / np.sum(weights**2)
    )
    assert abs(signal / np.sqrt(old_variance) - empirical) > 4 * se
    target = 10.0
    plan = exposure_time(
        target,
        rates,
        detector,
        quad,
        frame_time_s=1.0,
        post_processing=PostProcessing(0, 0),
    )
    np.testing.assert_array_equal(
        plan.n_frames, [np.ceil(target**2 * variance / signal**2)]
    )
