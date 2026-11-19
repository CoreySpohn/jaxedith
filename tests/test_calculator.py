"""Detector-owned variance and analytic exposure planning."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from optixstuff.grids import Measure
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
    _, path, rates = example
    q = rates.quadrature
    rates = eqx.tree_at(
        lambda r: (r.star_leakage, r.exozodi),
        rates,
        (jnp.full(q.nodes_nm.shape, 0.001), jnp.full(q.nodes_nm.shape, 0.002)),
    )
    e = path.detector.electrons(rates)
    plan = ox.FramePlan(100.0, 36)
    _, base = path.detector.mean_and_variance(e, plan)
    for f, g in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.2)):
        actual = snr(
            rates,
            path.detector,
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
    _, path, rates = example
    q = rates.quadrature
    rates = eqx.tree_at(
        lambda r: r.star_leakage, rates, jnp.full(q.nodes_nm.shape, 1e-4)
    )
    pp = PostProcessing(0.1, 0.2)
    counts = jnp.array([1.0, 7.0, 36.0, 100.0])
    target = jax.vmap(
        lambda n: snr(
            rates,
            path.detector,
            ox.FramePlan(100.0, n),
            post_processing=pp,
        )[0]
    )(counts)

    def solve(t):
        return exposure_time(
            t, rates, path.detector, frame_time_s=100.0, post_processing=pp
        )

    plans = jax.jit(jax.vmap(solve))(target)
    np.testing.assert_array_equal(plans.n_frames[:, 0], counts)
    plan = solve(9.5)
    assert snr(rates, path.detector, plan, post_processing=pp)[0] >= 9.5
    previous = ox.frames_for_time(plan.live_time_s - 100.0, 100.0)
    assert snr(rates, path.detector, previous, post_processing=pp)[0] < 9.5
    e = path.detector.electrons(rates)
    ceiling = e.planet / (0.1 * e.star_leakage)
    assert jnp.isinf(solve(ceiling * 1.001).live_time_s).all()
    assert (solve(0.0).n_frames == 0).all()
    # A target exactly at the speckle ceiling is infeasible despite rounding,
    # and a NaN background reaches S/N as NaN instead of a valid-looking value.
    exact = snr(
        rates,
        path.detector,
        ox.FramePlan(100.0, jnp.inf),
        post_processing=pp,
    )
    assert jnp.isinf(solve(exact[0]).live_time_s).all()
    unknown = eqx.tree_at(
        lambda r: r.variance.local_zodi, rates, jnp.full(q.nodes_nm.shape, jnp.nan)
    )
    plan = ox.FramePlan(100.0, 36.0)
    assert jnp.isnan(snr(unknown, path.detector, plan, post_processing=pp)).all()


@vv.case("aperture-photon-budget", "code-verification")
def test_reader_jit_vmap_and_scene_gradient(example):
    """Fails if JAX transformations lose scene dependence or use wrong derivatives.

    Basis: 1e-12 eager/JIT algebra; central difference h=1e-4 truncation
    is O(h^2), allowing 1e-7 relative for a smooth flux-dependent S/N.
    """
    system, path, reference = example
    q = reference.quadrature
    pp = PostProcessing(0.0, 0.0)

    def evaluate(scale, time):
        scene = eqx.tree_at(lambda s: s.star.flux_jy, system, 36.31 * scale)
        rates = aperture_rates(scene, path, q, time_d=time, planet_index=0, zodi=None)
        return snr(rates, path.detector, ox.FramePlan(100, 36), post_processing=pp)[0]

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
            system, path, q, time_d=0.0, planet_index=index, zodi=None
        )
        return rates.planet

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
    """Check weighted exposure against pixels on a sloped leakage field.

    The four-sigma sample-variance band uses the fourth Poisson cumulant
    and the exact sampling variance of the unbiased variance estimator.
    """
    from conftest import FlatCompanions, TabulatedCoronagraph, make_path, make_system
    from hwoutils.conversions import arcsec_to_rad
    from hwoutils.planck import planck_photon_radiance
    from optixstuff.apertures import Circular, Matched
    from optixstuff.grids import Image, PixelGrid
    from optixstuff.performance import performance_curves
    from optixstuff.spectral import Band
    from skyscapes.scene import FlatStar

    path = make_path(chromatic=False)
    frame = path.primary.frame(550.0)
    grid = PixelGrid.focal(9, 0.4, frame)
    x, y = grid.coordinates()
    offset = 0.4
    gaussian = jnp.exp(-((x - offset) ** 2 + y**2) / (2 * 0.42**2))
    psf = 0.25 * gaussian / gaussian.sum()
    star = 2e-10 * jnp.exp(-3 * (x - offset))
    sky = 0.5
    stellar_image = Image(star, grid, (), Measure.integral(), "fraction")

    class PixelTable(TabulatedCoronagraph):
        def _native_stellar_map(self, stellar_diam_lod):
            return stellar_image

        def _native_source_psfs(self, x_lod, y_lod):
            planes = jnp.broadcast_to(psf, (jnp.size(x_lod), *grid.shape))
            return ox.SourcePSFs(
                Image(planes, grid, ("source",), Measure.integral(), "fraction"),
                jnp.isfinite(x_lod) & jnp.isfinite(y_lod),
            )

    record = performance_curves(
        lambda separation: Image(psf, grid, (), Measure.integral(), "fraction"),
        stellar_image,
        Circular(0.7),
        separations_lod=[offset],
    )
    table = PixelTable(
        grid,
        Image(jnp.full(grid.shape, sky), grid, (), Measure.intensive(), "transmission"),
        record,
    )
    scale = frame.lod_to_arcsec(550.0)
    # Four detector pixels per table pixel. Two of the electrons per
    # detector pixel per second arise from CIC at the chosen frame time.
    frame_time = 2.0
    detector = ox.Detector(
        pixel_scale_arcsec=scale * grid.step / 2,
        shape=grid.shape,
        qe=0.8,
        survival_fraction=0.5,
        dark_current_e_per_pixel_s=1.0,
        clock_induced_charge_e_per_pixel_frame=4.0,
        read_noise_e_rms=1.5,
    )
    path = eqx.tree_at(lambda p: (p.coronagraph, p.detector), path, (table, detector))
    system = make_system(FlatStar(1e11), chromatic=False, disk=None)
    system = eqx.tree_at(lambda s: s.dist_pc, system, 1 / (offset * scale))
    band = Band(549.5, 550.5)
    solid_angle = grid.pixel_area * arcsec_to_rad(scale) ** 2
    # Two local-zodi, one companion, and three thermal photons per native
    # pixel per second. Thermal bypasses both throughput and sky transmission.
    radiance = 2.0 / (40 * 0.4 * sky * solid_angle)
    companion_density = 1.0 / (40 * 0.4 * sky * grid.pixel_area)
    emissivity = 3.0 / (40 * planck_photon_radiance(550.0, 1800.0) * solid_angle)

    class FlatSky:
        def spectral_breaks_nm(self):
            return np.empty(0)

        def photon_radiance(self, wave, **kwargs):
            return jnp.full_like(wave, radiance)

    kwargs = dict(
        time_d=0.0,
        planet_index=0,
        zodi=FlatSky(),
        temperature_K=1800.0,
        eps_warm_T_cold=emissivity,
        companion_flux_density=FlatCompanions(companion_density),
        quadrature_order=1,
    )
    rates = aperture_rates(
        system, path, band, rule=Matched(), frame_time_s=frame_time, **kwargs
    )
    plan = ox.FramePlan(frame_time, 1)
    pp = PostProcessing(0, 0)
    calculated = snr(rates, detector, plan, post_processing=pp)[0]

    # Independently form per-native-pixel electron rates before applying
    # any aperture weights. The planet never enters the optimized weights.
    star_incident = 1e11 * 40 * 0.4 * 0.8 * 0.5
    planet = star_incident * 1e-10 * np.asarray(psf)
    poisson = (
        star_incident * np.asarray(star) + (2 + 1 + 3) * 0.8 * 0.5 + 4 * (1 + 4 / 2)
    )
    read_variance = 4 * 1.5**2
    background = poisson + read_variance / frame_time
    weights = np.asarray(psf) / background
    np.testing.assert_allclose(rates.n_pix, 4 * weights.sum(), rtol=1e-12)
    np.testing.assert_allclose(rates.variance.n_pix, 4 * (weights**2).sum(), rtol=1e-12)
    signal = np.sum(weights * planet)
    variance = np.sum(weights**2 * (planet + background))
    np.testing.assert_allclose(calculated, signal * np.sqrt(frame_time / variance))

    draws = 8192
    rng = np.random.default_rng(0)
    counts = rng.poisson(frame_time * poisson, (draws, *grid.shape))
    counts = counts + rng.normal(0, np.sqrt(read_variance), counts.shape)
    sample_variance = np.var(np.sum(counts * weights, axis=(1, 2)), ddof=1)
    expected = frame_time * np.sum(weights**2 * background)
    cumulant4 = frame_time * np.sum(weights**4 * poisson)
    se = np.sqrt(cumulant4 / draws + 2 * expected**2 / (draws - 1))
    assert abs(sample_variance - expected) < 4 * se

    white_snr = np.sum(psf * planet) * np.sqrt(
        frame_time / np.sum(psf**2 * (background + planet))
    )
    circular = aperture_rates(system, path, band, rule=Circular(0.7), **kwargs)
    circular_snr = snr(circular, detector, plan, post_processing=pp)[0]
    assert calculated > white_snr
    assert calculated > circular_snr
    target = 10.0
    plan = exposure_time(
        target, rates, detector, frame_time_s=frame_time, post_processing=pp
    )
    np.testing.assert_array_equal(
        plan.n_frames, [np.ceil(target**2 * variance / (frame_time * signal**2))]
    )
