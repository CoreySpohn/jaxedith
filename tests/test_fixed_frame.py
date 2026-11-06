"""Finite-frame regressions absent from the legacy rate-equation tests."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
import test_intermediates
import test_stellar_diameter
from coronalyze import PPConfig

from jaxedith import (
    ETCScene,
    count_rates_ayo,
    count_rates_from_system_ayo,
    count_variance_components,
    count_variance_components_ayo,
    detector_variance_counts,
    exptime_ayo,
    exptime_from_system_ayo,
    exptime_min_duration,
    exptime_whole_frames,
    frame_count,
    intermediates,
    snr_ayo,
    snr_fixed_frame,
    snr_from_system_ayo,
)
from jaxedith._checks import DEBUG_CHECKS

scalar_optical_path = test_intermediates.scalar_optical_path
system = test_stellar_diameter.system


@pytest.fixture
def fixed_path(scalar_optical_path):
    return eqx.tree_at(
        lambda p: (
            p.detector.dark_current_rate_e_per_s,
            p.detector.read_noise_e,
            p.detector.clock_induced_charge_rate_e_per_frame,
            p.detector.frame_time_s,
        ),
        scalar_optical_path,
        (0.03, 4.0, 0.002, 20.0),
    )


@pytest.fixture
def scene():
    return ETCScene(
        F0=1e8,
        Fs_over_F0=0.01,
        Fp_over_Fs=1e-7,
        Fzodi=1e-9,
        Fexozodi=2e-9,
        Fbinary=3e-9,
        dist_pc=10.0,
        sep_arcsec=0.2,
    )


def test_counts_and_independent_detector_terms():
    """Fails if dark is frame-rounded or a partial read/CIC is omitted."""
    with jax.enable_x64():
        t = jnp.array([0.0, 0.001, 5.0, 20.0, 60.0, 61.0])
        frames = jnp.array([0.0, 1.0, 1.0, 1.0, 3.0, 4.0])
        np.testing.assert_array_equal(frame_count(t, 20.0), frames)
        dark = detector_variance_counts(t, 2.0, 0.1, 0.0, 0.0, 20.0)
        read = detector_variance_counts(t, 2.0, 0.0, 3.0, 0.0, 20.0)
        cic = detector_variance_counts(t, 2.0, 0.0, 0.0, 0.4, 20.0)
        np.testing.assert_allclose(dark, [0.0, 0.0002, 1.0, 4.0, 12.0, 12.2])
        np.testing.assert_array_equal(read, [0.0, 18.0, 18.0, 18.0, 54.0, 72.0])
        np.testing.assert_allclose(cic, [0.0, 0.8, 0.8, 0.8, 2.4, 3.2])
        np.testing.assert_allclose(
            detector_variance_counts(t, 2.0, 0.1, 3.0, 0.4, 20.0), dark + read + cic
        )
        near = jnp.array([1e-20, 60.0 * (1 + 5e-13), 60.0 * (1 + 2e-12)])
        np.testing.assert_array_equal(frame_count(near, 20.0), [1.0, 3.0, 4.0])
        assert frame_count(0.07, 0.01) == 7
        assert (
            frame_count(1e-200, 1e200) == 1
        )  # Ratio underflows, duration is positive.
        np.testing.assert_array_equal(
            jax.vmap(jax.grad(frame_count), in_axes=(0, None))(t, 20.0), 0.0
        )
        assert snr_fixed_frame(1.0, 2.0, 0.0, 0.0, 2.0, 0.1, 3.0, 0.4, 20.0) == 0


@pytest.mark.parametrize("x64", [False, True])
def test_pinned_legacy_values(fixed_path, scene, x64):
    """Fails if the default or explicit legacy cadence changes a recorded bit."""
    with jax.enable_x64(x64):
        expected = (
            [
                2.827433388230814,
                52.2282749531881,
                0.06283185307179585,
                2.724754284930627,
                339.69188041482425,
            ]
            if x64
            else [
                2.827433388230814,
                52.228275299072266,
                0.06283185631036758,
                2.7247543334960938,
                339.69189453125,
            ]
        )
        if not x64 and DEBUG_CHECKS:
            expected[0] = 2.8274333477020264
        args = (fixed_path, scene, 500.0, 5.0, 20.0)
        for options in ({}, dict(cadence="photon_counting_ayo")):
            actual = [
                *count_rates_ayo(*args, ppfact=30.0, **options),
                snr_ayo(*args, 100.0, ppfact=30.0, **options),
                exptime_ayo(*args, 5.0, ppfact=30.0, **options),
            ]
            np.testing.assert_array_equal([float(v) for v in actual], expected)


def test_scene_variance_and_cic_source_independence(fixed_path, scene, monkeypatch):
    """Fails if fixed CIC depends on source rate, or k multiplies detector noise."""

    def forbidden(*args, **kwargs):
        pytest.fail("fixed frames called photon_counting_time")

    monkeypatch.setattr(intermediates, "photon_counting_time", forbidden)
    with jax.enable_x64():
        args = (fixed_path, scene, 500.0, 5.0, 20.0)
        kwargs = dict(cadence="fixed_frame", ppfact=30.0, bg_multiplier=3.0)
        budget = count_variance_components_ayo(*args, 61.0, **kwargs)
        # Six aperture pixels: 2 (lambda/D)^2, one pixel/(lambda/D)^2,
        # and three channels. Four reads include the final one-second frame.
        expected_detector = 6 * (0.03 * 61 + (16 + 0.002) * 4)
        assert float(budget["detector"]) == pytest.approx(expected_detector)
        Cp, Cb, Cnf = count_rates_ayo(*args, cadence="fixed_frame", ppfact=30.0)
        np.testing.assert_allclose(budget["planet"], Cp * 61)
        np.testing.assert_allclose(budget["background"], 3 * Cb * 61)
        leakage = intermediates.stellar_leakage(
            fixed_path, 500.0, 5.0, 20.0, 1e8, 0.01, 3.0
        )
        np.testing.assert_allclose(Cnf, leakage / 30)
        np.testing.assert_allclose(budget["systematic"], (leakage * 61 / 30) ** 2)
        np.testing.assert_allclose(
            snr_fixed_frame(
                Cp, Cb, Cnf, 61.0, 6.0, 0.03, 4.0, 0.002, 20.0, bg_multiplier=3.0
            ),
            budget["planet"] / jnp.sqrt(budget["total"]),
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            snr_ayo(*args, 61.0, **kwargs), budget["planet"] / jnp.sqrt(budget["total"])
        )
        brighter = eqx.tree_at(lambda s: s.Fp_over_Fs, scene, 1e-4)
        other = count_variance_components_ayo(
            fixed_path, brighter, 500.0, 5.0, 20.0, 61.0, **kwargs
        )
        np.testing.assert_array_equal(other["detector"], budget["detector"])
        off = count_variance_components_ayo(
            *args, 61.0, cadence="fixed_frame", ppfact=jnp.inf
        )
        assert off["systematic"] == 0
        np.testing.assert_allclose(
            off["total"], Cp * 61 + 2 * Cb * 61 + expected_detector
        )


def test_textbook_snr_floor_regression():
    """Fails if the systematic term is missing from the 10 versus 8.94 case."""
    # Howell (2006), CCD equation: signal=100, shot variance=100;
    # a correlated residual of 0.5 e/s adds 25 e^2 after ten seconds.
    assert (
        float(snr_fixed_frame(10.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 20.0)) == 10.0
    )
    np.testing.assert_allclose(
        snr_fixed_frame(10.0, 0.0, 0.5, 10.0, 1.0, 0.0, 0.0, 0.0, 20.0),
        100 / np.sqrt(125),
        rtol=1e-7,
    )


@pytest.mark.parametrize("planner", [exptime_whole_frames, exptime_min_duration])
@pytest.mark.parametrize("x64", [False, True])
def test_solver_feasibility_and_minimality(planner, x64):
    """Fails if a root misses a target or skips an earlier feasible frame."""
    with jax.enable_x64(x64):
        rng = np.random.default_rng(0)
        rates = (
            jnp.asarray(rng.uniform(1.0, 3.0, 24)),
            jnp.asarray(rng.uniform(0.0, 5.0, 24)),
            0.02,
        )
        hardware = (4.0, 0.01, 3.0, 0.1, 10.0)
        targets = jnp.asarray(rng.uniform(0.1, 8.0, 24))
        solve = jax.jit(lambda s: planner(s, *rates, *hardware, n_max=128))
        times = solve(targets)
        achieved = snr_fixed_frame(*rates, times, *hardware)
        assert np.all(np.asarray(achieved) >= np.asarray(targets))
        whole = jax.jit(
            lambda s: exptime_whole_frames(s, *rates, *hardware, n_max=128)
        )(targets)
        assert np.all(np.asarray(times) <= np.asarray(whole))
        previous = jnp.maximum(frame_count(times, 10.0) - 1, 0) * 10
        assert np.all(
            np.asarray(snr_fixed_frame(*rates, previous, *hardware)) < targets
        )
        if planner is exptime_min_duration:
            # A few ULPs of upward rounding ensure the root meets the target.
            np.testing.assert_allclose(
                achieved, targets, rtol=8 * jnp.finfo(targets.dtype).eps
            )


def test_float32_fractional_frame_boundaries():
    """Fails if float32 roundoff charges an extra read or CIC at a boundary."""
    with jax.enable_x64(False):
        n = jnp.arange(1, 14, dtype=jnp.float32)
        t = n * 0.1

        def evaluate(t):
            return frame_count(t, 0.1), detector_variance_counts(
                t, 2.0, 0.0, 3.0, 0.5, 0.1
            )

        for evaluate_fn in (evaluate, jax.jit(evaluate)):
            frames, detector = evaluate_fn(t)
            np.testing.assert_array_equal(frames, n)
            np.testing.assert_array_equal(detector, 19 * n)
            # A resolved partial frame must still incur another read.
            frames, _ = evaluate_fn(t * (1 + 1e-5))
            np.testing.assert_array_equal(frames, n + 1)


@pytest.mark.parametrize("planner", [exptime_whole_frames, exptime_min_duration])
@pytest.mark.parametrize("n_max", [10, 32])
def test_whole_frame_roundtrip(planner, n_max):
    """Fails if JIT rejects an eagerly computed attainable boundary target."""
    with jax.enable_x64():
        rates = (2.0, 3.0, 0.02)
        hardware = (4.0, 0.01, 3.0, 0.1, 10.0)
        t = jnp.arange(1.0, 11.0) * 10
        target = snr_fixed_frame(*rates, t, *hardware)

        def solve(target):
            return planner(target, *rates, *hardware, n_max=n_max)

        np.testing.assert_array_equal(solve(target), t)
        np.testing.assert_array_equal(jax.jit(solve)(target), t)


@pytest.mark.parametrize("planner", [exptime_whole_frames, exptime_min_duration])
@pytest.mark.parametrize("x64", [False, True])
def test_solver_floor_ceiling_and_bounds(planner, x64):
    """Fails if impossible exposures or their SNR round trips return NaN."""
    with jax.enable_x64(x64):

        def solve(targets, Cp):
            return planner(targets, Cp, 3.0, 0.02, 4.0, 0.01, 3.0, 0.1, 10.0, n_max=1)

        targets = jnp.array([0.0, 100.0, 101.0, 1.0, 15.0])
        Cp = jnp.array([2.0, 2.0, 2.0, 0.0, 2.0])

        def evaluate(t, floor):
            return snr_fixed_frame(Cp, 3.0, floor, t, 4.0, 0.01, 3.0, 0.1, 10.0)

        evaluate_fns = (evaluate, jax.jit(evaluate))
        for solve_fn in (solve, jax.jit(solve)):
            result = solve_fn(targets, Cp)
            np.testing.assert_array_equal(
                result, [0.0, jnp.inf, jnp.inf, jnp.inf, jnp.inf]
            )
            for evaluate_fn in evaluate_fns:
                np.testing.assert_array_equal(evaluate_fn(result, 0.02), 0.0)
                np.testing.assert_array_equal(
                    evaluate_fn(jnp.array([0.0, jnp.inf, -jnp.inf, jnp.nan, 0.0]), 0.0),
                    0.0,
                )


def test_partial_solver_respects_read_discontinuity():
    """Fails if global bisection jumps past the earliest feasible interval."""
    with jax.enable_x64():
        rates = (1.0, 0.0, 0.0)
        hardware = (1.0, 0.0, 10.0, 0.0, 10.0)
        target = 0.9
        first = exptime_min_duration(target, *rates, *hardware, n_max=8)
        # At 10 s SNR is 0.953; a second read drops it below 0.7.
        assert 9 < first < 10
        assert snr_fixed_frame(*rates, 10.001, *hardware) < target
        assert exptime_whole_frames(target, *rates, *hardware) == 10
        assert snr_fixed_frame(*rates, first, *hardware) >= target


@pytest.fixture
def system_args(fixed_path, system):
    exposure = ox.ExposureConfig(
        start_time_jd=jnp.array([2460000.5, 2460001.0]),
        exposure_time_s=jnp.asarray(100.0),
        central_wavelength_nm=jnp.asarray(500.0),
        bin_width_nm=jnp.asarray(20.0),
        position_angle_deg=jnp.asarray(0.0),
    )
    observatory = SimpleNamespace(
        temperature_K=270.0, overhead_multi=1.0, overhead_fixed_s=0.0
    )
    return system, fixed_path, observatory, exposure, PPConfig(ppfact=30.0, n_rolls=1)


def test_system_budget_floor_and_explicit_options(system_args):
    """Fails if system SNR omits its floor or ignores explicit k/ppfact/cadence."""
    options = dict(
        zodi_fn=lambda *_: 0.0, cadence="fixed_frame", bg_multiplier=3.0, ppfact=2e4
    )

    @jax.jit
    def evaluate(ppfact):
        kwargs = options | dict(ppfact=ppfact)
        budget = count_variance_components(*system_args, 100.0, **kwargs)
        snr = snr_from_system_ayo(*system_args, 100.0, **kwargs)
        _, _, floor = count_rates_from_system_ayo(
            *system_args,
            zodi_fn=options["zodi_fn"],
            cadence="fixed_frame",
            ppfact=ppfact,
        )
        return budget, snr, floor

    budget, snr, floor = evaluate(jnp.asarray(2e4))
    np.testing.assert_allclose(
        snr, budget["planet"] / jnp.sqrt(budget["total"]), rtol=1e-6
    )
    np.testing.assert_allclose(budget["systematic"], (floor * 100) ** 2, rtol=1e-6)
    off, snr_off, _ = evaluate(jnp.asarray(jnp.inf))
    np.testing.assert_array_equal(off["systematic"], 0.0)
    assert np.all(np.asarray(snr_off) > snr)


@pytest.mark.parametrize("planning", ["whole_frames", "min_duration"])
def test_system_planner_and_batch_jit(system_args, planning):
    """Fails if the planner/options cannot vmap over complete systems under JIT."""
    system, *rest = system_args
    dynamic, static = eqx.partition(system, eqx.is_array)
    batch = eqx.combine(jax.tree.map(lambda x: jnp.stack([x, x]), dynamic), static)
    options = dict(
        zodi_fn=lambda *_: 0.0, cadence="fixed_frame", ppfact=jnp.inf, bg_multiplier=3.0
    )

    @eqx.filter_jit
    @eqx.filter_vmap
    def evaluate(s):
        t = exptime_from_system_ayo(
            s, *rest, 1.0, planning=planning, n_max=128, **options
        )
        return t, snr_from_system_ayo(s, *rest, t[0, 0], **options)

    times, achieved = evaluate(batch)
    if planning == "min_duration":
        np.testing.assert_allclose(achieved, 1.0, rtol=1e-6)
    assert np.all(np.asarray(achieved) >= 1.0)
    whole = exptime_from_system_ayo(system, *rest, 1.0, n_max=128, **options)
    assert np.all(np.asarray(times) <= np.asarray(whole))
