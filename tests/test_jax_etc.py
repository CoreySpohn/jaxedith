# ruff: noqa: E402
"""Tests for the JAX ETC backend.

Validates:
1. Count rate parity between JAX and numpy pyEDITH functions
2. Solver round-trip consistency (solve_exptime ↔ solve_snr)
3. JIT compilation
4. vmap over batched inputs
"""

import numpy as np
import pytest

# Skip entire module if JAX not installed
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

from jaxedith.config import (
    AYO_CONFIG,
    CONFIG,
    EXOSIMS_CHARACTERIZATION_CONFIG,
    EXOSIMS_DETECTION_CONFIG,
)
from jaxedith.count_rates import (
    count_rate_binary,
    count_rate_detector,
    count_rate_exozodi,
    count_rate_planet,
    count_rate_stellar_leakage,
    count_rate_thermal,
    count_rate_zodi,
    noise_floor_exozodi,
    noise_floor_stellar,
    noise_floor_total,
    photon_counting_time,
)
from jaxedith.solver import (
    solve_exptime_ayo,
    solve_exptime_exosims_char,
    solve_exptime_exosims_det,
    solve_snr_ayo,
    solve_snr_exosims_char,
    solve_snr_exosims_det,
)

# ---------------------------------------------------------------------------
# Fixtures: reference values from test_exposure_time_calculator.py
# ---------------------------------------------------------------------------

# Common parameters (stripped of astropy units)
F0 = 1.34e8  # ph/s/m^2/nm
Fs_over_F0 = 0.005311289818550127  # dimensionless
Fp_over_Fs = 1e-9  # dimensionless
area_m2 = 42.759068268120557  # m^2
core_throughput = 0.2968371  # dimensionless
throughput = 0.35910000000000003  # dimensionless (optics * QE)
dlambda_nm = 100.0  # nm
n_channels = 2
lod_arcsec = 0.013104498490920989  # arcsec

# Detector parameters
det_npix = 9.054697
det_DC = 3e-05  # e/pix/s
det_RN = 2.0  # e/pix/read
det_tread = 1000.0  # s
det_CIC = 1e-3  # e/pix/frame
det_t_photon = 13.79303  # s/frame


# ── Count rate parity tests ──────────────────────────────────────────────────


class TestCountRateParity:
    """Verify JAX count rates match numpy pyEDITH values."""

    def test_count_rate_planet(self):
        """CRp should match pyEDITH's calculate_CRp."""
        result = count_rate_planet(
            F0,
            Fs_over_F0,
            Fp_over_Fs,
            area_m2,
            throughput,
            core_throughput,
            dlambda_nm,
            n_channels,
        )
        assert np.isclose(float(result), 0.64877874, rtol=1e-5)

    def test_count_rate_stellar_leakage(self):
        """CRbs via core_mean_intensity should match pyEDITH's calculate_CRbs.

        The reference value (0.0008138479) was computed from pyEDITH's 2D mode
        (Istar / pixscale^2), which yippy azimuthally averages into the 1D
        core_mean_intensity curve. So both paths give the same number.
        """
        Istar_2d = 2.3272595994978797e-14
        pixscale_lod = 0.25
        core_mean_intensity = Istar_2d / (pixscale_lod**2)
        core_area_lod2 = 1.0

        result = count_rate_stellar_leakage(
            F0,
            Fs_over_F0,
            area_m2,
            throughput,
            dlambda_nm,
            n_channels,
            core_area_lod2,
            core_mean_intensity,
        )
        assert np.isclose(float(result), 0.0008138479, rtol=1e-5)

    def test_count_rate_zodi(self):
        """CRbz should match pyEDITH's calculate_CRbz."""
        Fzodi = 3.5213620474344346e-10
        sky_trans = 0.4006394155914143
        result = count_rate_zodi(
            F0,
            Fzodi,
            lod_arcsec,
            sky_trans,
            area_m2,
            throughput,
            dlambda_nm,
            n_channels,
            core_area_lod2=1.0,
        )
        assert np.isclose(float(result), 0.0099697346, rtol=1e-5)

    def test_count_rate_exozodi(self):
        """CRbez should match pyEDITH's calculate_CRbez."""
        Fexozodi = 7.1490465158365465e-09
        sky_trans = 0.6161309232588068
        dist_pc = 18.195476531982425
        sep_arcsec_local = 0.02784705929320709

        result = count_rate_exozodi(
            F0,
            Fexozodi,
            lod_arcsec,
            sky_trans,
            area_m2,
            throughput,
            dlambda_nm,
            n_channels,
            core_area_lod2=1.0,
            dist_pc=dist_pc,
            sep_arcsec=sep_arcsec_local,
        )
        assert np.isclose(float(result), 1.2124248, rtol=1e-5)

    def test_count_rate_binary_zero(self):
        """CRbbin should be zero when Fbinary=0."""
        result = count_rate_binary(
            F0,
            0.0,
            0.65,
            area_m2,
            throughput,
            dlambda_nm,
            n_channels,
            core_area_lod2=1.0,
        )
        assert float(result) == 0.0

    def test_count_rate_thermal(self):
        """CRbth should match pyEDITH's calculate_CRbth."""
        wavelength_nm = 500.0
        temp_K = 290.0
        lod_rad = 6.353240152477764e-08
        emis = 0.468
        QE = 0.675
        dQE = 1.0

        result = count_rate_thermal(
            wavelength_nm,
            area_m2,
            dlambda_nm,
            temp_K,
            lod_rad,
            emis,
            QE,
            dQE,
            core_area_lod2=1.0,
        )
        assert np.isclose(float(result), 2.848015e-30, rtol=0.05)

    def test_count_rate_detector(self):
        """CRbd should match pyEDITH's calculate_CRbd."""
        result = count_rate_detector(
            det_npix, det_DC, det_RN, det_tread, det_CIC, det_t_photon
        )
        assert np.isclose(float(result), 0.037146898, rtol=1e-5)

    def test_noise_floor_stellar(self):
        """CRnf should match pyEDITH's calculate_CRnf."""
        noisefloor_raw = 7.25659425003725e-18
        pixscale_lod = 0.25
        noisefloor_value = noisefloor_raw / (pixscale_lod**2)
        snr = 7.0

        result = noise_floor_stellar(
            F0,
            Fs_over_F0,
            area_m2,
            throughput,
            dlambda_nm,
            n_channels,
            snr,
            noisefloor_value,
            core_area_lod2=1.0,
        )
        assert np.isclose(float(result), 1.7763531e-6, rtol=1e-5)

    def test_noise_floor_exozodi(self):
        """CRnf_ez is simply SNR * CRbez / ez_ppf."""
        CRbez = 1.2
        snr = 7.0
        ez_ppf = 30.0
        result = noise_floor_exozodi(CRbez, snr, ez_ppf)
        assert np.isclose(float(result), 7.0 * 1.2 / 30.0, rtol=1e-10)

    def test_noise_floor_total_with_exozodi(self):
        """Combined noise floor in quadrature when include_ez=True."""
        result = noise_floor_total(3.0, 4.0, include_ez=True)
        assert np.isclose(float(result), 5.0, rtol=1e-10)

    def test_noise_floor_total_without_exozodi(self):
        """Without exozodi, total = stellar only."""
        result = noise_floor_total(3.0, 4.0, include_ez=False)
        assert np.isclose(float(result), 3.0, rtol=1e-10)

    def test_photon_counting_time(self):
        """t_photon = 1 / (6.73 * det_CR / det_npix)."""
        det_CR = 0.723971066592388
        result = photon_counting_time(det_CR, det_npix)
        assert np.isclose(float(result), 1.8583934, rtol=1e-5)


# ── Config preset tests ──────────────────────────────────────────────────────


class TestConfigPresets:
    """Verify preset configurations have correct variant and parameters."""

    def test_ayo_config(self):
        """Test AYO preset variant."""
        assert AYO_CONFIG.variant == "ayo"
        assert AYO_CONFIG.bg_multiplier == 2.0
        assert AYO_CONFIG.include_exozodi_noise_floor is False

    def test_config(self):
        """Test default JAXEDITH preset variant."""
        assert CONFIG.variant == "jaxedith"
        assert CONFIG.bg_multiplier == 2.0
        assert CONFIG.include_exozodi_noise_floor is True

    def test_exosims_detection(self):
        """Test EXOSIMS detection preset variant."""
        assert EXOSIMS_DETECTION_CONFIG.variant == "exosims_det"
        assert EXOSIMS_DETECTION_CONFIG.bg_multiplier == 1.0
        assert EXOSIMS_DETECTION_CONFIG.include_exozodi_noise_floor is False

    def test_exosims_characterization(self):
        """Test EXOSIMS characterization preset variant."""
        assert EXOSIMS_CHARACTERIZATION_CONFIG.variant == "exosims_char"
        assert EXOSIMS_CHARACTERIZATION_CONFIG.bg_multiplier == 1.0


# ── Solver tests ─────────────────────────────────────────────────────────────


class TestSolver:
    """Verify solver correctness and round-trip consistency."""

    def test_solve_exptime_ayo_simple(self):
        """T = SNR² * (Cp + 2Cb) / (Cp² - Cnf²)."""
        # t = 49 * (1 + 1) / (1 - 0) = 98
        t = solve_exptime_ayo(1.0, 0.5, 0.0, 7.0)
        assert np.isclose(float(t), 98.0, rtol=1e-10)

    def test_solve_exptime_with_noise_floor(self):
        """With noise floor, denominator decreases ⟹ longer time."""
        # t = 49 * 2 / (1 - 0.25) = 130.667
        t = solve_exptime_ayo(1.0, 0.5, 0.5, 7.0)
        assert np.isclose(float(t), 49 * 2 / 0.75, rtol=1e-5)

    def test_planet_below_noise_floor(self):
        """If Cnf ≥ Cp, time should be infinity."""
        t = solve_exptime_ayo(1.0, 0.5, 1.5, 7.0)
        assert jnp.isinf(t)

    def test_solve_exptime_exosims_det(self):
        """T = SNR² * Cb / (Cp² - (SNR * Csp)²)."""
        # t = 49 * 0.5 / (1 - 0.07²) = 49 * 0.5 / 0.9951
        t = solve_exptime_exosims_det(1.0, 0.5, 0.01, 7.0)
        expected = 49 * 0.5 / (1 - 0.0049)
        assert np.isclose(float(t), expected, rtol=1e-6)

    def test_solve_exptime_exosims_char(self):
        """Characterization adds Cp to Cb in numerator."""
        t_det = solve_exptime_exosims_det(1.0, 0.5, 0.01, 7.0)
        t_char = solve_exptime_exosims_char(1.0, 0.5, 0.01, 7.0)
        # Char has larger numerator (Cb + Cp) vs (Cb), so t_char > t_det
        assert float(t_char) > float(t_det)
        # Check ratio: (Cb + Cp) / Cb = 1.5/0.5 = 3.0
        assert np.isclose(float(t_char / t_det), 3.0, rtol=1e-5)

    def test_round_trip_ayo(self):
        """solve_snr_ayo(solve_exptime_ayo(snr)) ≈ snr."""
        Cnf_rate = 0.01
        target_snr = 10.0
        Cnf = target_snr * Cnf_rate  # solve_exptime takes full Cnf

        t = solve_exptime_ayo(1.0, 0.5, Cnf, target_snr)
        recovered_snr = solve_snr_ayo(1.0, 0.5, Cnf_rate, float(t))
        assert np.isclose(float(recovered_snr), target_snr, rtol=1e-3)

    def test_round_trip_exosims_det(self):
        """solve_snr_exosims_det(solve_exptime_exosims_det(snr)) ≈ snr."""
        target_snr = 10.0
        t = solve_exptime_exosims_det(1.0, 0.5, 0.01, target_snr)
        recovered_snr = solve_snr_exosims_det(1.0, 0.5, 0.01, float(t))
        assert np.isclose(float(recovered_snr), target_snr, rtol=1e-4)

    def test_round_trip_exosims_char(self):
        """solve_snr_exosims_char(solve_exptime_exosims_char(snr)) ≈ snr."""
        target_snr = 10.0
        t = solve_exptime_exosims_char(1.0, 0.5, 0.01, target_snr)
        recovered_snr = solve_snr_exosims_char(1.0, 0.5, 0.01, float(t))
        assert np.isclose(float(recovered_snr), target_snr, rtol=1e-4)

    def test_zero_observation_time(self):
        """SNR should be effectively zero for zero observation time."""
        snr = solve_snr_ayo(1.0, 0.5, 0.0, 0.0)
        assert float(snr) < 1e-10

    def test_overhead_handling(self):
        """Overheads should increase exposure time."""
        t_no = solve_exptime_ayo(
            1.0, 0.5, 0.0, 7.0, overhead_multi=1.0, overhead_fixed=0.0
        )
        t_oh = solve_exptime_ayo(
            1.0, 0.5, 0.0, 7.0, overhead_multi=1.1, overhead_fixed=100.0
        )
        assert float(t_oh) > float(t_no)

    def test_rolls_multiply_time(self):
        """Multiple rolls should multiply total time."""
        t1 = solve_exptime_ayo(1.0, 0.5, 0.0, 7.0, n_rolls=1)
        t2 = solve_exptime_ayo(1.0, 0.5, 0.0, 7.0, n_rolls=2)
        assert np.isclose(float(t2), 2.0 * float(t1), rtol=1e-10)


# ── JIT compilation tests ────────────────────────────────────────────────────


class TestJIT:
    """Verify functions compile under jax.jit."""

    def test_jit_count_rate_planet(self):
        """Test count_rate_planet JIT compilation."""
        jitted = jax.jit(count_rate_planet)
        result = jitted(
            F0,
            Fs_over_F0,
            Fp_over_Fs,
            area_m2,
            throughput,
            core_throughput,
            dlambda_nm,
            n_channels,
        )
        assert np.isclose(float(result), 0.64877874, rtol=1e-5)

    def test_jit_solve_exptime_ayo(self):
        """Test solve_exptime_ayo JIT compilation."""

        @jax.jit
        def _solve(Cp, Cb, snr):
            return solve_exptime_ayo(Cp, Cb, 0.0, snr)

        result = _solve(1.0, 0.5, 7.0)
        assert np.isclose(float(result), 98.0, rtol=1e-10)

    def test_jit_solve_snr_ayo(self):
        """Test solve_snr_ayo JIT compilation."""

        @jax.jit
        def _solve(Cp, Cb, t):
            return solve_snr_ayo(Cp, Cb, 0.0, t)

        result = _solve(1.0, 0.5, 98.0)
        assert np.isclose(float(result), 7.0, rtol=1e-4)

    def test_jit_solve_exptime_exosims(self):
        """Test solve_exptime_exosims JIT compilation."""

        @jax.jit
        def _solve(Cp, Cb, Csp, snr):
            return solve_exptime_exosims_det(Cp, Cb, Csp, snr)

        result = _solve(1.0, 0.5, 0.01, 7.0)
        assert np.isfinite(float(result))

    def test_jit_count_rate_stellar_leakage(self):
        """Stellar leakage 1D mode should JIT-compile."""
        jitted = jax.jit(
            lambda cmi: count_rate_stellar_leakage(
                F0,
                Fs_over_F0,
                area_m2,
                throughput,
                dlambda_nm,
                n_channels,
                1.0,
                cmi,
            )
        )
        cmi = 2.3272595994978797e-14 / (0.25**2)
        result = jitted(cmi)
        assert np.isfinite(float(result))


# ── vmap tests ────────────────────────────────────────────────────────────────


class TestVmap:
    """Verify functions work with jax.vmap."""

    def test_vmap_count_rate_planet(self):
        """Vmap over different contrasts."""
        contrasts = jnp.array([1e-9, 1e-10, 1e-11])
        vmapped = jax.vmap(
            lambda c: count_rate_planet(
                F0,
                Fs_over_F0,
                c,
                area_m2,
                throughput,
                core_throughput,
                dlambda_nm,
                n_channels,
            )
        )
        results = vmapped(contrasts)
        assert results.shape == (3,)
        assert np.isclose(float(results[0] / results[1]), 10.0, rtol=1e-6)

    def test_vmap_solve_exptime_ayo(self):
        """Vmap over different SNRs."""
        snrs = jnp.array([5.0, 7.0, 10.0])
        vmapped = jax.vmap(lambda s: solve_exptime_ayo(1.0, 0.5, 0.0, s))
        results = vmapped(snrs)
        assert results.shape == (3,)
        # Exposure time ∝ SNR²
        assert np.isclose(float(results[1] / results[0]), 49.0 / 25.0, rtol=1e-6)

    def test_vmap_solve_exptime_exosims(self):
        """Vmap EXOSIMS solver over different SNRs."""
        snrs = jnp.array([5.0, 7.0, 10.0])
        vmapped = jax.vmap(lambda s: solve_exptime_exosims_det(1.0, 0.5, 0.01, s))
        results = vmapped(snrs)
        assert results.shape == (3,)
        assert all(jnp.isfinite(results))

    def test_vmap_wavelength_batch(self):
        """Vmap over wavelengths (simulating spectral mode)."""
        wavelengths = jnp.array([400.0, 500.0, 600.0])
        F0_arr = jnp.array([15000.0, 13400.0, 11000.0])

        vmapped = jax.vmap(
            lambda f0, wl: count_rate_planet(
                f0,
                Fs_over_F0,
                Fp_over_Fs,
                area_m2,
                throughput,
                core_throughput,
                wl * 0.2,
                n_channels,
            )
        )
        results = vmapped(F0_arr, wavelengths)
        assert results.shape == (3,)
        assert all(jnp.isfinite(results))


# ── Gradient tests ────────────────────────────────────────────────────────────


class TestGrad:
    """Verify differentiability."""

    def test_grad_count_rate_planet_wrt_contrast(self):
        """d(CRp)/d(contrast) should be positive and finite."""
        grad_fn = jax.grad(
            lambda c: count_rate_planet(
                F0,
                Fs_over_F0,
                c,
                area_m2,
                throughput,
                core_throughput,
                dlambda_nm,
                n_channels,
            )
        )
        grad_val = grad_fn(1e-9)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) > 0

    def test_grad_solve_exptime_ayo_wrt_contrast(self):
        """d(t_exp)/d(contrast) should be negative — brighter planet → shorter time."""

        def exptime_of_contrast(c):
            Cp = count_rate_planet(
                F0,
                Fs_over_F0,
                c,
                area_m2,
                throughput,
                core_throughput,
                dlambda_nm,
                n_channels,
            )
            return solve_exptime_ayo(Cp, 0.5, 0.0, 7.0)

        grad_val = jax.grad(exptime_of_contrast)(1e-9)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) < 0

    def test_grad_solve_exptime_exosims_wrt_contrast(self):
        """d(t_exp)/d(contrast) should be negative for EXOSIMS too."""

        def exptime_of_contrast(c):
            Cp = count_rate_planet(
                F0,
                Fs_over_F0,
                c,
                area_m2,
                throughput,
                core_throughput,
                dlambda_nm,
                n_channels,
            )
            return solve_exptime_exosims_det(Cp, 0.5, 0.01, 7.0)

        grad_val = jax.grad(exptime_of_contrast)(1e-9)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) < 0

    def test_grad_below_noise_floor_ayo(self):
        """Gradient should be finite even when planet is below noise floor.

        This validates the _safe_divide fix — before the fix, the
        denominator (Cp^2 - Cnf^2) going negative produced NaN in
        the gradient even though jnp.where selected inf for the result.
        """

        def f(Cp):
            return solve_exptime_ayo(Cp, 0.5, 1.5, 7.0)  # Cnf > Cp

        grad_val = jax.grad(f)(0.5)
        assert jnp.isfinite(grad_val)

    def test_grad_below_noise_floor_exosims(self):
        """Same as above for EXOSIMS detection — Csp dominates Cp."""

        def f(Cp):
            return solve_exptime_exosims_det(Cp, 0.5, 1.0, 7.0)  # SNR*Csp > Cp

        grad_val = jax.grad(f)(0.5)
        assert jnp.isfinite(grad_val)
