"""End-to-end integration tests: real coronagraph through jaxedith.

Loads a real coronagraph YIP via pooch, builds a full
optixstuff.OpticalPath, and pushes it through jaxedith to verify
the entire pipeline works and produces physically reasonable results.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yippy.datasets import fetch_coronagraph

import optixstuff as ox
from jaxedith import ETCScene, count_rates_ayo, exptime_ayo, snr_ayo


# ---------------------------------------------------------------------------
# Session-scoped fixtures (data downloaded once per test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def yip_path():
    """Download the default AAVC coronagraph YIP via pooch."""
    return fetch_coronagraph()


@pytest.fixture(scope="session")
def optical_path(yip_path):
    """Full optixstuff OpticalPath with real coronagraph."""
    primary = ox.SimplePrimary(diameter_m=6.0, obscuration=0.14)
    coronagraph = ox.YippyCoronagraph(yip_path)
    detector = ox.Detector(
        pixel_scale=0.010,
        shape=(100, 100),
        quantum_efficiency=0.9,
        dark_current_rate=3e-5,
        read_noise_electrons=0.0,
        cic_rate=1.3e-3,
        frame_time=1000.0,
        read_time=1000.0,
        dqe=1.0,
    )
    optics_filter = ox.ConstantThroughputElement(throughput=0.5, name="optics")
    return ox.OpticalPath(
        primary=primary,
        coronagraph=coronagraph,
        attenuating_elements=(optics_filter,),
        detector=detector,
    )


@pytest.fixture(scope="session")
def sun_like_scene():
    """Sun-like star at 10 pc with an Earth-like planet at 1e-10 contrast."""
    return ETCScene(
        F0=1.34e8,
        Fs_over_F0=0.005311,
        Fp_over_Fs=1e-10,
        Fzodi=3.5e-10,
        Fexozodi=7.15e-9,
        dist_pc=10.0,
        sep_arcsec=0.1,
        Fbinary=0.0,
    )


# ---------------------------------------------------------------------------
# End-to-end count rate and solver tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline: optixstuff OpticalPath -> jaxedith equations."""

    def test_count_rates_ayo_finite(self, optical_path, sun_like_scene):
        Cp, Cb, Cnf_rate = count_rates_ayo(
            optical_path, sun_like_scene,
            wavelength_nm=500.0, separation_lod=5.0, dlambda_nm=100.0,
        )
        assert jnp.isfinite(Cp), f"Cp not finite: {Cp}"
        assert jnp.isfinite(Cb), f"Cb not finite: {Cb}"
        assert jnp.isfinite(Cnf_rate), f"Cnf_rate not finite: {Cnf_rate}"

    def test_count_rates_ayo_positive_planet(self, optical_path, sun_like_scene):
        Cp, _, _ = count_rates_ayo(
            optical_path, sun_like_scene,
            wavelength_nm=500.0, separation_lod=5.0, dlambda_nm=100.0,
        )
        assert float(Cp) > 0

    def test_exptime_ayo_finite_positive(self, optical_path, sun_like_scene):
        t_exp = exptime_ayo(
            optical_path, sun_like_scene,
            wavelength_nm=500.0, separation_lod=5.0, dlambda_nm=100.0, snr=7.0,
        )
        assert jnp.isfinite(t_exp), f"t_exp not finite: {t_exp}"
        assert float(t_exp) > 0

    def test_snr_ayo_round_trip(self, optical_path, sun_like_scene):
        target_snr = 7.0
        t_exp = exptime_ayo(
            optical_path, sun_like_scene,
            wavelength_nm=500.0, separation_lod=5.0, dlambda_nm=100.0,
            snr=target_snr,
        )
        recovered_snr = snr_ayo(
            optical_path, sun_like_scene,
            wavelength_nm=500.0, separation_lod=5.0, dlambda_nm=100.0,
            t_obs=float(t_exp),
        )
        assert np.isclose(float(recovered_snr), target_snr, rtol=0.01)


# ---------------------------------------------------------------------------
# JIT compilation tests
# ---------------------------------------------------------------------------


class TestJITCompilation:
    """Verify the full pipeline JIT-compiles."""

    def test_jit_exptime_ayo(self, optical_path, sun_like_scene):
        @eqx.filter_jit
        def _calc(wl, sep, dlam, snr):
            return exptime_ayo(optical_path, sun_like_scene, wl, sep, dlam, snr)

        result = _calc(500.0, 5.0, 100.0, 7.0)
        assert jnp.isfinite(result)

    def test_jit_snr_ayo(self, optical_path, sun_like_scene):
        @eqx.filter_jit
        def _calc(wl, sep, dlam, t_obs):
            return snr_ayo(optical_path, sun_like_scene, wl, sep, dlam, t_obs)

        result = _calc(500.0, 5.0, 100.0, 3600.0)
        assert jnp.isfinite(result)

    def test_vmap_over_separations(self, optical_path, sun_like_scene):
        separations = jnp.array([3.0, 5.0, 7.0, 10.0])

        @eqx.filter_jit
        def _batch(seps):
            return jax.vmap(
                lambda sep: exptime_ayo(
                    optical_path, sun_like_scene, 500.0, sep, 100.0, 7.0,
                )
            )(seps)

        results = _batch(separations)
        assert results.shape == (4,)
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results > 0)


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


class TestPhysicalSanity:
    """Verify results are physically reasonable."""

    def test_brighter_planet_shorter_time(self, optical_path):
        bright = ETCScene(F0=1.34e8, Fs_over_F0=0.005, Fp_over_Fs=1e-9)
        faint = ETCScene(F0=1.34e8, Fs_over_F0=0.005, Fp_over_Fs=1e-10)

        t_bright = exptime_ayo(optical_path, bright, 500.0, 5.0, 100.0, 7.0)
        t_faint = exptime_ayo(optical_path, faint, 500.0, 5.0, 100.0, 7.0)
        assert float(t_bright) < float(t_faint)

    def test_higher_snr_longer_time(self, optical_path, sun_like_scene):
        t_low = exptime_ayo(optical_path, sun_like_scene, 500.0, 5.0, 100.0, 5.0)
        t_high = exptime_ayo(optical_path, sun_like_scene, 500.0, 5.0, 100.0, 10.0)
        assert float(t_high) > float(t_low)
