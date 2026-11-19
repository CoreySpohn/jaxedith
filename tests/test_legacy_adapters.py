"""Comparison dialects receive scene values through explicit adapters."""

import equinox as eqx
import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
from conftest import make_path, make_system, oracle
from optixstuff.spectral import Band, Quadrature
from reference.legacy import (
    etc_scene_from_system,
    frame_plan_photon_counting,
    legacy_pyedith_noise_rate,
)
from reference.primitives import planet_rate, stellar_leakage_rate, zodi_rate
from skyscapes.scene import FlatStar
from zodi.jax import AYOExozodi, PrecomputedZodi

from jaxedith import aperture_rates


@vv.case("etc-benchmark", "cross-code-benchmark", reference_code="pyedith")
def test_adapter_fed_legacy_limit_and_chromatic_bias(example):
    """Fails if F0 ratios change units or scalar comparisons duplicate rates.

    Basis: flat photon spectra and constant curves agree to 1e-12; the
    worked chromatic band has independently calculated 0.7-1.2% bias.
    The 6.73 rule and historical noise arithmetic use 1e-12 rounding.
    """
    system = make_system(FlatStar(1e6), chromatic=False)
    path = make_path(chromatic=False, leakage=2e-10)
    path = eqx.tree_at(
        lambda p: p.detector,
        path,
        ox.IdealDetector(pixel_scale_arcsec=0.01, shape=(1, 1)),
    )
    band = Band(549.999, 550.001)
    zodi = PrecomputedZodi(jnp.array([500.0, 600.0]), jnp.full(2, 1e10))
    scene = etc_scene_from_system(system, 550.0, time_d=0.0, planet_index=0, zodi=zodi)
    q = Quadrature.build((band,), system, path, zodi, AYOExozodi())
    rates = aperture_rates(
        system, path, q, time_d=0.0, planet_index=0, zodi=zodi
    ).integrate()
    cp = planet_rate(
        scene.F0, scene.Fs_over_F0, scene.Fp_over_Fs, 40.0, 0.4, 0.3, band.width_nm, 1
    )
    leak = stellar_leakage_rate(
        scene.F0,
        scene.Fs_over_F0,
        40.0,
        0.4,
        band.width_nm,
        1,
        np.pi * 0.7**2,
        2e-10 / (np.pi * 0.7**2),
    )
    np.testing.assert_allclose(rates.planet, cp, rtol=1e-12, atol=0)
    np.testing.assert_allclose(rates.star_leakage, leak, rtol=1e-12, atol=0)
    sky = zodi_rate(
        scene.F0,
        scene.Fzodi,
        550e-9 / 8 / (np.pi / (180 * 3600)),
        0.65,
        40.0,
        0.4,
        band.width_nm,
        1,
        np.pi * 0.7**2,
    )
    np.testing.assert_allclose(rates.local_zodi, sky, rtol=1e-12, atol=0)
    system, path, r = example
    scene = etc_scene_from_system(system, 550.0, time_d=0.0, planet_index=0, zodi=None)
    chromatic = planet_rate(
        scene.F0, scene.Fs_over_F0, scene.Fp_over_Fs, 40.0, 0.4 * 0.8 * 0.9, 0.3, 100, 1
    )
    assert 0.007 < abs(chromatic / oracle()[1] - 1) < 0.012
    np.testing.assert_allclose(
        path.detector.electrons(r).planet, oracle()[1], rtol=1e-9, atol=0
    )
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
