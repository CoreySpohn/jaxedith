"""Finite stellar size in photon rates, shot noise and exposure planning."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import arcsec2rad, nm2m
from optixstuff import FramePlan
from skyscapes.scene import BlackbodyStar, FlatStar

from jaxedith import PostProcessing, aperture_rates, exposure_time, snr


@vv.case("aperture-photon-budget", "code-verification")
def test_finite_diameter_physics_and_transforms(finite_star_setup):
    """Exact-algebra/floating-point: exp(5*d) uniform YIP, rtol=1e-10.

    Independent SI angular conversion and pi*r^2 predict absolute photons.
    Larger diameter must raise shot-noise exposure time and lower fixed-time
    S/N, while planet photons stay unchanged. JIT/vmap covers target diameters.
    """
    system, path, bands, q, prepared = finite_star_setup

    def rates(d):
        scene = eqx.tree_at(lambda s: s.star.diameter_arcsec, system, d)
        return aperture_rates(
            scene,
            path,
            bands,
            time_d=0.0,
            planet_index=0,
            prepared_aperture=prepared,
            zodi=None,
        )

    diameters = jnp.array([0.0, 0.0005, 0.001, 0.002])
    # Planet position is immaterial to the uniform stellar map; use the
    # scene's unmodified orbit for both finite and point-source rates.
    result = jax.jit(jax.vmap(rates))(diameters)
    waves = np.asarray(q.nodes_nm)
    d_lod = np.asarray(diameters)[:, None, None] * arcsec2rad * 8 / (waves * nm2m)
    expected = 40 * 0.4 * 1e6 * 1e-9 * np.pi * 0.7**2 * np.exp(5 * d_lod)
    np.testing.assert_allclose(result.star_leakage, expected, rtol=1e-10, atol=0)
    eager = jnp.stack([rates(d).star_leakage for d in diameters])
    np.testing.assert_allclose(result.star_leakage, eager, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(
        result.planet, jnp.broadcast_to(result.planet[0], result.planet.shape)
    )
    pp = PostProcessing(0.0, 0.0)
    times, signal_to_noise = [], []
    for d in diameters:
        r = rates(d)
        times.append(
            exposure_time(
                5.0,
                r,
                path.detector,
                q,
                frame_time_s=1.0,
                n_pix=10.0,
                post_processing=pp,
            ).live_time_s
        )
        signal_to_noise.append(
            snr(
                r,
                path.detector,
                q,
                FramePlan(100.0, 36),
                n_pix=10.0,
                post_processing=pp,
            )
        )
    assert np.all(np.diff(times, axis=0) > 0)
    assert np.all(np.diff(signal_to_noise, axis=0) < 0)


@vv.case("aperture-photon-budget", "code-verification")
def test_blackbody_diameter_uses_distance(finite_star_setup):
    """Exact-algebra: physical/apparent stars agree to 1e-12 at two distances.

    The angular diameter is independently 2R/d; a missing distance or a
    bound-method value fails the absolute leakage prediction.
    """
    from hwoutils.constants import pc2m

    system, path, bands, q, prepared = finite_star_setup
    # Product diameter is 8 m; a different primary must not set stellar lambda/D.
    path = eqx.tree_at(lambda p: p.primary.circumscribed_diameter_m, path, 6.0)
    star = BlackbodyStar(teff_K=5772.0, radius_m=6.957e8)
    for distance in (10.0, 20.0):
        scene = eqx.tree_at(lambda s: (s.star, s.dist_pc), system, (star, distance))
        r = aperture_rates(
            scene,
            path,
            bands,
            time_d=0.0,
            separation_arcsec=0.05,
            prepared_aperture=prepared,
            zodi=None,
        )
        diameter_rad = 2 * star.radius_m / (distance * pc2m)
        expected = (
            40
            * 0.4
            * star.photon_flux_density(q.nodes_nm, 0.0, dist_pc=distance)
            * 1e-9
            * np.pi
            * 0.7**2
            * jnp.exp(5 * diameter_rad * 8 / (q.nodes_nm * nm2m))
        )
        np.testing.assert_allclose(r.star_leakage, expected, rtol=1e-12, atol=0)
        for i, band in enumerate(bands):
            apparent = FlatStar(
                star.photon_flux_density(band.center_nm, 0.0, dist_pc=distance),
                diameter_arcsec=diameter_rad / arcsec2rad,
            )
            other = eqx.tree_at(lambda s: s.star, scene, apparent)
            actual = aperture_rates(
                other,
                path,
                bands,
                time_d=0.0,
                separation_arcsec=0.05,
                prepared_aperture=prepared,
                zodi=None,
            )
            np.testing.assert_allclose(
                actual.star_leakage[i], r.star_leakage[i], rtol=1e-12, atol=0
            )
