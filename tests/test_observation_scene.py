"""Observation-layer adapters against skyscapes.scene.System."""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils import constants as const
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.system.orbit import KeplerianOrbit
from skyscapes.atmosphere import GridAtmosphere
from skyscapes.scene import Planet, SpectrumStar, System

from jaxedith.observation import _system_to_etc_scene


def _make_system():
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    star = SpectrumStar(
        Ms_kg=const.Msun2kg,
        dist_pc=10.0,
        ra_deg=0.0,
        dec_deg=0.0,
        diameter_arcsec=0.0,
        luminosity_lsun=1.0,
        wavelengths_nm=jnp.array([400.0, 800.0]),
        times_jd=jnp.array([0.0, 1.0]),
        flux_density_jy=jnp.full((2, 2), 3631.0),
    )
    orbit = KeplerianOrbit(
        a_AU=jnp.array([1.0]),
        e=jnp.array([0.0]),
        W_rad=jnp.array([0.0]),
        i_rad=jnp.array([0.0]),
        w_rad=jnp.array([0.0]),
        M0_rad=jnp.array([0.0]),
        t0_d=jnp.array([0.0]),
    )
    atmosphere = GridAtmosphere(
        Rp_Rearth=jnp.array([1.0]),
        wavelengths_nm=jnp.array([400.0, 800.0]),
        phase_angle_deg=jnp.array([0.0, 180.0]),
        contrast_grid=jnp.full((1, 2, 2), 1e-10),
    )
    return System(
        star=star,
        planets=(Planet(orbit=orbit, atmosphere=atmosphere),),
        trig_solver=solver,
    )


def test_system_to_etc_scene_accepts_scene_system():
    system = _make_system()
    scene = _system_to_etc_scene(
        system=system,
        planet_index=0,
        wavelength_nm=550.0,
        time_jd=0.5,
        Fzodi=0.0,
    )
    assert jnp.isfinite(scene.F0)
    assert scene.F0 > 0
    assert jnp.isfinite(scene.Fp_over_Fs)
    assert jnp.isclose(scene.Fp_over_Fs, 1e-10, rtol=1e-3)
    assert jnp.isfinite(scene.sep_arcsec)
    assert scene.sep_arcsec > 0
