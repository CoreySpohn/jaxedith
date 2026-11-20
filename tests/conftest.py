# ruff: noqa: E402
"""Shared fixtures for jaxedith tests."""

import importlib.util
import os

import jax

jax.config.update("jax_enable_x64", True)

from hypothesis import settings
from jaxtyping import install_import_hook

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))

# Library checks (finite, non-negative rates; NaN-free exposure division
# intermediates) raise during tests. SNR solvers and overheads are not checked.
os.environ.setdefault("JAXEDITH_DEBUG_CHECKS", "1")

# Runtime shape and dtype checking of the jaxtyping annotations, tests only.
# The hook must be installed before jaxedith is first imported. Set
# JAXEDITH_TYPECHECK=0 to run without it.
if os.environ.get("JAXEDITH_TYPECHECK", "1") == "1":
    install_import_hook(["jaxedith", "reference"], "beartype.beartype")


import jax.numpy as jnp
import optixstuff as ox
import pytest
from optixstuff.spectral import Band, Quadrature


@pytest.fixture(scope="module")
def example():
    """A detector and the photon rates of one weighted aperture over one band."""
    detector = ox.Detector(
        pixel_scale_arcsec=0.01,
        shape=(12, 12),
        qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
        survival_fraction=0.9,
        dark_current_e_per_pixel_s=3e-4,
        clock_induced_charge_e_per_pixel_frame=0.005,
        read_noise_e_rms=0.2,
    )
    quadrature = Quadrature.build((Band(500, 600),))
    ones = jnp.ones(quadrature.nodes_nm.shape)

    def photon_rates(n_pix, scale, variance=None):
        return ox.Rates(
            quadrature,
            "photon",
            n_pix=jnp.array([n_pix]),
            variance=variance,
            planet=scale * 4e-4 * ones,
            star_leakage=scale * 2e-4 * ones,
            local_zodi=scale * 1e-4 * ones,
            exozodi=scale * 5e-5 * ones,
        )

    return detector, photon_rates(12.0, 1.0, variance=photon_rates(9.0, 0.8))
