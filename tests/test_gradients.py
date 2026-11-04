"""Derivative checks for the closed-form equations via jax.test_util.check_grads.

``check_grads`` compares forward-mode and reverse-mode derivatives, to second
order, against finite differences of the function itself. Its finite-difference
step is absolute, so every argument is scaled to order one before the call.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from jaxedith.etc import (
    exptime_from_rates_ayo,
    exptime_from_rates_exosims_char,
    exptime_from_rates_exosims_det,
    snr_from_rates_ayo,
    snr_from_rates_exosims_char,
    snr_from_rates_exosims_det,
)
from jaxedith.primitives import planet_rate


@pytest.fixture(autouse=True)
def _x64():
    """Finite differences at eps=1e-3 need float64 to resolve second order."""
    with jax.enable_x64():
        yield


# Arguments sit in the regime where the planet is above the noise floor, away
# from the jnp.maximum / jnp.where kinks that no finite difference can cross.
CASES = {
    "exptime_ayo": (exptime_from_rates_ayo, (1.0, 0.5, 0.05, 7.0)),
    "exptime_exosims_det": (exptime_from_rates_exosims_det, (1.0, 0.5, 0.01, 7.0)),
    "exptime_exosims_char": (exptime_from_rates_exosims_char, (1.0, 0.5, 0.01, 7.0)),
    "snr_ayo": (snr_from_rates_ayo, (1.0, 0.5, 0.05, 98.0)),
    "snr_exosims_det": (snr_from_rates_exosims_det, (1.0, 0.5, 0.01, 98.0)),
    "snr_exosims_char": (snr_from_rates_exosims_char, (1.0, 0.5, 0.01, 98.0)),
}


@pytest.mark.parametrize("name", CASES)
def test_equation_derivatives(name):
    fn, args = CASES[name]
    check_grads(
        fn,
        tuple(jnp.asarray(a) for a in args),
        order=2,
        modes=("fwd", "rev"),
        eps=1e-4,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "name", [name for name in CASES if name.startswith("exptime_")]
)
def test_planet_rate_derivatives(name):
    """Check AD and physical contrast signs, scaled by 1e-9 and 100 nm."""

    def rate(contrast_scaled, dlambda_scaled):
        return planet_rate(
            1.34e8,
            5.3e-3,
            contrast_scaled * 1e-9,
            42.76,
            0.359,
            0.297,
            dlambda_scaled * 100.0,
            2,
        )

    check_grads(
        rate,
        (jnp.asarray(1.0), jnp.asarray(1.0)),
        order=2,
        modes=("fwd", "rev"),
        eps=1e-4,
        atol=1e-4,
        rtol=1e-4,
    )

    planet_slope = jax.grad(rate, argnums=0)(1.0, 1.0)
    assert jnp.isfinite(planet_slope)
    assert planet_slope > 0.0

    exptime, args = CASES[name]

    def exposure(contrast_scaled):
        return exptime(rate(contrast_scaled, 1.0), *args[1:])

    assert jnp.isfinite(exposure(1.0))
    exposure_slope = jax.grad(exposure)(1.0)
    assert jnp.isfinite(exposure_slope)
    assert exposure_slope < 0.0
