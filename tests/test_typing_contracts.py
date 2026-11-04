"""Guard that the jaxtyping import hook in conftest.py is active."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jaxedith.etc import exptime_from_rates_ayo
from jaxedith.primitives import (
    noise_floor_stellar,
    planet_rate,
    speckle_residual,
    stellar_leakage_rate,
)


def test_hook_rejects_complex_input():
    """A silently disabled hook would make every annotation decorative."""
    with pytest.raises(TypeCheckError):
        exptime_from_rates_ayo(1.0 + 0.0j, 0.5, 0.0, 7.0)


def test_hook_checks_at_trace_time_under_jit():
    with pytest.raises(TypeCheckError):
        jax.jit(exptime_from_rates_ayo)(jnp.ones(3, dtype=complex), 0.5, 0.0, 7.0)


@pytest.mark.parametrize(
    "product, n_factors",
    [
        (planet_rate, 8),
        (stellar_leakage_rate, 8),
        (noise_floor_stellar, 8),
        (speckle_residual, 3),
    ],
)
def test_scalars_arrays_and_batches_pass(product, n_factors):
    for cp in (1.0, jnp.float32(1.0), jnp.ones(4), jnp.ones((2, 3))):
        assert exptime_from_rates_ayo(cp, 0.5, 0.0, 7.0) is not None

    # Product-only primitives retain the input dtype kind, including integers.
    for factor in (
        2,
        2.0,
        np.int64(2),
        np.float64(2.0),
        np.full((2, 3), 2),
        np.full((2, 3), 2.0),
        jnp.asarray(2),
        jnp.asarray(2.0),
        jnp.full((2, 3), 2),
        jnp.full((2, 3), 2.0),
    ):
        result = product(*([factor] * n_factors))
        np.testing.assert_array_equal(result, np.asarray(factor) ** n_factors)
        assert np.asarray(result).dtype.kind == np.asarray(factor).dtype.kind
