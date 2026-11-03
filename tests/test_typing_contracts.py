"""Guard that the jaxtyping import hook in conftest.py is active."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaxedith.etc import exptime_from_rates_ayo


def test_hook_rejects_complex_input():
    """A silently disabled hook would make every annotation decorative."""
    with pytest.raises(TypeCheckError):
        exptime_from_rates_ayo(1.0 + 0.0j, 0.5, 0.0, 7.0)


def test_hook_checks_at_trace_time_under_jit():
    with pytest.raises(TypeCheckError):
        jax.jit(exptime_from_rates_ayo)(jnp.ones(3, dtype=complex), 0.5, 0.0, 7.0)


def test_scalars_arrays_and_batches_pass():
    for cp in (1.0, jnp.float32(1.0), jnp.ones(4), jnp.ones((2, 3))):
        assert exptime_from_rates_ayo(cp, 0.5, 0.0, 7.0) is not None
