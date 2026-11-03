"""The opt-in invariants in ``jaxedith._checks`` fire under jit, vmap and grad."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxedith._checks import DEBUG_CHECKS, check_finite_nonneg
from jaxedith.etc import exptime_from_rates_ayo

pytestmark = pytest.mark.skipif(not DEBUG_CHECKS, reason="JAXEDITH_DEBUG_CHECKS off")


def test_nan_rate_raises_eagerly():
    with pytest.raises(Exception, match="must not be NaN"):
        exptime_from_rates_ayo(jnp.nan, 0.5, 0.02, 7.0)


def test_nan_rate_raises_under_jit_vmap_and_grad():
    f = eqx.filter_jit(exptime_from_rates_ayo)
    with pytest.raises(Exception, match="must not be NaN"):
        f(jnp.nan, 0.5, 0.02, 7.0)
    batched = jax.vmap(exptime_from_rates_ayo, (0, None, None, None))
    with pytest.raises(Exception, match="must not be NaN"):
        batched(jnp.array([1.0, jnp.nan]), 0.5, 0.02, 7.0)
    with pytest.raises(Exception, match="must not be NaN"):
        jax.grad(exptime_from_rates_ayo)(jnp.nan, 0.5, 0.02, 7.0)


def test_below_floor_infinity_is_not_an_error():
    assert jnp.isinf(exptime_from_rates_ayo(0.5, 0.5, 1.5 / 7.0, 7.0))


@pytest.mark.parametrize("bad", [-1.0, jnp.nan, jnp.inf])
def test_rates_must_be_finite_and_nonnegative(bad):
    with pytest.raises(Exception, match="finite and non-negative"):
        check_finite_nonneg(jnp.asarray(bad), "Cp")
