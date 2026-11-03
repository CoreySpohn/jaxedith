"""Opt-in runtime invariants, enabled with ``JAXEDITH_DEBUG_CHECKS=1``.

The flag is read once at import and is a plain Python bool, so when it is off
the helpers return their argument untouched and add nothing to the traced
program. When it is on, each check is an ``equinox.error_if`` that raises from
inside ``jit``, ``vmap`` and ``grad`` alike.
"""

import os

import equinox as eqx
import jax.numpy as jnp

DEBUG_CHECKS = os.environ.get("JAXEDITH_DEBUG_CHECKS", "0") == "1"


def check_finite_nonneg(x, name):
    """Raise if ``x`` has a NaN, an infinity or a negative entry."""
    if not DEBUG_CHECKS:
        return x
    x = jnp.asarray(x)  # error_if needs an array to thread the error onto
    bad = ~jnp.isfinite(x) | (x < 0)
    return eqx.error_if(x, bad, f"{name} must be finite and non-negative")


def check_not_nan(x, name):
    """Raise if ``x`` has a NaN entry (infinity passes)."""
    if not DEBUG_CHECKS:
        return x
    x = jnp.asarray(x)
    return eqx.error_if(x, jnp.isnan(x), f"{name} must not be NaN")
