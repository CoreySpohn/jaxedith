"""Property tests with hypothesis.extra.numpy replacing hand-picked grids."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxedith.etc import (
    exptime_from_rates_ayo,
    exptime_from_rates_exosims_char,
    exptime_from_rates_exosims_det,
    snr_from_rates_ayo,
    snr_from_rates_exosims_char,
    snr_from_rates_exosims_det,
)

# (exptime, snr, name of the third rate argument's role does not matter)
VARIANTS = {
    "ayo": (exptime_from_rates_ayo, snr_from_rates_ayo, 0.05),
    "exosims_det": (exptime_from_rates_exosims_det, snr_from_rates_exosims_det, 0.05),
    "exosims_char": (
        exptime_from_rates_exosims_char,
        snr_from_rates_exosims_char,
        0.05,
    ),
}

SMALL = settings(deadline=None)


def _rate_arrays(shapes):
    """Planet above the floor (Cp >= 0.5, floor term <= 0.05 * 10 = 0.5)."""
    cp = hnp.arrays(np.float64, shapes[0], elements=st.floats(1.0, 10.0))
    cb = hnp.arrays(np.float64, shapes[1], elements=st.floats(0.01, 10.0))
    floor = hnp.arrays(np.float64, shapes[2], elements=st.floats(0.0, 0.05))
    return st.tuples(cp, cb, floor)


broadcastable = hnp.mutually_broadcastable_shapes(
    num_shapes=3, min_dims=0, max_dims=3, max_side=3
).map(lambda s: s.input_shapes)


@pytest.mark.parametrize("variant", VARIANTS)
@SMALL
@given(
    data=broadcastable.flatmap(_rate_arrays),
    snr=st.floats(1.0, 10.0),
)
def test_round_trip_and_broadcast(variant, data, snr):
    """snr(exptime(snr)) recovers snr; the result has the broadcast shape."""
    exptime, snr_from, _ = VARIANTS[variant]
    cp, cb, floor = data
    with jax.enable_x64():
        t = exptime(cp, cb, floor, snr)
        recovered = snr_from(cp, cb, floor, t)
    assert t.shape == np.broadcast_shapes(cp.shape, cb.shape, floor.shape)
    assert recovered.shape == t.shape
    np.testing.assert_allclose(np.asarray(recovered), snr, rtol=1e-6)


@SMALL
@given(
    cp=st.floats(1.0, 10.0),
    cb=st.floats(0.01, 10.0),
    floor=st.floats(0.0, 0.05),
    snr=st.floats(1.0, 9.0),
)
def test_exptime_grows_with_snr(cp, cb, floor, snr):
    """Raising the target SNR never shortens the required time."""
    with jax.enable_x64():
        t_lo = exptime_from_rates_ayo(cp, cb, floor, snr)
        t_hi = exptime_from_rates_ayo(cp, cb, floor, snr * 1.1)
    assert float(t_hi) > float(t_lo) > 0.0
    assert jnp.isfinite(t_lo)


@SMALL
@given(
    cp=st.floats(1.0, 10.0),
    cb=st.floats(0.01, 10.0),
    floor=st.floats(0.0, 0.05),
    snr=st.floats(1.0, 9.0),
)
def test_exptime_shrinks_with_planet_rate(cp, cb, floor, snr):
    """A brighter planet never needs a longer exposure."""
    with jax.enable_x64():
        t_faint = exptime_from_rates_ayo(cp, cb, floor, snr)
        t_bright = exptime_from_rates_ayo(cp * 1.1, cb, floor, snr)
    assert float(t_bright) < float(t_faint)
