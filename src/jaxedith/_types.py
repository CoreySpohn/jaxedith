"""Shape and dtype aliases for jaxtyping annotations.

The annotations are documentation in production and are enforced only in the
test suite, where ``tests/conftest.py`` installs a jaxtyping import hook.
``"..."`` accepts any shape, so scalars, batches and broadcasting all pass;
the check is on dtype kind and on array-likeness.
"""

from jaxtyping import Array, ArrayLike, Float, Real

# Input: Python or NumPy scalar, or JAX array, of any shape. Integers are
# allowed because counts such as ``n_channels`` and ``n_rolls`` are ints.
Num = Real[ArrayLike, "..."]

# Output: floating result. ``ArrayLike`` (not ``Array``) because a function
# fed only Python floats returns a Python float.
FloatOut = Float[ArrayLike, "..."]

# Per-planet, per-epoch grids used by the ``*_from_system_*`` wrappers. The
# names ``k`` (planets) and ``t`` (epochs) bind across every annotation in one
# function, so a (K, T) result built from a (K, T') input fails.
PerKT = Float[Array, "k t"]
PerT = Float[Array, " t"]
Scalar = Float[ArrayLike, ""]
ScalarArray = Float[Array, ""]
