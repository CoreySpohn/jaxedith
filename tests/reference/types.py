"""Shape and dtype aliases for the reference code's jaxtyping annotations.

``"..."`` accepts any shape, so scalars, batches and broadcasting all pass;
the check is on dtype kind and on array-likeness.
"""

from jaxtyping import ArrayLike, Float, Real

# Input: Python or NumPy scalar, or JAX array, of any shape. Integers are
# allowed for counts such as the target number of frames.
Num = Real[ArrayLike, "..."]

# Output: floating result. ``ArrayLike`` (not ``Array``) because a function
# fed only Python floats returns a Python float.
FloatOut = Float[ArrayLike, "..."]
