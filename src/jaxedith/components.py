"""Layer 2: Structured count-rate components.

Each function accepts an ``optixstuff.OpticalPath`` and astrophysical /
observation scalars, unpacks the optical path, and calls the corresponding
pure Layer 1 function in :mod:`jaxedith.count_rates`.

Layer 2 turns the 8-to-12-argument pure-float call sites in
:mod:`jaxedith.count_rates` into structured calls that take an
``OpticalPath`` + a few scalars, without introducing heavy scene objects.

Each Layer 2 wrapper mirrors a single invocation inside
``jaxedith.core._compute_count_rates``; parity is tested in
``tests/test_components.py``.
"""
