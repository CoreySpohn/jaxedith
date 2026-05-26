# ETC variants

`jaxedith` ships three ETC equations side by side, each exposed as
a parallel family of functions:

| Variant | Origin | Background multiplier | Noise term |
|---|---|---|---|
| `ayo` | AYO / [pyEDITH](https://github.com/eleonoraalei/pyEDITH/) | 2× (ADI assumption) | analytical `Cnf_rate = raw_contrast × ppfact` |
| `exosims_det` | EXOSIMS detection (Nemati model) | 1× | speckle residual `Csp = stellar_leakage × ppfact × stability_fact` |
| `exosims_char` | EXOSIMS characterization | 1× + `Cp` self-noise | speckle residual `Csp` |

The variant is baked into the function name; there is no runtime
dispatch. Pick the variant that matches your reference convention
and call directly.

## When to use each

**AYO** implements the AYO / [pyEDITH](https://github.com/eleonoraalei/pyEDITH/)
ETC equations, which are used by the HWO project office for
mission-concept yield work. It assumes angular differential imaging
(ADI) processing -- two telescope rolls combined to suppress
speckles, with the photon-noise budget doubled to reflect the
variance penalty of subtraction.

**EXOSIMS detection** is the inner-loop ETC for the EXOSIMS Nemati
formulation. Use it for cross-validation against EXOSIMS yield runs
or for studies that need to match the EXOSIMS convention. The speckle
term is `Csp = stellar_leakage × ppfact × stability_fact`, modelled
explicitly rather than via a noise floor.

**EXOSIMS characterization** is the same Nemati equation with an
extra `+ Cp` term in the noise budget representing the planet's own
shot-noise contribution -- relevant when integrating long enough on
a bright-enough source that the planet itself contributes
non-negligibly to the variance.

## Three function flavours per variant

For each variant, three entry points:

- `count_rates_<variant>(...)` -- returns the rate triple
  `(Cp, Cb, Cnf_rate)` for AYO or `(Cp, Cb, Csp)` for EXOSIMS.
  Useful for inspecting which term dominates the budget.
- `exptime_<variant>(..., snr)` -- solves for the exposure time
  required to reach a target SNR.
- `snr_<variant>(..., t_obs)` -- inverts the previous: given a fixed
  observing time, what SNR do you achieve?

System-level wrappers (`*_from_system_<variant>`) accept a
{class}`skyscapes.System` + {class}`optixstuff.OpticalPath` +
observatory + exposure + post-processing config; they `jax.vmap` the
scalar core over `(K planets, T epochs)`.

## What the variants share

All three:
- Use the same Layer 2 intermediates for planet signal, stellar
  leakage, zodi background, exozodi background, binary background,
  thermal background, and detector noise.
- Apply the same overhead model: multiplicative overhead, fixed
  overhead in seconds, and a `n_rolls` factor for total observing
  time.
- Are JAX-pure: JIT-compatible, `vmap`-able, and differentiable
  end-to-end.

The only thing that changes between variants is the *closed-form
ETC equation* applied to the rate triple -- see
{mod}`jaxedith.etc` for the three exptime-solve + three snr-solve
formulas side by side.

## Picking a variant in code

```python
# AYO
from jaxedith import exptime_ayo, exptime_from_system_ayo

# EXOSIMS detection
from jaxedith import exptime_exosims_det, exptime_from_system_exosims_det

# EXOSIMS characterization
from jaxedith import exptime_exosims_char, exptime_from_system_exosims_char
```

All three have the same call signature shape (modulo a handful of
variant-specific kwargs like `stability_fact` for EXOSIMS or
`bg_multiplier` for AYO). Swapping variants in a study is a
one-line import change.

## Cross-variant tests

The test suite includes round-trip parity tests for each variant:
solve for `t_exp` at target SNR, then run `snr_*` at that `t_exp`
and check we get back the target. The Layer 1 primitives + Layer 2
intermediates are shared, so any drift between variants shows up as
an `etc.py` math difference rather than a hidden component-level
mismatch.

## See also

- [Architecture](architecture) -- the four-layer pipeline (primitives,
  intermediates, etc, public) and how the variants compose with the
  shared lower layers.
