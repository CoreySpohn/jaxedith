"""ETCConfig: equinox module that selects the ETC equation variant.

The ``variant`` field is a plain Python string used for trace-time dispatch —
it selects which solver function to call rather than branching inside a single
function.  This means JIT compiles only the chosen code path and ``vmap``
traces only one solver, avoiding dead-branch overhead.
"""

import equinox as eqx


class ETCConfig(eqx.Module):
    """Configuration that determines which ETC equation variant to use.

    The ``variant`` field drives solver dispatch at trace time:

    - ``"ayo"`` — AYO / Stark et al. equations (ADI background doubling,
      data-driven noise floor).
    - ``"jaxedith"`` — Same as AYO plus exozodi noise floor extension.
    - ``"exosims_det"`` — EXOSIMS detection (no ADI factor, speckle residual).
    - ``"exosims_char"`` — EXOSIMS characterization (Cp added to Cb).

    All other fields are static parameters passed directly to the selected
    solver — no boolean flags or ``jnp.where`` branching.

    Attributes:
        variant: Equation variant string (see above).
        bg_multiplier: ADI background multiplier.  AYO/jaxedith: 2.0;
            EXOSIMS: 1.0.
        ppfact: Post-processing factor (EXOSIMS only).
        stability_fact: Wavefront stability factor (EXOSIMS only).
        overhead_multi: Multiplicative overhead factor.
        overhead_fixed_s: Fixed overhead in seconds.
        n_rolls: Number of telescope rolls.
        include_exozodi_noise_floor: Whether the exozodi noise floor is
            included (jaxedith extension).
        npix_multiplier: Pixel count correction factor for the photometric
            aperture.  ETC-specific, not a hardware property.
    """

    # --- Solver variant (trace-time dispatch) ---
    variant: str = "jaxedith"

    # --- Equation parameters ---
    bg_multiplier: float = 2.0
    ppfact: float = 1.0
    stability_fact: float = 1.0

    # --- Overheads ---
    overhead_multi: float = 1.0
    overhead_fixed_s: float = 0.0
    n_rolls: int = 1

    # --- Exozodi noise floor (jaxedith extension) ---
    include_exozodi_noise_floor: bool = True

    # --- Aperture correction ---
    npix_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

AYO_CONFIG = ETCConfig(
    variant="ayo",
    bg_multiplier=2.0,
    include_exozodi_noise_floor=False,
)
"""AYO preset — original Stark et al. equations without exozodi noise floor."""

CONFIG = ETCConfig(
    variant="jaxedith",
    bg_multiplier=2.0,
    include_exozodi_noise_floor=True,
)
"""jaxedith preset — AYO equations plus exozodi noise floor."""

EXOSIMS_DETECTION_CONFIG = ETCConfig(
    variant="exosims_det",
    bg_multiplier=1.0,
    include_exozodi_noise_floor=False,
)
"""EXOSIMS detection preset — no ADI factor, speckle residual instead of noise floor."""

EXOSIMS_CHARACTERIZATION_CONFIG = ETCConfig(
    variant="exosims_char",
    bg_multiplier=1.0,
    include_exozodi_noise_floor=False,
)
"""EXOSIMS characterization preset — Cp added to Cb."""
