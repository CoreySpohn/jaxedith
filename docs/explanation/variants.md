# Comparison references

jaxedith provides `aperture_rates`, `PostProcessing`, `snr` and `exposure_time`. Fixed angular
aperture preparation, detector yield, variance and frame counting come from
optixstuff.

The source checkout's `tests/reference/` package holds plain-number
pyEDITH, EXOSIMS and AYO reference implementations. Its `primitives.py`
contains scalar electron-rate formulas, `etc.py` the rate-to-S/N and
time equations, `scene.py` their monochromatic input record, and
`legacy.py` test adapters for scene, processing and cadence conventions.
This test support is excluded from both the wheel and source distribution.

Cross-code benchmarks import these references from the test suite and carry
`vv.case(..., "cross-code-benchmark", reference_code=...)` markers.
Algebra, differentiation and broadcasting checks of the reference equations
remain code verification. These tests compare conventions and recorded
pyEDITH outputs; they do not execute live upstream codes or establish
agreement with measured data.

The comparisons match photon zero points, aperture areas, detector yield and
spectral weighting explicitly. A centre-times-width reference agrees with
band integration for flat photon spectra and constant response, while a
chromatic band needs its stated approximation bound.
