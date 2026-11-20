# jaxedith

`jaxedith` computes S/N and whole-frame exposure plans from an aperture photon
budget, using the detector's shared statistics contract.

It reads an `optixstuff.Rates` record: photon integrands weighted by an
aperture, with six fields in photons/s/nm of shape `(W, Q)`: `planet`,
`star_leakage`, `local_zodi`, `exozodi`, `thermal`, and `stellar_companions`.
Its `variance` field is a second `Rates` record weighted by the squared weights,
and `rates.n_pix` and `rates.variance.n_pix` give the weighted detector pixel
counts. `rates.integrate()` returns mean photons/s.
`detector.electrons(rates)` applies spectral QE and survival before integration
to give mean electrons/s, and converts the variance record with it.

## Usage

This example requires a `rates` record and an `instrument`.

```python notest
from optixstuff import frames_for_time
from jaxedith import PostProcessing, exposure_time, snr

post_processing = PostProcessing(
    speckle_residual_fraction=0.1,
    exozodi_residual_fraction=0.0,
)
plan = frames_for_time(3600.0, frame_time_s=100.0)
achieved = snr(
    rates,
    instrument.detector,
    plan,
    post_processing=post_processing,
)
required = exposure_time(
    10.0,
    rates,
    instrument.detector,
    frame_time_s=100.0,
    post_processing=post_processing,
)
seconds = required.live_time_s
occupied_seconds = required.occupied_time_s(instrument.detector)
```

The record comes from the owner of the photon budget; this package does not
build one.

## Noise and exposure time

`PostProcessing` requires two fractions in `[0, 1]`; zero means perfect
subtraction. With detected rates `p`, `ell`, `d` and live time `t`,

```text
signal = p*t
variance = detector_variance + (f_speckle*ell*t)^2 + (f_exozodi*d*t)^2
```

The detector supplies source shot noise, dark, CIC, excess noise and read
variance through `mean_and_variance`, using the squared-weight rates and pixel
count. Signal and residual speckle and disk rates use the mean weights. No implicit ADI/background multiplier
is applied. Entire final frames collect photons; `frames_for_time` is the
only frame counter. Read overhead is separate from live time.

Write `b = variance_of_one_frame / frame_time_s` and
`f2 = (f_speckle*ell)^2 + (f_exozodi*d)^2`. The continuous solution is
`t = target_snr^2*b / (p^2 - target_snr^2*f2)`, then rounded to whole frames.
For a positive target the solution is infinite when
`p^2 <= target_snr^2*f2`; its `FramePlan` has infinite count and live time.
Nonpositive targets require zero frames. S/N at an infinite plan returns its
asymptotic ceiling. Planning is discrete; S/N and rate calculations support
continuous-parameter gradients. Enable JAX x64 for faint-source accuracy.

## Cross-code references

Scalar pyEDITH, EXOSIMS and AYO comparison implementations live in
`tests/reference/` in the source checkout. They support cross-code benchmarks
and convention checks, and are excluded from the wheel and source distribution.
See [comparison references](docs/explanation/variants.md) for their scope.

## Installation

```bash
pip install jaxedith
```

The calculator requires the shared `Rates`, `Quadrature` and detector contracts. The development tests use their
coordinated unreleased versions; published dependency floors have not yet
been verified for this API.
