# jaxedith

`jaxedith` reads a `skyscapes.System` and an `optixstuff.OpticalPath` into
an aperture photon budget, then computes S/N and whole-frame exposure plans
using the detector's shared statistics contract.

`aperture_rates` returns `optixstuff.PhotonRates`: six photon integrands in
photons/s/nm with shape `(W, Q)` and no spatial axes. The fields are `planet`,
`star_leakage`, `local_zodi`, `exozodi`, `thermal`, and `stellar_companions`.
`rates.integrate(quadrature)` returns photons/s. Only
`detector.electrons(rates, quadrature)` converts them to electrons/s, applying
spectral QE and survival at the nodes before integration.

## Usage

This example requires a configured `system` and `optical_path`.

```python notest
from hwoutils.spectral import Band
from optixstuff import frames_for_time
from jaxedith import (
    PostProcessing,
    aperture_quadrature,
    aperture_rates,
    exposure_time,
    prepare_aperture,
    snr,
)

band = Band(500.0, 600.0)
quadrature = aperture_quadrature(system, optical_path, band)
aperture = prepare_aperture(optical_path, band, quadrature)
rates = aperture_rates(
    system,
    optical_path,
    band,
    time_d=0.0,
    planet_index=0,
    prepared_aperture=aperture,
)
post_processing = PostProcessing(
    speckle_residual_fraction=0.1,
    exozodi_residual_fraction=0.0,
)
plan = frames_for_time(3600.0, frame_time_s=100.0)
achieved = snr(
    rates,
    optical_path.detector,
    quadrature,
    plan,
    n_pix=12,
    post_processing=post_processing,
)
required = exposure_time(
    10.0,
    rates,
    optical_path.detector,
    quadrature,
    frame_time_s=100.0,
    n_pix=12,
    post_processing=post_processing,
)
seconds = required.live_time_s
occupied_seconds = required.occupied_time_s(optical_path.detector)
```

Select one flattened `planet_index`, or pass `separation_arcsec` and
`position_angle_deg` for a background-only aperture. Position angle is north
through east; `time_d` is scalar days from the system epoch. Multiple `Band`
objects produce independent rows. Use `vmap` over times or planet indices.
The default rule is `yippy.Circular(0.7)`. Its solid angle is fixed across each
band at the centre, or at an explicit `reference_wavelength_nm`, using the
primary's frame and the curve record's diameter convention. Leakage reads
`leakage_fraction` directly, never a density times an inferred pixel area.
Table coronagraphs supply background transmission from their native image at
`(separation_lod, 0)`; scalar coronagraphs supply their calibrated scalar
transmission. Thus this is a radial aperture performance reader.

At wavelength `lambda_q`, both the radius and separation in lambda/D scale
by `lambda_ref / lambda_q`. `prepare_aperture` prepares each scaled rule
through the coronagraph API; pass its `PreparedAperture` to the reader before
JIT, and rebuild it when the instrument, rule, reference or quadrature changes.
An eager call prepares it automatically when omitted. `HalfMaximum` and
`Matched` need upstream support for dilating a frozen reference template;
polychromatic table models raise `NotImplementedError` for these rules.
Scalar coronagraphs instead describe a calibrated fixed angular aperture:
their constants and wavelength functions already refer to that aperture,
and their area is in reference lambda/D squared. No PSF scaling is inferred.

Quadrature splits at stellar, planetary, optical-throughput, local-zodi,
analytic-exozodi, QE and multiband-dispatch breakpoints. Disk knots enter
only with `exozodi="scene"`. Custom spectral models declare
`quadrature_knots_nm` or `wavelengths_nm`; wavenumber tables use `nu_grid`.
The reader builds quadrature if omitted. Prepare it explicitly to reuse the
identical weights in the detector, or before tracing model table arrays.
Host preparation uses NumPy; node evaluation and statistics use JAX.

## Backgrounds and sampling

Local zodi defaults to `zodi.jax.AYOZodi()`. Supply a local-zodi
model and `ecliptic_latitude_deg` / `solar_longitude_deg` for pointing dependence,
or `zodi=None` to disable it. Solar longitude means target minus solar
longitude. The caller computes pointing geometry from its observatory.

Exozodi defaults to `jaxedith.AYOExozodi(level_zodis=3.0,
surface_brightness_mag_arcsec2=22.0)`. Its photon radiance is the exact AB
magnitude-zero photon flux times `level_zodis * 10**(-0.4 * 22)` per arcsec²
at 1 AU, converted to steradians and scaled by
`(system.dist_pc * separation_arcsec)**-2`. The brightness parameter replaces
22 in that expression. This is the scalar AYO calculator dialect: the ratio's
denominator is the AB zero-magnitude source, with flat AB colour and no host
luminosity or inclination multiplier. The model has no spectral table knots.
A model supplied as `exozodi=` must provide
`photon_radiance(wavelength_nm, *, separation_au)` in photons/s/m²/nm/sr;
pass the same model to `aperture_quadrature` to include any declared knots.

The default evaluates one scalar per wavelength node and ignores `system.disk`
and `disk_grid`, including for background-only apertures. `exozodi=None`
disables exozodi. Select `exozodi="scene"` in both `aperture_quadrature` and
`aperture_rates` to use the scene disk instead.

In scene mode, disk light comes from `System.photon_radiance`, sampled at the aperture centre
on a one-pixel sky grid and multiplied by the fixed solid angle. A missing
disk gives zero. This approximates a footprint integral: the calibrated
smooth face-on test bounds the difference by 5% for its stated geometry,
not for arbitrary disk structure. For traced moving positions, supply a
fixed `disk_grid` in arcseconds with `SkyOrientation`, covering the orbit.
The reader renders the disk on that grid and bilinearly samples the centre;
out-of-grid samples are NaN. Grid convergence is the caller's responsibility.
Comparison with a redistributed disk image additionally requires a bound on
PSF smoothing and image resampling; a centre-to-footprint bound alone does
not cover those operations.

Thermal emission retains the Planck photon law with `temperature_K=270` and
`eps_warm_T_cold=0` by default. Its emissivity times cold transmission replaces
optical throughput. Optional `companion_flux_density(wavelength_nm, time_d)`
returns photons/s/m2/nm per reference lambda/D squared, retaining the legacy
stray-light area law. It defaults to zero. No F0-relative ratios enter the reader.
One call is one optical path: historical `n_channels` and `npix_multiplier`
metadata do not multiply the new rates; supply the effective `n_pix` explicitly.

## Noise and exposure time

`PostProcessing` requires two fractions in `[0, 1]`; zero means perfect
subtraction. With detected rates `p`, `ell`, `d` and live time `t`,

```text
signal = p*t
variance = detector_variance + (f_speckle*ell*t)^2 + (f_exozodi*d*t)^2
```

The detector supplies source shot noise, dark, CIC, excess noise and read
variance through `mean_and_variance`. No implicit ADI/background multiplier
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

## Legacy comparisons

`jaxedith.primitives` retains the scalar AYO rate primitives and
`jaxedith.etc` retains the six AYO/EXOSIMS rate-to-S/N or time equations,
with their original electron units and numerics. `jaxedith.legacy` provides
`etc_scene_from_system`, `post_processing_from_pyedith`,
`post_processing_from_exosims`, `frame_plan_photon_counting`, and
`legacy_pyedith_noise_rate`. `ETCScene` remains in `jaxedith.scene` and is
produced by the scene adapter. These are comparison tools, separate from the
shared photon reader. The old optical-path `count_rates_*`, `exptime_*`,
`snr_*` and `*_from_system_*` families, `intermediates`, local zodi wrappers,
and duplicate cadence calculators have been removed without aliases.

The photon-counting adapter chooses a frame time with mean `1/6.73` electrons
per pixel. pyEDITH's `ppfact` divides leakage; EXOSIMS `ppFact` is the residual
fraction itself. The adapters make that distinction explicit.

## Installation

```bash
pip install jaxedith
```

This redesign requires the shared `Band`, `Quadrature`, `PixelGrid`, photon
scene, aperture-curve and detector contracts. The development tests use their
coordinated unreleased versions; published dependency floors have not yet
been verified for this API.
