# Architecture

The primary path is `System -> aperture_rates -> ApertureRates -> detector`.
The reader accepts bands, a shared `Quadrature`, or a `PreparedAperture`.
For bands it builds nodes and dimensional weights on the host from the
active spectra and instrument.
The reader returns mean and squared-weight photon rates with shape `(W, Q)`
and both per-band detector pixel counts. Integration gives
photons/s, while `detector.electrons` applies wavelength-dependent yield before
integrating to electrons/s. `snr` and `exposure_time` use the detector's
`mean_and_variance` and `optixstuff.frames_for_time`.

The default aperture is `Circular(0.7)`, fixed over each band at its reference
wavelength. The primary frame converts the record's area into steradians.
`optixstuff.prepare_aperture(instrument, quadrature)` returns an
`optixstuff.PreparedAperture` for `aperture_rates(system, instrument, aperture, ...)`.
It prepares `Circular(r_ref * lambda_ref / lambda_q)` records eagerly and
evaluates separation in node lambda/D, keeping the reference solid angle.
Scalar models describe a calibrated fixed angular aperture without a PSF
scaling law. Polychromatic `HalfMaximum` and `Matched` table apertures require
upstream frozen-template dilation and currently raise `NotImplementedError`.
Stellar leakage is already an aperture sum. Exozodi defaults to `zodi.jax.AYOExozodi()`:
an analytic AB-zero-point surface brightness at three zodis, 22 mag/arcsec²
per zodi at 1 AU, scaled by `(dist_pc * separation_arcsec)**-2`.
`exozodi=None` gives zero. A supplied analytic model declares
its own breakpoints through `spectral_breaks_nm()`.
With `exozodi="scene"`, `System.photon_radiance_at` evaluates the disk at
the aperture centre and the reader multiplies by aperture solid angle.
The sky offsets and scene time support JAX transformations.
An image comparison must bound PSF redistribution and reconstruction as well
as centre-to-footprint radiance differences.

`PostProcessing` owns the two residual fractions. It adds squared residual
counts after detector variance; it never changes photon or electron rates.

The scalar cross-code references live only in `tests/reference/`; see
[comparison references](variants.md). They do not supply any calculator rates
or detector statistics.
