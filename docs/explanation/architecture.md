# Architecture

The primary path is `System -> aperture_rates -> PhotonRates -> detector`.
`aperture_quadrature` prepares shared wavelength nodes and dimensional weights.
The reader returns node photon rates with shape `(W, Q)`; integration gives
photons/s, while `detector.electrons` applies wavelength-dependent yield before
integrating to electrons/s. `snr` and `exposure_time` use the detector's
`mean_and_variance` and `optixstuff.frames_for_time`.

The default aperture is `Circular(0.7)`, fixed over each band at its reference
wavelength. The primary frame converts the record's area into steradians.
`optixstuff.prepare_aperture(optical_path, bands, quadrature)` returns an
`optixstuff.PreparedAperture` for `aperture_rates(..., prepared_aperture=aperture)`.
It prepares `Circular(r_ref * lambda_ref / lambda_q)` records eagerly and
evaluates separation in node lambda/D, keeping the reference solid angle.
Scalar models describe a calibrated fixed angular aperture without a PSF
scaling law. Polychromatic `HalfMaximum` and `Matched` table apertures require
upstream frozen-template dilation and currently raise `NotImplementedError`.
Stellar leakage is already an aperture sum. Exozodi defaults to `zodi.jax.AYOExozodi()`:
an analytic AB-zero-point surface brightness at three zodis, 22 mag/arcsec²
per zodi at 1 AU, scaled by `(dist_pc * separation_arcsec)**-2`.
The scene disk and `disk_grid` are ignored unless `exozodi="scene"`;
`exozodi=None` gives zero. Pass the same exozodi model to
`aperture_quadrature`, which collects its spectral knots too.
In scene mode, disk radiance is sampled at the
planet and multiplied by aperture solid angle; no extra host flux is applied.
The scene-mode one-pixel grid needs concrete sky geometry. Traced moving positions
use a caller-supplied fixed sky grid and bilinear centre sampling.
An image comparison must bound PSF redistribution and reconstruction as well
as centre-to-footprint radiance differences.

`PostProcessing` owns the two residual fractions. It adds squared residual
counts after detector variance; it never changes photon or electron rates.

The scalar cross-code references live only in `tests/reference/`; see
[comparison references](variants.md). They do not supply any calculator rates
or detector statistics.
