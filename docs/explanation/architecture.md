# Architecture

The primary path is `System -> aperture_rates -> PhotonRates -> detector`.
`aperture_quadrature` prepares shared wavelength nodes and dimensional weights.
The reader returns node photon rates with shape `(W, Q)`; integration gives
photons/s, while `detector.electrons` applies wavelength-dependent yield before
integrating to electrons/s. `snr` and `exposure_time` use the detector's
`mean_and_variance` and `optixstuff.frames_for_time`.

The default aperture is `Circular(0.7)`, fixed over each band at its reference
wavelength. The primary frame converts the record's area into steradians.
Stellar leakage is already an aperture sum. Disk radiance is sampled at the
planet and multiplied by aperture solid angle; no extra host flux is applied.
The one-pixel default needs concrete sky geometry. Traced moving positions
use a caller-supplied fixed sky grid and bilinear centre sampling.

`PostProcessing` owns the two residual fractions. It adds squared residual
counts after detector variance; it never changes photon or electron rates.

The comparison layer is separate: `primitives` holds plain-number rates,
`etc` holds legacy equations, and `legacy` adapts scene and processing
conventions. It has no second optical-path derivation. `ETCScene` retains its
historical monochromatic units and is produced by `etc_scene_from_system`.
