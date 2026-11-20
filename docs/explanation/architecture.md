# Architecture

The path is `Rates -> detector -> snr or exposure_time`. A `Rates` record holds
mean and squared-weight photon rates with shape `(W, Q)` and both per-band
detector pixel counts. Integration gives photons/s, while `detector.electrons`
applies wavelength-dependent yield before integrating to electrons/s. `snr` and
`exposure_time` use the detector's `mean_and_variance` and
`optixstuff.frames_for_time`.

`PostProcessing` owns the two residual fractions. It adds squared residual
counts after detector variance; it never changes photon or electron rates.

The scalar cross-code references live only in `tests/reference/`; see
[comparison references](variants.md). They do not supply any calculator rates
or detector statistics.
