# Architecture

The path is `curves, scene, instrument -> rates -> detector -> snr or
exposure_time`. `jaxedith.rates` reads the curves at the planet's separation
in lambda/D at each band centre and multiplies each source by the collecting
area, the system throughput and the aperture, per spectral node. It returns an
`optixstuff.Rates` record: mean and squared-weight photon rates of shape
`(W, Q)` and per-band pixel counts. `jaxedith.scene_exposure_time` and `jaxedith.scene_snr`
are the one-call entry points: `rates` followed by the optixstuff function on
`instrument.detector`, with no arithmetic of their own.

`detector.electrons` applies wavelength-dependent yield and integrates over
each band. `optixstuff.snr` and `optixstuff.exposure_time` read the detector's
`mean_and_variance` and the `optixstuff.PostProcessing` record; the same
reduction serves the image path of coronagraphoto. jaxedith owns neither the
reduction, the zodi models (skyscapes), the curves (yippy measures them) nor
overheads and rolls (the survey simulator).

The same reduction, reached from the image path, is described in optixstuff's
[how an exposure time is computed](https://optixstuff.readthedocs.io/en/latest/explanation/exposure_time.html).

The scalar cross-code references live only in `tests/reference/`; see
[comparison references](variants.md). They do not supply any calculator rates
or detector statistics.
