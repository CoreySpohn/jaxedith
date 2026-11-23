# Architecture

The path is `curves, scene, instrument -> rates -> detector -> snr or
exposure_time`. `jaxedith.rates` reads the curves at the planet's separation
in lambda/D at each band centre and multiplies each source by the collecting
area, the system throughput and the aperture, per spectral node. It reads no
spectrum: the star, the local zodi and the disk's host star are node rows from
`scene.sources`, which integrates a tabulated spectrum exactly over each band.
It returns an
`optixstuff.Rates` record: mean and squared-weight photon rates of shape
`(W, Q)` and per-band pixel counts, for one planet. `jaxedith.planet_exposure_time`
and `jaxedith.planet_snr` are the one-call entry points for one planet: `rates`
followed by the optixstuff function on `instrument.detector`, with no arithmetic
of their own. `jaxedith.scene_rates` is a `vmap` of `rates` over every planet of
the scene and every time, which computes `scene.sources` once outside the map and
passes `sources.contrast`, in the layout of `coronagraphoto.scene_aperture_rates`:
fields of shape `(W, Q, K, T)` and `n_pix` of shape `(W, K, T)`, for `K` planets
and `T` times. A planet with `scene.valid` False reads zero, so its exposure time
is invalid. A scene with a batch of systems is not read; map the function over them.

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
