# jaxedith

`jaxedith` turns performance curves, a scene, an instrument and a planet
position into analytic aperture rates: the scalar AYO and pyEDITH count-rate
terms, evaluated in JAX. It returns an `optixstuff.Rates` record, which
`optixstuff.snr` and `optixstuff.exposure_time` reduce to a signal-to-noise
ratio or an exposure time. The package has five functions: `rates`, `scene_rates`,
`photon_counting_frame_time`, and the one-call `planet_exposure_time` and `planet_snr`.

## Usage

This example needs `curves` (an `optixstuff.PerformanceCurves` measured for the
telescope and star, for example from yippy's `coronagraph.curves(rule)`), a
`skyscapes.Scene`, an `optixstuff.Instrument` and the bands to read.

The curves are read at the band-centre lambda/D for every node, so the response
does not slide across a pixel and the quadrature is built with `field_lod=0`: a
wider field would add nodes and change nothing. The scene's tabulated spectra
(a PHOENIX star, a local-zodi table, a planet model with a grid) are integrated
exactly over each band by an operator built on the host, and `scene.sources` reads
them once on the nodes. A band must lie inside every source table.

```python notest
import jaxedith
import optixstuff as ox
from optixstuff.spectral import Quadrature, Resolution

quadrature = Quadrature.build(
    bands, instrument, scene.local_zodi, field_lod=0
)  # the instrument may be passed as a break model: field_lod=0 keeps `order` nodes
resolution = Resolution.build(quadrature, *scene.spectral_tables())
sources = scene.sources(quadrature, resolution, time_d)
contrast = sources.contrast[:, :, planet, 0]  # (W, Q), or pass a scalar contrast

rates = jaxedith.rates(
    curves,
    scene,
    instrument,
    quadrature,
    east_arcsec,
    north_arcsec,
    contrast,
    time_d=time_d,
    sources=sources,  # node rows of the one time read
    uniform_sky=uniform_sky,  # the sky transmission profile
)
frame_time_s = jaxedith.photon_counting_frame_time(rates, instrument.detector)
time = ox.exposure_time(
    rates,
    instrument.detector,
    7.0,
    frame_time_s=frame_time_s,
    post_processing=ox.PostProcessing(1 / 30, 0.0, background_variance_factor=2.0),
)
frames = ox.frames_for_time(time.time_s, frame_time_s)
```

For pyEDITH and EXOSIMS users the one-call form, which answers for one planet, takes the same arguments plus
the target and the frame time, and returns what the two steps above return:

```python notest
time = jaxedith.planet_exposure_time(
    curves,
    scene,
    instrument,
    quadrature,
    east_arcsec,
    north_arcsec,
    contrast,
    time_d=time_d,
    sources=sources,
    uniform_sky=uniform_sky,
    target_snr=7.0,
    frame_time_s=frame_time_s,
    post_processing=ox.PostProcessing(1 / 30, 0.0, background_variance_factor=2.0),
)
snr = jaxedith.planet_snr(
    curves,
    scene,
    instrument,
    quadrature,
    east_arcsec,
    north_arcsec,
    contrast,
    time_d=time_d,
    sources=sources,
    uniform_sky=uniform_sky,
    time_s=3600.0,
    frame_time_s=frame_time_s,
    post_processing=ox.PostProcessing(1 / 30, 0.0, background_variance_factor=2.0),
)
```

`jaxedith.planet_exposure_time` is `rates` followed by `optixstuff.exposure_time` on
`instrument.detector`, and `jaxedith.planet_snr` is `rates` followed by
`optixstuff.snr`; neither adds arithmetic. They return an
`optixstuff.ExposureTime` and the S/N per band.

`rates` is a scalar read of one planet that reads no spectrum itself: the star, the local
zodi and the disk's host star are the node rows of `sources`, and `contrast` is a scalar or
the star-weighted node contrast `sources.contrast`. Map it over positions and contrasts with
`jax.vmap`.
Fields are photons/s/nm of shape `(W, Q)`, with `n_pix` per band and a
squared-weight `variance` record. Detector yield is not applied there:
`detector.electrons(rates)` applies QE and survival once, downstream.

To read every planet of the scene at every time, `scene_rates` is a `vmap` of `rates`
over the planets and the times that reads `scene.sources` once outside the map, and takes the same layout as
`coronagraphoto.scene_aperture_rates`, so the two reads swap:

```python notest
record = jaxedith.scene_rates(
    curves,
    scene,
    instrument,
    quadrature,
    time_d,
    resolution=resolution,
    uniform_sky=uniform_sky,
)  # time_d: (T,) days; fields (W, Q, K, T), n_pix (W, K, T)
time = ox.exposure_time(
    record,
    instrument.detector,
    7.0,
    frame_time_s=frame_time_s,
    post_processing=post_processing,
)  # one time per band, planet and time
```

Planets with `scene.valid` False read zero, so their exposure time is invalid.

The reduction itself is described in optixstuff's
[how an exposure time is computed](https://optixstuff.readthedocs.io/en/latest/explanation/exposure_time.html).
The conventions of the terms, and how AYO, pyEDITH and EXOSIMS map onto
`PostProcessing`, are on the [conventions page](docs/explanation/conventions.md).
Thermal emission is read from `Instrument.warm_emission` (optixstuff's
`WarmEmission`) and is zero without it; stellar companions are zero in this
release. `coronagraphoto` fills the same field from its images and aperture
records.

## Cross-code references

Scalar pyEDITH, EXOSIMS and AYO comparison implementations live in
`tests/reference/` in the source checkout. They support cross-code benchmarks
and convention checks, and are excluded from the wheel and source distribution.
See [comparison references](docs/explanation/variants.md) for their scope.

## Installation

```bash
pip install jaxedith
```

The development tests use coordinated unreleased versions of optixstuff and
skyscapes; published dependency floors have not yet been verified for this API.
