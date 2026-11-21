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
`skyscapes.Scene`, an `optixstuff.Instrument` and a `Quadrature` built for the
star and instrument.

```python notest
import jaxedith
import optixstuff as ox

rates = jaxedith.rates(
    curves,
    scene,
    instrument,
    quadrature,
    east_arcsec,
    north_arcsec,
    contrast,
    time_d=0.0,
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
    time_d=0.0,
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
    time_d=0.0,
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

`rates` is a scalar read of one planet; map it over positions and contrasts with `jax.vmap`.
Fields are photons/s/nm of shape `(W, Q)`, with `n_pix` per band and a
squared-weight `variance` record. Detector yield is not applied there:
`detector.electrons(rates)` applies QE and survival once, downstream.

To read every planet of the scene at every time, `scene_rates` is a `vmap` of `rates`
over the planets and the times, and takes the same layout as
`coronagraphoto.scene_aperture_rates`, so the two reads swap:

```python notest
record = jaxedith.scene_rates(
    curves, scene, instrument, quadrature, time_d, uniform_sky=uniform_sky
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
Thermal emission and stellar companions are zero in this release.

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
