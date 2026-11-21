# jaxedith

Analytic aperture rates from performance curves, a scene and an instrument,
in the AYO and pyEDITH conventions, with `jaxedith.scene_rates` for every planet
of a scene and one-call `jaxedith.planet_exposure_time` and `jaxedith.planet_snr`
for one planet. The reduction itself is `optixstuff.snr` and
`optixstuff.exposure_time`; optixstuff's page on
[how an exposure time is computed](https://optixstuff.readthedocs.io/en/latest/explanation/exposure_time.html) follows this path and the image path
of coronagraphoto to it.

The [README](https://github.com/CoreySpohn/jaxedith#readme) contains the full
calling example. The [conventions](explanation/conventions) page lists every
term and the post-processing settings of each community code.

```{toctree}
:maxdepth: 2

installation
explanation/architecture
explanation/conventions
explanation/variants
autoapi/index
```
