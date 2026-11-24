# Conventions

## The terms

Per spectral node, in photons/s/nm. `A` is the collecting area, `T` the system
throughput, `F` the star's photon flux density, `P` the planet's photon flux density, `s` the separation in
lambda/D at the band centre, and `Omega = core_area(s) (lambda_c / D)**2` the
aperture solid angle.

| `Rates` field | Formula | Convention |
|---|---|---|
| `planet` | `A T P throughput(s)` or `A T F contrast throughput(s)` | AYO `CR_p`, pyEDITH `calculate_CRp` |
| `star_leakage` | `A T F leakage_density(s) core_area(s)` | AYO `CR_bs`; the leakage rule chosen when the curves are requested sets whether the density is the aperture mean or the azimuthal ring mean; the density is read at the star's angular diameter in lambda/D at each node, so the star's size sets the leakage; between the record's tabulated diameters the read is limited by the spacing of that table (see the optixstuff architecture page), and a star beyond the last diameter raises, which aborts a jitted call |
| `local_zodi` | `A T L_zodi sky(s) Omega` | AYO `CR_bz`: a uniform surface brightness times sky transmission times aperture solid angle |
| `exozodi` | `A T L_disk sky(s) Omega` | AYO `CR_bez`: the disk radiance at the aperture centre times the aperture solid angle |
| `thermal` | `A_geo e B Omega` | pyEDITH `calculate_CRbth`: `B` the Planck photon radiance at the temperature of `Instrument.warm_emission`, `e` its effective emissivity (warm emissivity times cold transmission), `A_geo` the collecting area without `T` or the flux scale; zero when the instrument has no `warm_emission`, the default |
| `stellar_companions` | zero | returns with a companion source |
| `n_pix` | `Omega / pixel_scale**2`, per band | AYO and pyEDITH `det_npix` |
| `variance` | the same record from the squared-weight moments | equal to the mean for `Circular` and `HalfMaximum` |

`F`, `P` and `L_zodi` are the node-effective rows of `scene.sources`, and `L_disk` is read
against the node-effective star: a tabulated spectrum is integrated exactly over each
band whatever its roughness, and the rule is exact in the response to the degree of the
quadrature. `rates` itself reads no spectrum. The curves are read at the band-centre
lambda/D for every node, so build the quadrature with `field_lod=0`, also when the
instrument is passed as a break model; a field radius would add nodes and change nothing.

The scene's `flux_scale` leaves act through the sources, so `F`, `L_zodi` and `L_disk`
above already include them: the star scale multiplies `star_leakage` and
reflected planet light, the zodi scale `local_zodi` and the disk scale `exozodi`.
A planet's scale is carried by `Sources.planets`; thermal emission remains
independent of the stellar scale. The
instrument's own `flux_scale` multiplies every sky term through `A`.

The aperture is fixed in angle within a band. A planet outside the curves' support
(`curves.supported`: between the first and last sample, with the aperture inside
`data_extent_lod`) reads zero in every field
with `Rates.supported` False, so the exposure time and S/N are NaN there. The curves must be measured for the
telescope's diameter, and for the star's angular diameter: jaxedith does not
broaden them.

`photon_counting_frame_time(rates, detector)` returns `1 / (6.73 r)` seconds per
band, with `r` the electron rate of all sources per aperture pixel, the
cadence at which a photon-counting EMCCD sees 1/6.73 electrons per pixel per
frame.

Read a scene's planets by passing `planet=` with its node-effective photon flux
density in photons/s/m^2/nm from `Sources.planets`.
Alternatively, `contrast=` states a planet-to-star hypothesis relative to
`sources.star` at the read time; exactly one of the two keywords is required.

```python notest
sources = scene.sources(quadrature, resolution, time_d)
by_flux = jaxedith.rates(
    curves, scene, instrument, quadrature, east_arcsec, north_arcsec,
    planet=sources.planets[:, :, 0, 0], time_d=time_d, sources=sources, uniform_sky=uniform_sky,
)
by_contrast = jaxedith.rates(
    curves, scene, instrument, quadrature, east_arcsec, north_arcsec,
    contrast=1e-10, time_d=time_d, sources=sources, uniform_sky=uniform_sky,
)
```

## One-call entry points for one planet

`jaxedith.planet_exposure_time(..., target_snr=, frame_time_s=, post_processing=)` and
`jaxedith.planet_snr(..., time_s=, frame_time_s=, post_processing=)` take the arguments
of `rates` (including `sources`, with a positional `contrast` hypothesis) and compose it with `optixstuff.exposure_time` and `optixstuff.snr`
on `instrument.detector`. The frame time is explicit; use
`photon_counting_frame_time` for the photon-counting cadence.

## Every planet of a scene

`jaxedith.scene_rates(curves, scene, instrument, quadrature, time_d, *, resolution, uniform_sky)`
reads each planet at each time with the terms of `rates`, taking positions from the scene
and the node rows and planet photon flux densities from one `scene.sources` call made outside the map over
planets and times. Its `Rates` has fields of shape `(W, Q, K, T)` and
`n_pix` of shape `(W, K, T)`, the layout of `coronagraphoto.scene_aperture_rates`,
and goes to `optixstuff.snr` or `optixstuff.exposure_time` unchanged.

## Post-processing settings

`optixstuff.PostProcessing(speckle_residual_fraction, exozodi_residual_fraction,
background_variance_factor)` carries the assumption of the subtraction. The
factor multiplies the variance of every source except the planet's own shot
noise. It is required and keyword-only: two when a reference image is
subtracted (differential imaging), one when it is not.

| Convention | `speckle_residual_fraction` | `exozodi_residual_fraction` | `background_variance_factor` |
|---|---|---|---|
| AYO and pyEDITH | `1 / ppf` | `1 / ez_ppf` | 2 |
| EXOSIMS characterization | `ppFact * stability_fact` | 0 | 1 |
| EXOSIMS detection | not expressible: the planet's own shot noise is left out of the variance | | |

The scalar detection equation of EXOSIMS and pyEDITH's separate read time and
photon-counting time (when read noise is nonzero) are kept as reference code in
`tests/reference/`.
