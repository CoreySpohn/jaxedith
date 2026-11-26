# Conventions

## The terms

Per spectral node, in photons/s/nm. `A` is the collecting area, `T` the system
throughput, `F` the star's photon flux density, `P` the planet's photon flux density, `s` the separation in
lambda/D at the band centre, and `Omega = core_area(s) (lambda_c / D)**2` the
aperture solid angle.

| `Rates` field | Formula | Convention |
|---|---|---|
| `planet` | `A T P throughput(s)` or `A T F contrast throughput(s)` | AYO `CR_p`, pyEDITH `calculate_CRp` |
| `star_leakage` | `A T F leakage_density(s) core_area(s)` | AYO `CR_bs`; the leakage rule chosen when the curves are requested sets whether the density is the aperture mean or the azimuthal ring mean; the density is read at the star's angular diameter in lambda/D at the observing band centre, so the star's size sets the leakage; between the record's tabulated diameters the read is limited by the spacing of that table (see the optixstuff architecture page), and a star beyond the last diameter raises, which aborts a jitted call |
| `local_zodi` | `A T L_zodi sky(s) Omega` | AYO `CR_bz`: a uniform surface brightness times sky transmission times aperture solid angle |
| `exozodi` | `A T L_disk sky(s) Omega` | AYO `CR_bez`: the disk radiance at the aperture centre times sky transmission times aperture solid angle |
| `thermal` | `A_geo e B Omega` | pyEDITH `calculate_CRbth`: `B` the Planck photon radiance at the temperature of `Instrument.warm_emission`, `e` its effective emissivity (warm emissivity times cold transmission), `A_geo` the collecting area without `T` or the flux scale; zero when the instrument has no `warm_emission`, the default |
| `stellar_companions` | zero | returns with a companion source |
| `n_pix` | `Omega / pixel_scale**2`, per band | AYO and pyEDITH `det_npix` |
| `variance` | the same record from the squared-weight moments | equal to the mean for `Circular` and `HalfMaximum` |

Sky transmission for `local_zodi` and `exozodi` travels with the curves as
`PerformanceCurves.uniform_sky`. Each band reads its own profile. A scene
with a disk or a local zodi requires a profile on every band's record.

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

The aperture is fixed in angle within a band. Analytic support is the overlap
of the curve sample span and the separations where the aperture fits inside
`data_extent_lod`. A scene with a disk or local zodi also requires coverage
from the band's sky transmission profile. A nonempty interval includes its
endpoints, with a roundoff allowance of two float32 machine epsilons at each
endpoint's scale for the radial norm and the angular conversion. Both precision
lanes use the same float32 grid in arcseconds. Outside support, rates and pixel
counts are zero, `Rates.supported` is false, and exposure time and S/N are NaN.
The curves must be measured for the telescope's diameter and the star's angular
diameter.

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
    curves,
    scene,
    instrument,
    quadrature,
    east_arcsec,
    north_arcsec,
    planet=sources.planets[:, :, 0, 0],
    time_d=time_d,
    sources=sources,
)
by_contrast = jaxedith.rates(
    curves,
    scene,
    instrument,
    quadrature,
    east_arcsec,
    north_arcsec,
    contrast=1e-10,
    time_d=time_d,
    sources=sources,
)
```

## One-call entry points for one planet

`jaxedith.planet_exposure_time(..., target_snr=, frame_time_s=, post_processing=)` and
`jaxedith.planet_snr(..., time_s=, frame_time_s=, post_processing=)` take the arguments
of `rates` (including `sources`, with a positional `contrast` hypothesis) and compose it with `optixstuff.exposure_time` and `optixstuff.snr`
on `instrument.detector`. The frame time is explicit; use
`photon_counting_frame_time` for the photon-counting cadence.

For a flat-spectrum star at 10 pc observed with a 6 m telescope, the precision
comparison uses contrasts of `1e-6` and `3e-10` at 0.05 arcsec separation.
The 500–600 and 600–700 nm bands have target S/N of 5 and 7 and frame times
of 100 and 60 s, respectively, with zero residual fractions and a background
variance factor of one.
In float32, the largest differences relative to each band's float64 value
were `4.51e-7` for exposure time, `1.30e-9` for the planet count rate,
`2.97e-7` for stellar leakage and `3.93e-7` for the total background
(leakage plus local and exozodiacal light here), so use float64 when you
need finer agreement.

## Every planet of a scene

`jaxedith.scene_rates(curves, scene, instrument, quadrature, time_d, *, resolution)`
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
