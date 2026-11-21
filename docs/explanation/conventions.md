# Conventions

## The terms

Per spectral node, in photons/s/nm. `A` is the collecting area, `T` the system
throughput, `F` the star's photon flux density, `s` the separation in
lambda/D at the band centre, and `Omega = core_area(s) (lambda_c / D)**2` the
aperture solid angle.

| `Rates` field | Formula | Convention |
|---|---|---|
| `planet` | `A T F contrast throughput(s)` | AYO `CR_p`, pyEDITH `calculate_CRp` |
| `star_leakage` | `A T F leakage_density(s) core_area(s)` | AYO `CR_bs`; the leakage rule chosen when the curves are requested sets whether the density is the aperture mean or the azimuthal ring mean |
| `local_zodi` | `A T L_zodi sky(s) Omega` | AYO `CR_bz`: a uniform surface brightness times sky transmission times aperture solid angle |
| `exozodi` | `A T L_disk sky(s) Omega` | AYO `CR_bez`: the disk radiance at the aperture centre times the aperture solid angle |
| `thermal` | zero | returns with a warm-telescope description |
| `stellar_companions` | zero | returns with a companion source |
| `n_pix` | `Omega / pixel_scale**2`, per band | AYO and pyEDITH `det_npix` |
| `variance` | the same record from the squared-weight moments | equal to the mean for `Circular` and `HalfMaximum` |

The aperture is fixed in angle within a band. A separation beyond the curves'
`data_extent_lod` gives NaN rates. The curves must be measured for the
telescope's diameter, and for the star's angular diameter: jaxedith does not
broaden them.

`photon_counting_frame_time(rates, detector)` returns `1 / (6.73 r)` seconds per
band, with `r` the electron rate of all sources per aperture pixel, the
cadence at which a photon-counting EMCCD sees 1/6.73 electrons per pixel per
frame.

## One-call entry points

`jaxedith.scene_exposure_time(..., target_snr=, frame_time_s=, post_processing=)` and
`jaxedith.scene_snr(..., time_s=, frame_time_s=, post_processing=)` take the arguments
of `rates` and compose it with `optixstuff.exposure_time` and `optixstuff.snr`
on `instrument.detector`. The frame time is explicit; use
`photon_counting_frame_time` for the photon-counting cadence.

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
