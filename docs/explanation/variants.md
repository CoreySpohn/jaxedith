# Legacy comparison equations

The main calculator has one detector contract. AYO and EXOSIMS dialects
remain only as scalar comparison equations in `jaxedith.etc`.

| Equation suffix | Random variance rate | Systematic residual |
| --- | --- | --- |
| `ayo` | `Cp + bg_multiplier*Cb` | `Cnf_rate` |
| `exosims_det` | `Cb` | `Csp` |
| `exosims_char` | `Cp + Cb` | `Csp` |

The three `exptime_from_rates_*` functions return seconds and the three
`snr_from_rates_*` functions return S/N. Inputs are plain electron rates.
Their existing overhead and roll conventions remain unchanged.

```python
from jaxedith.etc import exptime_from_rates_ayo, snr_from_rates_ayo
from jaxedith.legacy import post_processing_from_pyedith

processing = post_processing_from_pyedith(10.0)
t = exptime_from_rates_ayo(1.0, 0.5, 0.01, 7.0)
achieved = snr_from_rates_ayo(1.0, 0.5, 0.01, t)
```

Use `etc_scene_from_system` for F0-relative inputs to the comparison rate
primitives. Fzodi is per arcsec squared relative to the AB zero point.
Fexozodi is an effective 1-AU value: sampled disk brightness times projected
radius squared. The primitive's inverse-square factor reconstructs the local
brightness without imposing that profile on the disk model.

`post_processing_from_pyedith(ppfact)` gives `1/ppfact`;
`post_processing_from_exosims(ppFact)` gives `ppFact`. Both set the disk
residual to zero. `frame_plan_photon_counting` implements the mean occupancy
of `1/6.73` electrons per pixel per frame. `legacy_pyedith_noise_rate` retains
separate historical read and CIC times for comparisons.

A centre-times-width comparison is exact for flat photon spectra and
constant response. It has about 0.7% electron-rate bias in the worked
500-600 nm chromatic example. Use the photon reader for band integration.
