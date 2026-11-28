# Changelog

## [4.0.0](https://github.com/CoreySpohn/jaxedith/compare/v3.0.0...v4.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **jaxedith:** enforce curve and sky support with float32-resolution inclusive endpoints in both lanes
* **analytic:** the curves route reads the leakage at the star's diameter in lambda/D at the band centre instead of at each node.
* **analytic:** the uniform_sky keyword of rates, scene_rates, planet_snr and planet_exposure_time is removed; each band reads PerformanceCurves.uniform_sky from its own record, and a scene with a disk or a local zodi raises on the host, naming the band, when a band's record carries no profile.
* **analytic:** jaxedith.rates takes exactly one of the keyword-only contrast= or planet= (photon flux density); scene_rates passes Sources.planets, so a planet around a star with a zero node reads finite; planet_snr and planet_exposure_time keep their contrast argument
* **rates:** rates reads the leakage density at the star's angular diameter in lambda/D per node; a record of one profile raises for a star of a different diameter.
* **rates:** rates, planet_exposure_time and planet_snr take a required sources keyword (from Scene.sources) and read no spectrum themselves; scene_rates takes a required resolution keyword and reads the sources once outside the map over planets and times.
* **rates:** read support from curves.supported, zero past the last sample
* **rates:** read zero with a supported flag beyond the curves' extent instead of NaN
* **analytic:** fill Rates.thermal from Instrument.warm_emission
* **analytic:** rename scene_snr and scene_exposure_time to planet_snr and planet_exposure_time, add scene_rates
* **analytic:** require post_processing in scene_snr and scene_exposure_time
* **analytic:** rename snr and exposure_time to scene_snr and scene_exposure_time
* **conventions:** state background_variance_factor in every PostProcessing
* **analytic:** add one-call snr and exposure_time that compose rates with the optixstuff calculator
* **rates:** add rates and photon_counting_frame_time for analytic aperture rates
* **calculator:** delete the S/N and exposure-time calculator and its helper tests in favor of optixstuff
* **api:** remove aperture_rates and its fixtures
* **api:** take the zodi models from skyscapes and read the sightline from the scene sky frame
* **api:** return one optixstuff Rates from aperture_rates and remove ApertureRates
* import bands, grids and resampling from optixstuff; read quadrature weights by name
* **api:** read exozodi from scene.disk instead of the exozodi keyword
* take the optixstuff Instrument (renamed from OpticalPath); rename the optical_path argument to instrument
* build quadratures from the evaluated models and drop aperture_quadrature
* integrate the uniform-sky response over the aperture for every rule
* read stellar diameter and disk radiance from skyscapes and drop the disk_grid route
* build inverse-variance Matched weights from the scene and detector rates
* use variance moments in aperture noise terms
* take aperture preparation from optixstuff and drop the direct yippy and numpy dependencies
* jaxedith ships one calculator; the scalar comparison dialects move into the test suite as cross-code references
* the analytic AYO exozodi default comes from zodi.jax
* local-zodi models come from zodi.jax
* analytic AYO exozodi is the default in aperture_rates; the scene disk is opt-in
* hold the aperture fixed in angle across a band by evaluating each node at its own separation and radius
* read the shared photon-rate budget through one aperture reader with post-processing and detector statistics

### Features

* analytic AYO exozodi is the default in aperture_rates; the scene disk is opt-in ([74b59a8](https://github.com/CoreySpohn/jaxedith/commit/74b59a862009755268e32a93af121a79c30b0c33))
* **analytic:** add one-call snr and exposure_time that compose rates with the optixstuff calculator ([008c2da](https://github.com/CoreySpohn/jaxedith/commit/008c2da7fffd1fde533f9a1bc95d9399168635d9))
* **analytic:** fill Rates.thermal from Instrument.warm_emission ([a43566d](https://github.com/CoreySpohn/jaxedith/commit/a43566d9a8df5d4674b5ef6f60d20a513b3012f9))
* **analytic:** read the sky transmission from each band's curves ([c39eb06](https://github.com/CoreySpohn/jaxedith/commit/c39eb06fe4f7483fad8fa5ce7858c41e3270a5cf))
* **analytic:** read the stellar diameter at the observing band centre ([07dcd64](https://github.com/CoreySpohn/jaxedith/commit/07dcd64e60dbdfd1385aa36961c4378e4ee00cb4))
* **analytic:** rename scene_snr and scene_exposure_time to planet_snr and planet_exposure_time, add scene_rates ([f22fc13](https://github.com/CoreySpohn/jaxedith/commit/f22fc13e85350be4417230e26fb14b3936e02b2a))
* **analytic:** require post_processing in scene_snr and scene_exposure_time ([ca845b3](https://github.com/CoreySpohn/jaxedith/commit/ca845b3e20ba68804d0da28a52047e943699fee2))
* **api:** read exozodi from scene.disk instead of the exozodi keyword ([402682e](https://github.com/CoreySpohn/jaxedith/commit/402682ebb640c56ac93a9410fd2f8b531e0e0df2))
* **api:** return one optixstuff Rates from aperture_rates and remove ApertureRates ([6e312d2](https://github.com/CoreySpohn/jaxedith/commit/6e312d2acb50defd014b87a8b3022b948647c90a))
* **api:** take the zodi models from skyscapes and read the sightline from the scene sky frame ([b0fbe69](https://github.com/CoreySpohn/jaxedith/commit/b0fbe6925640b900994b8c81d7fa196916bda201))
* build inverse-variance Matched weights from the scene and detector rates ([f6f6954](https://github.com/CoreySpohn/jaxedith/commit/f6f695404d154157815b31b189314b3b7158d9d3))
* **rates:** add rates and photon_counting_frame_time for analytic aperture rates ([cd709b3](https://github.com/CoreySpohn/jaxedith/commit/cd709b3d0d3354bf76f8ca4f78cd5ce403d6a57f))
* **rates:** read the star leakage at the star's diameter ([c6e34ec](https://github.com/CoreySpohn/jaxedith/commit/c6e34ec97a9ad20328f5b1ad1a2e5d387115e308))
* **rates:** take node-effective sources and compute them once in scene_rates ([4880cfe](https://github.com/CoreySpohn/jaxedith/commit/4880cfe8b1899fc2b889ab76c8c404434cdc48ee))
* read stellar diameter and disk radiance from skyscapes and drop the disk_grid route ([7cbc3f1](https://github.com/CoreySpohn/jaxedith/commit/7cbc3f1f6c457c647d82dbbe4dc381253748a17d))
* read the shared photon-rate budget through one aperture reader with post-processing and detector statistics ([8ff3518](https://github.com/CoreySpohn/jaxedith/commit/8ff3518cc2845b1fa45c26189058b043d2fcea23))


### Bug Fixes

* **analytic:** read scene planets by photon flux density in rates ([ff37dbf](https://github.com/CoreySpohn/jaxedith/commit/ff37dbf7c728d5d52275ab23e358090436ae2b16))
* **etc:** pass each star's angular diameter to the coronagraph leakage ([5059320](https://github.com/CoreySpohn/jaxedith/commit/5059320b16478ed703aaa8d40e878078ad1d539e))
* hold the aperture fixed in angle across a band by evaluating each node at its own separation and radius ([138c143](https://github.com/CoreySpohn/jaxedith/commit/138c143cd69eacc1879904d109820f08000ca360))
* integrate the uniform-sky response over the aperture for every rule ([ad5ca56](https://github.com/CoreySpohn/jaxedith/commit/ad5ca565a4433fa4b5d74d4ceab2b74d1e5eebfb))
* **jaxedith:** declare equinox and numpy as runtime dependencies ([bc44795](https://github.com/CoreySpohn/jaxedith/commit/bc44795e25ffeb3b707b760865667a41408e5dcf))
* **jaxedith:** enforce curve and sky support with float32-resolution inclusive endpoints in both lanes ([6df658e](https://github.com/CoreySpohn/jaxedith/commit/6df658e701d25c182ed7795e7799781dd29d649c))
* keep every aperture rule field when scaling the radius to a wavelength node ([8e42a4d](https://github.com/CoreySpohn/jaxedith/commit/8e42a4db45c7eb495a2382995b05ad976bf4d87d))
* propagate NaN variance, treat the speckle ceiling as infeasible, and mark out-of-range traced planets ([dc9af1e](https://github.com/CoreySpohn/jaxedith/commit/dc9af1e07f1ebb3ef9a71fcab821bfd86c8dba2d))
* **rates:** read support from curves.supported, zero past the last sample ([3ce996d](https://github.com/CoreySpohn/jaxedith/commit/3ce996d76aa71132d620b549ccc58889711f1212))
* **rates:** read zero with a supported flag beyond the curves' extent instead of NaN ([6c32327](https://github.com/CoreySpohn/jaxedith/commit/6c32327139b2fdaee83770506a5b8ce532c3ee15))
* use variance moments in aperture noise terms ([0adbd58](https://github.com/CoreySpohn/jaxedith/commit/0adbd58c9758d9ab37410061f8014254757b9f19))


### Documentation

* **conventions:** state background_variance_factor in every PostProcessing ([a07a301](https://github.com/CoreySpohn/jaxedith/commit/a07a3014a69fbe6884702bce859dc6616dfc6316))


### Code Refactoring

* **analytic:** rename snr and exposure_time to scene_snr and scene_exposure_time ([2ea4ded](https://github.com/CoreySpohn/jaxedith/commit/2ea4ded74649bce91266ce0bbb8a4f1b0b7698e5))
* **api:** remove aperture_rates and its fixtures ([deb4865](https://github.com/CoreySpohn/jaxedith/commit/deb486573f18096255969008cb00f7241842aa74))
* build quadratures from the evaluated models and drop aperture_quadrature ([24989b9](https://github.com/CoreySpohn/jaxedith/commit/24989b9989b2631f2ac08ca800bbc8e42fcdaed6))
* **calculator:** delete the S/N and exposure-time calculator and its helper tests in favor of optixstuff ([2f30d1e](https://github.com/CoreySpohn/jaxedith/commit/2f30d1e2bdd64e2b66adff19dc76fae7be5543d1))
* import bands, grids and resampling from optixstuff; read quadrature weights by name ([9482dcb](https://github.com/CoreySpohn/jaxedith/commit/9482dcbf146939b14f46e10271aec46545e957ce))
* jaxedith ships one calculator; the scalar comparison dialects move into the test suite as cross-code references ([2461f06](https://github.com/CoreySpohn/jaxedith/commit/2461f06a0115f3f0dce6af5c506e21409d50f118))
* local-zodi models come from zodi.jax ([a328a5c](https://github.com/CoreySpohn/jaxedith/commit/a328a5cb8db66197de84fad61ce8944845bfa915))
* take aperture preparation from optixstuff and drop the direct yippy and numpy dependencies ([7eddbed](https://github.com/CoreySpohn/jaxedith/commit/7eddbedf0492a1112263ba0dbbf566e5caa1ea6a))
* take the optixstuff Instrument (renamed from OpticalPath); rename the optical_path argument to instrument ([3c2f5aa](https://github.com/CoreySpohn/jaxedith/commit/3c2f5aa4338129a5c101a109879802640cfec9f6))
* the analytic AYO exozodi default comes from zodi.jax ([839d8b3](https://github.com/CoreySpohn/jaxedith/commit/839d8b3bdf27561c1f1ea82de42984034fa67073))

## [3.0.0](https://github.com/CoreySpohn/jaxedith/compare/v2.1.1...v3.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* read distance from the scene dynamics and take exposure times as days from the scene epoch

### Features

* read distance from the scene dynamics and take exposure times as days from the scene epoch ([c15293e](https://github.com/CoreySpohn/jaxedith/commit/c15293e1d67c003a98449c3b63ac6343e2b6b587))

## [2.1.1](https://github.com/CoreySpohn/jaxedith/compare/v2.1.0...v2.1.1) (2026-11-15)


### Bug Fixes

* **jaxedith:** support orbix 4 and skyscapes 3, whose time and luminosity keywords were renamed ([dede988](https://github.com/CoreySpohn/jaxedith/commit/dede9881b257a27156c07908149ccfec762591bc))

## [2.1.0](https://github.com/CoreySpohn/jaxedith/compare/v2.0.0...v2.1.0) (2026-11-12)


### Features

* **etc:** add fixed-frame finite-exposure variance, SNR and exposure planners ([8c18683](https://github.com/CoreySpohn/jaxedith/commit/8c18683c9af1511c1a3a72dc7559a16c282c3077))


### Bug Fixes

* **etc:** tolerate roundoff in fixed-frame planners and frame counts; zero SNR for non-finite durations ([9f1c13b](https://github.com/CoreySpohn/jaxedith/commit/9f1c13b7ac3dcc7c866d4c38b102765511cf689b))
* **intermediates:** apply detector QE to photon rates and use frame time for read noise ([d4e766f](https://github.com/CoreySpohn/jaxedith/commit/d4e766f62c994c8d6dfe5788cdd58f7dd8187dbf))
* **intermediates:** treat dqe as a multiplicative QE degradation factor ([a0db634](https://github.com/CoreySpohn/jaxedith/commit/a0db63474518ef2431f8e0d19c9f97439d633ca3))
* **jaxedith:** jit the zodi Fzodi helpers to keep eager compile counts down ([e1c5274](https://github.com/CoreySpohn/jaxedith/commit/e1c527468ee08207d881bdf3fe45d372cada2d90))

## [2.0.0](https://github.com/CoreySpohn/jaxedith/compare/v1.0.1...v2.0.0) (2026-11-05)


### ⚠ BREAKING CHANGES

* **deps:** the test extra needs optixstuff>3.0.0 (4.0.0 or later), whose core_mean_intensity is per (lambda/D)^2. With it, stellar leakage, background and the EXOSIMS speckle residual from YippyCoronagraph rise by 1/pixel_scale_lod**2 (16x at 0.25 lambda/D pixels) while its AYO noise floor is unchanged. Noise floors from coronagraphs that already report a density, such as physical-optics backends and custom scalar coronagraphs, drop by pixel_scale_lod**2; recalibrate any ppfact tuned to the old floor.

### Features

* **etc:** opt-in error_if invariants on rates and solver inputs, drop finite-only tests ([ea27f8d](https://github.com/CoreySpohn/jaxedith/commit/ea27f8d404ab3172f85b773bfe1a961659bba922))
* **etc:** pass stellar angular diameter to coronagraph leakage ([202b4c3](https://github.com/CoreySpohn/jaxedith/commit/202b4c37c6dce3cd73eb1c2a253426b5df04f669))


### Bug Fixes

* **deps:** require optixstuff with per-(lambda/D)^2 core intensity ([3bba081](https://github.com/CoreySpohn/jaxedith/commit/3bba08135867c5144bd68995eb1c840699349813))
* **intermediates:** read core mean intensity per (lambda/D)^2 in the noise floor ([ab82e3b](https://github.com/CoreySpohn/jaxedith/commit/ab82e3b9a0e09995337e48a9632f27172f760945))
* **jaxedith:** allow integer returns from product-only primitives ([02128f7](https://github.com/CoreySpohn/jaxedith/commit/02128f7de2e30a0f5745768f5c8a3e0a51b7b6a0))

## [1.0.1](https://github.com/CoreySpohn/jaxedith/compare/v1.0.0...v1.0.1) (2026-07-18)


### Bug Fixes

* **from_system:** reference zodi/exozodi to a mag-0 F0 zero point, not the star flux ([c339272](https://github.com/CoreySpohn/jaxedith/commit/c33927226fce7472bebd0417fc368ca33d291292))

## [1.0.0](https://github.com/CoreySpohn/jaxedith/compare/v0.1.0...v1.0.0) (2026-05-26)


### ⚠ BREAKING CHANGES

* rename layer modules to primitives/intermediates/etc and count_rate_X to X_rate

### Features

* rename layer modules to primitives/intermediates/etc and count_rate_X to X_rate ([7a6d83f](https://github.com/CoreySpohn/jaxedith/commit/7a6d83ff2f3e3c6ea48db37df7ba8cf890ced6c5))


### Bug Fixes

* Calculating the zodi using the helio ecliptic longitude instead of the ecliptic latitude ([ec19cfb](https://github.com/CoreySpohn/jaxedith/commit/ec19cfbc8de71cef362340421331f9bfa924706b))

## [0.1.0](https://github.com/CoreySpohn/jaxedith/compare/v0.0.2...v0.1.0) (2026-04-23)


### Features

* Adding full skyscapes support ([997279a](https://github.com/CoreySpohn/jaxedith/commit/997279ac7d17df477b9c42793e642fa55e4024af))
* **jaxedith:** add AYO *_from_system_ayo wrappers with (K, T) vmap ([2a6a3b6](https://github.com/CoreySpohn/jaxedith/commit/2a6a3b65146e056abe8709c92a505c3ba947447b))
* **jaxedith:** add AYO scalar publics (count_rates_ayo, exptime_ayo, snr_ayo) ([8a6e0ca](https://github.com/CoreySpohn/jaxedith/commit/8a6e0cabdf086f971f3d79c6834d729e543ad719))
* **jaxedith:** add EXOSIMS det/char *_from_system_* wrappers ([3aa415b](https://github.com/CoreySpohn/jaxedith/commit/3aa415bd67b94f5b0be319cf6aeae954c0a6e9f7))
* **jaxedith:** add EXOSIMS det/char scalar publics ([cd19f62](https://github.com/CoreySpohn/jaxedith/commit/cd19f6204318fe458cf97168215fa23b1ad0711d))
* **jaxedith:** add zodi_fn_ayo and zodi_fn_leinert callables ([1058f37](https://github.com/CoreySpohn/jaxedith/commit/1058f37fdb5b8625841b34dbb4fb1c07e821df2d))
* **jaxedith:** Layer 2 binary_background wrapper ([21b2e14](https://github.com/CoreySpohn/jaxedith/commit/21b2e14fb6b8464430a6d5ae36e2a1fdd48b8c41))
* **jaxedith:** Layer 2 detector_noise wrapper ([1dba6e9](https://github.com/CoreySpohn/jaxedith/commit/1dba6e951bde08458111f69b9d5a2bdac5479a14))
* **jaxedith:** Layer 2 exozodi_background wrapper ([0cbe9b3](https://github.com/CoreySpohn/jaxedith/commit/0cbe9b3035025d1844b7de4c7a8074dea119673e))
* **jaxedith:** Layer 2 planet_signal wrapper ([5469618](https://github.com/CoreySpohn/jaxedith/commit/54696185a52928487cf4a6d3650fd14c121a64e2))
* **jaxedith:** Layer 2 stellar_leakage wrapper ([85e1b16](https://github.com/CoreySpohn/jaxedith/commit/85e1b16d129a34221cb6c319f90553bf3d065006))
* **jaxedith:** Layer 2 stellar_noise_floor wrapper ([2726e04](https://github.com/CoreySpohn/jaxedith/commit/2726e043699a9b4cdbf9429c721ce79a084de6e1))
* **jaxedith:** Layer 2 thermal_background wrapper ([f0c3ec3](https://github.com/CoreySpohn/jaxedith/commit/f0c3ec3fd34809ac5ae0aef67f7851b63dbaa892))
* **jaxedith:** Layer 2 zodi_background wrapper ([d42e47b](https://github.com/CoreySpohn/jaxedith/commit/d42e47b6a0ffca6fe6e9e7b6c6d07c7f517b2abc))
* **jaxedith:** re-export Layer 2 components from package ([24f7eb6](https://github.com/CoreySpohn/jaxedith/commit/24f7eb6d059df655d1d65623407674ad7ebb2367))
* **jaxedith:** scaffold Layer 2 components module ([a41467c](https://github.com/CoreySpohn/jaxedith/commit/a41467cff3679b9c847cf612d206a6abd8cee777))

## [0.0.2](https://github.com/CoreySpohn/jaxedith/compare/v0.0.1...v0.0.2) (2026-04-13)


### Miscellaneous Chores

* release 0.0.2 ([0403b67](https://github.com/CoreySpohn/jaxedith/commit/0403b67df165274c5a31468e4c8a113c49650adf))

## 0.0.1 (2026-04-13)


### Features

* Add all things from the pyEDITH branch ([b0a0f5d](https://github.com/CoreySpohn/jaxEDITH/commit/b0a0f5d49ae77608d1d75abb16723a519e3b1ed7))


### Miscellaneous Chores

* release 0.0.1 ([164ba31](https://github.com/CoreySpohn/jaxEDITH/commit/164ba3136a758595a70fc4cf50fb93a9d801a3bb))
