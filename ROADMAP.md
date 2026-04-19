# jaxedith Roadmap

Forward-looking items not on the active plan. Active work lives in
`hwo-mission-control/burn/jaxedith-refactor/brain/plans/`.

## Future / potential

- **`_from_scene_*` wrappers.** Thin convenience entry points accepting
  a slimmed `ETCScene` (8 astrophysical scalars: `F0`, `Fs_over_F0`,
  `Fp_over_Fs`, `dist_pc`, `sep_arcsec`, `Fzodi`, `Fexozodi`,
  `noisefloor_value`). Deferred at the 2026-04-19 redesign because the
  primary user workflow is `_from_system` (skyscapes objects directly)
  and there is no current concrete use case for `ETCScene` as an input.
  Revisit if a user actually asks for scalar bag input via an
  intermediate dataclass.

- **Per-variant `get_*` factory helpers.** Analogous to `orbix.get_grid_solver`,
  these would pre-configure and JIT-compile an `exptime_*` or `snr_*`
  function with fixed `optical_path` / `ppconfig` / `observatory`. Useful
  if repeated-call scenarios (yield grids, sensitivity studies) show
  that repeated trace overhead is measurable.

- **Real multi-channel support.** `OpticalPath.n_channels` is currently
  a scalar "N parallel identical optical paths" multiplier. If the
  mission concept ever requires truly distinct channels (different
  coronagraphs, different detectors per channel), replace the scalar
  with a tuple-of-`OpticalPath` structure and update Layer 2 / Layer 3
  accordingly.
