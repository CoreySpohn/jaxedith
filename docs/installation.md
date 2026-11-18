# Installation

```bash
pip install jaxedith
```

This pulls in jaxedith and its JAX-CPU dependencies. For most ETC
work -- yield curves, sensitivity sweeps, retrievals -- the CPU
build is fast enough.

## GPU install

If you're running large vmap loops over targets, wavelengths, or
exposure-time grids, GPU acceleration helps. Install JAX with CUDA
12:

```bash
pip install jaxedith jax[cuda12]
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for shared-system caveats and CUDA-runtime troubleshooting.

## Sibling libraries

The reader requires the photon scene, aperture performance, shared-grid and
detector contracts described in the README. This branch is tested against
coordinated unreleased versions; published dependency floors are not yet
verified for those contracts.

| Package | Role |
|---|---|
| `optixstuff` | Instrument, aperture preparation, photon rates, detector statistics and frame plans |
| `skyscapes` | System, photon spectra and disk radiance |
| `hwoutils` | Band, Quadrature, PixelGrid, constants and conversions |
| `zodi[jax]` | Local-zodi photon radiance, spectral tables and dust calibration |

Runtime dependencies also include Equinox, JAX and jaxtyping.
Install test dependencies with:

```bash
pip install 'jaxedith[test]'
```

The comparison suite itself requires a source checkout; it is not shipped
in the installed distributions.

## Verifying the install

```python
import jaxedith

print(jaxedith.__version__)
```

The first JAX import can take 10-20 s on cold cache as XLA
initializes. This is normal.
