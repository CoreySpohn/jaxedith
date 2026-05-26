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

jaxedith assumes the rest of the HWO direct-imaging stack as input
types:

| Package | Role |
|---|---|
| `optixstuff` | `OpticalPath` (primary, throughput, coronagraph, detector) + `ExposureConfig` |
| `skyscapes` | `Scene` / `System` with `Star`, `Planet`, `Disk`, physical models, backgrounds |
| `orbix` | `Observatory` -- needed for the `_from_system_*` zodi geometry helpers |
| `coronalyze` | `PPConfig` (post-processing knobs) |
| `hwoutils` | Shared unit conversions, transforms, constants |

Runtime dependencies are just `hwoutils`, `jax`, `jaxlib`, and
`orbix`. The other workspace libs are needed only for the
`_from_system_*` wrappers and are installed via the `test` extra (or
explicitly):

```bash
pip install jaxedith[test]
```

## Working from source

```bash
git clone https://github.com/CoreySpohn/jaxedith
cd jaxedith
uv sync --all-packages
```

`--all-packages` ensures every workspace member (skyscapes,
optixstuff, coronagraphoto, ...) installs as an editable workspace
member, which is what you want when developing across the stack.

## Verifying the install

```python
import jaxedith
print(jaxedith.__version__)
```

The first JAX import can take 10-20 s on cold cache as XLA
initializes. This is normal.
