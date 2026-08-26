# mojo-numpy-stl

`mojo-numpy-stl` is a Mojo-accelerated, compatible subset of
[numpy-stl](https://pypi.org/project/numpy-stl/). It keeps numpy-stl's structured
50-byte facet layout and Python-shaped `Mesh` API while moving per-facet normal
calculation and the signed polyhedral volume/inertia integrals into one compiled
Mojo shared library.

The covered import is deliberately `mojo_stl` so it can be installed and
parity-tested beside upstream `stl`:

```python
import numpy as np
from mojo_stl import Mode, mesh

data = np.zeros(1, dtype=mesh.Mesh.dtype)
data["vectors"][0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
model = mesh.Mesh(data)
model.save("triangle.stl", mode=Mode.BINARY)
print(model.get_unit_normals())
```

## Covered subset

`Mesh` accepts a one-dimensional NumPy array with exactly `Mesh.dtype` (the
same 50-byte structured dtype as numpy-stl). Covered and tested: the structured
array views (`vectors`, `normals`, `points`, `v0`/`v1`/`v2`, `x`/`y`/`z`, `attr`),
normal/area/centroid/bounds/unit-normal updates, translation and rotation,
empty-facet and single-duplicate filtering, signed mass properties (including
density), exact closed-surface checks, and binary/ASCII read/write through
filenames or binary handles with automatic format detection.

Not covered: 3MF input, numpy-stl's optional C speedups, exhaustive
closed-surface diagnostics, multi-file loaders, arbitrary input-dtype coercion,
and every convenience/API compatibility feature upstream exposes. ASCII parsing
supports standard STL facet grammar.

## Install and run

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`numpy-stl` is installed in the Pixi environment and the test suite asserts
array, file-format, normal, volume, centre-of-gravity, and inertia parity with
that real upstream package.

## How it works

The Python package passes the address of the existing structured NumPy array
through `ctypes`. Mojo reconstructs a mutable pointer for each 50-byte record:
three float32 normal values, nine float32 vertex values, then one uint16 STL
attribute. This means normal updates operate in place without converting the
facet buffer. Binary serialization remains a direct write of that interoperable
NumPy layout; ASCII serialization is a small standards-compatible Python layer.
The normal kernel fuses normal, area, and centroid output in one pass. The mass
kernel accumulates the ten Geometric Tools polyhedral integrals in float64 SIMD
lanes, and Python applies the final centre-of-gravity shift to inertia. Both
kernels switch to eight CPU workers at one million facets; smaller meshes stay
serial to avoid thread-launch overhead.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 (72 logical CPUs),
using numpy-stl 3.2.0 and the pinned Mojo nightly:

| kernel | mojo-numpy-stl | numpy-stl | speedup |
| --- | ---: | ---: | ---: |
| `update_normals`, 200k facets | 1.53 ms | 24.83 ms | 16.19x |
| `get_unit_normals`, 200k facets | 1.01 ms | 15.99 ms | 15.90x |
| `get_mass_properties`, 49,992 closed facets | 100.77 ms | 892.34 ms | 8.86x |

Mass-property calls retain numpy-stl's signed-integral calculation and run its
exact closed-surface check using packed 24-byte edge keys. The result of that
upstream-compatible check is informational, as it is in numpy-stl. There is no
GPU path: normals are memory-bound, and the float64 integral pass takes under
one millisecond at benchmark size while the required topology check dominates.
Including host/device transfer leaves no end-to-end kernel with enough effective
arithmetic intensity for a GPU path that wins.

MIT.
