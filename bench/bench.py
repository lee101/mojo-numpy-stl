"""Measured comparison with the installed numpy-stl reference implementation."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from stl import mesh as upstream_mesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from mojo_stl import mesh  # noqa: E402


def timeit(fn, reps=4):
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter(); fn(); best = min(best, time.perf_counter() - start)
    return best


def row(name, mojo, reference):
    print(f"| {name} | {mojo * 1e3:.2f} ms | {reference * 1e3:.2f} ms | {reference / mojo:.2f}x |", flush=True)


def main():
    rng = np.random.default_rng(0)
    n = 200_000
    data = np.zeros(n, dtype=mesh.Mesh.dtype)
    data["vectors"] = rng.normal(size=(n, 3, 3)).astype(np.float32)
    ours = mesh.Mesh(data.copy(), calculate_normals=False)
    reference = upstream_mesh.Mesh(data.copy(), calculate_normals=False)
    print("| kernel | mojo-numpy-stl | numpy-stl | speedup |", flush=True)
    print("| --- | ---: | ---: | ---: |", flush=True)
    row("update_normals, 200k facets", timeit(ours.update_normals), timeit(reference.update_normals))
    ours.update_normals(); reference.update_normals()
    row("get_unit_normals, 200k facets", timeit(ours.get_unit_normals), timeit(reference.get_unit_normals))
    cube_vectors = np.array([[[0, 0, 0], [1, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
                             [[0, 0, 1], [1, 0, 1], [1, 1, 1]], [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
                             [[0, 0, 0], [1, 0, 0], [1, 0, 1]], [[0, 0, 0], [1, 0, 1], [0, 0, 1]],
                             [[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 1]],
                             [[1, 1, 0], [0, 1, 0], [0, 1, 1]], [[1, 1, 0], [0, 1, 1], [1, 1, 1]],
                             [[0, 1, 0], [0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1], [0, 1, 1]]], dtype=np.float32)
    mass_data = np.zeros(49_992, dtype=mesh.Mesh.dtype)
    mass_data["vectors"] = np.tile(cube_vectors, (4_166, 1, 1))
    mass_data["vectors"][:, :, 0] += np.repeat(np.arange(4_166, dtype=np.float32) * 2, 12)[:, None]
    mass_ours = mesh.Mesh(mass_data.copy(), calculate_normals=False)
    mass_reference = upstream_mesh.Mesh(mass_data.copy(), calculate_normals=False)
    row("mass properties, 49,992 closed facets", timeit(mass_ours.get_mass_properties), timeit(mass_reference.get_mass_properties))


if __name__ == "__main__": main()
