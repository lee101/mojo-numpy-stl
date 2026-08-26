from __future__ import annotations

import io

import numpy as np
import pytest
from stl import mesh as upstream_mesh
from stl import base as upstream_base

from mojo_stl import Mode, RemoveDuplicates
from mojo_stl import mesh


def cube_data(dtype=mesh.Mesh.dtype):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    data = np.zeros(len(faces), dtype=dtype)
    data["vectors"] = vertices[faces]
    return data


def test_dtype_and_surface_views_match_upstream():
    ours = mesh.Mesh(cube_data())
    reference = upstream_mesh.Mesh(cube_data(upstream_mesh.Mesh.dtype))
    assert mesh.Mesh.dtype == upstream_mesh.Mesh.dtype
    for name in ("vectors", "normals", "points", "v0", "v1", "v2", "x", "y", "z", "areas", "centroids", "units", "min_", "max_"):
        assert np.allclose(getattr(ours, name), getattr(reference, name))
    ours.v0[0] = [3, 4, 5]
    assert np.array_equal(ours[0, :3], [3, 4, 5])


def test_mojo_normals_match_numpy_stl_on_random_facets():
    rng = np.random.default_rng(4)
    data = np.zeros(10_003, dtype=mesh.Mesh.dtype)
    data["vectors"] = rng.normal(size=(len(data), 3, 3)).astype(np.float32)
    ours = mesh.Mesh(data.copy(), calculate_normals=False)
    reference = upstream_mesh.Mesh(data.copy(), calculate_normals=False)
    ours.update_normals(); reference.update_normals()
    assert np.allclose(ours.normals, reference.normals, rtol=1e-6, atol=1e-6)
    assert np.allclose(ours.get_unit_normals(), reference.get_unit_normals(), rtol=1e-6, atol=1e-6)


def test_simd_mass_tail_matches_numpy_stl():
    data = np.concatenate((cube_data(), cube_data()[:1]))
    ours = mesh.Mesh(data.copy())
    reference = upstream_mesh.Mesh(data.copy())
    got = ours.get_mass_properties()
    expected = reference.get_mass_properties()
    assert got[0] == pytest.approx(expected[0])
    assert got[1] == pytest.approx(expected[1])
    assert got[2] == pytest.approx(expected[2])


def test_parallel_threshold_geometry_and_mass_reduction():
    n = 1_000_003
    base = cube_data()
    data = np.resize(base, n)
    ours = mesh.Mesh(data, calculate_normals=False)
    ours.update_normals()

    expected = upstream_mesh.Mesh(base.copy())
    probes = np.concatenate((
        np.arange(24),
        np.arange(n // 8 - 2, n // 8 + 3),
        np.arange(n // 2 - 2, n // 2 + 3),
        np.arange(n - 24, n),
    ))
    faces = probes % len(base)
    assert np.allclose(ours.normals[probes], expected.normals[faces], rtol=1e-6, atol=1e-6)
    assert np.allclose(ours.areas[probes], expected.areas[faces], rtol=1e-6, atol=1e-6)
    assert np.allclose(ours.centroids[probes], expected.centroids[faces], rtol=1e-6, atol=1e-6)

    repeats, tail = divmod(n, len(base))
    base_integrals = mesh.Mesh(base.copy())._integrals()
    tail_integrals = mesh.Mesh(base[:tail].copy())._integrals()
    assert ours._integrals() == pytest.approx(
        repeats * base_integrals + tail_integrals, rel=1e-12, abs=1e-12
    )


def test_native_boundary_rejects_dtype_narrowing_and_handles_empty_data():
    with pytest.raises(TypeError, match="Mesh.dtype"):
        mesh.Mesh(np.zeros((1, 12), dtype=np.float64), calculate_normals=False)
    with pytest.raises(TypeError, match="Mesh.dtype"):
        mesh.Mesh(np.zeros(1, dtype=[("vectors", "<f8", (3, 3))]), calculate_normals=False)

    empty = mesh.Mesh(np.empty(0, dtype=mesh.Mesh.dtype))
    empty.update_normals()
    assert empty.get_unit_normals().shape == (0, 3)
    assert np.array_equal(empty._integrals(), np.zeros(10))


def test_read_only_and_strided_data_are_safe_to_update():
    source = cube_data()
    readonly = source.copy(); readonly.flags.writeable = False
    copied = mesh.Mesh(readonly)
    assert copied.data.flags.writeable
    assert np.allclose(copied.normals, upstream_mesh.Mesh(source.copy()).normals)

    strided = np.concatenate((source, source))[::2]
    compacted = mesh.Mesh(strided)
    assert compacted.data.flags.c_contiguous
    assert np.allclose(compacted.normals, upstream_mesh.Mesh(strided.copy()).normals)


def test_mass_volume_center_and_inertia_match_numpy_stl():
    data = cube_data()
    ours = mesh.Mesh(data.copy())
    reference = upstream_mesh.Mesh(data.copy())
    got = ours.get_mass_properties()
    expected = reference.get_mass_properties()
    assert got[0] == pytest.approx(expected[0])
    assert got[1] == pytest.approx(expected[1])
    assert got[2] == pytest.approx(expected[2])
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx([0.5, 0.5, 0.5])
    mass, cog, inertia = ours.get_mass_properties_with_density(2.7)
    assert mass == pytest.approx(2.7) and cog == pytest.approx(got[1])
    assert inertia == pytest.approx(got[2] * 2.7)


def test_exact_closed_surface_check_matches_numpy_stl():
    data = cube_data()
    ours = mesh.Mesh(data.copy())
    reference = upstream_mesh.Mesh(data.copy())
    assert ours.check(exact=True) == reference.is_closed(exact=True)

    open_data = data[:-1]
    ours = mesh.Mesh(open_data.copy())
    reference = upstream_mesh.Mesh(open_data.copy())
    assert ours.check(exact=True) == reference.is_closed(exact=True)

    duplicate = np.concatenate((data, data[:1]))
    ours = mesh.Mesh(duplicate.copy())
    reference = upstream_mesh.Mesh(duplicate.copy())
    assert ours.check(exact=True) == reference.is_closed(exact=True)

    signed_zero = data.copy()
    signed_zero["vectors"][signed_zero["vectors"] == 0] = -0.0
    ours = mesh.Mesh(signed_zero.copy())
    reference = upstream_mesh.Mesh(signed_zero.copy())
    assert ours.check(exact=True) == reference.is_closed(exact=True)


def test_binary_write_and_read_are_interoperable(tmp_path):
    ours = mesh.Mesh(cube_data(), name="unit-cube")
    ours_path = tmp_path / "ours.stl"
    ours.save(str(ours_path), mode=Mode.BINARY)
    upstream = upstream_mesh.Mesh.from_file(str(ours_path))
    assert np.array_equal(upstream.data, ours.data)

    reference_path = tmp_path / "reference.stl"
    source = upstream_mesh.Mesh(cube_data())
    source.save(str(reference_path), mode=upstream_mesh.stl.Mode.BINARY)
    loaded = mesh.Mesh.from_file(str(reference_path))
    assert np.array_equal(loaded.data, source.data)


def test_ascii_write_and_read_are_interoperable(tmp_path):
    ours = mesh.Mesh(cube_data(), name="cube name")
    path = tmp_path / "mesh-ascii.stl"
    ours.save(str(path), mode=Mode.ASCII)
    assert path.read_bytes().startswith(b"solid cube name")
    upstream = upstream_mesh.Mesh.from_file(str(path), mode=upstream_mesh.stl.Mode.ASCII)
    assert np.allclose(upstream.vectors, ours.vectors, atol=1e-6)

    reference_path = tmp_path / "reference-ascii.stl"
    upstream_mesh.Mesh(cube_data()).save(str(reference_path), mode=upstream_mesh.stl.Mode.ASCII)
    loaded = mesh.Mesh.from_file(str(reference_path), mode=Mode.ASCII)
    assert np.allclose(loaded.vectors, ours.vectors, atol=1e-6)


def test_file_handles_and_automatic_detection():
    original = mesh.Mesh(cube_data(), name="memory")
    binary = io.BytesIO(); original.save("memory.stl", fh=binary, mode=Mode.BINARY)
    binary.seek(0)
    loaded = mesh.Mesh.from_file("memory.stl", fh=binary)
    assert loaded.name.startswith("mojo-numpy-stl")
    assert np.array_equal(loaded.data, original.data)
    ascii_data = io.BytesIO(); original.save("memory.stl", fh=ascii_data, mode=Mode.ASCII)
    ascii_data.seek(0)
    assert np.allclose(mesh.Mesh.from_file("memory.stl", fh=ascii_data).vectors, original.vectors)


def test_empty_and_duplicate_filters_match_upstream():
    data = cube_data()
    data = np.concatenate((data, data[:1], np.zeros(1, dtype=data.dtype)))
    ours = mesh.Mesh(data, calculate_normals=False, remove_empty_areas=True, remove_duplicate_polygons=RemoveDuplicates.SINGLE)
    reference = upstream_mesh.Mesh(data, calculate_normals=False, remove_empty_areas=True,
                                   remove_duplicate_polygons=upstream_base.RemoveDuplicates.SINGLE)
    assert np.array_equal(ours.data, reference.data)


def test_transforms_and_mapping_contract():
    ours, reference = mesh.Mesh(cube_data()), upstream_mesh.Mesh(cube_data())
    ours.translate([1, 2, 3]); reference.translate([1, 2, 3])
    ours.rotate([0, 0, 1], np.pi / 3, point=[1, 2, 3]); reference.rotate([0, 0, 1], np.pi / 3, point=[1, 2, 3])
    assert np.allclose(ours.vectors, reference.vectors)
    assert len(ours) == len(list(ours)) == 12
    assert repr(ours).startswith("<Mesh:")
