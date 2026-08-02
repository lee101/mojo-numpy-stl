"""Binary and ASCII STL interchange compatible with numpy-stl's covered API."""

from __future__ import annotations

import enum
import io
import os
import struct

import numpy as np

from .base import BaseMesh

BUFFER_SIZE, HEADER_SIZE, COUNT_SIZE, MAX_COUNT = 4096, 80, 4, int(1e8)


class Mode(enum.IntEnum):
    AUTOMATIC = 0
    ASCII = 1
    BINARY = 2


AUTOMATIC, ASCII, BINARY = Mode.AUTOMATIC, Mode.ASCII, Mode.BINARY


def _binary(fh, header, check_size=False):
    count_raw = fh.read(COUNT_SIZE)
    count = struct.unpack("<I", count_raw)[0] if len(count_raw) == COUNT_SIZE else 0
    if count >= MAX_COUNT:
        raise AssertionError(f"File too large, got {count} triangles which exceeds the maximum of {MAX_COUNT}")
    if check_size and hasattr(fh, "seek"):
        here = fh.tell(); fh.seek(0, os.SEEK_END); size = fh.tell() - HEADER_SIZE - COUNT_SIZE; fh.seek(here)
        if size != count * BaseMesh.dtype.itemsize:
            raise AssertionError(f"Expected {size // BaseMesh.dtype.itemsize} vectors but header indicates {count}")
    raw = fh.read(count * BaseMesh.dtype.itemsize)
    if len(raw) < count * BaseMesh.dtype.itemsize:
        raise ValueError("truncated binary STL facet data")
    return header.strip(), np.frombuffer(raw, dtype=BaseMesh.dtype, count=count).copy()


def _ascii_bytes(raw):
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or not lines[0].lower().startswith(b"solid"):
        raise ValueError("not an ASCII STL")
    name = lines[0][5:].strip()
    facets, i = [], 1
    while i < len(lines):
        line = lines[i].lower()
        if line.startswith((b"endsolid", b"end solid")):
            break
        if not line.startswith(b"facet normal"):
            raise ValueError(f"expected facet normal, got {lines[i]!r}")
        normal = [float(x) for x in lines[i].split()[2:5]]; i += 1
        if i >= len(lines) or lines[i].lower() != b"outer loop": raise ValueError("expected outer loop")
        i += 1; vertices = []
        for _ in range(3):
            if i >= len(lines) or not lines[i].lower().startswith(b"vertex"): raise ValueError("expected vertex")
            vertices.append([float(x) for x in lines[i].split()[1:4]]); i += 1
        if i >= len(lines) or lines[i].lower() != b"endloop": raise ValueError("expected endloop")
        i += 1
        if i >= len(lines) or lines[i].lower() != b"endfacet": raise ValueError("expected endfacet")
        i += 1; facets.append((normal, vertices, [0]))
    data = np.asarray(facets, dtype=BaseMesh.dtype) if facets else np.empty(0, dtype=BaseMesh.dtype)
    return name, data


class BaseStl(BaseMesh):
    @classmethod
    def load(cls, fh, mode=AUTOMATIC, speedups=True):
        del speedups
        header = fh.read(HEADER_SIZE)
        if not header: return None
        if isinstance(header, str): header = header.encode()
        if mode is Mode.BINARY: return _binary(fh, header)
        if mode is Mode.ASCII:
            return _ascii_bytes(header + fh.read())
        if header.lstrip().lower().startswith(b"solid"):
            try: return _ascii_bytes(header + fh.read())
            except ValueError:
                if not fh.seekable(): raise
                fh.seek(HEADER_SIZE); return _binary(fh, header, check_size=True)
        return _binary(fh, header)

    @classmethod
    def from_file(cls, filename, calculate_normals=True, fh=None, mode=Mode.AUTOMATIC, speedups=True, **kwargs):
        if fh is None:
            with open(filename, "rb") as file_handle: loaded = cls.load(file_handle, mode, speedups)
        else: loaded = cls.load(fh, mode, speedups)
        if loaded is None: raise ValueError("empty STL file")
        name, data = loaded
        return cls(data, calculate_normals, name=name.decode(errors="replace") if isinstance(name, bytes) else name,
                   speedups=speedups, **kwargs)

    @classmethod
    def from_files(cls, filenames, calculate_normals=True, mode=Mode.AUTOMATIC, speedups=True, **kwargs):
        meshes = [cls.from_file(name, calculate_normals, mode=mode, speedups=speedups, **kwargs) for name in filenames]
        return cls(np.concatenate([item.data for item in meshes]), calculate_normals=calculate_normals, **kwargs)

    @classmethod
    def from_multi_file(cls, filename, calculate_normals=True, fh=None, mode=Mode.AUTOMATIC, speedups=True, **kwargs):
        if fh is None:
            with open(filename, "rb") as file_handle: raw = file_handle.read()
        else: raw = fh.read()
        if isinstance(raw, str): raw = raw.encode()
        if mode is Mode.BINARY or not raw.lstrip().lower().startswith(b"solid"):
            name, data = _binary(io.BytesIO(raw[80:]), raw[:80])
            yield cls(data, calculate_normals, name=name.decode(errors="replace"), speedups=speedups, **kwargs)
            return
        starts = [i for i, line in enumerate(raw.splitlines(keepends=True)) if line.strip().lower().startswith(b"solid")]
        lines = raw.splitlines(keepends=True)
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(lines)
            name, data = _ascii_bytes(b"".join(lines[start:end]))
            yield cls(data, calculate_normals, name=name.decode(errors="replace"), speedups=speedups, **kwargs)

    def get_header(self, name):
        return (f"mojo-numpy-stl 0.1.0 {name}")[:80].ljust(80, " ")

    def _write_binary(self, fh, name):
        fh.write(self.get_header(name).encode())
        fh.write(struct.pack("<I", len(self)))
        fh.write(self.data.tobytes())

    def _write_ascii(self, fh, name):
        def write(line): fh.write((line + "\n").encode())
        write(f"solid {name}")
        for item in self.data:
            n = item["normals"]; v = item["vectors"]
            write("facet normal {:f} {:f} {:f}".format(*map(float, n))); write("  outer loop")
            for vertex in v: write("    vertex {:f} {:f} {:f}".format(*map(float, vertex)))
            write("  endloop"); write("endfacet")
        write(f"endsolid {name}")

    def save(self, filename, fh=None, mode=Mode.AUTOMATIC, update_normals=True):
        if not filename: raise AssertionError("Filename is required for the STL headers")
        if update_normals: self.update_normals()
        if mode not in (Mode.AUTOMATIC, Mode.ASCII, Mode.BINARY): raise ValueError(f"Mode {mode!r} is invalid")
        write = self._write_ascii if mode is Mode.ASCII else self._write_binary
        name = self.name or os.path.basename(filename)
        if fh is not None:
            if isinstance(fh, io.TextIOBase): raise TypeError("File handles should be in binary mode - even when writing an ASCII STL.")
            write(fh, name)
        else:
            with open(filename, "wb") as file_handle: write(file_handle, name)


StlMesh = BaseStl.from_file
