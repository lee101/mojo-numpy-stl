"""ctypes binding for the Mojo geometry kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_STL_LIB") or os.path.join(ROOT, "dist", "libmojo-numpy-stl.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mnst_update_normals": ([I, I], None),
    "mnst_unit_normals": ([I, I, I], None),
    "mnst_mass_integrals": ([I, I, I], None),
}
_loaded = None


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    if os.environ.get("MOJO_STL_LIB"):
        raise RuntimeError(f"MOJO_STL_LIB does not exist or is stale: {LIB}")
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")],
                          cwd=ROOT, text=True, capture_output=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stdout + proc.stderr).strip() or "Mojo library build failed")
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = args, result
    return _loaded


def _address(array: np.ndarray, *, writable: bool = False) -> int:
    """Return a checked address for an array passed to the native library.

    ctypes accepts arbitrary integer addresses.  Keep all validation on this
    side of the ABI, where NumPy can describe the allocation safely.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("native kernels require a NumPy array")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("native kernel arrays must be C-contiguous and aligned")
    if writable and not array.flags.writeable:
        raise ValueError("native kernel output must be writable")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("native kernel array has a null data pointer")
    return address


def update_normals(data: np.ndarray) -> None:
    if data.ndim != 1:
        raise ValueError("facet data must be a one-dimensional record array")
    if not len(data):
        return
    lib().mnst_update_normals(_address(data, writable=True), len(data))


def unit_normals(data: np.ndarray, result: np.ndarray) -> None:
    if data.ndim != 1 or result.shape != (len(data), 3) or result.dtype != np.dtype("<f4"):
        raise ValueError("invalid unit-normal kernel buffers")
    if not len(data):
        return
    lib().mnst_unit_normals(_address(data), len(data), _address(result, writable=True))


def mass_integrals(data: np.ndarray, result: np.ndarray) -> None:
    if data.ndim != 1 or result.shape != (10,) or result.dtype != np.dtype("<f8"):
        raise ValueError("invalid mass-integral kernel buffers")
    if not len(data):
        result.fill(0)
        return
    lib().mnst_mass_integrals(_address(data), len(data), _address(result, writable=True))
