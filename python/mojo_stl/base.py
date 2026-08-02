"""numpy-stl compatible mesh container backed by Mojo geometry kernels."""

from __future__ import annotations

import enum
import math

import numpy as np

from ._lib import mass_integrals, unit_normals, update_normals

AREA_SIZE_THRESHOLD = 0
VECTORS = DIMENSIONS = 3


class Dimension(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


X, Y, Z = Dimension.X, Dimension.Y, Dimension.Z


class RemoveDuplicates(enum.Enum):
    NONE = 0
    SINGLE = 1
    ALL = 2

    @classmethod
    def map(cls, value):
        if value is True:
            return cls.SINGLE
        if value and value in cls:
            return value
        return cls.NONE


class BaseMesh:
    dtype = np.dtype([("normals", "<f4", (3,)), ("vectors", "<f4", (3, 3)), ("attr", "<u2", (1,))])

    def __init__(self, data, calculate_normals=True, remove_empty_areas=False,
                 remove_duplicate_polygons=RemoveDuplicates.NONE, name="", speedups=True, **kwargs):
        del kwargs
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a NumPy array with Mesh.dtype")
        if data.dtype != self.dtype:
            raise TypeError(f"data dtype must be Mesh.dtype ({self.dtype!r}), got {data.dtype!r}")
        if data.ndim != 1:
            raise ValueError("data must be a one-dimensional structured array")
        if remove_empty_areas:
            data = self.remove_empty_areas(data)
        if RemoveDuplicates.map(remove_duplicate_polygons) is not RemoveDuplicates.NONE:
            data = self.remove_duplicate_polygons(data, remove_duplicate_polygons)
        self.data = np.ascontiguousarray(data)
        if not self.data.flags.writeable:
            self.data = self.data.copy()
        self.name, self.speedups = name, speedups
        if calculate_normals:
            self.update_normals()

    @property
    def attr(self): return self.data["attr"]
    @attr.setter
    def attr(self, value): self.data["attr"] = value
    @property
    def normals(self): return self.data["normals"]
    @normals.setter
    def normals(self, value): self.data["normals"] = value
    @property
    def vectors(self): return self.data["vectors"]
    @vectors.setter
    def vectors(self, value): self.data["vectors"] = value
    @property
    def points(self): return self.vectors.reshape(self.data.size, 9)
    @points.setter
    def points(self, value): self.points[:] = value
    @property
    def v0(self): return self.vectors[:, 0]
    @v0.setter
    def v0(self, value): self.vectors[:, 0] = value
    @property
    def v1(self): return self.vectors[:, 1]
    @v1.setter
    def v1(self, value): self.vectors[:, 1] = value
    @property
    def v2(self): return self.vectors[:, 2]
    @v2.setter
    def v2(self, value): self.vectors[:, 2] = value
    @property
    def x(self): return self.points[:, 0::3]
    @x.setter
    def x(self, value): self.points[:, 0::3] = value
    @property
    def y(self): return self.points[:, 1::3]
    @y.setter
    def y(self, value): self.points[:, 1::3] = value
    @property
    def z(self): return self.points[:, 2::3]
    @z.setter
    def z(self, value): self.points[:, 2::3] = value

    @classmethod
    def remove_empty_areas(cls, data):
        v = data["vectors"]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return data[(n * n).sum(axis=1) > AREA_SIZE_THRESHOLD ** 2]

    @classmethod
    def remove_duplicate_polygons(cls, data, value=RemoveDuplicates.SINGLE):
        value = RemoveDuplicates.map(value)
        if value is RemoveDuplicates.NONE:
            return data
        polygons = data["vectors"].sum(axis=1)
        idx = np.lexsort(polygons.T)
        diff = np.any(polygons[idx[1:]] != polygons[idx[:-1]], axis=1)
        if value is RemoveDuplicates.SINGLE:
            return data[np.sort(idx[np.concatenate(([True], diff))])]
        diff_a, diff_b = np.concatenate(([True], diff)), np.concatenate((diff, [True]))
        filtered = data[np.sort(idx[diff_a & diff_b])]
        return data[np.sort(idx[diff_a])] if len(filtered) <= len(data) / 2 else data[np.sort(idx[np.concatenate((diff, [False]))])]

    def update_normals(self, update_areas=True, update_centroids=True):
        update_normals(self.data)
        if update_areas:
            self.update_areas(self.normals)
        if update_centroids:
            self.update_centroids()

    def get_unit_normals(self):
        result = np.empty((len(self), 3), dtype=np.float32)
        unit_normals(self.data, result)
        return result

    def update_areas(self, normals=None):
        normals = self.normals if normals is None else normals
        areas = 0.5 * np.sqrt((normals * normals).sum(axis=1))
        self.areas = areas.reshape((-1, 1))

    def update_centroids(self): self.centroids = (self.v0 + self.v1 + self.v2) / 3
    def update_min(self): self._min = self.vectors.min(axis=(0, 1))
    def update_max(self): self._max = self.vectors.max(axis=(0, 1))
    def update_units(self): self.units = self.get_unit_normals()

    @property
    def min_(self):
        if not hasattr(self, "_min"): self.update_min()
        return self._min
    @min_.setter
    def min_(self, value): self._min = value
    @property
    def max_(self):
        if not hasattr(self, "_max"): self.update_max()
        return self._max
    @max_.setter
    def max_(self, value): self._max = value
    @property
    def areas(self):
        if not hasattr(self, "_areas"): self.update_areas()
        return self._areas
    @areas.setter
    def areas(self, value): self._areas = value
    @property
    def centroids(self):
        if not hasattr(self, "_centroids"): self.update_centroids()
        return self._centroids
    @centroids.setter
    def centroids(self, value): self._centroids = value
    @property
    def units(self):
        if not hasattr(self, "_units"): self.update_units()
        return self._units
    @units.setter
    def units(self, value): self._units = value

    def check(self, exact=False):
        if exact:
            reversed_triangles = (np.cross(self.v1 - self.v0, self.v2 - self.v0) * self.normals).sum(axis=1) < 0
            n = len(self)
            edges = np.empty((3, n, 2, 3), dtype=np.float32)
            edges[0] = self.vectors[:, (0, 1)]
            edges[1] = self.vectors[:, (1, 2)]
            edges[2] = self.vectors[:, (2, 0)]
            edges[:, reversed_triangles] = edges[:, reversed_triangles, ::-1]
            directed = edges.reshape(3 * n, 6)
            if len(np.unique(directed, axis=0)) != 3 * n:
                return False
            undirected = directed.copy()
            left, right = undirected[:, :3], undirected[:, 3:]
            reverse = ((left[:, 0] > right[:, 0]) |
                       ((left[:, 0] == right[:, 0]) & (left[:, 1] > right[:, 1])) |
                       ((left[:, 0] == right[:, 0]) & (left[:, 1] == right[:, 1]) & (left[:, 2] > right[:, 2])))
            undirected[reverse] = undirected[reverse][:, (3, 4, 5, 0, 1, 2)]
            return 3 * n == 2 * len(np.unique(undirected, axis=0))
        allowed = np.abs(self.normals).sum(axis=0) * np.finfo(np.float32).eps
        return bool((np.abs(self.normals.sum(axis=0)) <= allowed).all())

    is_closed = check

    def _integrals(self):
        value = np.empty(10, dtype=np.float64)
        mass_integrals(self.data, value)
        return value

    def get_mass_properties(self):
        self.check(True)
        intg = self._integrals()
        volume = intg[0]
        cog = intg[1:4] / volume
        cogsq = cog * cog
        inertia = np.zeros((3, 3))
        inertia[0, 0] = intg[5] + intg[6] - volume * (cogsq[1] + cogsq[2])
        inertia[1, 1] = intg[4] + intg[6] - volume * (cogsq[2] + cogsq[0])
        inertia[2, 2] = intg[4] + intg[5] - volume * (cogsq[0] + cogsq[1])
        inertia[0, 1] = inertia[1, 0] = -(intg[7] - volume * cog[0] * cog[1])
        inertia[1, 2] = inertia[2, 1] = -(intg[8] - volume * cog[1] * cog[2])
        inertia[0, 2] = inertia[2, 0] = -(intg[9] - volume * cog[2] * cog[0])
        return volume, cog, inertia

    def get_mass_properties_with_density(self, density):
        volume, cog, inertia = self.get_mass_properties()
        return volume * density, cog, inertia * density

    @classmethod
    def rotation_matrix(cls, axis, theta):
        axis = np.asarray(axis)
        if not axis.any(): return np.identity(3)
        axis = axis / np.linalg.norm(axis)
        a = math.cos(theta / 2); b, c, d = -axis * math.sin(theta / 2)
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c+a*d), 2*(b*d-a*c)], [2*(b*c-a*d), a*a+c*c-b*b-d*d, 2*(c*d+a*b)], [2*(b*d+a*c), 2*(c*d-a*b), a*a+d*d-b*b-c*c]])

    def rotate_using_matrix(self, rotation_matrix, point=None):
        point = np.zeros(3) if point is None else np.asarray(point)
        self.normals[:] = self.normals.dot(rotation_matrix)
        self.vectors[:] = (self.vectors - point).dot(rotation_matrix) + point
    def rotate(self, axis, theta=0, point=None):
        if theta: self.rotate_using_matrix(self.rotation_matrix(axis, theta), point)
    def translate(self, translation): self.vectors[:] += np.asarray(translation)
    def transform(self, matrix):
        matrix = np.asarray(matrix); assert matrix.shape == (4, 4)
        assert np.allclose(np.linalg.det(matrix[:3, :3]), 1.0)
        self.vectors[:] = self.vectors.dot(matrix[:3, :3].T) + matrix[:3, 3]

    def __getitem__(self, key): return self.points[key]
    def __setitem__(self, key, value): self.points[key] = value
    def __len__(self): return self.data.size
    def __iter__(self): yield from self.points
    def __repr__(self): return f"<Mesh: {self.name!r} {self.data.size} vertices>"
