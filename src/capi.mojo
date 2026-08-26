"""Hot geometry kernels for mojo-numpy-stl's Python compatibility layer."""

from max.algorithm import parallelize
from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime NORMAL_PARALLEL_THRESHOLD = 1_000_000
comptime MASS_PARALLEL_THRESHOLD = 1_000_000
comptime WORKERS = 8


def facet(data: Int, i: Int) -> F32Ptr:
    # numpy-stl records are 12 float32 values followed by a uint16 attribute.
    return F32Ptr(unsafe_from_address=data + i * 50)


def update_normals_range(data: Int, start: Int, end: Int):
    for i in range(start, end):
        var t = facet(data, i)
        var ax = t[6] - t[3]
        var ay = t[7] - t[4]
        var az = t[8] - t[5]
        var bx = t[9] - t[3]
        var by = t[10] - t[4]
        var bz = t[11] - t[5]
        t[0] = ay * bz - az * by
        t[1] = az * bx - ax * bz
        t[2] = ax * by - ay * bx


def update_geometry_range(
    data: Int,
    areas: F32Ptr,
    centroids: F32Ptr,
    start: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    for i in range(start, end):
        var t = facet(data, i)
        comptime if W == 4:
            var v0 = t.load[width=W, alignment=1](3)
            var v1 = t.load[width=W, alignment=1](6)
            var v2 = t.load[width=W, alignment=1](8).shuffle[1, 2, 3, 0]()
            var a = v1 - v0
            var b = v2 - v0
            var cross = (
                a.shuffle[1, 2, 0, 3]() * b.shuffle[2, 0, 1, 3]()
                - a.shuffle[2, 0, 1, 3]() * b.shuffle[1, 2, 0, 3]()
            )
            cross[3] = 0.0
            t[0] = cross[0]
            t[1] = cross[1]
            t[2] = cross[2]
            areas[i] = 0.5 * sqrt((cross * cross).reduce_add())
            var center = (v0 + v1 + v2) / 3.0
            centroids[3 * i] = center[0]
            centroids[3 * i + 1] = center[1]
            centroids[3 * i + 2] = center[2]
        else:
            var ax = t[6] - t[3]
            var ay = t[7] - t[4]
            var az = t[8] - t[5]
            var bx = t[9] - t[3]
            var by = t[10] - t[4]
            var bz = t[11] - t[5]
            var nx = ay * bz - az * by
            var ny = az * bx - ax * bz
            var nz = ax * by - ay * bx
            t[0] = nx
            t[1] = ny
            t[2] = nz
            areas[i] = 0.5 * sqrt(nx * nx + ny * ny + nz * nz)
            centroids[3 * i] = (t[3] + t[6] + t[9]) / 3.0
            centroids[3 * i + 1] = (t[4] + t[7] + t[10]) / 3.0
            centroids[3 * i + 2] = (t[5] + t[8] + t[11]) / 3.0


@export("mnst_update_normals")
def mnst_update_normals(data: Int, n: Int) abi("C"):
    if n >= NORMAL_PARALLEL_THRESHOLD:
        @__parameter
        def work(part: Int) capturing:
            update_normals_range(data, part * n // WORKERS, (part + 1) * n // WORKERS)

        parallelize[work](WORKERS, WORKERS)
    else:
        update_normals_range(data, 0, n)


@export("mnst_update_geometry")
def mnst_update_geometry(data: Int, n: Int, areas_addr: Int, centroids_addr: Int) abi("C"):
    var areas = F32Ptr(unsafe_from_address=areas_addr)
    var centroids = F32Ptr(unsafe_from_address=centroids_addr)
    if n >= NORMAL_PARALLEL_THRESHOLD:
        @__parameter
        def work(part: Int) capturing:
            update_geometry_range(
                data,
                areas,
                centroids,
                part * n // WORKERS,
                (part + 1) * n // WORKERS,
            )

        parallelize[work](WORKERS, WORKERS)
    else:
        update_geometry_range(data, areas, centroids, 0, n)


@export("mnst_unit_normals")
def mnst_unit_normals(data: Int, n: Int, dst: Int) abi("C"):
    var u = F32Ptr(unsafe_from_address=dst)
    for i in range(n):
        var t = facet(data, i)
        var x = t[0]
        var y = t[1]
        var z = t[2]
        var length = sqrt(Float64(x * x + y * y + z * z))
        if length > 0.0:
            u[i * 3] = x / Float32(length)
            u[i * 3 + 1] = y / Float32(length)
            u[i * 3 + 2] = z / Float32(length)
        else:
            u[i * 3] = x
            u[i * 3 + 1] = y
            u[i * 3 + 2] = z


def mass_integrals_range(data: Int, start: Int, end: Int, q: F64Ptr, offset: Int):
    comptime W = simdwidthof[DType.float64]()
    var i0 = SIMD[DType.float64, W](0.0)
    var i1 = SIMD[DType.float64, W](0.0)
    var i2 = SIMD[DType.float64, W](0.0)
    var i3 = SIMD[DType.float64, W](0.0)
    var i4 = SIMD[DType.float64, W](0.0)
    var i5 = SIMD[DType.float64, W](0.0)
    var i6 = SIMD[DType.float64, W](0.0)
    var i7 = SIMD[DType.float64, W](0.0)
    var i8 = SIMD[DType.float64, W](0.0)
    var i9 = SIMD[DType.float64, W](0.0)
    var vector_end = start + (end - start) // W * W
    var i = start
    while i < vector_end:
        var x0 = SIMD[DType.float64, W]()
        var y0 = SIMD[DType.float64, W]()
        var z0 = SIMD[DType.float64, W]()
        var x1 = SIMD[DType.float64, W]()
        var y1 = SIMD[DType.float64, W]()
        var z1 = SIMD[DType.float64, W]()
        var x2 = SIMD[DType.float64, W]()
        var y2 = SIMD[DType.float64, W]()
        var z2 = SIMD[DType.float64, W]()
        for lane in range(W):
            var t = facet(data, i + lane)
            x0[lane] = Float64(t[3])
            y0[lane] = Float64(t[4])
            z0[lane] = Float64(t[5])
            x1[lane] = Float64(t[6])
            y1[lane] = Float64(t[7])
            z1[lane] = Float64(t[8])
            x2[lane] = Float64(t[9])
            y2[lane] = Float64(t[10])
            z2[lane] = Float64(t[11])
        var d0 = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
        var d1 = (x2 - x0) * (z1 - z0) - (x1 - x0) * (z2 - z0)
        var d2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

        var tx = x0 + x1
        var f1x = tx + x2
        var t2x = x0 * x0 + x1 * tx
        var f2x = t2x + x2 * f1x
        var f3x = x0 * x0 * x0 + x1 * t2x + x2 * f2x
        var g0x = f2x + x0 * (f1x + x0)
        var g1x = f2x + x1 * (f1x + x1)
        var g2x = f2x + x2 * (f1x + x2)

        var ty = y0 + y1
        var f1y = ty + y2
        var t2y = y0 * y0 + y1 * ty
        var f2y = t2y + y2 * f1y
        var f3y = y0 * y0 * y0 + y1 * t2y + y2 * f2y
        var g0y = f2y + y0 * (f1y + y0)
        var g1y = f2y + y1 * (f1y + y1)
        var g2y = f2y + y2 * (f1y + y2)

        var tz = z0 + z1
        var f1z = tz + z2
        var t2z = z0 * z0 + z1 * tz
        var f2z = t2z + z2 * f1z
        var f3z = z0 * z0 * z0 + z1 * t2z + z2 * f2z
        var g0z = f2z + z0 * (f1z + z0)
        var g1z = f2z + z1 * (f1z + z1)
        var g2z = f2z + z2 * (f1z + z2)

        i0 += d0 * f1x
        i1 += d0 * f2x
        i2 += d1 * f2y
        i3 += d2 * f2z
        i4 += d0 * f3x
        i5 += d1 * f3y
        i6 += d2 * f3z
        i7 += d0 * (y0 * g0x + y1 * g1x + y2 * g2x)
        i8 += d1 * (z0 * g0y + z1 * g1y + z2 * g2y)
        i9 += d2 * (x0 * g0z + x1 * g1z + x2 * g2z)
        i += W

    var s0 = i0.reduce_add()
    var s1 = i1.reduce_add()
    var s2 = i2.reduce_add()
    var s3 = i3.reduce_add()
    var s4 = i4.reduce_add()
    var s5 = i5.reduce_add()
    var s6 = i6.reduce_add()
    var s7 = i7.reduce_add()
    var s8 = i8.reduce_add()
    var s9 = i9.reduce_add()
    while i < end:
        var t = facet(data, i)
        var x0 = Float64(t[3])
        var y0 = Float64(t[4])
        var z0 = Float64(t[5])
        var x1 = Float64(t[6])
        var y1 = Float64(t[7])
        var z1 = Float64(t[8])
        var x2 = Float64(t[9])
        var y2 = Float64(t[10])
        var z2 = Float64(t[11])
        var d0 = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
        var d1 = (x2 - x0) * (z1 - z0) - (x1 - x0) * (z2 - z0)
        var d2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        var tx = x0 + x1
        var f1x = tx + x2
        var t2x = x0 * x0 + x1 * tx
        var f2x = t2x + x2 * f1x
        var f3x = x0 * x0 * x0 + x1 * t2x + x2 * f2x
        var g0x = f2x + x0 * (f1x + x0)
        var g1x = f2x + x1 * (f1x + x1)
        var g2x = f2x + x2 * (f1x + x2)
        var ty = y0 + y1
        var f1y = ty + y2
        var t2y = y0 * y0 + y1 * ty
        var f2y = t2y + y2 * f1y
        var f3y = y0 * y0 * y0 + y1 * t2y + y2 * f2y
        var g0y = f2y + y0 * (f1y + y0)
        var g1y = f2y + y1 * (f1y + y1)
        var g2y = f2y + y2 * (f1y + y2)
        var tz = z0 + z1
        var f1z = tz + z2
        var t2z = z0 * z0 + z1 * tz
        var f2z = t2z + z2 * f1z
        var f3z = z0 * z0 * z0 + z1 * t2z + z2 * f2z
        var g0z = f2z + z0 * (f1z + z0)
        var g1z = f2z + z1 * (f1z + z1)
        var g2z = f2z + z2 * (f1z + z2)
        s0 += d0 * f1x
        s1 += d0 * f2x
        s2 += d1 * f2y
        s3 += d2 * f2z
        s4 += d0 * f3x
        s5 += d1 * f3y
        s6 += d2 * f3z
        s7 += d0 * (y0 * g0x + y1 * g1x + y2 * g2x)
        s8 += d1 * (z0 * g0y + z1 * g1y + z2 * g2y)
        s9 += d2 * (x0 * g0z + x1 * g1z + x2 * g2z)
        i += 1
    q[offset] = s0
    q[offset + 1] = s1
    q[offset + 2] = s2
    q[offset + 3] = s3
    q[offset + 4] = s4
    q[offset + 5] = s5
    q[offset + 6] = s6
    q[offset + 7] = s7
    q[offset + 8] = s8
    q[offset + 9] = s9


@export("mnst_mass_integrals")
def mnst_mass_integrals(data: Int, n: Int, dst: Int) abi("C"):
    """The ten signed polyhedral integrals from Geometric Tools' formula."""
    var q = F64Ptr(unsafe_from_address=dst)
    if n >= MASS_PARALLEL_THRESHOLD:
        @__parameter
        def work(part: Int) capturing:
            mass_integrals_range(
                data,
                part * n // WORKERS,
                (part + 1) * n // WORKERS,
                q,
                10 * part,
            )

        parallelize[work](WORKERS, WORKERS)
        for value in range(10):
            var total = 0.0
            for part in range(WORKERS):
                total += q[10 * part + value]
            q[value] = total
    else:
        mass_integrals_range(data, 0, n, q, 0)
    q[0] /= 6.0
    q[1] /= 24.0
    q[2] /= 24.0
    q[3] /= 24.0
    q[4] /= 60.0
    q[5] /= 60.0
    q[6] /= 60.0
    q[7] /= 120.0
    q[8] /= 120.0
    q[9] /= 120.0
