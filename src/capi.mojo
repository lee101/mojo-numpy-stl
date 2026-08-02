"""Hot geometry kernels for mojo-numpy-stl's Python compatibility layer."""

from std.math import sqrt

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def facet(data: Int, i: Int) -> F32Ptr:
    # numpy-stl records are 12 float32 values followed by a uint16 attribute.
    return F32Ptr(unsafe_from_address=data + i * 50)


@export("mnst_update_normals")
def mnst_update_normals(data: Int, n: Int) abi("C"):
    for i in range(n):
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


@export("mnst_mass_integrals")
def mnst_mass_integrals(data: Int, n: Int, dst: Int) abi("C"):
    """The ten signed polyhedral integrals from Geometric Tools' formula."""
    var q = F64Ptr(unsafe_from_address=dst)
    var i0 = 0.0
    var i1 = 0.0
    var i2 = 0.0
    var i3 = 0.0
    var i4 = 0.0
    var i5 = 0.0
    var i6 = 0.0
    var i7 = 0.0
    var i8 = 0.0
    var i9 = 0.0
    for i in range(n):
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
    q[0] = i0 / 6.0
    q[1] = i1 / 24.0
    q[2] = i2 / 24.0
    q[3] = i3 / 24.0
    q[4] = i4 / 60.0
    q[5] = i5 / 60.0
    q[6] = i6 / 60.0
    q[7] = i7 / 120.0
    q[8] = i8 / 120.0
    q[9] = i9 / 120.0
