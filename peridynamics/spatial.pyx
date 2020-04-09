from libc.math cimport sqrt


def euclid(r1, r2):
    return ceuclid(r1, r2)


cdef inline double ceuclid(double[:] r1, double[:] r2):
    cdef int imax = 3
    cdef double[3] dr

    for i in range(imax):
        dr[i] = r2[i] - r1[i]
        dr[i] = dr[i] * dr[i]

    return sqrt(dr[0] + dr[1] + dr[2])


def strain(r1, r2, r10, r20):
    return cstrain(r1, r2, r10, r20)


cdef inline double cstrain(double[:] r1, double[:] r2,
                           double[:] r10, double[:] r20):
    cdef double l, dl

    l = ceuclid(r1, r2)
    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0


def strain2(l, r10, r20):
    return cstrain2(l, r10, r20)


cdef inline double cstrain2(double l, double[:] r10, double[:] r20):
    cdef double dl

    l0 = ceuclid(r10, r20)
    dl = l - l0

    return dl/l0


