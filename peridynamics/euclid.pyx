from libc.math cimport sqrt, abs


def euclid(r1, r2):
    return ceuclid(r1, r2)


cdef inline double ceuclid(double[:] r1, double[:] r2):
    cdef int imax = 3
    cdef double[3] dr

    for i in range(imax):
        dr[i] = r2[i] - r1[i]
        dr[i] = dr[i] * dr[i]

    return sqrt(dr[0] + dr[1] + dr[2])


