"""Test OpenCL kernels."""
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
from scipy.spatial.distance import cdist


def _strain(r, d0):
    d = cdist(r, r)
    strain = (d - d0) / (d0 + np.eye(r.shape[0]))
    return strain


if __name__ == "__main__":
    # Create test data
    n = 5000
    r = np.random.random((n, 3)).astype(np.float32)
    d = np.empty((n, n), dtype=np.float32)

    # Create context and queue
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Create program and kernel functions
    kernel_source = open("./peridynamics.cl").read()
    program = cl.Program(context, kernel_source).build()
    dist = program.dist
    strain = program.strain

    # Distance
    r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    d_d = cl.Buffer(context, mf.WRITE_ONLY, d.nbytes)

    dist(queue, (n, n), None, r_d, d_d)
    cl.enqueue_copy(queue, d, d_d)
    assert np.allclose(d, cdist(r, r))

    # Strain
    strain_h = np.empty_like(d)
    u = np.random.random((n, 3)).astype(np.float32)
    r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r+u)
    strain_d = cl.Buffer(context, mf.WRITE_ONLY, strain_h.nbytes)

    strain(queue, (n, n), None, r_d, d_d, strain_d)
    cl.enqueue_copy(queue, strain_h, strain_d)
    assert np.allclose(strain_h, _strain(r+u, d), atol=1.e-6)
