"""Tests for the OpenCL kernels."""
from ..cl import kernel_source
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pytest
from scipy.spatial.distance import cdist


@pytest.fixture(scope="module")
def gpu_context():
    """Create a context with the first GPU on the default platform."""
    platform = cl.get_platforms()
    devices = platform[0].get_devices(cl.device_type.GPU)
    return cl.Context([devices[0]])


@pytest.fixture(scope="module")
def queue(gpu_context):
    """Create a CL command queue."""
    return cl.CommandQueue(gpu_context)


@pytest.fixture(scope="module")
def program(gpu_context):
    """Create a program object from the kernel source."""
    return cl.Program(gpu_context, kernel_source).build()


@pytest.fixture(scope="module")
def example():
    """Create an example test case to compare against OpenCL."""
    class Example():
        def __init__(self):
            n = 1000
            r0 = np.random.random((n, 3)).astype(np.float32)
            d0 = cdist(r0, r0).astype(np.float32)
            u = np.random.random((n, 3)).astype(np.float32)
            r = r0+u

            self.n = n
            self.d0 = d0
            self.r = r
            self.strain = self._strain(r, d0)

        def _strain(self, r, d0):
            d = cdist(r, r)
            strain = (d - d0) / (d0 + np.eye(r.shape[0]))
            return strain

    return Example()


def test_distance(gpu_context, queue, program, example):
    """Test Euclidean distance calculation."""
    # Retrieve test data
    n = example.n
    r = example.r
    d = np.empty((n, n), dtype=np.float32)

    # Kernel functor
    dist = program.dist

    # Create buffers
    r_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    d_d = cl.Buffer(gpu_context, mf.WRITE_ONLY, d.nbytes)

    dist(queue, (n, n), None, r_d, d_d)
    cl.enqueue_copy(queue, d, d_d)
    assert np.allclose(d, cdist(r, r))


def test_strain(gpu_context, queue, program, example):
    """Test strain calculation."""
    # Retrieve test data
    n = example.n
    r = example.r
    d0 = example.d0
    expected_strain = example.strain
    strain_h = np.empty_like(d0)

    # Kernel functor
    strain = program.strain

    # Create buffers
    r_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    d0_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d0)
    strain_d = cl.Buffer(gpu_context, mf.WRITE_ONLY, strain_h.nbytes)

    strain(queue, (n, n), None, r_d, d0_d, strain_d)
    cl.enqueue_copy(queue, strain_h, strain_d)
    assert np.allclose(strain_h, expected_strain, atol=1.e-6)


def test_neighbourhood(gpu_context, queue, program, example):
    """Test neighbourhood calculation."""
    # Retrieve test data
    n = example.n
    r = example.r
    neighbourhood_h = np.empty((n, n), dtype=np.bool_)

    neighbourhood_expected = cdist(r, r) < 0.5

    # Kernel functor
    neighbourhood = program.neighbourhood

    # Create buffers
    r_d = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    neighbourhood_d = cl.Buffer(gpu_context, mf.WRITE_ONLY,
                                neighbourhood_h.nbytes)
    neighbourhood(queue, (n, n), None, r_d, np.float32(0.5), neighbourhood_d)
    cl.enqueue_copy(queue, neighbourhood_h, neighbourhood_d)
    assert np.all(neighbourhood_h == neighbourhood_expected)
