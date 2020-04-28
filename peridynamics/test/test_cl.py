"""Tests for the OpenCL kernels."""
from ..cl import kernel_source, get_context, pad
from ..cl.utilities import DOUBLE_FP_SUPPORT
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pytest
from scipy.spatial.distance import cdist


def test_get_context():
    context = get_context()

    if type(context) == cl._cl.Context:
        devices = context.devices
        assert len(devices) == 1
        assert (devices[0].get_info(cl.device_info.DOUBLE_FP_CONFIG)
                == DOUBLE_FP_SUPPORT)
    else:
        assert context is None


class TestPad():
    """Test padding helper function."""

    def test_pad_1d(self):
        """Test padding for a 1D array."""
        dimension = 258
        group_size = 256
        expected_dimension = 512

        array = np.random.random(dimension)
        array = pad(array, group_size)

        assert array.shape == (expected_dimension,)
        assert np.all(
            array[dimension:] == np.zeros(expected_dimension-dimension)
            )

    def test_no_padding(self):
        """Test padding when non is required."""
        dimension = 512
        group_size = 256
        expected_dimension = 512

        array = np.random.random(dimension)
        array = pad(array, group_size)

        assert array.shape == (expected_dimension,)
        assert np.all(
            array[dimension:] == np.zeros(expected_dimension-dimension)
            )

    def test_pad_2d_axis0(self):
        """Test padding a 2D array along axis 0."""
        dimension = 755
        other_dimension = 5
        group_size = 256
        expected_dimension = 768

        array = np.random.random((dimension, other_dimension))
        array = pad(array, group_size)

        assert array.shape == ((expected_dimension, other_dimension,))
        assert np.all(
            array[dimension:, :] ==
            np.zeros((expected_dimension-dimension, other_dimension))
            )

    def test_pad_2d_axis1(self):
        """Test padding a 2D array along axis 1."""
        dimension = 400
        other_dimension = 17
        group_size = 256
        expected_dimension = 512

        array = np.random.random((other_dimension, dimension))
        array = pad(array, group_size, axis=1)

        assert array.shape == ((other_dimension, expected_dimension, ))
        assert np.all(
            array[:, dimension:] ==
            np.zeros((other_dimension, expected_dimension-dimension))
            )


@pytest.fixture(scope="module")
def context():
    """Create a context using the default platform, prefer GPU."""
    platform = cl.get_platforms()
    devices = platform[0].get_devices(cl.device_type.GPU)
    if not devices:
        devices = platform[0].get_devices(cl.device_type.DEFAULT)
    return cl.Context([devices[0]])


@pytest.fixture(scope="module")
def queue(context):
    """Create a CL command queue."""
    return cl.CommandQueue(context)


@pytest.fixture(scope="module")
def program(context):
    """Create a program object from the kernel source."""
    return cl.Program(context, kernel_source).build()


@pytest.fixture(scope="module")
def example():
    """Create an example test case to compare against OpenCL."""
    class Example():
        def __init__(self):
            n = 1000
            critical_strain = 0.05
            r0 = np.random.random((n, 3)).astype(np.float64)
            d0 = cdist(r0, r0).astype(np.float64)
            u = np.random.random((n, 3)).astype(np.float64)
            r = r0+u

            self.n = n
            self.critical_strain = critical_strain
            self.r0 = r0
            self.d0 = d0
            self.r = r
            self.strain = self._strain(r, d0)
            self.neighbourhood = self._neighbourhood(d0)
            self.family = np.sum(self.neighbourhood, axis=0).astype(np.int32)
            self.connectivity = (
                self.neighbourhood * ~(abs(self.strain) > critical_strain)
                ).astype(np.bool_)
            self.damage = (
                (self.family - np.sum(self.connectivity, axis=0))/self.family
                )

        def _strain(self, r, d0):
            d = cdist(r, r)
            strain = (d - d0) / (d0 + np.eye(r.shape[0]))
            return strain

        def _neighbourhood(self, d0):
            neighbourhood = d0 < 0.5
            np.fill_diagonal(neighbourhood, False)
            return neighbourhood

    return Example()


def test_neighbourhood(context, queue, program, example):
    """Test neighbourhood calculation."""
    # Retrieve test data
    n = example.n
    r0 = example.r0
    expected_neighbourhood = example.neighbourhood
    neighbourhood_h = np.empty((n, n), dtype=np.bool_)

    # Kernel functor
    neighbourhood = program.neighbourhood

    # Create buffers
    r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r0)
    neighbourhood_d = cl.Buffer(context, mf.WRITE_ONLY, neighbourhood_h.nbytes)

    neighbourhood(queue, (n, n), None, r_d, np.float64(0.5), neighbourhood_d)
    cl.enqueue_copy(queue, neighbourhood_h, neighbourhood_d)
    assert np.all(neighbourhood_h == expected_neighbourhood)


def test_distance(context, queue, program, example):
    """Test Euclidean distance calculation."""
    # Retrieve test data
    n = example.n
    r = example.r
    nhood = example.neighbourhood
    d = np.empty((n, n), dtype=np.float64)

    # Kernel functor
    dist = program.dist

    # Create buffers
    r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    nhood_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=nhood)
    d_d = cl.Buffer(context, mf.WRITE_ONLY, d.nbytes)

    dist(queue, (n, n), None, r_d, nhood_d, d_d)
    cl.enqueue_copy(queue, d, d_d)
    assert np.allclose(d[nhood], cdist(r, r)[nhood])


def test_strain(context, queue, program, example):
    """Test strain calculation."""
    # Retrieve test data
    n = example.n
    r = example.r
    d0 = example.d0
    nhood = example.neighbourhood
    expected_strain = example.strain
    strain_h = np.empty_like(d0)

    # Kernel functor
    strain = program.strain

    # Create buffers
    r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    d0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d0)
    nhood_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=nhood)
    strain_d = cl.Buffer(context, mf.WRITE_ONLY, strain_h.nbytes)

    strain(queue, (n, n), None, r_d, d0_d, nhood_d, strain_d)
    cl.enqueue_copy(queue, strain_h, strain_d)
    assert np.allclose(strain_h[nhood], expected_strain[nhood])


def test_break_bonds(context, queue, program, example):
    """Test bond breaking."""
    # Retrieve test data
    n = example.n
    nhood = example.neighbourhood
    strain = example.strain
    critical_strain = example.critical_strain
    expected_nhood_new = example.connectivity
    nhood_new = nhood.copy()

    # Kernel functor
    break_bonds = program.break_bonds

    # Create buffers
    strain_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=strain)
    nhood_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                        hostbuf=nhood)

    break_bonds(queue, (n, n), None, strain_d, np.float64(critical_strain),
                nhood_d)
    cl.enqueue_copy(queue, nhood_new, nhood_d)

    assert np.all(nhood_new == expected_nhood_new)


def test_break_bonds2(context, queue, program):
    """
    Test bond breaking with a hand crafted example.

    A negative strain is used to ensure absolute values are used.
    The first row of the neighbourhood begins as false to ensure strains below
    the critical value do not cause bonds to be reformed.
    """
    n = 3
    strain = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, -9.0]
        ])
    nhood = np.ones((n, n), dtype=np.bool_)
    nhood[0, :] = False
    nhood_new = nhood.copy()

    critical_strain = 6
    expected_nhood_new = np.array([
        [False, False, False],
        [True, True, True],
        [False, False, False]
        ])

    # Kernel functor
    break_bonds = program.break_bonds

    # Create buffers
    strain_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=strain)
    nhood_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                        hostbuf=nhood)

    break_bonds(queue, (n, n), None, strain_d, np.float64(critical_strain),
                nhood_d)
    cl.enqueue_copy(queue, nhood_new, nhood_d)
    assert np.all(nhood_new == expected_nhood_new)


def test_damage(context, queue, program, example):
    """Test damage kernels."""
    connectivity = example.connectivity
    family = example.family
    expected_damage = example.damage
    damage_h = np.empty_like(expected_damage)

    # Initialise
    group_size = 256

    connectivity = pad(connectivity, group_size)

    n_cols = connectivity.shape[1]
    col_length = connectivity.shape[0]
    work_groups_per_col = col_length//group_size

    p_h = np.zeros((work_groups_per_col, n_cols), dtype=np.int32)
    local_memory_size = (
        connectivity[:, 0].astype(np.int32).nbytes//work_groups_per_col
        )

    # Kernel functors
    damage1 = program.damage1
    damage2 = program.damage2

    # Create buffers
    connectivity_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=connectivity)
    b_d = cl.LocalMemory(local_memory_size)
    p_d = cl.Buffer(context, mf.READ_WRITE, p_h.nbytes)

    # Call first kernel
    damage1(queue, connectivity.shape, (group_size, 1,), connectivity_d, b_d,
            p_d)
    # In production copying this array now is unecessary and will hinder
    # performance
    cl.enqueue_copy(queue, p_h, p_d)
    assert np.allclose(np.sum(p_h, axis=0), np.sum(connectivity, axis=0))

    # Create buffers
    family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=family)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage_h.nbytes)

    # Call second kernel
    damage2(queue, (n_cols,), None, p_d, np.int32(work_groups_per_col),
            family_d, damage_d)
    cl.enqueue_copy(queue, damage_h, damage_d)
    assert np.allclose(damage_h, expected_damage)
