"""Tests for the neighbour list module."""
from .conftest import context_available
from peripy.create_crack import (create_crack)
from ..integrators import Euler, EulerCL
from ..model import Model
import numpy as np
from scipy.spatial.distance import cdist
import pytest


@pytest.fixture()
def basic_model_3d(data_path, simple_displacement_boundary):
    """Create a basic 3D model object."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    euler = Euler(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  dimensions=3,
                  is_displacement_boundary=simple_displacement_boundary)
    return model, euler


@pytest.fixture()
def basic_model_3d_cl(data_path, simple_displacement_boundary):
    """Create a basic 3D model object using an OpenCL integrator."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    euler = EulerCL(dt=1e-3)
    model = Model(mesh_file, integrator=euler, horizon=0.1,
                  critical_stretch=0.05,
                  bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                  dimensions=3,
                  is_displacement_boundary=simple_displacement_boundary)
    return model, euler


def test_family(basic_model_3d):
    """Test family function."""
    model, integrator = basic_model_3d
    r = np.random.random((100, 3))
    horizon = 0.2

    (family_actual,
     *_) = model._set_neighbour_list(
                 r, horizon, 100)
    family_expected = np.sum(cdist(r, r) < horizon, axis=0) - 1

    assert np.all(family_actual == family_expected)


class TestNeigbourList():
    """Test neighbour list function."""

    def test_neighbour_list_cython(self, basic_model_3d):
        """Test cython version of the neighbour list function."""
        model, integrator = basic_model_3d
        r = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0]
            ])

        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(r, 1.1, 4)
        nl_expected = np.array([
            [1, 2],
            [0, 3],
            [0, 0],
            [1, 0]
            ])
        n_neigh_expected = np.array([2, 2, 1, 1])

        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)
        assert max_neighbours_actual == 2

    @context_available
    def test_neighbour_list_cl1(self, basic_model_3d_cl):
        """Test OpenCL version of the neighbourlist function."""
        model, integrator = basic_model_3d_cl
        r = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0]
            ])

        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(
             r, 1.1, 4, context=integrator.context)
        nl_expected = np.array([
            [1, 2],
            [0, 3],
            [0, -1],
            [1, -1]
            ])
        n_neigh_expected = np.array([2, 2, 1, 1])

        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)
        assert max_neighbours_actual == 2

    @context_available
    def test_neighbour_list_cl2(self, basic_model_3d_cl):
        """Test OpenCL version of the neighbourlist function."""
        model, integrator = basic_model_3d_cl
        r = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]
            ])

        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(
             r, 1.1, 4, context=integrator.context)
        nl_expected = np.array([
            [1, 2, 3, -1],
            [0, 3, -1, -1],
            [0, -1, -1, -1],
            [0, 1, -1, -1]
            ])
        n_neigh_expected = np.array([3, 2, 1, 2])

        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)
        assert max_neighbours_actual == 4

    def test_create_crack_cython(self, basic_model_3d):
        """Test crack creations function."""
        model, integrator = basic_model_3d
        r = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            ])
        horizon = 1.1
        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(r, horizon, 5)

        nl_expected = np.array([
            [1, 2, 4],
            [0, 3, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]
            ])
        n_neigh_expected = np.array([3, 2, 1, 1, 1])
        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)

        crack = np.array([(0, 2), (1, 3)], dtype=np.int32)
        create_crack(crack, nlist_actual, n_neigh_actual)

        nl_expected = np.array([
            [1, 4, -1],
            [0, -1, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [0, 0, 0]
            ])
        n_neigh_expected = np.array([2, 1, 0, 0, 1])
        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)

        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(
             r, horizon, 5, initial_crack=crack)

        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)

    @context_available
    def test_create_crack_cl(self, basic_model_3d_cl):
        """Test crack creations function."""
        model, integrator = basic_model_3d_cl
        r = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            ])
        horizon = 1.1
        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(
             r, horizon, 5, context=integrator.context)

        nl_expected = np.array([
            [1, 2, 4, -1],
            [0, 3, -1, -1],
            [0, -1, -1, -1],
            [1, -1, -1, -1],
            [0, -1, -1, -1]
            ])
        n_neigh_expected = np.array([3, 2, 1, 1, 1])
        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)

        crack = np.array([(0, 2), (1, 3)], dtype=np.int32)
        create_crack(crack, nlist_actual, n_neigh_actual)

        nl_expected = np.array([
            [1, 4, -1, -1],
            [0, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [0, -1, -1, -1]
            ])
        n_neigh_expected = np.array([2, 1, 0, 0, 1])
        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)

        (_,
         nlist_actual,
         n_neigh_actual,
         max_neighbours_actual) = model._set_neighbour_list(
             r, horizon, 5, initial_crack=crack, context=integrator.context)

        assert np.all(nlist_actual == nl_expected)
        assert np.all(n_neigh_actual == n_neigh_expected)
