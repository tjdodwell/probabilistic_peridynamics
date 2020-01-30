"""
A simple regression test simulating a basic model for nine steps using the
Euler integrator.
"""
from ..model import Model
import numpy as np
import pytest


@pytest.fixture(scope="module")
def simple_square(data_path):
    path = data_path
    mesh_file = path / "example_mesh.msh"

    class SimpleSquare(Model):
        def __init__(self):
            super().__init__()

            # Material Parameters from classical material model
            self.horizon = 0.1
            self.kscalar = 0.05
            self.s00 = 0.005

            self.crack_length = 0.3

            self.read_mesh(mesh_file)
            self.set_volume()

            self.lhs = []
            self.rhs = []

            # Find the Boundary
            for i in range(0, self.nnodes):
                bnd = self.find_boundary(self.coords[i][:])
                if bnd < 0:
                    (self.lhs).append(i)
                elif bnd > 0:
                    (self.rhs).append(i)

            self.L = []
            # bottom left
            self.X0 = [0.0, 0.0]
            self.nfem = []

            for i in range(0, self.dimensions):
                self.L.append(np.max(self.coords[:, i]))
                self.nfem.append(int(np.ceil(self.L[i] / self.horizon)))

        def find_boundary(self, x):
            # Function which marks constrain particles
            # Does not live on a boundary
            bnd = 0
            if x[0] < 1.5 * self.horizon:
                bnd = -1
            elif x[0] > 1.0 - 1.5 * self.horizon:
                bnd = 1
            return bnd

        def is_crack(self, x, y):
            output = 0
            p1 = x
            p2 = y
            if x[0] > y[0]:
                p2 = x
                p1 = y
            # 1e-6 makes it fall one side of central line of particles
            if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
                # draw a straight line between them
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                c = p1[1] - m * p1[0]
                # height a x = 0.5
                height = m * 0.5 + c
                if (height > 0.5 * (1 - self.crack_length)
                        and height < 0.5 * (1 + self.crack_length)):
                    output = 1
            return output

    model = SimpleSquare()
    return model


@pytest.fixture(scope="module")
def regression(simple_square):
    model = simple_square

    model.set_connectivity(0.1)
    model.set_H()

    u = []
    u.append(np.zeros((model.nnodes, 3)))

    damage = []
    damage.append(np.zeros(model.nnodes))

    # Number of nodes
    nnodes = model.nnodes

    tim = 0.
    dt = 1e-3
    load_rate = 0.00001
    for t in range(1, 11):
        tim += dt

        # Compute the force with displacement u[t-1]
        damage.append(np.zeros(nnodes))

        model.bond_stretch(u[t-1])
        damage[t] = model.damage()
        f = model.bond_force()

        # Simple Euler update of the Solution
        u.append(np.zeros((nnodes, 3)))
        u[t] = u[t-1] + dt * f

        # Apply boundary conditions
        u[t][model.lhs, 1:3] = np.zeros((len(model.lhs), 2))
        u[t][model.rhs, 1:3] = np.zeros((len(model.rhs), 2))

        u[t][model.lhs, 0] = -0.5 * t * load_rate * np.ones(len(model.rhs))
        u[t][model.rhs, 0] = 0.5 * t * load_rate * np.ones(len(model.rhs))

    return model, u[t], damage[t]


class TestRegression:
    def test_displacements(self, regression, data_path):
        _, displacements, *_ = regression
        path = data_path

        expected_displacements = np.load(
            path/"expected_displacements.npy"
            )
        assert np.all(displacements == expected_displacements)

    def test_damage(self, regression, data_path):
        _, _, damage = regression
        path = data_path

        expected_damage = np.load(
            path/"expected_damage.npy"
            )
        assert np.all(np.array(damage) == expected_damage)

    def test_mesh(self, regression, data_path, tmp_path):
        model, displacements, damage = regression
        path = data_path

        mesh = tmp_path / "mesh.vtk"
        model.write_mesh(mesh, damage, displacements)

        expected_mesh = path / "expected_mesh.vtk"

        assert mesh.read_bytes() == expected_mesh.read_bytes()
