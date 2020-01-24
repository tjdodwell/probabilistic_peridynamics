"""
A simple regression test simulating a basic model for nine steps using the
Euler integrator.
"""
from ..SeqPeriVectorized import SeqModel as MODEL
from ..fem import grid as fem
import numpy as np
import pathlib
import pytest


@pytest.fixture
def simple_square():
    mesh_file = (
        pathlib.Path(__file__).parent.absolute() / "data/example_mesh.msh")

    class simpleSquare(MODEL):
        def __init__(self):
            super().__init__()

            # Material Parameters from classical material model
            self.horizon = 0.1
            self.kscalar = 0.05
            self.s00 = 0.005

            self.crackLength = 0.3

            self.read_mesh(mesh_file)
            self.setVolume()

            self.lhs = []
            self.rhs = []

            # Find the Boundary
            for i in range(0, self.nnodes):
                bnd = self.findBoundary(self.coords[i][:])
                if bnd < 0:
                    (self.lhs).append(i)
                elif bnd > 0:
                    (self.rhs).append(i)

            # Build Finite Element Grid Overlaying particles
            myGrid = fem.Grid()

            self.L = []
            # bottom left
            self.X0 = [0.0, 0.0]
            self.nfem = []

            for i in range(0, self.dimensions):
                self.L.append(np.max(self.coords[:, i]))
                self.nfem.append(int(np.ceil(self.L[i] / self.horizon)))

            myGrid.buildStructuredMesh2D(self.L, self.nfem, self.X0, 1)

            self.p_localCoords, self.p2e = myGrid.particletoCell_structured(
                self.coords[:, :self.dimensions])

        def findBoundary(self, x):
            # Function which marks constrain particles
            # Does not live on a boundary
            bnd = 0
            if x[0] < 1.5 * self.horizon:
                bnd = -1
            elif x[0] > 1.0 - 1.5 * self.horizon:
                bnd = 1
            return bnd

        def isCrack(self, x, y):
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
                if (height > 0.5 * (1 - self.crackLength)
                        and height < 0.5 * (1 + self.crackLength)):
                    output = 1
            return output

    model = simpleSquare()
    return model


def test_regression(simple_square):
    model = simple_square

    model.setConn(0.1)
    model.setH()

    u = []
    u.append(np.zeros((model.nnodes, 3)))

    damage = []
    damage.append(np.zeros(model.nnodes))

    # Number of nodes
    nnodes = model.nnodes

    tim = 0.
    dt = 1e-3
    load_rate = 0.00001
    for t in range(9):
        tim += dt

        # Compute the force with displacement u[t-1]
        damage.append(np.zeros(nnodes))

        model.calcBondStretchNew(u[t-1])
        damage[t] = model.checkBonds()
        f = model.computebondForce()

        # Simple Euler update of the Solution
        u.append(np.zeros((nnodes, 3)))
        u[t] = u[t-1] + dt * f

        # Apply boundary conditions
        u[t][model.lhs, 1:3] = np.zeros((len(model.lhs), 2))
        u[t][model.rhs, 1:3] = np.zeros((len(model.rhs), 2))

        u[t][model.lhs, 0] = -0.5 * t * load_rate * np.ones(len(model.rhs))
        u[t][model.rhs, 0] = 0.5 * t * load_rate * np.ones(len(model.rhs))

    path = pathlib.Path(__file__).parent.absolute()
    expected_coords = np.load(path / "data/regression_expected_output.npy")
    assert np.all(model.coords == expected_coords)
