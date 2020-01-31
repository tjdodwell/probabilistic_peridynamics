"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics.grid import Grid
from peridynamics.SeqPeriVectorized import SeqModel as MODEL
from pstats import SortKey, Stats


class simpleSquare(MODEL):
    # A User defined class for a particular problem which defines all necessary
    # parameters

    def __init__(self):
        super().__init__()
        self.dim = 2

        # Material Parameters from classical material model
        self.horizon = 0.1
        self.kscalar = 0.05
        self.s00 = 0.005

        self.crackLength = 0.3

        mesh_file = pathlib.Path(__file__).parent.absolute() / "test.msh"
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
        myGrid = Grid()

        self.L = []
        # bottom left
        self.X0 = [0.0, 0.0]
        self.nfem = []

        for i in range(0, self.dim):
            self.L.append(np.max(self.coords[:, i]))
            self.nfem.append(int(np.ceil(self.L[i] / self.horizon)))

        myGrid.buildStructuredMesh2D(self.L, self.nfem, self.X0)

        self.p_localCoords, self.p2e = myGrid.particletoCell_structured(
            self.coords[:, :self.dim])

    def findBoundary(self, x):
        # Function which markes constrain particles
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


def multivar_normal(L, num_nodes):
    """
    Fn for taking a single multivar normal sample covariance matrix with
    Cholesky factor, L
    """
    zeta = np.random.normal(0, 1, size=num_nodes)
    zeta = np.transpose(zeta)

    # vector
    w_tild = np.dot(L, zeta)

    return w_tild


def noise(L, samples, num_nodes):
    """
    takes multiple samples from multivariate normal distribution with
    covariance matrix whith Cholesky factor, L
    """
    noise = []
    for i in range(samples):
        noise.append(multivar_normal(L, num_nodes))

    return np.transpose(noise)


def sim(myModel=simpleSquare(), numSteps=400,
        loadRate=0.00001, dt=1e-3, print_every=10):
    print("Peridynamic Simulation -- Starting")

    myModel.setConn(0.1)
    myModel.setH()

    u = []

    damage = []

    u.append(np.zeros((myModel.nnodes, 3)))

    damage.append(np.zeros(myModel.nnodes))

    tim = 0.0

    # Number of nodes
    nnodes = myModel.nnodes

    for t in range(1, numSteps+1):
        tim += dt

        # Compute the force with displacement u[t-1]
        damage.append(np.zeros(nnodes))

        myModel.calcBondStretchNew(u[t-1])
        damage[t] = myModel.checkBonds()
        f = myModel.computebondForce()

        # Simple Euler update of the Solution + Add the Stochastic Random Noise
        u.append(np.zeros((nnodes, 3)))

        u[t] = u[t-1] + dt * f

        # Apply boundary conditions
        u[t][myModel.lhs, 1:3] = np.zeros((len(myModel.lhs), 2))
        u[t][myModel.rhs, 1:3] = np.zeros((len(myModel.rhs), 2))

        u[t][myModel.lhs, 0] = -0.5 * t * loadRate * np.ones(len(myModel.rhs))
        u[t][myModel.rhs, 0] = 0.5 * t * loadRate * np.ones(len(myModel.rhs))

        if t % print_every == 0:
            myModel.write_mesh("U_"+"t"+str(t)+".vtk", damage[t], u[t])


def main():
    """
    Stochastic Peridynamics, takes multiple stable states (fully formed cracks)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    model = simpleSquare()
    sim(model, numSteps=10)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()