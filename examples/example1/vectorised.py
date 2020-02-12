"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import Euler
from pstats import SortKey, Stats

mesh_file = pathlib.Path(__file__).parent.absolute() / "test.msh"


@initial_crack_helper
def is_crack(x, y):
    output = 0
    crack_length = 0.3
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
        if (height > 0.5 * (1 - crack_length)
                and height < 0.5 * (1 + crack_length)):
            output = 1
    return output


def sim(model, steps=400, load_rate=0.00001, dt=1e-3, print_every=10):
    print("Peridynamic Simulation -- Starting")

    integrator = Euler(dt=1e-3)

    u = np.zeros((model.nnodes, 3))

    tim = 0.0
    for t in range(1, steps+1):
        tim += dt

        model.bond_stretch(u)
        damage = model.damage()
        f = model.bond_force()

        u = integrator.step(u, f)

        # Apply boundary conditions
        u[model.lhs, 1:3] = np.zeros((len(model.lhs), 2))
        u[model.rhs, 1:3] = np.zeros((len(model.rhs), 2))

        u[model.lhs, 0] = -0.5 * t * load_rate * np.ones(len(model.rhs))
        u[model.rhs, 0] = 0.5 * t * load_rate * np.ones(len(model.rhs))

        if t % print_every == 0:
            model.write_mesh("U_"+"t"+str(t)+".vtk", damage, u)


def main():
    """
    Stochastic peridynamics, takes multiple stable states (fully formed cracks)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
                  elastic_modulus=0.05, initial_crack=is_crack)

    # Set left-hand side and right-hand side of boundary
    indices = np.arange(model.nnodes)
    model.lhs = indices[model.coords[:, 0] < 1.5*model.horizon]
    model.rhs = indices[model.coords[:, 0] > 1.0 - 1.5*model.horizon]

    sim(model, steps=10)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()
