import numpy as np

import grid as fem

import multiscaleGrid as ms

from mpi4py import MPI



# test grid.py

dim = 2

numParticles = 10

L = [20 , 10] #

n = [2, 1]

hf = np.ones(dim)

order = 1

X0 = [0.0, 0.0] # bottom left

# myGrid = fem.Grid()
#
# myGrid.buildStructuredMesh2D(L,n,X0,order)
#
# particleCoords = L[0] * np.random.rand(numParticles, dim)
#
# print(particleCoords)
#
# localCoords, p2e = myGrid.particletoCell_structured(particleCoords)
#
# print(p2e)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_Size = comm.Get_size()

parGrid = ms.MultiscaleGrid(comm)

myGrid = parGrid.buildCoarseGrid("node_centered", L, n, X0, hf)
