import numpy as np

import grid as fem

import multiscaleGrid as ms

from mpi4py import MPI



# test grid.py

dim = 2

numParticles = 200

L = [1 , 1] #

n = [10, 10]

hf = np.ones(dim)

order = 1

X0 = [0.0, 0.0] # bottom left

myGrid = fem.Grid()

myGrid.buildStructuredMesh2D(L,n,X0,order)

particleCoords = np.random.rand(numParticles, dim)

for i in range(0, numParticles):
    particleCoords[i][0] *= L[0]
    particleCoords[i][1] *= L[1]

print(particleCoords)

localCoords, p2e = myGrid.particletoCell_structured(particleCoords)

print(p2e)

#print(localCoords)

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# comm_Size = comm.Get_size()
#
# parGrid = ms.MultiscaleGrid(comm)
#
# parGrid.buildCoarseGrid("node_centered", L, n, X0, hf)
#
# parGrid.buildPartitionOfUnity()
