import numpy as np
import grid as fem

dim = 2
numParticles = 200
L = [1, 1]
n = [10, 10]
hf = np.ones(dim)
order = 1

# bottom left
X0 = [0.0, 0.0]

myGrid = fem.Grid()
myGrid.buildStructuredMesh2D(L, n, X0, order)

particleCoords = np.random.rand(numParticles, dim)
for i in range(0, numParticles):
    particleCoords[i][0] *= L[0]
    particleCoords[i][1] *= L[1]

print(particleCoords)

localCoords, p2e = myGrid.particletoCell_structured(particleCoords)

print(p2e)
print(localCoords)
