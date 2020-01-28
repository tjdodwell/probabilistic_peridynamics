import numpy as np


class Grid:
    def __init__(self):
        self.dim = 2
        self.nel = 0
        self.numNodes = 0

    def findNeighbours(self):
        # Simple implementation for now as coarse mesh will be small
        M = np.zeros((self.numNodes, self.numNodes), dtype=int)

        for ie in range(0, self.nel):
            nodes = self.connectivity[ie][:]
            for i in range(0, 4):
                for j in range(0, 4):
                    if i != j:
                        M[nodes[i]][nodes[j]] = 1

        # Build List of Neighbours
        self.neighbours = [[] for i in range(self.numNodes)]
        for i in range(0, self.numNodes):
            for j in range(0, self.numNodes):
                if M[i][j] == 1:
                    self.neighbours[i].append(int(j))

        # Build list of macroscale elements in which a node lives
        tmpN2E = [[] for i in range(self.numNodes)]
        # For each elements
        for ie in range(0, self.nel):
            nodes = self.connectivity[ie][:]
            for j in range(0, nodes.size):
                tmpN2E[nodes[j]].append(ie)

        self.Node2Elements = [[] for i in range(self.numNodes)]

        for i in range(0, self.numNodes):
            tmpArray = np.array(tmpN2E[i])
            self.Node2Elements[i] = np.unique(tmpArray)

        return self.neighbours, self.Node2Elements

    def buildStructuredMesh2D(self, L, n, X0):
        self.n = n

        # Function builds a structured finite element mesh in 2D
        numNodesX = n[0] + 1
        numNodesY = n[1] + 1
        self.nodePerElement = 4
        self.numNodes = numNodesX * numNodesY
        self.nel = n[0] * n[1]

        x = np.linspace(X0[0], X0[0] + L[0], numNodesX)
        y = np.linspace(X0[1], X0[1] + L[1], numNodesY)

        self.h = np.zeros(self.dim)

        for i in range(0, self.dim):
            self.h[i] = L[i] / n[i]

        # Nodes will be formed from a tensor product of this two vectors
        self.coords = np.zeros((self.numNodes, 2))

        count = 0
        for i in range(0, n[1] + 1):
            for j in range(0, n[0] + 1):
                self.coords[count][0] = x[j]
                self.coords[count][1] = y[i]
                count += 1

        # Build Connectivity Matrix
        self.nel = n[0] * n[1]
        self.connectivity = np.zeros((self.nel, self.nodePerElement),
                                     dtype=int)
        count = 0
        ncount = 0
        for j in range(0, n[1]):
            for i in range(0, n[0]):
                self.connectivity[count][0] = ncount
                self.connectivity[count][1] = ncount + 1
                self.connectivity[count][2] = ncount + numNodesX + 1
                self.connectivity[count][3] = ncount + numNodesX
                count += 1
                ncount += 1
            ncount += 1

    def particletoCell_structured(self, pCoords):
        numParticles = int(pCoords[:].size / self.dim)
        p2e = np.zeros(numParticles, dtype=int)
        p_localCoords = np.zeros((numParticles, self.dim))
        # For each of the particles
        for i in range(0, numParticles):
            # particle coordinates
            xP = pCoords[i][:]
            id = np.zeros(self.dim)
            for j in range(0, self.dim):
                id[j] = np.floor(xP[j] / self.h[j])
                # Catch boundary case
                if id[j] == self.n[j]:
                    id[j] -= 1
            if self.dim == 2:
                p2e[i] = self.n[0] * id[1] + id[0]
            else:
                p2e[i] = (self.n[0] * self.n[1] * id[1]
                          + self.n[0] * id[1] + id[0])
            # Global to local mapping is easy as structured grid / domain
            node = self.connectivity[p2e[i]][:]
            for j in range(0, self.dim):
                p_localCoords[i][j] = (
                    (2 / self.h[j])
                    * (pCoords[i][j]
                       - (self.coords[node[0]][j] + 0.5 * self.h[j]))
                    )

        return p_localCoords, p2e

    def evalPhi(self, x, order=1):
        if self.dim == 2:
            if order == 1:
                phi = np.zeros(4)
                phi[0] = (1 - x[0]) * (1 - x[1])
                phi[1] = (1 + x[0]) * (1 - x[1])
                phi[2] = (1 + x[0]) * (1 + x[1])
                phi[3] = (1 - x[0]) * (1 + x[1])
                phi *= 0.25
        return phi
