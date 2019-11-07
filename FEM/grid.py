import numpy as np

class Grid:

    def __init__(self):

        self.dim = 2
        self.nel = 0
        self.numNodes = 0

    def findNeighbours(self):
        # Simple implementation for now as coarse mesh will be small
        M = np.zeros((self.numNodes, self.numNodes), dtype = int)

        for ie in range(0, self.nel):
            nodes = self.connectivity[ie][:]
            for i in range(0, 4):
                for j in range(0,4):
                    if(i != j):
                        M[nodes[i]][nodes[j]] = 1
        # Build List of Neighbours
        self.neighbours = [ [] for i in range(self.numNodes) ]
        for i in range(0, self.numNodes):
            for j in range(0, self.numNodes):
                if (M[i][j] == 1):
                    self.neighbours[i].append(int(j))

        # Build list of macroscale elements in which a node lives
        tmpN2E = [ [] for i in range(self.numNodes) ]
        for ie in range(0, self.nel): # For each elements
            nodes = self.connectivity[ie][:]
            for j in range(0,nodes.size):
                tmpN2E[nodes[j]].append(ie)

        self.Node2Elements = [ [] for i in range(self.numNodes) ]

        for i in range(0, self.numNodes):
            tmpArray = np.array(tmpN2E[i])
            self.Node2Elements[i] = np.unique(tmpArray)


        return self.neighbours, self.Node2Elements

    def buildStructuredMesh2D(self,L,n,X0,order,verb = 2):

        if(verb > 0):
            print('Building Structured 2D Grid!')

        self.n = n

        # Function builds a structured finite element mesh in 2D
        if(order == 1):
            numNodesX = n[0] + 1
            numNodesY = n[1] + 1
            self.nodePerElement = 4
            self.numNodes = numNodesX * numNodesY
            self.nel = n[0] * n[1]

        else:
            print('Grids of order 2 or high are not currently supported, assuming order 1')
            order == 1;

        x = np.linspace(X0[0], X0[0] + L[0], numNodesX)
        y = np.linspace(X0[1], X0[1] + L[1], numNodesY)

        self.h = np.zeros(self.dim)

        for i in range(0,self.dim):
            self.h[i] = L[i] / n[i]

        # Nodes will be formed from a tensor product of this two vectors
        self.coords = np.zeros((self.numNodes, 2))


        count = 0
        for i in range(0, n[1] + 1):
            for j in range(0, n[0] + 1):
                self.coords[count][0] = x[j]
                self.coords[count][1] = y[i]
                count += 1 # increment node counter

        # Build Connectivity Matrix
        self.nel = n[0] * n[1]
        self.connectivity = np.zeros((self.nel, self.nodePerElement), dtype = int)
        count = 0
        ncount = 0
        for j in range(0, n[1]):
            for i in range(0, n[0]):
                self.connectivity[count][0] = ncount
                self.connectivity[count][1] = ncount + 1
                self.connectivity[count][2] = ncount + numNodesX + 1
                self.connectivity[count][3] = ncount + numNodesX
                count += 1 # increment element counter
                ncount += 1

            ncount+=1

        if(verb > 1):
            print('... Grid Built!')
            print('Number of Nodes ' + str(self.numNodes))
            print('Number of Elements ' + str(self.nel))
            for i in range(0, self.numNodes):
                print(str(self.coords[i][:]))
            for i in range(0, self.nel):
                print(str(self.connectivity[i][:]))

    def particletoCell_structured(self,pCoords):
        numParticles = int(pCoords[:].size / self.dim)
        particle2Cell_Maps = np.zeros(numParticles, dtype = int)
        localCoords = np.zeros((numParticles, self.dim))

        for i in range(0, numParticles):
            xP = pCoords[i][:]
            id = np.zeros(self.dim)
            for j in range(0,self.dim):
                id[j] = np.floor(xP[j] / self.h[j])
            if(self.dim == 2):
                p2e[i] = self.n[0] * id[1] + id[0]
            else:
                p2e[i] = self.n[0] * self.n[1] * id[1] + self.n[0] * id[1] + id[0]
            # Global to local mapping is easy as structured grid / domain
            node = self.connectivity[p2e[i]][:]
            for j in range(0,self.dim):
                p_localCoords[i][j] = (2 / self.h[j]) * (pCoords[i][j] - (self.coords[node[0]][j] + 0.5 * self.h[j]))


        return p_localCoords, p2e
