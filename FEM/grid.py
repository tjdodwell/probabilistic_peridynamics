import numpy as np

class Grid:

    def __init__(self):

        self.dim = 2


    def buildStructuredMesh2D(self,L,n,X0,order):

        print('Building Structured 2D Grid!')

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
        for i in range(0, n[0] + 1):
            for j in range(0, n[1] + 1):
                self.coords[count][0] = x[j]
                self.coords[count][1] = y[i]
                count += 1 # increment node counter

        # Build Connectivity Matrix
        self.nel = n[0] * n[1]
        self.connectivity = np.zeros((self.nel, self.nodePerElement))
        count = 0
        ncount = 0
        for i in range(0, n[0]):
            for j in range(0, n[1]):
                self.connectivity[count][0] = ncount
                self.connectivity[count][1] = ncount + 1
                self.connectivity[count][2] = ncount + numNodesX + 1
                self.connectivity[count][3] = ncount + numNodesX
                count += 1 # increment element counter
                ncount += 1
            ncount += 1
        print('... Grid Build')
        print('Number of Nodes ' + str(self.numNodes))
        print('Number of Elements ' + str(self.nel))
        for i in range(0, self.numNodes):
            print(str(self.coords[i][:]))
        for i in range(0, self.nel):
            print(str(self.connectivity[i][:]))
