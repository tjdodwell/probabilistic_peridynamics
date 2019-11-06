import grid as fem

import numpy as np

class MultiscaleGrid:

    def __init__(self, comm_, dim_ = 2):

        self.comm = comm_

        self.rank = comm_.Get_rank()

        self.dim = dim_


    def buildCoarseGrid(self, type, L, n, X0, hf):

        # Identical Coarse Grid is built on each processor

        self.macroGrid = fem.Grid()

        self.macroGrid.buildStructuredMesh2D(L,n,X0,1,0)

        # Type: element or node centric

        # Node centric - Overlapping, each domain support of node (assumming linear macro finite element)

        allNeighbours = self.macroGrid.findNeighbours()

        self.NN = allNeighbours[self.rank]

        self.numNeighbours = len(self.NN)

        # Find boundary box for local grid

        bottomLeft = self.macroGrid.coords[self.rank][:]
        topRight = self.macroGrid.coords[self.rank][:]

        bL = np.abs(np.sum(bottomLeft))
        tR = np.abs(np.sum(topRight))

        flag = -1

        for i in range(0, self.numNeighbours):
            current = self.macroGrid.coords[self.NN[i]][:]
            if(self.rank == flag):
                print("This is rank " + str(self.rank) + " and Neighbour " + str(self.NN[i]))
                print("This has coords " + str(current))
            l1 = np.abs(np.sum(current))
            if (l1 < bL):
                bottomLeft = current
            if (l1 > tR):
                topRight = current
            if(self.rank == flag):
                print("bottomLeft " + str(bottomLeft))
                print("topRight " + str(topRight))


        #print("This is rank " + str(self.rank) + " bottom left is " + str(bottomLeft))

        # Build local finite element grid

        L_local = topRight - bottomLeft

        print("L_local " + str(L_local))

        nf = [];

        for i in range(0, self.dim):
            nf.append(int(np.floor(L_local[i] / hf[i])))


        self.fineGrid = fem.Grid()

        self.fineGrid.buildStructuredMesh2D(L_local,nf,bottomLeft,1,0)
