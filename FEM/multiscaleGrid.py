import grid as fem

import numpy as np

class MultiscaleGrid:

    def __init__(self, comm_, dim_ = 2):

        self.comm = comm_

        self.rank = comm_.Get_rank()

        self.dim = dim_


    def buildCoarseGrid(self, type, L, n, X0, hf):

        # buildCoarseGrid constructs an indentical coarse grid on each processor

        # it then builds a local mesh for each subdomain - including overlap

        # Identical Coarse Grid is built on each processor

        self.macroGrid = fem.Grid()

        self.macroGrid.buildStructuredMesh2D(L,n,X0,1,0)

        # Type: element or node centric

        # Node centric - Overlapping, each domain support of node (assumming linear macro finite element)

        allNeighbours, allN2E = self.macroGrid.findNeighbours()

        self.NN = allNeighbours[self.rank]

        self.N2E = allN2E[self.rank]

        print("This is rank " + str(self.rank) + " - " + str(self.N2E))

        self.numNeighbours = len(self.NN)

        # Find boundary box for local grid

        bottomLeft = self.macroGrid.coords[self.rank][:]
        topRight = self.macroGrid.coords[self.rank][:]

        l1 = np.zeros(self.numNeighbours + 1)

        l1[0] = np.sum(np.abs(self.macroGrid.coords[self.rank][:]))
        for i in range(0, self.numNeighbours):
            l1[i+1] = np.sum(np.abs(self.macroGrid.coords[self.NN[i]][:]))
        tmp_id_min = np.argmin(l1)
        tmp_id_max = np.argmax(l1)

        if(tmp_id_min == 0):
            bottomLeft = self.macroGrid.coords[self.rank][:]
        else:
            bottomLeft = self.macroGrid.coords[self.NN[tmp_id_min-1]][:]

        if(tmp_id_max == 0):
            topRight = self.macroGrid.coords[self.rank][:]
        else:
            topRight = self.macroGrid.coords[self.NN[tmp_id_max-1]][:]


        # Build local finite element grid
        L_local = topRight - bottomLeft
        nf = [];
        for i in range(0, self.dim):
            nf.append(int(np.floor(L_local[i] / hf[i])))
        self.fineGrid = fem.Grid()
        self.fineGrid.buildStructuredMesh2D(L_local,nf,bottomLeft,1,0)

    # def buildPartitionOfUnity(self):
    #
    #     # In this case we will assume POU is linear shape functions on macroGrid
    #
    #     self.xi = np.zeros(self.fineGrid.numNodes) # Setup vector to store POU - xi
    #
    #     for i in range(0, self.fineGrid.numNodes):
