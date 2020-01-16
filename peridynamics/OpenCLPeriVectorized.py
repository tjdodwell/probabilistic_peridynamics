import numpy as np
import periFunctions as func


class SeqModel:
    def __init__(self):
        # self.nnodes defined when instance of readMesh called, cannot
        # initialise any other matrix until we know the nnodes
        self.v = True
        self.dim = 2

        self.meshFileName = "test.msh"

        self.meshType = 2
        self.boundaryType = 1
        self.numBoundaryNodes = 2
        self.numMeshNodes = 3

        if self.dim == 3:
            self.meshType = 4
            self.boundaryType = 2
            self.numBoundaryNodes = 3
            self.numMeshNodes = 4

    def readMesh(self, fileName):
        f = open(fileName, "r")

        if f.mode == "r":
            iline = 0

            # Read the Nodes in the Mesh First
            findNodes = 0
            while (findNodes == 0):
                iline += 1
                line = f.readline()
                if line.strip() == '$Nodes':
                    findNodes = 1

            line = f.readline()
            self.nnodes = int(line.strip())
            self.coords = np.zeros((self.nnodes, 3), dtype=np.float64)

            for i in range(0, self.nnodes):
                iline += 1
                line = f.readline()
                rowAsList = line.split()
                self.coords[i][0] = rowAsList[1]
                self.coords[i][1] = rowAsList[2]
                self.coords[i][2] = rowAsList[3]

            # This line will read $EndNodes - Could add assert on this
            line = f.readline()
            # This line will read $Elements
            line = f.readline()

            # Read the Elements from the mesh for the volume calculations
            # connectivity
            # This gives the total number of elements - but includes all types
            # of elements
            line = f.readline()
            self.totalNel = int(line.strip())
            self.connectivity = []
            self.connectivity_bnd = []

            for ie in range(0, self.totalNel):
                iline += 1
                line = f.readline()
                rowAsList = line.split()

                if int(rowAsList[1]) == self.boundaryType:
                    tmp = np.zeros(self.dim)
                    for k in range(0, self.dim):
                        tmp[k] = int(rowAsList[5 + k]) - 1
                    self.connectivity_bnd.append(tmp)

                elif int(rowAsList[1]) == self.meshType:
                    tmp = np.zeros(self.dim + 1)
                    for k in range(0, self.dim + 1):
                        tmp[k] = int(rowAsList[5 + k]) - 1
                    self.connectivity.append(tmp)

            self.nelem = len(self.connectivity)
            self.nelem_bnd = len(self.connectivity_bnd)

        f.close()

    def setVolume(self):
        V = np.zeros(self.nnodes, dtype=np.float64)

        for ie in range(0, self.nelem):
            n = self.connectivity[ie]
            # Compute Area / Volume
            val = 1. / n.size

            # Define area of element
            if self.dim == 2:
                xi = self.coords[int(n[0])][0]
                yi = self.coords[int(n[0])][1]
                xj = self.coords[int(n[1])][0]
                yj = self.coords[int(n[1])][1]
                xk = self.coords[int(n[2])][0]
                yk = self.coords[int(n[2])][1]

                val *= 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))

            for j in range(0, n.size):
                V[int(n[j])] += val
        self.V = V.astype(np.float64)

    def setNetwork(self, horizon):
        """
        Sets the family matrix, and converts to horizons matrix. Calculates
        horizons_lengths
        """

        # Container for nodal family
        self.family = []

        # Container for number of nodes (including self) that each of the nodes
        # is connected to
        self.horizons_lengths = np.zeros(self.nnodes, dtype=int)

        for i in range(0, self.nnodes):
            tmp = []

            for j in range(0, self.nnodes):
                if i != j:
                    l2_sqr = func.l2_sqr(self.coords[i, :], self.coords[j, :])
                    if np.sqrt(l2_sqr) < horizon:
                        tmp.append(j)
            self.family.append(np.zeros(len(tmp), dtype=np.intc))
            self.horizons_lengths[i] = np.intc((len(tmp)))
            for j in range(0, len(tmp)):
                self.family[i][j] = np.intc((tmp[j]))

        # Maximum number of nodes that any one of the nodes is connected to
        self.MAX_HORIZON_LENGTH = np.intc(
            len(max(self.family, key=lambda x: len(x)))
            )

        horizons = -1 * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.family):
            horizons[i][0:len(j)] = j

        # Make sure it is in a datatype that C can handle
        self.horizons = horizons.astype(np.intc)

        # Initiate crack
        for i in range(0, self.nnodes):

            for k in range(0, self.MAX_HORIZON_LENGTH):
                j = self.horizons[i][k]
                if self.isCrack(self.coords[i, :], self.coords[j, :]):
                    self.horizons[i][k] = np.intc(-1)
