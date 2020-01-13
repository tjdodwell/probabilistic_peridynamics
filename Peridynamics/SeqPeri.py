import numpy as np
import PeriParticle as peri
import periFunctions as func


class SeqModel:
    def __init__(self):
        self.dim = 2

        self.meshFileName = "test.msh"

        self.meshType = 2
        self.boundaryType = 1
        self.numBoundaryNodes = 2
        self.numMeshNodes = 3

        # Material Parameters from classical material model
        self.horizon = 0.1
        self.K = 0.05
        self.s00 = 0.05

        self.c = 18.0 * self.K / (np.pi * (self.horizon**4))

        if self.dim == 3:
            self.meshType = 4
            self.boundaryType = 2
            self.numBoundaryNodes = 3
            self.numMeshNodes = 4

    def findBoundary(self, x):
        # Default function return no boundary overwritten by user
        return 0

    def isCrack(self, x, y):
        # Default funciton return no pre-defined crack overwritten by user
        return 0

    def checkBonds(self, U, broken, damage):
        for i in range(0, self.nnodes):
            # deformed coordinates of particle i
            yi = self.coords[i, :] + U[i, :]
            count = 0

            # extract family for particle i
            family = self.family[i]
            for k in range(0, len(family)):
                j = self.family[i][k]

                # if bond is not broken
                if broken[i][k] == 0:
                    # deformed coordinates of particle j
                    yj = self.coords[j, :] + U[j, :]
                    bondLength = func.l2norm(
                        self.coords[i, :] - self.coords[j, :]
                        )
                    rvec = yj - yi
                    r = func.l2norm(rvec)
                    strain = (r - bondLength) / bondLength

                    if strain > self.s00:
                        broken[i][k] = 1
                        count += 1.0
                else:
                    count += 1.0
            damage[i] = float(count / len(family))
        return broken, damage

    def computebondForce(self, U, broken):
        self.c = 18.0 * self.K / (np.pi * (self.horizon**4))

        # Container for the forces on each particel in each dimension
        F = np.zeros((self.nnodes, 3))

        # For each particles
        for i in range(0, self.nnodes):
            yi = self.coords[i, :] + U[i, :]

        # This is a list of id's for family members
        family = self.family[i]
        for k in range(0, len(family)):
            j = self.family[i][k]

            # Bond between particle i and j has not been broken
            if broken[i][k] == 0:
                yj = self.coords[j, :] + U[j, :]
                bondLength = func.l2norm(self.coords[i, :] - self.coords[j, :])
                rvec = yj - yi
                r = func.l2norm(rvec)
                dr = r - bondLength
                # Kills an issue with rounding error
                if dr < 1e-12:
                    dr = 0.0
                kb = (self.c / bondLength) * self.V[i] * dr
                F[i, :] += (kb / r) * rvec
        return F

    def initialiseCrack(self, broken, damage):
        for i in range(0, self.nnodes):
            family = self.family[i]
            count = 0

            for k in range(0, len(family)):
                j = self.family[i][k]
                if self.isCrack(self.coords[i, :], self.coords[j, :]):
                    broken[i][k] = 1
                    count += 1
            damage[i] = float(count / len(family))
        return broken, damage

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
            self.coords = np.zeros((self.nnodes, 3))

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

            # Read the Elements
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

    def setNetwork(self, horizon):
        # List to store the network
        self.net = []

        # For each of the particles
        for i in range(0, self.nnodes):
            self.net.append(peri.Particle())
            self.net[i].setId(i)
            self.net[i].setCoord(self.coords[i])

        self.V = np.zeros(self.nnodes)

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
                self.V[int(n[j])] += val

        # Setup the family for each particle and define the covariance matrix
        # There is more efficient ways to do this, but doesn't really matter
        # since one of calculation

        # Initiate Covariance matrix, C and length scale, lambda
        lambd = 900
        self.family = []

        X = np.sum(pow(self.coords, 2), axis=1)

        tiled_X = np.tile(X, (self.nnodes, 1))
        tiled_Xt = np.transpose(tiled_X)

        outer_product = np.dot(self.coords, np.transpose(self.coords))

        sqr_l2norm = (tiled_X - np.multiply(outer_product, 2) + tiled_Xt)

        # Construct covariance matrix
        kernel_values = np.multiply(-lambd, sqr_l2norm)
        self.COVARIANCE = np.exp(kernel_values)

# =============================================================================
# # TODO: Vectorize family matrix construction
# l2norm = np.sqrt(sqr_l2norm)
# A = np.add(l2norm, -1.*horizon)
# self.family = (A<0)
# print(self.family)
# =============================================================================

        for i in range(0, self.nnodes):
            tmp = []

            for j in range(0, self.nnodes):
                if i != j:
                    l2_sqr = func.l2_sqr(self.coords[i, :], self.coords[j, :])
                    if np.sqrt(l2_sqr) < horizon:
                        tmp.append(j)
            self.family.append(np.zeros(len(tmp), dtype=int))
            for j in range(0, len(tmp)):
                self.family[i][j] = int(tmp[j])
