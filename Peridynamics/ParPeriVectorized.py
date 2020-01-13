import numpy as np
import PeriParticle as peri
import periFunctions as func
import parallelFunctions as par
from PostProcessing import vtu
from mpi4py import MPI
from scipy import sparse
import warnings
import time


class ParModel:
    def __init__(self):
        self.testCode = 1
        self.plotPartiton = 1

        self.v = True
        self.dim = 2

        self.meshFileName = "test.msh"

        self.meshType = 2
        self.boundaryType = 1
        self.numBoundaryNodes = 2
        self.numMeshNodes = 3

        # Material Parameters from classical material model
        self.horizon = 0.1
        # renamed from self.K in earlier versions, as K is now the covariance
        # matrix
        self.kscalar = 0.05
        self.s00 = 0.05

        self.c = 18.0 * self.kscalar / (np.pi * self.horizon**4)

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
            # Does the domain decomposition
            self.partition, self.NN = par.decomposeDomain(
                self.coords, self.connectivity, self.comm.Get_size(),
                self.partitionType)
        f.close()

    def setNetwork(self, horizon):
        myRank = self.comm.Get_rank()

        # List to store the network
        self.net = []
        # wasnt previously a class variable
        self.neighbour_ids = []
        # local to global index list
        self.l2g = []
        # global to local index list
        self.g2l = []
        localCount = 0
        localplusghostCount = 0

        # for each of the particles
        for i in range(0, self.nnodes):
            self.g2l.append(-1)
            # particle belongs to this processor
            if myRank == int(self.partition[i]):
                self.net.append(peri.Particle())
                self.net[localCount].setId(i)
                self.net[localCount].setCoord(self.coords[i])
                localCount += 1
                localplusghostCount += 1
                self.l2g.append(i)
                self.g2l[i] = localCount
            # This is the case where particle lives on another process but is
            # in a neighouring subdomain, so may be in family
            else:
                check = 0
                for k in range(0, len(self.NN[myRank])):
                    # If one of the neighbours of the particle is our rank
                    # process
                    if self.NN[myRank][k] == int(self.partition[i]):
                        check += 1
                # Number greater than zero indicates that particle is one of
                # the neighbouring subdomains
                if check > 0:
                    self.neighbour_ids.append(i)

        # Store the number of particles directly in subdomains
        self.numlocalNodes = localCount

        if self.testCode:
            # Check that no nodes have to be lost or added by bug
            mylocal = np.zeros(1)
            totalNodes = np.zeros(1)
            mylocal[0] = localCount
            self.comm.Reduce(mylocal, totalNodes, MPI.SUM, 0)
            if myRank == 0:
                assert int(totalNodes[0]) == self.nnodes
            self.comm.Barrier()

        Vols = np.zeros(self.nnodes)

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
                Vols[int(n[j])] += val

        self.V = np.zeros(self.nnodes)
        for i in range(0, self.numlocalNodes):
            self.V[i] = Vols[self.net[i].id]

        # Setup the family for each particle

        # For each local cell, loop over each set of local particles
        # There is more efficient ways to do this, but doesn't really matter
        # since one off calculation

        # Initiate the connectivity matrix as non sparse
        conn = np.zeros((self.nnodes, self.nnodes))

        # Initiate uncracked connectivity matrix
        conn_0 = np.zeros((self.nnodes, self.nnodes))

        # Setup the family for each particle
        # in place of family matrix, have now got connectivity matrix
        self.family = []
        tmpGhost = []

        # For each local cell loop over each set of local particles
        # Check if nodes are connected, but only if it is a local node or ghost
        # particle
        # This may take a while
        for i in range(0, self.nnodes):
            for j in range(0, self.nnodes):
                bool1 = bool2 = -1
                # if at least one of these nodes is in the local partition
                if i in self.l2g:
                    bool1 = True
                if j in self.l2g:
                    bool2 = True

                bool3 = bool1 or bool2
                if bool3:

                    if func.l2(self.coords[i, :], self.coords[j, :]) < horizon:
                        conn_0[i, j] = 1
                        if i == j:
                            # do not fill diagonal
                            pass
                        elif (self.isCrack(self.coords[i, :],
                                           self.coords[j, :]) is False):
                            conn[i, j] = 1

        # Initial bond damages
        count = np.sum(conn, axis=0)
        self.family = np.sum(conn_0, axis=0)
        damage = np.divide((self.family - count), self.family)
        damage.resize(self.nnodes)

        # return damage for our local nodes only
        damage_local = np.zeros(self.numlocalNodes)

        for i in range(self.numlocalNodes):
            globalId_i = self.net[i].id
            # could be an issue here
            count = np.sum(conn[globalId_i])
            damage = np.divide(self.family[globalId_i] - count,
                               self.family[globalId_i])
            damage_local[i] = damage

        if self.v:
            print('time, t = 0')
            print(np.max(damage_local), 'max_damage')
            print(np.min(damage_local), 'min_damage')
        # Lower triangular - count bonds only once
        # make diagonal values 0
        # conn = np.tril(conn, -1) #TODO

        # Convert to sparse matrix
        self.conn = sparse.csr_matrix(conn)
        self.conn_0 = sparse.csr_matrix(conn_0)

        # need to eliminate zeros?
        self.conn.eliminate_zeros()

        if self.v:
            print('self.conn is', self.conn)

        # Find a better way to do this with the sparse matrices?
        # which is a list of ghost particles global ids
        self.ghostList = np.unique(tmpGhost)

        self.ghostListProcessors = []

        for i in range(0, len(self.ghostList)):
            self.ghostListProcessors.append(
                int(self.partition[self.ghostList[i]])
                )

        # Integer value which stores number of ghost requests for displacements
        # at each time step
        self.numGhostRequests_to_send = 0
        # Integer value which stores numbe of ghost requests that will be
        # received
        self.numGhostRequests_to_recv = 0
        # Will be a list of length self.numGhostRequests containing the
        # processor number for each communicator
        self.GhostRequestProcessorIds_send = []
        # Will be a list of length self.numGhostRequests containing the
        # processor number for each communicator
        self.GhostRequestProcessorIds_recv = []
        # List of list contain the global ids which will be sent.
        self.IdListGhostRequests_send = []
        # List of list contain the global ids which will be recv.
        self.IdListGhostRequests_recv = []

        if self.comm.Get_size() == 1:
            # In the case where we have one process, then we have sequential
            # simulation
            pass
        else:
            # Loop over each processor
            for i in range(0, self.comm.Get_size()):
                # Container for vector to mark which partitions are nearest
                # neighbours for partition i
                areNN = np.zeros(self.comm.Get_size(), dtype=int)

                # Each partition has a corresponding process
                if myRank == i:
                    for k in range(0, len(self.NN[i])):
                        # Mark as a nearest neighbour for processor i
                        areNN[self.NN[i][k]] = 1

                # Communicate to all other processors
                self.comm.Bcast(areNN, root=i)
                # wait here until all processes have reached this point, i.e.
                # that the full areNN is constructed.
                self.comm.Barrier()

                for k in range(0, len(self.NN[i])):
                    proc_id = int(self.NN[i][k])
                    # If this is the control processor

                    if myRank == i:
                        # Create a list of ghost which live on a given
                        # neighbour
                        tmp = []

                        # Loop over all ghost particles
                        for j in range(0, self.ghostList.size):
                            proc = self.ghostListProcessors[j]
                            # if particle is in processor self.NN[k]
                            if proc == proc_id:
                                tmp.append(self.ghostList[j])

                        # tmp contains list of particles in ghost of i -
                        # required from processor self.NN[k]
                        self.numGhostRequests_to_recv += 1
                        self.GhostRequestProcessorIds_recv.append(proc_id)

                        # messages with different tags will be buffered by the
                        # network until this processor is ready for them
                        self.comm.send(int(len(tmp)), dest=proc_id, tag=1)
                        tmpArray = np.zeros(len(tmp), dtype=int)

                        for ii in range(0, tmpArray.size):
                            tmpArray[ii] = int(tmp[ii])

                        self.IdListGhostRequests_recv.append(tmpArray)
                        self.comm.Send(tmpArray, dest=proc_id, tag=2)
                    elif myRank == proc_id:
                        self.numGhostRequests_to_send += 1
                        self.GhostRequestProcessorIds_send.append(i)
                        numParticles_tmp = self.comm.recv(source=i, tag=1)
                        tmpNumpy = np.empty(numParticles_tmp, dtype=int)
                        self.comm.Recv(tmpNumpy, source=i, tag=2)
                        self.IdListGhostRequests_send.append(tmpNumpy)
                self.comm.Barrier()

            # for writing the Ghost information to file
            if(self.testCode):
                # Dirty hack as my vtkWriter only works for lists for scalar
                # variables
                data = []
                for i in range(0, self.nnodes):
                    # Initialise by default all particles to -1
                    data.append(-1)
                for i in range(0, self.numlocalNodes):
                    id_ = self.net[i].id
                    # All those in the subdomain set to number of rank
                    data[id_] = myRank
                for i in range(0, self.numGhostRequests_to_send):
                    for j in range(0, self.IdListGhostRequests_send[i].size):
                        # All those in a Ghost Request list set to value of
                        # processor to which they will be sent
                        id_ = self.IdListGhostRequests_send[i][j]
                        data[id_] = self.GhostRequestProcessorIds_send[i]
                x = np.zeros((self.numlocalNodes, 3))
                data_local = []

                for i in range(0, self.numlocalNodes):
                    x[i, :] = self.coords[self.l2g[i], :]
                    data_local.append(data[self.l2g[i]])

    def communicateGhostParticles(self, u):
        """
        Carries out communication of displacements required at the beginning of
        each step by exploiting nearest neighbour communication as set up in
        setNetwork()
        Uses non-blocking communication
        Input: displacements
        Output: Displacements required for that specific process
        """
        uNew = u

        # Sending block
        fullCommunication = 1

        if fullCommunication == 0:
            # 1st process for sending ghost particles
            if self.comm.Get_rank() == 1:
                ids = self.IdListGhostRequests_send[1]
                proc_recv = self.GhostRequestsProcessorIds_send[1]
                req = self.comm.Isend(u[ids, :], dest=0)
                # Returns when the operation identified by requests is
                # complete
                req.Wait()
            # 0th process for receiving ghost particles
            if self.comm.Get_rank() == 0:
                ids = self.IdListGhostRequests_recv[0]
                tmpDisp = np.empty((ids.size, 3), dtype=float)
                # Irecv will return immediately, indicating to the system that
                # it will be receiving a message
                req = self.comm.Irecv(tmpDisp, source=1)
                # proceed beyond Irecv to do other useful work, and then check
                # back later to see if the message has arrived. This can be
                # used to dramatically improve performance
                req.Wait()

        if fullCommunication == 1:
            # For each coordinate dimension, XYZ
            for j in range(0, 3):
                self.comm.Barrier()

                for i in range(0, self.numGhostRequests_to_send):
                    # Collect information to be sent
                    ids = self.IdListGhostRequests_send[i]
                    proc_recv = self.GhostRequestProcessorIds_send[i]
                    self.comm.Isend(u[ids, j], dest=proc_recv)

                self.comm.Barrier()

                # Receiving block
                for i in range(0, self.numGhostRequests_to_recv):
                    ids = self.IdListGhostRequests_recv[i]
                    tmpDisp = np.empty((ids.size, 1), dtype=float)
                    proc_send = self.GhostRequestProcessorIds_recv[i]
                    self.comm.Irecv(tmpDisp, source=proc_send)
                    for k in range(0, ids.size):
                        # Received as np scalar array
                        uNew[ids[k], j] = tmpDisp[0]
            self.comm.Barrier()

        return uNew

    def setConnPar(self, horizon):
        """
        Sets the sparse connectivity matrix for the partition, should only ever
        be called once
        """
        myRank = self.comm.Get_rank()

        # Initiate the connectivity matrix as non sparse
        conn = np.zeros((self.nnodes, self.nnodes))

        # Initiate uncracked connectivity matrix
        conn_0 = np.zeros((self.nnodes, self.nnodes))

        # Setup the family for each particle
        # in place of family matrix, have now got connectivity matrix
        self.family = []
        tmpGhost = []

        # For each local cell loop over each set of local particles
        for i in range(0, self.numlocalNodes):
            globalId_i = self.net[i].id

            # For each node in the same partition
            for j in range(0, self.numlocalNodes):
                # j is also in the set of local nodes
                globalId_j = self.net[j].id
                # Loop over local nodes in same partition
                # do not fill diagonals
                if globalId_i != globalId_j:
                    if func.l2(self.coords[globalId_i, :],
                               self.coords[globalId_j, :]) < horizon:
                        conn_0[globalId_i, globalId_j] = 1
                        if (self.isCrack(self.coords[globalId_i, :],
                                         self.coords[globalId_j, :]) is False):
                            conn[globalId_i, globalId_j] = 1

            # loop over nodes in the nearest neighbour partitions
            for k in range(0, len(self.neighbour_ids)):
                # k is in the set of nearest neighbour nodes
                globalId_k = self.neighbour_ids[k]
                if (func.l2(self.coords[globalId_i, :],
                            self.coords[globalId_k, :]) < horizon):
                    # Must fill both triangles of matrix, since we aren't
                    # iterating over the neighbour id's in the parent loop

                    # needed? There is definately an issue here
                    conn_0[globalId_i, globalId_k] = 1
                    if (self.isCrack(self.coords[globalId_i, :],
                                     self.coords[globalId_k, :]) is False):
                        tmpGhost.append(globalId_k)
                        conn[globalId_i, globalId_k] = 1

        # Initial bond damages
        count = np.sum(conn, axis=0)
        self.family = np.sum(conn_0, axis=0)
        damage = np.divide((self.family - count), self.family)
        damage.resize(self.nnodes)

        # Lower triangular - count bonds only once
        # make diagonal values 0
        conn = np.tril(conn, -1)

        # Convert to sparse matrix
        self.conn = sparse.csr_matrix(conn)
        self.conn_0 = sparse.csr_matrix(conn_0)

        if self.v:
            print('self.conn is', self.conn)

        # Find a better way to do this with the sparse matrices?
        # which is a list of ghost particles
        self.ghostList = np.unique(tmpGhost)

        self.ghostListProcessors = []

        for i in range(0, len(self.ghostList)):
            self.ghostListProcessors.append(
                int(self.partition[self.ghostList[i]])
                )

        # Integer value which stores number of ghost requests for displacements
        # at each time step
        self.numGhostRequests_to_send = 0
        # Integer value which stores numbe of ghost requests that will be
        # received
        self.numGhostRequests_to_recv = 0
        # Will be a list of length self.numGhostRequests containing the
        # processor number for each communicator
        self.GhostRequestProcessorIds_send = []
        # Will be a list of length self.numGhostRequests containing the
        # processor number for each communicator
        self.GhostRequestProcessorIds_recv = []
        # List of list contain the global ids which will be sent.
        self.IdListGhostRequests_send = []
        # List of list contain the global ids which will be recv.
        self.IdListGhostRequests_recv = []

        # Loop over each processor
        for i in range(0, self.comm.Get_size()):
            # Container for vector to mark which partitions are nearest
            # neighbours for partition i
            areNN = np.zeros(self.comm.Get_size(), dtype=int)

            # Each partition has a corresponding process
            if myRank == i:
                for k in range(0, len(self.NN[i])):
                    # Mark as a nearest neighbour for processor i
                    areNN[self.NN[i][k]] = 1

            # Communicate to all other processors
            self.comm.Bcast(areNN, root=i)

            # wait here until all processes have reached this point, i.e. that
            # the full areNN is constructed.
            self.comm.Barrier()

            for k in range(0, len(self.NN[i])):
                proc_id = int(self.NN[i][k])

                # If this is the control processor
                if myRank == i:
                    # Create a list of ghost which live on a given neighbour
                    tmp = []

                    # Loop over all ghost particles
                    for j in range(0, self.ghostList.size):
                        proc = self.ghostListProcessors[j]
                        # if particle is in processor self.NN[k]
                        if proc == proc_id:
                            tmp.append(self.ghostList[j])

                    # tmp contains list of particles in ghost of i - required
                    # from processor self.NN[k]
                    self.numGhostRequests_to_recv += 1
                    self.GhostRequestProcessorIds_recv.append(proc_id)

                    # messages with different tags will be buffered by the
                    # network until this processor is ready for them
                    self.comm.send(int(len(tmp)), dest=proc_id, tag=1)
                    tmpArray = np.zeros(len(tmp), dtype=int)

                    for ii in range(0, tmpArray.size):
                        tmpArray[ii] = int(tmp[ii])

                    self.IdListGhostRequests_recv.append(tmpArray)
                    self.comm.Send(tmpArray, dest=proc_id, tag=2)
                elif myRank == proc_id:
                    self.numGhostRequests_to_send += 1
                    self.GhostRequestProcessorIds_send.append(i)
                    numParticles_tmp = self.comm.recv(source=i, tag=1)
                    tmpNumpy = np.empty(numParticles_tmp, dtype=int)
                    self.comm.Recv(tmpNumpy, source=i, tag=2)
                    self.IdListGhostRequests_send.append(tmpNumpy)
            self.comm.Barrier()

        # for writing the Ghost information to file
        if self.testCode:
            # Dirty hack as my vtkWriter only works for lists for scalar
            # variables
            data = []
            for i in range(0, self.nnodes):
                # Initialise by default all particles to -1
                data.append(-1)
            for i in range(0, self.numlocalNodes):
                id_ = self.net[i].id
                # All those in the subdomain set to number of rank
                data[id_] = myRank
            for i in range(0, self.numGhostRequests_to_send):
                for j in range(0, self.IdListGhostRequests_send[i].size):
                    # All those in a Ghost Request list set to value of
                    # processor to which they will be sent
                    id_ = self.IdListGhostRequests_send[i][j]
                    data[id_] = self.GhostRequestProcessorIds_send[i]
            x = np.zeros((self.numlocalNodes, 3))
            data_local = []

            for i in range(0, self.numlocalNodes):
                x[i, :] = self.coords[self.l2g[i], :]
                data_local.append(data[self.l2g[i]])
            vtu.writeParallel("GhostInformation", self.comm,
                              self.numlocalNodes, x, data_local,
                              np.zeros((self.numlocalNodes, 3)))
        return damage

    def setH(self):
        """
        Constructs the covariance matrix, K, failure strains matrix and H
        matrix, which is a sparse matrix containing distances
        """
        st = time.time()
        coords = self.coords

        # Extract the coordinates
        V_x = coords[:, 0]
        V_y = coords[:, 1]
        V_z = coords[:, 2]

        # Tiled matrices - there is a more efficient way to do this, like in
        # calcBondStretchNew
        lam_x = np.tile(V_x, (self.nnodes, 1))
        lam_y = np.tile(V_y, (self.nnodes, 1))
        lam_z = np.tile(V_z, (self.nnodes, 1))

        # Dense matrices
        H_x0 = -lam_x + lam_x.transpose()
        H_y0 = -lam_y + lam_y.transpose()
        H_z0 = -lam_z + lam_z.transpose()

        norms_matrix = (np.power(H_x0, 2) + np.power(H_y0, 2)
                        + np.power(H_z0, 2))

        self.L_0 = np.sqrt(norms_matrix)
        self.H_x0 = sparse.csr_matrix(self.conn_0.multiply(H_x0))
        self.H_y0 = sparse.csr_matrix(self.conn_0.multiply(H_y0))
        self.H_z0 = sparse.csr_matrix(self.conn_0.multiply(H_z0))
        self.H_x0.eliminate_zeros()
        self.H_y0.eliminate_zeros()
        self.H_z0.eliminate_zeros()

        # Length scale for the covariance matrix
        scale = 0.05

        # Scale of the covariance matrix
        nu = 1e-5

        # inv length scale parameter
        inv_length_scale = np.divide(-1., 2.*pow(scale, 2))

        # radial basis functions
        rbf = np.multiply(inv_length_scale, norms_matrix)

        # Exponential of radial basis functions
        K = np.exp(rbf)

        # Multiply by the vertical scale to get covariance matrix, K
        self.K = np.multiply(pow(nu, 2), K)

        # Create L matrix for sampling perturbations
        # epsilon is a numerical trick so that M is positive semi definite
        epsilon = 1e-5

        # add epsilon before scaling by a vertical variance scale, nu
        Iden = np.identity(self.nnodes)
        K_tild = K + np.multiply(epsilon, Iden)

        K_tild = np.multiply(pow(nu, 2), K_tild)

        self.C = np.linalg.cholesky(K_tild)

        if (self.H_x0.shape != self.H_y0.shape
                or self.H_x0.shape != self.H_z0.shape):
            raise Exception(
                'The sizes of H_x0, H_y0 and H_z0 did not match!'
                'The sizes were {}, {}, {}, respectively'.format(
                    self.H_x0.shape, self.H_y0.shape, self.H_z0.shape
                    )
                )

        if self.v == 2:
            print(self.L_0, self.L_0.shape, 'here is L_0')

        if self.L_0.shape != self.H_x0.shape:
            print('The size of the connectivity matrix is {}'.format(
                self.conn.shape
                ))
            warnings.warn(
                'L_0.size was {}, whilst H_x0.size was {}, they should be the'
                ' same size'.format(self.L_0.shape, self.H_x0.shape)
                )

        # initiate fail stretches matrix as a linked list format
        self.fail_strains = np.full((self.nnodes, self.nnodes), self.s00)
        # Store in sparse structure (even though it is dense)
        self.fail_strains = sparse.csr_matrix(self.fail_strains)

        if self.v:
            print(
                'Type of fail strains is {} and the shape is {}'.format(
                    type(self.fail_strains), self.fail_strains.shape
                    )
                )

        print('Constructed H in {} seconds'.format(time.time() - st))

    def calcBondStretchNew(self, U):
        """
        Calculate the bond strains
        This is the same as the sequential code apart from we only loop over
        particles
        U will contain the displacements from the ghost particles because of
        communication step at the beginning of the timestep
        The improvement on calcBondStretch() is not calculating the relative
        bond distances for every node, just the connected ones
        """
        self.comm.Barrier()
        st = time.time()

        cols, rows, data_x, data_y, data_z = [], [], [], [], []

        for i in range(self.nnodes):
            row = self.conn_0.getrow(i)
            if row.nnz == 0:
                pass
            else:
                # could be a problem here if there is nothing in the row?
                rows.extend(row.indices)
                cols.extend(np.full((row.nnz), i))
                data_x.extend(np.full((row.nnz), U[i, 0]))
                data_y.extend(np.full((row.nnz), U[i, 1]))
                data_z.extend(np.full((row.nnz), U[i, 2]))

        # Must not be lower triangular
        lam_x = sparse.csr_matrix((data_x, (rows, cols)),
                                  shape=(self.nnodes, self.nnodes))
        lam_y = sparse.csr_matrix((data_y, (rows, cols)),
                                  shape=(self.nnodes, self.nnodes))
        lam_z = sparse.csr_matrix((data_z, (rows, cols)),
                                  shape=(self.nnodes, self.nnodes))

        delH_x = -lam_x + lam_x.transpose()
        delH_y = -lam_y + lam_y.transpose()
        delH_z = -lam_z + lam_z.transpose()

        # Sparse matrices
        self.H_x = delH_x + self.H_x0
        self.H_y = delH_y + self.H_y0
        self.H_z = delH_z + self.H_z0

        norms_matrix = (np.power(self.H_x, 2) + np.power(self.H_y, 2)
                        + np.power(self.H_z, 2))

        self.L = norms_matrix.sqrt()

        if self.v == 2:
            print(' The shape of lamx is {}, {}'.format(lam_x.shape, lam_x))
            print('The shape of delH_x is {}, {}'.format(delH_x.shape, delH_x))
            print('The shape of H_x is {}, {}'.format(self.H_x.shape,
                                                      self.H_x))
            print('The shape of L is {} {}'.format(self.L.shape, self.L))

        del_L = self.L - self.L_0

        # Doesn't this kill compressive strains? - seems it is consistent with
        # peridynamic theory
        del_L[del_L < 1e-12] = 0

        # Step 1. initiate as a sparse matrix
        strain = sparse.csr_matrix(self.conn.shape)

        # Step 2. elementwise division
        # TODO: investigate indexing with [self.L_0.nonzero()]  instead of
        # [self.conn.nonzero()]
        strain[self.conn.nonzero()] = sparse.csr_matrix(
            del_L[self.conn.nonzero()]/self.L_0[self.conn.nonzero()]
            )

        self.strain = sparse.csr_matrix(strain)
        self.strain.eliminate_zeros()

        if strain.shape != self.L_0.shape:
            warnings.warn(
                'strain.shape was {}, whilst L_0.shape was {}'.format(
                    strain.shape, self.L_0.shape
                    )
                )
        if self.v:
            print('time taken to calc bond stretch was {}'.format(
                -st + time.time()
                ))

    def calcBondStretch(self, U):
        """
        Calculate the bond strains
        This is the same as the sequential code apart from we only loop over
        paticles
        U will contain the displacements from the ghost particles because of
        communication step at beginning of the timestep
        """
        # wait until all processes have reached this routine
        self.comm.Barrier()

        st = time.time()

        delV_x = U[:, 0]
        lam_x = np.tile(delV_x, (self.nnodes, 1))

        delV_y = U[:, 1]
        lam_y = np.tile(delV_y, (self.nnodes, 1))

        delV_z = U[:, 2]
        lam_z = np.tile(delV_z, (self.nnodes, 1))

        # dense matrices
        delH_x = -lam_x + lam_x.transpose()
        delH_y = -lam_y + lam_y.transpose()
        delH_z = -lam_z + lam_z.transpose()

        # dense matrices
        self.H_x = delH_x + self.H_x0
        self.H_y = delH_y + self.H_y0
        self.H_z = delH_z + self.H_z0

        # Compute bond length matrix
        # bond lengths at current time step
        # Step 1. Initiate as a sparse matrix

        norms_matrix = (np.power(self.H_x, 2) + np.power(self.H_y, 2)
                        + np.power(self.H_z, 2))

        self.L = np.sqrt(norms_matrix)

        if self.v == 2:
            print(' The shape of L is {}, {}'.format(self.L.shape, self.L))
            print(delH_x, 'ABOVE is delH_x')
            print(self.H_x, 'ABOVE is H_x')

        del_L = self.L - self.L_0

        # Doesn't this kill compressive strains?
        del_L[del_L < 1e-12] = 0

        # Step 1. initiate as a sparse matrix
        strain = sparse.csr_matrix(self.conn.shape)

        # Step 2. elementwise division
        # TODO: investigate indexing with [self.L_0.nonzero()]  instead of
        # [self.conn.nonzero()]
        strain[self.conn.nonzero()] = sparse.csr_matrix(
            del_L[self.conn.nonzero()]/self.L_0[self.conn.nonzero()]
            )

        self.strain = sparse.csr_matrix(strain)
        self.strain.eliminate_zeros()

        if strain.shape != self.L_0.shape:
            warnings.warn(
                'strain.shape was {}, whilst L_0.shape was {}'.format(
                    strain.shape, self.L_0.shape
                    )
                )
        if self.v:
            print('time taken to calc bond stretch was {}'.format(
                -st + time.time()
                ))

    def checkBonds(self):
        """
        Calculates bond damage
        """
        st = time.time()
        # Make sure only calculating for bonds that exist

        # Step 1. initiate as sparse matrix
        bond_healths = sparse.csr_matrix(self.conn.shape)

        # Step 2. Find broken bonds, squared as strains can be negative
        bond_healths[self.conn.nonzero()] = sparse.csr_matrix(
            self.fail_strains.power(2)[self.conn.nonzero()]
            - self.strain.power(2)[self.conn.nonzero()]
            )

        # Update failed bonds
        # Does it break the bonds of the neighbour particles?
        bond_healths = bond_healths > 0

        self.conn = sparse.csr_matrix(bond_healths)
        # needed?
        self.conn.eliminate_zeros()

        # Bond damages
        # Using lower triangular connectivity matrix, so just mirror it for
        # bond damage calc
        temp = self.conn  # + self.conn.transpose()

        # return damage for our local nodes only
        damage_local = np.zeros(self.numlocalNodes)

        for i in range(self.numlocalNodes):
            globalId_i = self.net[i].id
            # could be an issue here
            count = np.sum(temp.getrow(globalId_i))
            damage = np.divide(self.family[globalId_i] - count,
                               self.family[globalId_i])
            damage_local[i] = damage

        if self.v:
            print(np.max(damage_local), 'max_damage')
            print(np.min(damage_local), 'min_damage')

        if self.v:
            print('time taken to check bonds was {}'.format(-st + time.time()))

        return damage_local

    def computebondForce(self):
        """
        Only computed for each of the particles in the subdomain
        """
        st = time.time()
        self.c = 18.0 * self.kscalar / (np.pi * (self.horizon**4))
        # Container for the forces on each particle in each dimension
        F = np.zeros((self.nnodes, 3))

        # Step 1. Initiate container as a sparse matrix, only need calculate
        # for bonds that exist
        force_normd = sparse.csr_matrix(self.conn.shape)

        # Step 2. find normalised forces
        force_normd[self.conn.nonzero()] = sparse.csr_matrix(
            self.strain[self.conn.nonzero()]/self.L[self.conn.nonzero()]
            )

        # Make lower triangular into full matrix
        force_normd = force_normd  # + force_normd.transpose()

        # Multiply by the direction and scale of each bond (just trigonometry,
        # we have already scaled for bond length in step 2)
        bond_force_x = force_normd.multiply(self.H_x)
        bond_force_y = force_normd.multiply(self.H_y)
        bond_force_z = force_normd.multiply(self.H_z)

        # now sum along the rows to calculate resultant force on nodes
        F_x = np.array(bond_force_x.sum(axis=0))
        F_y = np.array(bond_force_y.sum(axis=0))
        F_z = np.array(bond_force_z.sum(axis=0))

        F_x.resize(self.nnodes)
        F_y.resize(self.nnodes)
        F_z.resize(self.nnodes)

        # Finally multiply by volume and stiffness
        F_x = self.c * np.multiply(F_x, self.V)
        F_y = self.c * np.multiply(F_y, self.V)
        F_z = self.c * np.multiply(F_z, self.V)

        if self.v == 2:
            print(F_x, 'The shape of F_x is', F_x.shape, type(F_x))
            print(self.V, 'The shape of V is', self.V.shape, type(self.V))

        F[:, 0] = F_x
        F[:, 1] = F_y
        F[:, 2] = F_z

        assert F.shape == (self.nnodes, 3)
        if self.v:
            print(
                'time taken to compute bond force was {}'.format(
                    -st + time.time()
                    )
                )

        F_local = np.zeros((self.numlocalNodes, 3))
        for i in range(self.numlocalNodes):
            globalId_i = self.net[i].id
            F_local[i] = F[globalId_i]

        return F_local
