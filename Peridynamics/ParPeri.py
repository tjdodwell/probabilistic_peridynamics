import sys

import numpy as np

import PeriParticle as peri

import periFunctions as func

import parallelFunctions as par

from mpi4py import MPI

sys.path.insert(1, '../PostProcessing')

import vtk as vtk

import vtu as vtu

class ParModel:

	def __init__(self):

		self.testCode = 1



		self.plotPartition = 1

		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3

		# Material Parameters from classical material model
		self.horizon = 5.0
		self.k_scalar = 1.00
		self.s00 = 0.005

		self.c = 18.0 * self.k_scalar / (np.pi * (self.horizon**4));


		if (self.dim == 3):
			self.meshType = 4
			self.boundaryType = 2
			self.numBoundaryNodes = 3
			self.numMeshNodes = 4

		self.readMesh(self.meshFileName)

		self.setNetwork(self.horizon)

		self.lhs = []
		self.rhs = []

		# Find the Boundary
		for i in range(0, self.nnodes):
			bnd = self.findBoundary(self.coords[i][:])
			if (bnd < 0):
				(self.lhs).append(i)
			elif (bnd > 0):
				(self.rhs).append(i)

		self.u = [] # List for Solution at each time step
		self.damage = [] # List containing damage at each time step




	def checkBonds(self, U, broken, damage):
		""" Check bonds to see if they have been broken in this time step
			This is the same as the sequential code apart from we only loop over local particles
			Input:  'U' Particle displacements. 'U' will contain the displacements from the 
					ghost particles becuase of communication step at beginning of the timestep
					'broken' list of lists
					'damage' list
			Output: updated 'broken' and 'damage'
		"""
		self.comm.Barrier()
		for i in range(0, self.numlocalNodes):
			id_ = self.net[i].id
			yi = self.coords[id_,:] + U[id_,:] # deformed coordinates of particle i
			count = 0;
			family = self.family[i] # extract family for particle i
			for k in range(0, len(family)):
				j = self.family[i][k]
				d = self.distance2Boundary(self.coords[j,:])
				testMe = 0
				if(d > 1.0 * self.horizon):
					testMe = 1
				if( broken[i][k] == 0): # if bond is not previously broken
					yj = self.coords[j,:] + U[j,:] # deformed coordinates of particle j
					bondLength = func.l2norm(self.coords[id_,:] - self.coords[j,:])
					rvec = yj - yi
					r = func.l2norm(rvec)
					strain = (r - bondLength) / bondLength
					if ((strain > self.s00) and (testMe == 1)):
						broken[i][k] = 1 # Break the bond
						count += 1
				else :
					count += 1.0
			damage[i] = float(count / len(family))
		return broken, damage

	def computebondForce(self, U, broken):
		""" Computes the bond force and sums them
			Input: particle displacements, broken bonds
			Output: bond force vector for local nodes
		"""
		F = np.zeros((self.numlocalNodes,3)) # Container for the forces on each particle in each dimension
		for i in range(0, self.numlocalNodes): # For each particles in this subdomain
			id_ = self.net[i].id
			yi = self.coords[id_,:] + U[id_,:]
			family = self.family[i] # This is a list of id's for family members
			for k in range(0, len(family)):
				j = self.family[i][k]
				if(broken[i][k] == 0 ): # Bond between particle i and j has not been broken
					yj = self.coords[j,:] + U[j,:]
					bondLength = func.l2norm(self.coords[id_,:] - self.coords[j,:])
					rvec = yj - yi
					r = func.l2norm(rvec)
					dr = r - bondLength
					if( dr < 1e-12 ): # Kills an issue with rounding error
						dr = 0.0
					kb = (self.c / bondLength) * self.V[i] * dr
					F[i,:] += (kb / r) * rvec
		return F
	
	def computebondForceNew(self, U, broken):
		""" Computes the bond force and sums them to find resultant particle force
			Input: particle displacements, broken bonds
			Output: bond force vector for local nodes
		"""
		F = np.zeros((self.nnodes, 3))
		# non local particle forces will be 0, but they won't ever be used
		
		for i in range(0, self.numlocalNodes):
			id_ = self.net[i].id
			yi = self.coords[id_,:] + U[id_,:]
			family = self.family[i] # This is a list of id's for family members
			for k in range(0, len(family)):
				j = self.family[i][k]
				if(broken[i][k] == 0 ): # Bond between particle i and j has not been broken
					yj = self.coords[j,:] + U[j,:]
					bondLength = func.l2norm(self.coords[id_,:] - self.coords[j,:])
					rvec = yj - yi
					r = func.l2norm(rvec)
					dr = r - bondLength
					if( dr < 1e-12 ): # Kills an issue with rounding error
						dr = 0.0
					kb = (self.c / bondLength) * self.V[i] * dr
					F[id_,:] += (kb / r) * rvec
		return F

	def readMesh(self, fileName):
		""" Reads particle coordinates from the mesh, and mesh element connectivities. Domain decomposition is also done here.
			Input: mesh file name 'test.msh'
		"""
		f = open(fileName, "r")

		if f.mode == "r":

			iline = 0

			# Read the Nodes in the Mesh First

			findNodes = 0
			while (findNodes == 0):
				iline += 1
				line = f.readline()
				if(line.strip() == '$Nodes'):
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

			line = f.readline() # This line will read $EndNodes - Could add assert on this
			line = f.readline() # This line will read $Elements

			# Read the Elements

			line = f.readline() # This gives the total number of elements - but includes all types of elements
			self.totalNel = int(line.strip())
			self.connectivity = []
			self.connectivity_bnd = []


			for ie in range(0, self.totalNel):

				iline +=1
				line = f.readline()
				rowAsList = line.split()

				if(int(rowAsList[1]) == self.boundaryType):
					tmp = np.zeros(self.dim)
					for k in range(0, self.dim):
						tmp[k] = int(rowAsList[5 + k]) - 1
					self.connectivity_bnd.append(tmp)

				elif(int(rowAsList[1]) == self.meshType):
					tmp = np.zeros(self.dim + 1)
					for k in range(0, self.dim + 1):
						tmp[k] = int(rowAsList[5 + k]) - 1
					self.connectivity.append(tmp)

			self.nelem = len(self.connectivity)

			self.nelem_bnd = len(self.connectivity_bnd)

			self.partition, self.NN = par.decomposeDomain(self.coords, self.connectivity, self.comm.Get_size(), self.partitionType)

			if(self.plotPartition & self.comm.Get_rank() == 0):

				vtk.write("Partition.vtk","Partition", self.coords, self.partition, np.zeros((self.nnodes, 3)))


		f.close()

	def setNetwork(self, horizon):
		""" Creates local to global id lists, calculates volumes of the particles, creates family lists and lists of ghost particles. 
			Constructs the covariance matrix, K.
			Input: Peridynamic horizon distance
		"""
		myRank = self.comm.Get_rank();

		self.net = [] # List to store the network

		neighbour_ids = []

		self.l2g = [] # local to global

		self.g2l = [] # global to local
		
		
		localCount = 0
		for i in range(0, self.nnodes): # For each of the particles

			self.g2l.append(-1);

			if(myRank == int(self.partition[i])): # particle belongs to this processor
				self.net.append(peri.Particle())
				self.net[localCount].setId(i)
				self.net[localCount].setCoord(self.coords[i])
				localCount += 1
				self.l2g.append(i)
				self.g2l[i] = localCount

			else: # This is the case where particle lives on another process but is in a neighbouring subdomain, so has potential to be in family
				check = 0
				for k in range(0,len(self.NN[myRank])):
					if(self.NN[myRank][k] == int(self.partition[i])):
						check += 1
				if(check > 0): # Number greater than zero indicates that particle is in one of the neighbouring subdomains
					neighbour_ids.append(i)

		self.numlocalNodes = localCount # Store the number of particles directly in subdomains

		self.localCoords = np.zeros((self.numlocalNodes, 3))

		if(self.testCode):
			# Check that no nodes have be lost or added by bug
			mylocal = np.zeros(1)
			totalNodes = np.zeros(1)
			mylocal[0] = localCount
			self.comm.Reduce(mylocal, totalNodes, MPI.SUM, 0)
			if(myRank == 0):
				assert int(totalNodes[0]) == self.nnodes
			self.comm.Barrier()

		# Not ideal for now - but do on all processors over all elements - saves dealing with boundary cases
		Vols = np.zeros(self.nnodes)
		for ie in range(0,self.nelem):
			n = self.connectivity[ie]
			# Compute Area / Volume
			val = 1. / n.size
			if (self.dim == 2): # Define area of element
				xi = self.coords[int(n[0])][0]
				yi = self.coords[int(n[0])][1]
				xj = self.coords[int(n[1])][0]
				yj = self.coords[int(n[1])][1]
				xk = self.coords[int(n[2])][0]
				yk = self.coords[int(n[2])][1]
				val *= 0.5 * ( (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi) )
			for j in range(0,n.size):
				Vols[int(n[j])] += val

		self.V = np.zeros(self.numlocalNodes)
		for i in range(0, self.numlocalNodes):
			self.V[i] = Vols[self.net[i].id]
			
		# Generate the covariance matrix. Not ideal - but do on all processors
		# Length scale for the covariance matrix
		l = 0.05
		
		# Scale of the covariance matrix
		nu = 1e-5
		
		# inv length scale parameter
		inv_length_scale = (np.divide(-1., 2.*pow(l, 2)))
		
		X = np.sum(pow(self.coords, 2), axis=1)
		
		tiled_X = np.tile(X, (self.nnodes,1))
		tiled_Xt = np.transpose(tiled_X)
		
		outer_product= np.dot(self.coords, np.transpose(self.coords))
		
		norms_matrix = (tiled_X - np.multiply(outer_product, 2) + tiled_Xt)
		
		# radial basis functions
		rbf = np.multiply(inv_length_scale, norms_matrix)
		
		# Exponential of radial basis functions
		K = np.exp(rbf)
		
		# Multiply by the vertical scale to get covariance matrix, K
		self.K = np.multiply(pow(nu, 2), K)
		
		#Create L matrix for sampling perturbations
		#epsilon, numerical trick so that M is positive semi definite
		epsilon = 1e-5

		# add epsilon before scaling by a vertical variance scale, nu
		I = np.identity(self.nnodes)
		K_tild = K + np.multiply(epsilon, I)
		
		K_tild = np.multiply(pow(nu, 2), K_tild)
		
		self.C = np.linalg.cholesky(K_tild)	
		
		# -- Setup the family for each particle

		# For each local cell loop over each set of local particles
		# There is more efficient ways to do this, but doesn't really matter since one of calculation
		self.family = []

		tmpGhost = []
		#tmpGhostProcessors = []

		for i in range(0, self.numlocalNodes): # For each of the local nodes

			tmp = [] # temporary list
			globalId_i = self.net[i].id

			for j in range(0, self.numlocalNodes): # For each node in the same partition
				globalId_j = self.net[j].id
				# Loop over local nodes in same partition
				if(globalId_i != globalId_j):
					if(func.isNeighbour(self.coords[globalId_i,:], self.coords[globalId_j,:], horizon)):
						testCrack = self.isCrack(self.coords[globalId_i,:],self.coords[globalId_j,:])
						if(testCrack == 0):
							tmp.append(globalId_j)


			for k in range(0, len(neighbour_ids)): # loop over only nodes in the nearest neighbour partitions
				if(func.isNeighbour(self.coords[globalId_i,:], self.coords[neighbour_ids[k],:], horizon)):
					testCrack = self.isCrack(self.coords[globalId_i,:],self.coords[globalId_j,:])
					if(testCrack == 0):
						tmp.append(neighbour_ids[k])
						tmpGhost.append(neighbour_ids[k])

			# Add family for ith local node
			self.family.append(np.zeros(len(tmp), dtype = int))
			for j in range(0, len(tmp)):
				self.family[i][j] = int(tmp[j])

		self.ghostList = np.unique(tmpGhost)

		self.ghostListProcessors = []

		for i in range(0, len(self.ghostList)):
			self.ghostListProcessors.append(int(self.partition[self.ghostList[i]]))

		# Need to setup information to be communicated - THIS NOT THE WAY TO DO IT :)
		# Here since it is a one time setup cost using blocked communication

		self.numGhostRequests_to_send = 0 # Integer value which stores number of ghost requests for displacements at each time step
		self.numGhostRequests_to_recv = 0 # Integer value which stores numbe of ghost requests that will be received
		self.GhostRequestProcessorIds_send = [] # Will be a list of length self.numGhostRequests containing the processor number for each communicator
		self.GhostRequestProcessorIds_recv = [] # Will be a list of length self.numGhostRequests containing the processor number for each communicator
		self.IdListGhostRequests_send = [] # List of list contain the global ids which will be sent.
		self.IdListGhostRequests_recv = [] # List of list contain the global ids which will be recv.
		
		if(self.comm.Get_size() == 1):
			# In the case where we have one process, then we have sequential simulation
			# TODO: this bypass seems to work, but should be bug checked.
			pass
		else:
			for i in range(0, self.comm.Get_size()): # Loop over each processor
				areNN = np.zeros(self.comm.Get_size(), dtype = int) # Create Vector to mark which are neighest neighbours for processor i
				if(myRank == i): # this is your turn
					for k in range(0, len(self.NN[i])):
						areNN[self.NN[i][k]] = 1 # Mark as a neighest neighbour for processor i
				self.comm.Bcast(areNN, root=i) # Communicate to all other processors
				self.comm.Barrier()
				for k in range(0, len(self.NN[i])):
					proc_id = int(self.NN[i][k])
					if(myRank == i): # If this is the control processor
						tmp = [] # Create a list of ghost which live on a give neighbour
						for j in range(0, self.ghostList.size): # Loop over all ghost particles
							proc = self.ghostListProcessors[j]
							if(proc == proc_id): # if particle is in processor self.NN[k]
								tmp.append(self.ghostList[j])
						# tmp contains list of particles in ghost of i - required from processor self.NN[k]
						self.numGhostRequests_to_recv += 1
						self.GhostRequestProcessorIds_recv.append(proc_id)
						self.comm.send(int(len(tmp)), dest = proc_id, tag = 1)
						tmpArray = np.zeros(len(tmp), dtype = int)
						for ii in range(0, tmpArray.size):
							tmpArray[ii] = int(tmp[ii])
						self.IdListGhostRequests_recv.append(tmpArray)
						self.comm.Send(tmpArray, dest = proc_id, tag = 2)
					elif(myRank == proc_id):
						self.numGhostRequests_to_send += 1
						self.GhostRequestProcessorIds_send.append(i)
						numParticles_tmp = self.comm.recv(source = i, tag = 1)
						tmpNumpy = np.empty(numParticles_tmp, dtype=int)
						self.comm.Recv(tmpNumpy,source = i, tag = 2)
						self.IdListGhostRequests_send.append(tmpNumpy)
							# end else if
						# end if areNN[k] == 1
					# end for each NN
				# end if this processor is a nearest neighbour_ids
				# end for each processor
	
				self.comm.Barrier() # Processors Idle before we move to next processor

			if(self.testCode):
				data = [] # Dirty hack as my quick vtkWriter only works for lists for scalar variables
				for i in range(0, self.nnodes):
					data.append(-1) # Initialise by default all particles to -1
				for i in range(0, self.numlocalNodes):
					id_ = self.net[i].id
					data[id_] = myRank # All those in the subdomain set to number of rank
				for i in range(0, self.numGhostRequests_to_send):
					for j in range(0,self.IdListGhostRequests_send[i].size):
						# All those in a Ghost Request list set to value of processor to which they will be sent!
						id_ = self.IdListGhostRequests_send[i][j]
						data[id_] = self.GhostRequestProcessorIds_send[i]
	
				x = np.zeros((self.numlocalNodes,3))
				data_local = []
	
				for i in range(0, self.numlocalNodes):
					x[i,:] = self.coords[self.l2g[i],:]
					data_local.append(data[self.l2g[i]])
				print('data_local length', len(data_local), 'numlocalnodes', self.numlocalNodes, 'data length', len(data), 'l2g length', len(self.l2g))
				vtu.writeParallel("GhostInformation", self.comm, self.numlocalNodes, x, data_local, np.zeros((self.numlocalNodes, 3)))
	
				#vtk.write("GhostInformation_send" + str(myRank)+".vtk","Partition", self.coords, data, np.zeros((self.nnodes, 3)))


	def communicateGhostParticles(self, u):
		"""
		Code carries out communication of displacements required at the beginning of each step
		Carried out by exploiting nearest neighbour communicaiton as set up in setNetwork()
		Uses non-blocking communication
		Input: particle displacements
		"""
		uNew = u
		
		# Sending block

		fullCommunication = 1

		if(fullCommunication == 0):

			if(self.comm.Get_rank() == 1):

				ids = self.IdListGhostRequests_send[1]

				proc_recv = self.GhostRequestProcessorIds_send[1]

				req = self.comm.Isend(u[ids,:], dest = 0)

				req.Wait()


			if(self.comm.Get_rank() == 0):

				ids = self.IdListGhostRequests_recv[0]

				tmpDisp= np.empty((ids.size,3), dtype=float)

				req = self.comm.Irecv(tmpDisp, source = 1)

				req.Wait()

		if(fullCommunication == 1):

			for j in range(0,3):

				self.comm.Barrier() # Why do I need these? Someone can explain to me!

				for i in range(0, self.numGhostRequests_to_send):
					# Collect information to be sent
					ids = self.IdListGhostRequests_send[i]
					proc_recv = self.GhostRequestProcessorIds_send[i]
					self.comm.Isend(u[ids,j], dest = proc_recv)
					#print("This is processor = " + str(self.comm.Get_rank()) + " size = " + str(ids.size) + " , sending to proc = " + str(proc_recv) + " norm is = " + str(np.linalg.norm(u[ids,j])))

				self.comm.Barrier() # Why do I need these? Someone can explain to me!

				# Receiving block
				for i in range(0, self.numGhostRequests_to_recv):
					ids = self.IdListGhostRequests_recv[i]
					tmpDisp= np.empty((ids.size,1), dtype=float)
					proc_send = self.GhostRequestProcessorIds_recv[i]
					self.comm.Irecv(tmpDisp, source = proc_send)
					#print("This is processor = " + str(self.comm.Get_rank()) + " size = " + str(ids.size) + ", recv from proc = " + str(proc_send) + " norm is = " + str(np.linalg.norm(tmpDisp)))
					for k in range(0, ids.size):
						uNew[ids[k],j] = tmpDisp[0]

			self.comm.Barrier() # Why do I need these? Someone can explain to me!

		return uNew
