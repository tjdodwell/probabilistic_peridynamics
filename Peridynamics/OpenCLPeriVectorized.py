
import numpy as np

import PeriParticle as peri

import periFunctions as func

from scipy import sparse

import warnings

import time

#import pyopencl as cl


class SeqModel:
	def __init__(self):
		
        ## Scalars
		# self.nnodes defined when instance of readMesh called, cannot initialise any other matrix until we know the nnodes
		self.v = True  # is this needed here, since it was put in MODEL class, simplesquare?
		self.dim = 2

		self.meshFileName = "test.msh"
		
		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3
		
# =============================================================================
# 		# OpenCL stuff 
# 		self.context = context
# 		self.queue = cl.CommandQueue(context)
# =============================================================================
		
		
# =============================================================================
# 		# Material Parameters from classical material model
# 		self.horizon = 0.1
# 		self.kscalar = 0.05
# 		self.s00 = 0.05
# 
# 		self.c = 18.0 * self.kscalar / (np.pi * (self.horizon**4));
# =============================================================================


		if (self.dim == 3):
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
				if(line.strip() == '$Nodes'):
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

			line = f.readline() # This line will read $EndNodes - Could add assert on this
			line = f.readline() # This line will read $Elements

			# Read the Elements from the mesh for the volume calculations connectivity 

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

		f.close()
		
	def setVolume(self):

		V = np.zeros(self.nnodes, dtype=np.float64)

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
				V[int(n[j])] += val
		self.V = V.astype(np.float64)
	def setNetwork(self, horizon):
		""" Sets the family matrix, and converts to horizons matrix. Calculates horizons_lengths
		"""
		
		self.family = []
		self.horizons_lengths = np.zeros(self.nnodes, dtype= int)
		
		for i in range(0, self.nnodes):
			tmp = []
			
			for j in range(0, self.nnodes):
				if(i != j):
					l2_sqr = func.l2_sqr(self.coords[i,:], self.coords[j,:])
					if(np.sqrt(l2_sqr) < horizon):
						tmp.append(j)
			self.family.append(np.zeros(len(tmp), dtype = np.intc))
			self.horizons_lengths[i] = np.intc((len(tmp)))
			for j in range(0, len(tmp)):
				self.family[i][j] = np.intc((tmp[j]))
				
		self.MAX_HORIZON_LENGTH = np.intc((len(max(self.family,key = lambda x: len(x)))))
				

		horizons = -1 * np.ones([self.nnodes,self.MAX_HORIZON_LENGTH])
		for i,j in enumerate(self.family):
			horizons[i][0:len(j)] = j
		
		self.horizons = horizons.astype(np.intc)
		
		
		# Initiate crack
		
		for i in range(0, self.nnodes):
			
			for k in range(0, self.MAX_HORIZON_LENGTH):
				j = self.horizons[i][k]
				if(self.isCrack(self.coords[i,:], self.coords[j,:])):
					self.horizons[i][k] = np.intc(-1)
					
					
		
		
# =============================================================================
# 	def setConn(self, horizon):
# 		""" Sets the sparse connectivity matrix, should only ever be called once
# 		"""
# 		# Initiate connectivity matrix as non sparse
# 		conn = np.zeros((self.nnodes, self.nnodes))
# 		
# 		# Initiate uncracked connectivity matrix
# 		conn_0 = np.zeros((self.nnodes, self.nnodes))
# 		
# 		# Check if nodes are connected
# 		for i in range(0, self.nnodes):		
# 			for j in range(0, self.nnodes):
# 				if(func.l2(self.coords[i,:], self.coords[j,:]) < horizon):
# 					conn_0[i, j] = 1
# 					if i == j:
# 						pass # do not fill diagonal
# 					elif (self.isCrack(self.coords[i,:], self.coords[j,:]) == False):
# 						conn[i, j] = 1
# 						
# 		# Initial bond damages
# 		count = np.sum(conn, axis =0)
# 		self.family = np.sum(conn_0, axis=0)
# 		damage = np.divide((self.family - count), self.family)
# 		damage.resize(self.nnodes)
# 		
# 		print('initial damage vector is {}'.format(damage))
# 		
# 		# Lower triangular - count bonds only once
# 		# make diagonal values 0
# 		conn = np.tril(conn, -1)
# 		
# 		# Convert to sparse matrix
# 		self.conn = sparse.csr_matrix(conn)
# 		self.conn_0 = sparse.csr_matrix(conn_0)
# 		
# 		if self.v:
# 			print('self.conn is HERE', self.conn)
# 			
# 		return horizons, horizon_lengths
# =============================================================================
	
# =============================================================================
# 	def setH(self):
# 		""" Constructs the covariance matrix, K, failure strains matrix and H matrix, which is a sparse matrix containing distances
# 		"""
# 		# TODO redo this for optimised, and using kernels
# 		st = time.time()
# 		coords = self.coords
# 		
# 		nnodes = self.nnodes
# 		size = (nnodes, nnodes)
# 		# Extract the coordinates
# 		V_x = coords[:,0]
# 		V_y = coords[:, 1]
# 		V_z = coords[:, 2]
# 		
# 		# Tiled matrices
# 		h_lam_x = np.tile(V_x, (nnodes, 1))
# 		h_lam_y = np.tile(V_y, (nnodes, 1))
# 		h_lam_z = np.tile(V_z, (nnodes, 1))
# 		
# 		# Create empty arrays for transposes
# 		h_lam_x_t = np.empty(size).astype(np.float32)
# 		h_lam_y_t = np.empty(size).astype(np.float32)
# 		h_lam_z_t = np.empty(size).astype(np.float32)
# 		
# 		# Dense matrices
# 		h_H_x0 = np.empty(size).astype(np.float32)
# 		h_H_y0 = np.empty(size).astype(np.float32)
# 		h_H_z0 = np.empty(size).astype(np.float32)
# 		
# 		# Create OpenCL READ_ONLY buffers (input arrays) in device memory and copy data from host
# 		d_lam_x = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_lam_x)
# 		d_lam_y = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_lam_y)
# 		d_lam_z = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_lam_z)
# 		
# 		# Create OpenCL WRITE_ONLY buffers (output arrays) in device memory
# 		d_H_x0 = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_H_x0.nbytes)
# 		d_H_y0 = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_H_y0.nbytes)
# 		d_H_z0 = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_H_z0.nbytes)
# 		
# 		d_lam_x_t = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_lam_x_t.nbytes)
# 		d_lam_y_t = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_lam_y_t.nbytes)
# 		d_lam_z_t = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, h_lam_z_t.nbytes)
# 		
# 		
# 		kernelsource = open("kernelsource.cl").read()
# 		program = cl.Program(self.context, kernelsource).build()
# 		subtract = program.subtract
# 		subtract.set_scalar_arg_dtypes([np.int32, None, None, None])
# 		transpose_naive = program.transpose_naive
# 		transpose_naive.set_scalar_arg_dtypes([None, None, np.int32, np.int32, np.int32]) # add one more input argument
# 		
# 		# Why can't we just initiate with numpy zeros here?
# 		h_H_x0.fill(0.0)
# 		h_H_y0.fill(0.0)
# 		h_H_z0.fill(0.0)
# 		
# 		start_time =  time.time()
# 		globalrange = size
# 		localrange = None
# 		transpose_naive(self.queue, globalrange, localrange, d_lam_x_t, d_lam_x, 0, nnodes, nnodes)
# 		transpose_naive(self.queue, globalrange, localrange, d_lam_y_t, d_lam_y, 0, nnodes, nnodes)
# 		transpose_naive(self.queue, globalrange, localrange, d_lam_z_t, d_lam_z, 0, nnodes, nnodes)
# 		
# 		subtract(self.queue, globalrange, localrange, nnodes, d_H_x0, d_lam_x_t, d_lam_x)
# 		subtract(self.queue, globalrange, localrange, nnodes, d_H_y0, d_lam_y_t, d_lam_y)
# 		subtract(self.queue, globalrange, localrange, nnodes, d_H_z0, d_lam_z_t, d_lam_z)
# 		
# 		self.queue.finish()
# 		r_time = time.time() - start_time
# 		
# 		cl.enqueue_copy(self.queue, h_H_x0, d_H_x0)
# 		cl.enqueue_copy(self.queue, h_H_y0, d_H_y0)
# 		cl.enqueue_copy(self.queue, h_H_z0, d_H_z0)
# 
# 		norms_matrix = np.power(h_H_x0, 2) + np.power(h_H_y0, 2) + np.power(h_H_z0, 2)
# 		self.L_0 = np.sqrt(norms_matrix)
# 		
# # =============================================================================
# # 		# Into sparse matrices
# # 		self.H_x0 = sparse.csr_matrix(self.conn_0.multiply(h_H_x0))
# # 		self.H_y0 = sparse.csr_matrix(self.conn_0.multiply(h_H_y0))
# # 		self.H_z0 = sparse.csr_matrix(self.conn_0.multiply(h_H_z0))
# # 		self.H_x0.eliminate_zeros()
# # 		self.H_y0.eliminate_zeros()
# # 		self.H_z0.eliminate_zeros()	
# # =============================================================================
# 		
# 		# Length scale for the covariance matrix
# 		l = 0.05
# 		
# 		# Scale of the covariance matrix
# 		nu = 1e-5
# 		
# 		# inv length scale parameter
# 		inv_length_scale = (np.divide(-1., 2.*pow(l, 2)))
# 		
# 		# radial basis functions
# 		rbf = np.multiply(inv_length_scale, norms_matrix)
# 		
# 		# Exponential of radial basis functions
# 		K = np.exp(rbf)
# 		
# 		# Multiply by the vertical scale to get covariance matrix, K
# 		self.K = np.multiply(pow(nu, 2), K)
# 		
# 		#Create L matrix for sampling perturbations
# 		#epsilon, numerical trick so that M is positive semi definite
# 		epsilon = 1e-5
# 
# 		# add epsilon before scaling by a vertical variance scale, nu
# 		I = np.identity(self.nnodes)
# 		K_tild = K + np.multiply(epsilon, I)
# 		
# 		K_tild = np.multiply(pow(nu, 2), K_tild)
# 		
# 		self.C = np.linalg.cholesky(K_tild)
# 		norms_matrix = sparse.csr_matrix(self.H_x0.power(2) + self.H_y0.power(2) + self.H_z0.power(2))
# 		self.L_0 = norms_matrix.sqrt()
# 		
# 	
# 		if self.H_x0.shape != self.H_y0.shape or self.H_x0.shape != self.H_z0.shape:
# 			raise Exception(' The sizes of H_x0, H_y0 and H_z0 did not match! The sizes were {}, {}, {}, respectively'.format(self.H_x0.shape, self.H_y0.shape, self.H_z0.shape))
# 		
# 
# 		if self.v:
# 			print(self.L_0, self.L_0.shape, 'here is L_0')
# 		
# 		if self.L_0.shape != self.H_x0.shape:
# 			print(' The size of the connectivity matrix is {}'.format(self.conn.shape))
# 			warnings.warn('L_0.size was {}, whilst H_x0.size was {}, they should be the same size'.format(self.L_0.shape, self.H_x0.shape))
# 		
# 		# initiate fail_stretches matrix as a linked list format
# 		self.fail_strains = np.full((self.nnodes, self.nnodes), self.s00)
# 		# Make into a sparse matrix
# 		self.fail_strains = sparse.csr_matrix(self.fail_strains)
# 		
# 		if self.v:
# 			print('Type of fail strains is {} and the shape is {}'.format(type(self.fail_strains), self.fail_strains.shape))
# 		
# 		print('Did Element Wise Subtraction in {} seconds'.format(time.time() - st))
# 		print('Constructed H in {} seconds'.format(r_time))
# =============================================================================
		
	
