"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../Peridynamics')
sys.path.insert(1, '../../PostProcessing')
sys.path.insert(1, '../../FEM')

import PeriParticle as peri
from ParPeriVectorized import ParModel as MODEL
import numpy as np
import scipy.stats as sp
import vtk as vtk
import vtu as vtu
import time
from mpi4py import MPI

#import grid as fem


class simpleSquare(MODEL): # Should I pass comm into the arguments here

	# A User defined class for a particular problem which defines all necessary parameters

	def __init__(self, comm):
		
		# verbose
		self.v = True
		
		self.dim = 2
		
		# domain decomposition and mpi stuff
		self.comm = comm
		self.partitionType = 1 # hard coded for now!
		self.plotPartition = 1
		self.testCode = 1

		self.meshFileName = 'test.msh'
		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3

		# Material Parameters from classical material model
		self.horizon = 0.1
		self.kscalar = 0.05
		self.s00 = 0.005

		self.crackLength = 0.3

		self.readMesh(self.meshFileName)
		self.setNetwork(self.horizon)
		#self.setVolume() # SS by setNetwork
		
		self.lhs = []
		self.rhs = []

		# Find the Boundary maybe only needs to be done on one process then communicated in send recv?
		for i in range(0, self.nnodes):
			bnd = self.findBoundary(self.coords[i][:])
			if (bnd < 0):
				(self.lhs).append(i)
			elif (bnd > 0):
				(self.rhs).append(i)

# =============================================================================
# 		# Build Finite Element Grid Overlaying particles
# 		myGrid = fem.Grid()
# 
# 		self.L = []
# 		self.X0  = [0.0, 0.0] # bottom left
# 		self.nfem = []
# 
# 		for i in range(0, self.dim):
# 			self.L.append(np.max(self.coords[:,i]))
# 			self.nfem.append(int(np.ceil(self.L[i] / self.horizon)))
# 
# 		myGrid.buildStructuredMesh2D(self.L,self.nfem,self.X0,1)
# 
# 		self.p_localCoords, self.p2e = myGrid.particletoCell_structured(self.coords[:,:self.dim])
# =============================================================================

	def findBoundary(self,x):
		# Function which markes constrain particles
	    bnd = 0 # Does not live on a boundary
	    if (x[0] < 1.5 * self.horizon):
	        bnd = -1
	    elif (x[0] > 1.0 - 1.5 * self.horizon):
	        bnd = 1
	    return bnd

	def isCrack(self, x, y):
		output = 0
		p1 = x
		p2 = y
		if(x[0] > y[0]):
			p2 = x
			p1 = y
		if ((p1[0] < 0.5 + 1e-6) and (p2[0] > 0.5 + 1e-6)): # 1e-6 makes it fall one side of central line of particles
		    # draw a straight line between them
			m = (p2[1] - p1[1]) / (p2[0] - p1[0])
			c = p1[1] - m * p1[0]
			height = m * 0.5 + c # height a x = 0.5
			if ((height > 0.5 * (1 - self.crackLength)) and (height < 0.5 * (1 + self.crackLength))):
				output = 1
		return output

def multivar_normal(L, num_nodes):
		""" Fn for taking a single multivar normal sample covariance matrix with Cholesky factor, L
		"""
		zeta = np.random.normal(0, 1, size = num_nodes)
		zeta = np.transpose(zeta)

		w_tild = np.dot(L, zeta) #vector

		return w_tild

def noise(L, samples, num_nodes):
		""" takes multiple samples from multivariate normal distribution with covariance matrix whith Cholesky factor, L
		"""

		noise = []

		for i in range(samples):
			noise.append(multivar_normal(L, num_nodes))

		return np.transpose(noise)


def sim(sample, myModel, numSteps = 400, numSamples = 1, sigma = 1e-5, loadRate = 0.00001, dt = 1e-3, print_every = 10):
	print("Peridynamic Simulation -- Starting")
	
	#myModel.setConnPar(0.1) # May only need to set connectivity matrix up for each node
	myModel.setH()

	u = []

	damage = []


	u.append(np.zeros((myModel.nnodes, 3)))

	damage.append(np.zeros(myModel.numlocalNodes))

	verb = 0
	if(myModel.comm.Get_rank()):
		verb = 1
	
	# Need to sort this out - but hack for now need local boundary ids
	lhsLocal = []
	rhsLocal = []
	
	for i in range(0, myModel.numlocalNodes):
		# The boundary
		bnd = myModel.findBoundary(myModel.coords[myModel.l2g[i],:])
		if bnd < 0:
			lhsLocal.append(i)
		elif bnd > 0:
			rhsLocal.append(i)
	

	tim = 0.0;


	# Number of nodes
	nnodes = myModel.nnodes

	# Covariance matrix
	K = myModel.K
	
	# Cholesky decomposition of K
	L = myModel.L
	
	# Start the clock
	st = time.time()
	
	
	for t in range(1, numSteps):
		
		tim += dt;

		if(verb > 0):
			print("Time step = " + str(t) + ", Wall clock time for last time step= " + str(time.time() - st))
		
		st = time.time()
		
		# Communicate Ghost particles to required processors
		u[t-1] = myModel.communicateGhostParticles(u[t-1])
		gt = time.time()
		damage.append(np.zeros(myModel.numlocalNodes))

		myModel.calcBondStretch(u[t-1])
		damage[t] = myModel.checkBonds()
		f = myModel.computebondForce()

		# Simple Euler update of the Solution + Add the Stochastic Random Noise
		u.append(np.zeros((nnodes, 3)))
		
		# The nodes that are in myModel.l2g only are updated, i.e. nodes for each process
		u[t][myModel.l2g,:] = u[t-1][myModel.l2g,:] + dt*f[myModel.l2g,:] # + noise terms

		# Apply boundary conditions
		u[t][myModel.lhs,1:3] = np.zeros((len(myModel.lhs),2))
		u[t][myModel.rhs,1:3] = np.zeros((len(myModel.rhs),2))

		u[t][myModel.lhs,0] = -0.5 * t * loadRate * np.ones(len(myModel.rhs))
		u[t][myModel.rhs,0] = 0.5 * t * loadRate * np.ones(len(myModel.rhs))

		if(t % print_every == 0) :
			vtu.writeParallel("U_" + str(t),myModel.comm, myModel.numlocalNodes, myModel.coords[myModel.l2g,:], damage[t], u[t][myModel.l2g,:])			
		print('Timestep {} time to communicate ghost particles {} s '.format(t, gt - st))
		print('Timestep {} complete in {} s '.format(t, time.time() - st))
	return vtk.write("U_"+"sample"+str(sample)+".vtk","Solution time step = "+str(t), myModel.coords, damage[t], u[t])




def main():
	""" Stochastic Peridynamics, samples multiple stable states (fully formed cracks)
	"""
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	comm_Size = comm.Get_size()
	st = time.time()
	thisModel = simpleSquare(comm)
	no_samples = 1
	for s in range(no_samples):
		sim(s, thisModel)
	print('TOTAL TIME REQUIRED FOR PROCESS {} WAS {}s'.format(rank, time.time() - st))
main()
