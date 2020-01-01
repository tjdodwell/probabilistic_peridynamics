import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../Peridynamics')
sys.path.insert(1, '../../PostProcessing')
import PeriParticle as peri
from ParPeri import ParModel as MODEL
import numpy as np
import scipy.stats as sp
import vtk as vtk
import vtu as vtu
from timeit import default_timer as timer
from mpi4py import MPI
class simpleSquare(MODEL):
	# A User defined class for a particular problem which defines all necessary parameters
	def __init__(self,comm):
		self.dim = 2
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
		self.K = 0.05
		self.s00 = 0.005
		self.crackLength = 0.3
		self.c = 18.0 * self.K / (np.pi * (self.horizon**4))
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

	def findBoundary(self,x):
		# Function which markes constrained particles
	    bnd = 0 # Does not live on a boundary
	    if (x[0] < 1.5 * self.horizon):
	        bnd = -1
	    elif (x[0] > 1.0 - 1.5 * self.horizon):
	        bnd = 1
	    return bnd

	def distance2Boundary(self,x):
		d1 = np.abs(x[0])
		d2 = np.abs(1. - x[0])
		return min(d1,d2)

	def initialiseCrack(self, broken, damage):
		for i in range(0, self.numlocalNodes):
			id = self.net[i].id
			family = self.family[i]
			count = 0
			for k in range(0, len(family)):
				j = self.family[i][k]
				if(self.isCrack(self.coords[id,:], self.coords[j,:])):
					broken[i][k] = 1
					count += 1
			damage[i] = float(count / len(family))
		return broken, damage
	
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
		""" Fn for taking a single multivar normal sample covariance matrix with cholisky factor, L
		"""
		zeta = np.random.normal(0, 1, size = num_nodes)
		zeta = np.transpose(zeta)
		w_tild = np.dot(L, zeta) #vector
		return w_tild
	
def noise(L, samples, num_nodes):
		""" Takes multiple samples from multivariate normal distribution with covariance matrix whith Cholesky factor, L
		"""
		noise = []

		for i in range(samples):
			noise.append(multivar_normal(L, num_nodes))

		return np.transpose(noise)

def sim(sample, myModel, numSteps = 400, sigma = 1e-5, loadRate = 0.00001, dt = 1e-3, print_every = 1, noise_mode = 2):
	
	print("Peridynamic Simulation -- Starting")
	
	# MPI stuff
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	comm_Size = comm.Get_size()
	
	u = []
	damage = []
	# Setup broken flag
	broken = []
	for i in range(0, myModel.numlocalNodes):
		broken.append(np.zeros(len(myModel.family[i])))
	u.append(np.zeros((myModel.nnodes, 3)))
	damage.append(np.zeros(myModel.numlocalNodes))
	broken, damage[0] = myModel.initialiseCrack(broken, damage[0])
	verb = 0
	if(myModel.comm.Get_rank()):
		verb = 1
	time = 0.0;
	# Need to sort this out - but hack for now need local boundary ids
	lhsLocal = []
	rhsLocal = []
	for i in range(0, myModel.numlocalNodes):
		# The boundary
		bnd = myModel.findBoundary(myModel.coords[myModel.l2g[i],:])
		if(bnd < 0):
			lhsLocal.append(i)
		elif(bnd > 0):
			rhsLocal.append(i)
			

	# TODO: Tidy up this bit of the code
	
	# Total number of nodes
	nnodes = myModel.nnodes
	
	# Amplification factor
	sigma = 1e-4
	
	# Covariance matrix
	K = myModel.K
	
	# Cholesky decomposition of K
	C = myModel.C
	
	
	for t in range(1, numSteps):
		time += dt;
		# Communicate Ghost particles to required processors
		u[t-1] = myModel.communicateGhostParticles(u[t-1])
		
		damage.append(np.zeros(myModel.numlocalNodes))
		broken, damage[t] = myModel.checkBonds(u[t-1], broken, damage[t-1])
		f = myModel.computebondForceNew(u[t-1], broken)
		
		# Simple Euler update of the Solution + Add the Stochastic Random Noise
		u.append(np.zeros((myModel.nnodes, 3)))
		
		# Depending on the type of noise we are adding (tbc as maths develops)
		if noise_mode == 0:
			
			u[t][myModel.l2g,:] = u[t-1][myModel.l2g,:] + dt * f[myModel.l2g]
			
		elif noise_mode == 1:
			
			u[t][myModel.l2g,:] = u[t-1][myModel.l2g,:] + dt * f[myModel.l2g] +  np.random.normal(loc = 0.0, scale = sigma, size = (myModel.numlocalNodes, 3))
			
		elif noise_mode == 2:
			# TODO doesn't seem to be working, either really slow or idling
			# Generate the noise vector.
			# for the case when it is NOT independent
			
			#initializing variables. mpi4py requires that we pass numpy objects.
			noise_vector = np.zeros((myModel.nnodes, 3))
			force_vector = np.zeros((myModel.nnodes, 3))

			# Communication
			if rank ==0:
				# Full bond force vector is required for generating forcing term,
				# as force term is a linear combination of all other forces
				
				f = myModel.communicateGhostParticles(f)
				
				# Calculate force_vector, using linear combination 
				force_vector = np.dot(K, f)
				
				# noise term
				noise_vector = noise(C, 3, nnodes)
				local_noise_vector = noise_vector[myModel.l2g]
				
				comm.Send(noise_vector, tag = 1)
				comm.Send(force_vector, tag = 2)
				
			else:
			
				# Receieve the noise vector
				
				comm.Recv(noise_vector, source = 0, tag = 1)
				comm.Recv(force_vector, source = 0, tag = 2)
				
				local_noise_vector = noise_vector[myModel.l2g]
				local_force_vector = force_vector[myModel.l2g]
				
			u[t][myModel.l2g,:] = u[t-1][myModel.l2g,:] + dt * local_force_vector + local_noise_vector #exponential length squared kernel

		# Apply boundary conditions

		u[t][myModel.lhs,1:3] = np.zeros((len(myModel.lhs),2))
		u[t][myModel.rhs,1:3] = np.zeros((len(myModel.rhs),2))
		u[t][myModel.lhs,0] = - 0.5 * t * loadRate * np.ones(len(myModel.lhs))
		u[t][myModel.rhs,0] = 0.5 * t * loadRate * np.ones(len(myModel.rhs))
		if(t % print_every == 0) :
			print('Timestep {} complete'.format(t))
			vtu.writeParallel("U_" + str(t),myModel.comm, myModel.numlocalNodes, myModel.coords[myModel.l2g,:], damage[t], u[t][myModel.l2g,:])
def main():
	""" Stochastic Peridynamics, takes multiple stable states (fully formed cracks)
	"""
	# TODO: implement dynamic time step based on strain energy?
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	comm_Size = comm.Get_size()
	thisModel = simpleSquare(comm)
	no_samples = 1
	for s in range(no_samples):
		sim(s,thisModel)
  
main()