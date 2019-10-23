
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../Peridynamics')
sys.path.insert(1, '../../PostProcessing')

import PeriParticle as peri
from SeqPeri import SeqModel as MODEL
import numpy as np
import scipy.stats as sp
import vtk as vtk
from timeit import default_timer as timer
import time
from mpi4py import MPI
import sys


class simpleSquare(MODEL):

	# A User defined class for a particular problem which defines all necessary parameters

	def __init__(self):

		self.dim = 2

		self.meshFileName = 'test.msh'

		self.meshType = 2
		self.boundaryType = 1
		self.numBoundaryNodes = 2
		self.numMeshNodes = 3

		# Material Parameters from classical material model
		self.horizon = 0.1 # Note: this is large compared 
		self.K = 0.05
		self.s00 = 0.005

		self.crackLength = 0.3

		self.readMesh(self.meshFileName)

		self.setNetwork(self.horizon)

		self.lhs = []
		self.rhs = []
		self.nofail = []

		# Find the Boundary and No fail zone
		for i in range(0, self.nnodes):
			bnd = self.findBoundary(self.coords[i][:])
			nfl = self.noFail(self.coords[i][:])
			if (nfl == 1): # In no failure zone
				(self.nofail).append(i)
			if (bnd < 0): # Left hand boundary
				(self.lhs).append(i)
			elif (bnd > 0): # Right hand boundary
				(self.rhs).append(i)
			
				

	def findBoundary(self,x):
		# Function which markes constrain particles
	    bnd = 0 # Does not live on a boundary
	    if (x[0] < 1.5 * self.horizon):
	        bnd = -1
	    elif (x[0] > 1.0 - 1.5 * self.horizon):
	        bnd = 1
	    return bnd
	
	def noFail(self, x):
		""" Function which marks the no-failure zone for the simulations
			TODO: investigate if this can be avoided with applying a displacement more slowly, or on a curve that does not increase strain energy
		"""
		nfl = 0 # Does not live in nofail zone 
		if (x[0] < 2.5 * self.horizon):
			nfl = 1
		elif (x[0] > 1.0 - 2.5 * self.hohorizon):
			nfl = 1
		return nfl

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
		""" takes multiple samples from multivariate normal distribution with covariance matrix whith cholisky factor, L
		"""
		
		noise = []
	
		for i in range(samples):
			noise.append(multivar_normal(L, num_nodes))
		
		return np.transpose(noise)
	
	
def sim(sample, rank, myModel =simpleSquare(), numSteps = 600, sigma = 8e-6, loadRate = 0.00001, dt = 1e-3, print_every = 600):
	print("Peridynamic Simulation -- Starting")
	
	u = []
	
	damage = []
	
	# Setup broken flag
	
	broken = []
	for i in range(0, myModel.nnodes):
		broken.append(np.zeros(len(myModel.family[i])))
	
	u.append(np.zeros((myModel.nnodes, 3)))
	
	damage.append(np.zeros(myModel.nnodes))
	
	broken, damage[0] = myModel.initialiseCrack(broken, damage[0])
	
	verb = 1
	
	time = 0.0;
	
	
	# Number of nodes
	nnodes = myModel.nnodes
	
	
	# Covariance matrix
	M = myModel.COVARIANCE
	
	#Create L matrix
	#TODO: epsilon, numerical trick so that M is positive semi definite
	epsilon = 1e-4
	
	# Sample a random vector
	
	I = np.identity(nnodes)
	M_tild = M + np.multiply(epsilon, I)
	
	M_tild = np.multiply(pow(sigma, 2), M_tild)
	
	L = np.linalg.cholesky(M_tild)
	
	for t in range(1, numSteps):
	
		time += dt;
	
		if(verb > 0):
			print("Time step = " + str(t) + ", Time = " + str(time))
	
		# Compute the force with displacement u[t-1]
	
		damage.append(np.zeros(nnodes))
	
		broken, damage[t] = myModel.checkBonds(u[t-1], broken, damage[t-1], myModel.nofail)
	
		f = myModel.computebondForce(u[t-1], broken) # myModel.computeForce(u[t-1])
	
		# Simple Euler update of the Solution + Add the Stochastic Random Noise
	
		u.append(np.zeros((nnodes, 3)))
	
		
		#u[t] = u[t-1] + dt * f + np.random.normal(loc = 0.0, scale = sigma, size = (myModel.nnodes, 3)) #Brownian Noise
		u[t] = u[t-1] + dt * np.dot(M,f) + noise(L, 3, nnodes) #exponential length squared kernel
		#u[t] = u[t-1] + dt * f + noise(L, 3, nnodes)
	
		# Apply boundary conditions
		u[t][myModel.lhs,1:3] = np.zeros((len(myModel.lhs),2))
		u[t][myModel.rhs,1:3] = np.zeros((len(myModel.rhs),2))
	
		u[t][myModel.lhs,0] = -0.5 * t * loadRate * np.ones(len(myModel.rhs))
		u[t][myModel.rhs,0] = 0.5 * t * loadRate * np.ones(len(myModel.rhs))
		
		if(verb==1 and t % print_every == 0) :
			vtk.write("U_"+"sample"+str(sample)+"time"+str(t)+".vtk","Solution time step = "+str(t), myModel.coords, damage[t], u[t])
			print('Timestep {} complete'.format(t))
			
	return vtk.write("U_"+ "rank" + str(rank) + "_sample"+str(sample) + ".vtk","Solution time step = "+str(t), myModel.coords, damage[t], u[t])
	
	

	
def main():
	""" Stochastic Peridynamics, takes multiple stable states (fully formed cracks)
        >>>PARALLELIZED SAMPLING
	"""	
	# TODO: implement dynamic time step based on strain energy?

	# MPI4py stuff
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank() # which process
	size= comm.Get_size() # total no. processes 
    
    # read from command line
	n = int(sys.argv[1]) # no. of samples
	
	# test for conformability
	if rank == 0:
		time1 = time.time()
		
		# Currently, our program cannot handle sizes that are not evenly
        # divided by the number of processors
		if (n % size != 0):
			raise Exception('The number of processors should evenly divide the number of samples! \n The number of processes was: {}, the number of samples was: {}'.format(size, n))
	
		# number of samples taken by each process
		#local_n= np.array([n/size]) # TODO: for some reason getting a referenced before assignment error
	# communicate local array size to all processes
	#comm.Bcast(local_n, root=0)
	local_n= np.array([int(n/size)])
	
	# take some samples
	for s in range(local_n[0]):
		sim(sample = s, rank= rank)
	
	if rank == 0:
		time2 = time.time()
		print('Finished rank 0 in time: {}s'.format(time2 - time1))
main()
	
		
