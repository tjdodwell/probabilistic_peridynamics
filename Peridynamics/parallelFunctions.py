# A collection of functions for implementing peridynamics in parallel

import numpy as np

def decomposeDomain(coords, connectivity, M, partitionType):
	""" Simple domain decomposition
	Inputs: node coords, connectivity matrix, number of processes (M)
	Output: partition (array length nnodes of which paritition each node is in), nearestNeighbour (a list of lists, containing the partition neighbours for each partition, e.g. in X only, [[1], [0, 2],[1,3],...,[M-2]]
	"""
	part = []
	if partitionType == 1:
		# Simple Domain Decomposition for now in X only - Probably works well for examples considered 
		maxX = np.amax(coords[:,0])
		dX = (maxX/ M) + 1e-6
		for i in range(0, coords[:,0].size):
			part.append(np.floor(coords[i,0]/ dX))
			
		# Constructs nearest neighbour list for each subdomain. Simnple in this case
		nearestNeighbour  = []
		nearestNeighbour.append([1])
		for i in range(1, M-1):
			nearestNeighbour.append([i-1, i+1])
		nearestNeighbour.append([M-2])
		
    # if (partitionType == 2):
    #
    #     M0 = int(np.sqrt(M))
    #
    #     maxX = np.amax(coords[:,0])
    #     maxY = np.amax(coords[:,1]
    #
    #     dX = (maxX / M0) + 1e-6
    #     dY = (maxY / M0) + 1e-6
    #
    #     for i in range(0,coords[:,0].size):
    #         iPart = np.floor(coords[i,0] / dX)
    #         jPart = np.floor(coords[i,1] / dY)
	
	return part, nearestNeighbour
