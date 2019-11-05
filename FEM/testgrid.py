import grid as fem

# test grid.py


myGrid = fem.Grid()

L = [10, 10]

n = [5, 5]

order = 1

X0 = [0.0, 0.0]

myGrid.buildStructuredMesh2D(L,n,X0,order)
