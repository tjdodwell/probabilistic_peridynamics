# Probabilistic Peridynamics

Code base for 'Probabilistic Peridynamics' collaboration between Exeter, Cambridge &amp; Turing

## Dependencies

*  Numba - http://numba.pydata.org
*  mpi4py - https://mpi4py.readthedocs.io (for parallel runs only)
*  paraview - https://www.paraview.org (for visualisation)
*  Gmsh - http://gmsh.info (for generating initial geometry and particle distribution)

First two can be installed using 

`pip install <package>`

Paraview you get free from website given above

## Getting code & Running an example

You can get the code using git

`git clone https://git.exeter.ac.uk/td336/stochastic_peridynamics.git`

to run a simple sequential test - currently only code working - from main directory

`
cd Examples/Example1
`

followed by

`
python3 main.py
`

## Defining Geometry of problem

The code has been set up to use a finite element mesh as it's base geometry - where the nodes of the finite element mesh are converted into peridynamic particles. This allows the user to define reasonably complex geometries using standard meshing tools. In our case we have built .msh with Gmsh. See link above for information on their website
