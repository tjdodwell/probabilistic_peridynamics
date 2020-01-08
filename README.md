# Probabilistic Peridynamics

Code base for 'Probabilistic Peridynamics' collaboration between Exeter, Cambridge &amp; Turing

## Dependencies

*  Numba - http://numba.pydata.org
*  mpi4py - https://mpi4py.readthedocs.io (for parallel runs only)
*  paraview - https://www.paraview.org (for visualisation)
*  Gmsh - http://gmsh.info (for generating initial geometry and particle distribution)
*  OpenCL - https://youtu.be/KUTVnxCeC50 (for installing necessary drivers. It is not necessary to own a GPU to run the code."
First two can be installed using 

`pip install <package>`

Paraview you get free from website given above

## Getting code & Running an example

You can get the code using git

`git clone https://git.exeter.ac.uk/td336/stochastic_peridynamics.git`

to run a simple sequential test - from main directory

`
cd Examples/Example1
`

followed by

`
python3 main.py
`

to run a vectorized (fast) sequential test - from main directory

`
cd Examples/Example1
`

followed by

`
python3 main_vectorized.py
`

to run an inital bare bones OpenCL implementation (currently not working)

`
Install the dependencies (tutorial above)
`

from main directory

`
cd Examples/Example1
`

followed by

`
python3 main_OpenCL.py

`
kernels that can be found in 'opencl_peridynamics.cl'
based from the C++ source code by F.Mossaiby et. al
https://figshare.com/articles/Source_code_for_OpenCL_Peridynamics_solver/5097385
from the article
https://www.sciencedirect.com/science/article/pii/S0898122117304030

`
cd Examples/Example1
`

followed by

`
python3 main_vectorized.py
`

## Defining Geometry of problem

The code has been set up to use a finite element mesh as it's base geometry - where the nodes of the finite element mesh are converted into peridynamic particles. This allows the user to define reasonably complex geometries using standard meshing tools. In our case we have built .msh with Gmsh. See link above for information on their website
