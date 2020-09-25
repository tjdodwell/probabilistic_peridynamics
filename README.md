PeriPy
======

[![Build Status](https://travis-ci.com/alan-turing-institute/PeriPy.svg?branch=master)](https://travis-ci.com/alan-turing-institute/PeriPy)
[![codecov](https://codecov.io/gh/alan-turing-institute/PeriPy/branch/master/graph/badge.svg)](https://codecov.io/gh/alan-turing-institute/PeriPy)

PeriPy, a collaboration between Exeter, Cambridge &amp; the Alan Turing Institute, is a lightweight, open-source and high-performance python package for solving bond-based peridynamics (BBPD) problems in solid mechanics. It is implemented in [Python](https://www.python.org/) and the performance critical parts are implemented in [Cython](https://cython.org/) and [PyOpenCL](https://documen.tician.de/pyopencl/).

PeriPy allows users to write their code in pure Python. Simulations are then executed seamlessly using high performance OpenCL code.

Features
--------
- Easy to use. Get started with the latest documentation at [peripy.readthedocs.org](https://peripy.readthedocs.org)
- 2-5x faster than exisiting OpenCL solvers
- 'Outer-loop' applications including uncertainty quantification, optimisation and feature recognition are made possible
- Support for both regular and irregular mesh files. See [meshio](https://github.com/nschloe/meshio) for the full list of mesh formats
- Support for composite and interface material models
- Support for arbritrary n-linear 'microelastic' damage models
- Simulates force or displacement controlled boundary conditions and initial conditions
- Arbritrary subsets of particles are easily measured for their displacements, damages etc.
- Output files can be viewed in [Paraview](https://www.paraview.org/)
- Various 'partial volume correction' algorithms, 'surface correction' algorithms and 'micromodulus functions' are included. The code is easily extended to define your own
- Velocity-Verlet, Euler and Euler-Cromer integrators are included and the code is easily extended to define your own higher order and/or adaptive integrators


Get started (preferred)
-----------------------

### Building and Installation ###

- The package requires Python 3.7+
- Install cython, a build dependency, `pip install cython`
- Install PeriPy `pip install peripy`

### Running examples ###

- Run the first example by typing `peripy run example1` on the command line
- You can show the example code by typing `peripy run example1 --cat`
- Type `peripy run --list` for a list of examples
- For usage, type `peripy run --help`

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/). To run
the tests yourself,

- Install pytest using pip `pip install pytest`
- Type `peripy test` on the command line
- For coverage install `pytest-cov` and type `peripy coverage` on the command line

Get started from the GitHub repository (for developers)
-------------------------------------------------------

### Building and Installation ###

- The package requires Python 3.7+
- Clone the repository `git clone
  git@github.com:alan-turing-institute/probabilistic_peridynamics.git`
- Install cython, a build dependency, `pip install cython`
- Install using pip `pip install . -e` from the root directory of the repository

### Running examples ###

- You can find examples of how to use the package under:`examples/`. Run the first example by typing `python examples/example1/example.py`

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/). To run
the tests yourself,

- Install using pip `pip install -e .` from the root directory of the repository
- Install pytest using pip `pip install pytest`
- Run `pytest` from the root directory of the repository
- For coverage install `pytest-cov` and run `pytest --cov=./peripy`
