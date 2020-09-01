PeriPy
======

[![Build Status](https://travis-ci.com/alan-turing-institute/Probabilistic-Peridynamics.svg?branch=master)](https://travis-ci.com/alan-turing-institute/Probabilistic-Peridynamics)
[![codecov](https://codecov.io/gh/alan-turing-institute/Probabilistic-Peridynamics/branch/master/graph/badge.svg)](https://codecov.io/gh/alan-turing-institute/Probabilistic-Peridynamics)

PeriPy, a collaboration between Exeter, Cambridge &amp; the Alan Turing Institute.

Building and Installation
-------------------------

- The package requires Python 3.7+
- Clone the repository `git clone
  git@github.com:alan-turing-institute/probabilistic_peridynamics.git`
- Install cython, a build dependency, `pip install cython`
- Install using pip `pip install .` from the root directory of the repository

Examples
--------

- You can find an examples of how to use the package under:`examples/`. Run the first example by typing `python examples/example1/example.py`

Running the tests
-----------------

The tests for this project use [pytest](https://pytest.org/en/latest/). To run
the tests yourself,

- Install using pip `pip install -e .` from the root directory of the repository
- Install pytest using pip `pip install pytest`
- Run `pytest` from the root directory of the repository
- For coverage install `pytest-cov` and run `pytest --cov=./peripy`
