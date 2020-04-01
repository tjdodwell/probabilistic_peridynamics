"""Setup script for peridynamics."""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

setup(
    name="peridynamics",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(
        Extension(
            "peridynamics.neighbour_list",
            ["peridynamics/neighbour_list.pyx"],
            extra_compile_args=['-O3', '-fopenmp'],
            extra_link_args=['-fopenmp'],
            )
        ),
    setup_requires=[
        'cython'
        ],
    install_requires=[
        'meshio',
        'numpy',
        'scipy',
        'tqdm'
        ]
    )
