"""Setup script for peridynamics."""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extra_compile_args = ['-O3', '-fopenmp']
extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "peridynamics.neighbour_list",
        ["peridynamics/neighbour_list.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
        ),
    Extension(
        "peridynamics.peridynamics",
        ["peridynamics/peridynamics.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
        ),
    Extension(
        "peridynamics.spatial",
        ["peridynamics/spatial.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
        )
    ]


setup(
    name="peridynamics",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'meshio',
        'numpy',
        'scipy',
        'tqdm'
        ]
    )
