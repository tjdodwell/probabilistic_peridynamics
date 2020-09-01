"""Setup script for peridynamics."""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extra_compile_args = ['-O3']
extra_link_args = []

ext_modules = [
    Extension(
        "peripy.create_crack",
        ["peripy/create_crack.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
        ),
    Extension(
        "peripy.peridynamics",
        ["peripy/peridynamics.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
        ),
    Extension(
        "peripy.spatial",
        ["peripy/spatial.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
        )
    ]


setup(
    name="peripy",
    version="0.1",
    description="A fast OpenCL Peridynamics package for python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/alan-turing-institute/Probabilistic-Peridynamics",
    author="Jim Madge, Ben Boys, Tim Dodwell, Greg Mingas",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
    install_requires=[
        'meshio',
        'numpy',
        'pyopencl==2020.1',
        'scipy',
        'tqdm',
        'h5py',
        'sklearn'
        ]
    )
