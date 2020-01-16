from setuptools import setup, find_packages

setup(
    name="peridynamics",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numba',
        'numpy',
        'scipy'
        ]
    )
