"""OpenCL peridynamics implementation."""
from .utilities import pad
import pathlib

kernel_source_file = pathlib.Path(__file__).parent.absolute()/"peridynamics.cl"
kernel_source = open(kernel_source_file).read()

__all__ = ["kernel_source", "pad"]
