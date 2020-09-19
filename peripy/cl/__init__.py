"""OpenCL peridynamics implementation."""
from .utilities import double_fp_support, get_context, output_device_info
import pathlib

kernel_source_files = [
    pathlib.Path(__file__).parent.absolute()/source for source in [
        "peridynamics.cl",
        "euler.cl",
        "euler_cromer.cl",
        "velocity_verlet.cl"
        ]
    ]

kernel_source = "".join(
    [open(source).read() for source in kernel_source_files]
    )

__all__ = ["kernel_source", "double_fp_support", "get_context",
           "output_device_info"]
