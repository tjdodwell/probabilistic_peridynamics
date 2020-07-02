"""OpenCL peridynamics implementation."""
from .utilities import double_fp_support, get_context, pad
import pathlib

kernel_source_files = [
    pathlib.Path(__file__).parent.absolute()/source for source in [
        "euler.cl"
        ]
    ]

kernel_source = "".join(
    [open(source).read() for source in kernel_source_files]
    )

__all__ = ["kernel_source", "double_fp_support", "get_context", "pad"]
