"""A subprocess to compare the performance across different problem sizes."""
import subprocess

beams = ['1650beam792.msh',
         '1650beam2652.msh',
         '1650beam3570.msh',
         '1650beam4095.msh',
         '1650beam6256.msh']
with open("profiling.txt", "w+") as output:
    for beam in beams:
        subprocess.call(
            ["python", "examples/example2/example.py", beam, "--profile",
             "--opencl"], stdout=output)
