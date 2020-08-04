"""A subprocess to compare the performance across different problem sizes."""
import subprocess

beams = ['1650beam792t.msh',
         '1650beam2652t.msh',
         '1650beam3570t.msh',
         '1650beam4095t.msh',
         '1650beam6256t.msh']
with open("profiling.txt", "w+") as output:
    for beam in beams:
        subprocess.call(
            ["python", "examples/example2/example.py", beam, "--profile",
             "--opencl"], stdout=output)
