"""A subprocess to compare the performance across different problem sizes."""
import subprocess

beams = ['1650beam247500.msh',
	 '1650beam247500t.msh']
with open("profiling.txt", "w+") as output:
    for beam in beams:
        subprocess.call(
            ["python", "examples/example2/example.py", beam, "--profile",
             "--opencl"], stdout=output)
