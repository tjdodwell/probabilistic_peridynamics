"""A subprocess to compare the performance across different problem sizes."""
import subprocess

beams = ['1650beam74800.msh',
         '1650beam74800transfinite.msh',
         '1650beam144900.msh',
         '1650beam144900transfinite.msh',
         '1650beam247500.msh',
         '1650beam247500transfinite.msh']
with open("profiling.txt", "w+") as output:
    for beam in beams:
        subprocess.call(
            ["python", "examples/example3/example.py", beam, "--profile"],
            stdout=output)
