"""
Convenience script for running various PeriPy related tasks.

Source:https://github.com/pypr/pysph/blob/master/pysph/tools/cli.py
Copyright (c) 2009-2015, the PySPH developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from argparse import ArgumentParser
import sys


def run_examples(args):
    """Run examples from the command line."""
    from examples.run import main
    main(args)


def run_tests(args):
    """Run tests from the command line."""
    argv = ['--pyargs', 'peripy'] + args
    from pytest import cmdline
    cmdline.main(args=argv)


def run_coverage(args):
    """Run tests from the command line."""
    argv = ['--pyargs', '--cov=peripy'] + args
    from pytest import cmdline
    cmdline.main(args=argv)


def main():
    """Scipt for running PeriPy related tasks."""
    parser = ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )
    subparsers = parser.add_subparsers(help='sub-command help')

    runner = subparsers.add_parser(
        'run', help='Run PeriPy examples',
        add_help=False
    )
    runner.set_defaults(func=run_examples)

    tests = subparsers.add_parser(
        'test', help='Run entire PeriPy test-suite',
        add_help=False
    )
    tests.set_defaults(func=run_tests)

    coverage = subparsers.add_parser(
        'coverage', help='Run entire PeriPy test-suite with coverage report',
        add_help=False
    )
    coverage.set_defaults(func=run_coverage)

    if (len(sys.argv) == 1 or (len(sys.argv) > 1 and
                               sys.argv[1] in ['-h', '--help'])):
        parser.print_help()
        sys.exit()

    args, extra = parser.parse_known_args()
    args.func(extra)


if __name__ == '__main__':
    main()
