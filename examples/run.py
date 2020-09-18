"""
List and run PeriPy examples.

One can optionally supply the name of the example and any additional arguments.
Source: https://github.com/pypr/pysph/blob/master/pysph/examples/run.py

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
import argparse
import ast
import os
import sys
import subprocess

HERE = os.path.dirname(__file__) or '.'


def _exec_file(filename):
    ns = {'__name__': '__main__', '__file__': filename}
    co = compile(open(filename, 'rb').read(), filename, 'exec')
    exec(co, ns)


def _extract_full_doc(filename):
    p = ast.parse(open(filename, 'rb').read())
    return ast.get_docstring(p)


def _extract_short_doc(dirname, fname):
    return open(os.path.join(dirname, fname)).readline()[3:].strip()


def _get_module(fname):
    start = fname
    parts = ['peripy.examples']
    while os.path.dirname(start) != '':
        dirname, start = os.path.split(start)
        parts.append(dirname)
    return '.'.join(parts + [start[:-3]])


def example_info(module, filename):
    """Print example information."""
    print("Information for example: %s" % module)
    print(_extract_full_doc(filename))


def get_all_examples():
    """Get all of the examples."""
    basedir = HERE
    examples = []
    _ignore = [['run.py'],
               ['example3'],
               ['example3', 'profiling'],
               ['example3', 'damage models'],
               ['example3', 'SS'],
               ['example3', 'profiling', 'txt'],
               ['example4']]
    ignore = [os.path.abspath(os.path.join(basedir, *pth))
              for pth in _ignore]
    for dirpath, dirs, files in os.walk(basedir):
        rel_dir = os.path.relpath(dirpath, basedir)
        if rel_dir == '.':
            rel_dir = ''
        py_files = [x for x in files
                    if x.endswith('.py') and not x.startswith('_')]
        data = []
        for f in py_files:
            path = os.path.join(rel_dir, f)
            full_path = os.path.join(basedir, path)
            dirname = os.path.dirname(full_path)
            full_dirname = os.path.join(basedir, dirname)
            if ((os.path.abspath(full_path) in ignore)
                    or (os.path.abspath(full_dirname) in ignore)):
                continue
            module = _get_module(path)
            doc = _extract_short_doc(dirpath, f)
            data.append((module, doc))
        examples.extend(data)
    return examples


def get_input(prompt):
    """Get input (python 3 required)."""
    return input(prompt)


def get_path(module):
    """Return the path to the module filename given the module."""
    x = module[len('peripy.examples.'):].split('.')
    x[-1] = x[-1] + '.py'
    return os.path.join(HERE, *x)


def guess_correct_module(example):
    """
    Given some form of the example name guess and return a reasonable module.

    Examples
    --------
    >>> guess_correct_module('example1')
    'peripy.examples.example1.example'
    >>> guess_correct_module('peripy.examples.example1')
    'peripy.examples.example1.example'
    >>> guess_correct_module('examples.example1')
    'peripy.examples.example1.example'
    >>> guess_correct_module('example1/example.py')
    'peripy.examples.example1.example'
    >>> guess_correct_module('example1/example')
    'peripy.examples.example1.example'
    """
    if example.endswith('.py'):
        example = example[:-3]
    if not example.endswith('example'):
        example = example + '.example'
    example = example.replace('/', '.')
    if example.startswith('examples.'):
        module = 'peripy.' + example
        print(module, '1')
    elif not example.startswith('peripy.examples.'):
        module = 'peripy.examples.' + example
        print(module, '2')
    else:
        module = example
        print(module, '3')
    return module


def cat_example(module):
    """Cat example."""
    filename = get_path(module)
    print("# File: %s" % filename)
    print(open(filename).read())


def list_examples(examples):
    """List example."""
    for idx, (module, doc) in enumerate(examples):
        print("%d. %s" % (idx + 1, module[len('peripy.examples.'):]))
        print("   %s" % doc)


def run_command(module, args):
    """Run command."""
    print("Running example %s.\n" % module)
    filename = get_path(module)
    if '-h' not in args and '--help' not in args:
        example_info(module, filename)
    subprocess.call(
                ["python",
                 filename]
                +
                args)


def main(argv=None):
    """Run a PeriPy example."""
    if argv is None:
        argv = sys.argv[1:]
    examples = get_all_examples()
    parser = argparse.ArgumentParser(
        prog="run", description=__doc__.split("\n\n")[0], add_help=False
    )
    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", default=False, dest="list",
        help="List examples"
    )
    parser.add_argument(
        "--cat", action="store_true", default=False, dest="cat",
        help="Show/cat the example code on stdout"
    )
    parser.add_argument(
        "args", type=str, nargs="?",
        help='''optional example name (for example both cavity or
        peripy.examples.cavity will work) and arguments to the example.'''
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)
    if options.list:
        return list_examples(examples)
    if options.cat:
        module = guess_correct_module(options.args)
        return cat_example(module)
    if len(argv) > 0:
        module = guess_correct_module(argv[0])
        run_command(module, argv[1:])
    else:
        list_examples(examples)
        try:
            ans = int(get_input("Enter example number you wish to run: "))
        except ValueError:
            ans = 0
        if ans < 1 or ans > len(examples):
            print("Invalid example number, exiting!")
            sys.exit()

        args = str(get_input(
            "Enter additional arguments (leave blank to skip): "
        ))
        module, doc = examples[ans - 1]
        print("-" * 80)
        run_command(module, args.split())


if __name__ == '__main__':
    main()
