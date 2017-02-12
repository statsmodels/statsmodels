#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys
import argparse
from functools import partial
try:
    from concurrent import futures
    par = True
except ImportError:
    par = False


import nbformat
from nbconvert import HTMLExporter, RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor


EXAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "examples"))
SOURCE_DIR = os.path.join(EXAMPLE_DIR, "notebooks")
EXECUTED_DIR = os.path.join(EXAMPLE_DIR, "executed")
DST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "docs", "source", "examples",
                                       "notebooks", "generated"))
for dir in [EXECUTED_DIR, DST_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)


def execute_nb(src, dst, allow_errors=False, timeout=1000, kernel_name=None):
    '''
    Execute notebook in `src` and write the output to `dst`

    Parameters
    ----------
    src, dst: str
        path to notebook
    allow_errors: bool
    timeout: int
    kernel_name: str
        defualts to value set in notebook metadata

    Returns
    -------
    dst: str
    '''
    with io.open(src, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(allow_errors=allow_errors,
                             timeout=timeout,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadta': {'path': 'notebooks/'}})

    with io.open(dst, 'wt', encoding='utf-8') as f:
        nbformat.write(nb, f)
    return dst

def convert(src, dst, to='rst'):
    '''
    Convert a notebook `src`.

    Parameters
    ----------
    src, dst: str
        filepaths
    to: {'rst', 'html'}
        format to export to
    '''
    dispatch = {'rst': RSTExporter, 'html': HTMLExporter}
    exporter = dispatch[to.lower()]()

    (body, resources) = exporter.from_filename(src)
    with io.open(dst, 'wt', encoding='utf-8') as f:
        f.write(body)
    return dst

def find_notebooks(directory=None):
    if directory is None:
        directory = SOURCE_DIR
    nbs = (os.path.join(SOURCE_DIR, x)
           for x in os.listdir(SOURCE_DIR)
           if x.endswith('.ipynb'))
    return nbs

def do_one(nb, to=None, execute=None, allow_errors=None, timeout=None, kernel_name=None):
    from traitlets.traitlets import TraitError
    import jupyter_client

    name = os.path.basename(nb)
    if execute:
        dst = os.path.join(EXECUTED_DIR, name)
        print("Executeing %s to %s" % (nb, dst))
        nb = execute_nb(nb, dst, allow_errors=allow_errors, timeout=timeout,
                        kernel_name=kernel_name)
    dst = os.path.splitext(os.path.join(DST_DIR, name))[0] + '.' + to
    print("Converting %s to %s" % (nb, dst))
    try:
        convert(nb, dst, to=to)
    except TraitError:
        kernels = jupyter_client.kernelspec.find_kernel_specs()
        msg = ('Could not find kernel named `%s`, Available kernels:\n %s'
               % kernel_name, kernels)
        raise ValueError(msg)
    return dst

def do(fp=None, directory=None, to='html', execute=True,
       allow_errors=True, timeout=1000, kernel_name=''):
    if fp is None:
        nbs = find_notebooks(directory)
    else:
        nbs = [fp]

    if kernel_name is None:
        kernel_name = find_kernel_name()

    func = partial(do_one, to=to, execute=execute, allow_errors=allow_errors,
                   timeout=timeout, kernel_name=kernel_name)
    if par:
        with futures.ProcessPoolExecutor() as pool:
            for dst in pool.map(func, nbs):
                print("Finished %s" % dst)
    else:
        for nb in nbs:
            func(nb)
            print("Finished %s" % nb)


def find_kernel_name():
    import jupyter_client

    kernels = jupyter_client.kernelspec.find_kernel_specs()
    kernel_name = 'python%s' % sys.version_info.major
    if kernel_name not in kernels:
        return ''
    return kernel_name


parser = argparse.ArgumentParser(description="Process example notebooks")
parser.add_argument("--fp", type=str, default=None,
                    help="Path to notebook to convert. Converts all notebooks "
                         "in `directory` by default.")
parser.add_argument("--directory", type=str, default=None,
                    help="Path to notebook directory to convert")
parser.add_argument("--to", type=str, default="html",
                    help="Type to convert to. One of `{'html', 'rst'}`")
parser.add_argument("--execute", type=bool, default=True,
                    help="Execute notebook before converting")
parser.add_argument("--allow_errors", type=bool, default=True,
                    help="Allow errors while executing")
parser.add_argument("--timeout", type=int, default=1000,
                    help="Seconds to allow for each cell before timing out")
parser.add_argument("--kernel_name", type=str, default=None,
                    help="Name of kernel to execute with")

def main():
    args = parser.parse_args()
    do(fp=args.fp, directory=args.directory, to=args.to, execute=args.execute,
       allow_errors=args.allow_errors, timeout=args.timeout,
       kernel_name=args.kernel_name)

if __name__ == '__main__':
    main()

