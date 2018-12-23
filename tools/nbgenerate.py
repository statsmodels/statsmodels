#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import io
import os
import sys

from colorama import init, Fore

from functools import partial

try:
    from concurrent import futures

    has_futures = True
except ImportError:
    has_futures = False

import nbformat
from nbconvert import HTMLExporter, RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor

init()

here = os.path.dirname(__file__)
pkgdir = os.path.split(here)[0]
EXAMPLE_DIR = os.path.abspath(os.path.join(pkgdir, "examples"))
SOURCE_DIR = os.path.join(EXAMPLE_DIR, "notebooks")
EXECUTED_DIR = os.path.join(EXAMPLE_DIR, "executed")
DST_DIR = os.path.abspath(os.path.join(pkgdir,
                                       "docs", "source", "examples",
                                       "notebooks", "generated"))

error_message = """
******************************************************************************
ERROR: Error occured when running {notebook}
{exception}
{message}
******************************************************************************
"""
for dname in [EXECUTED_DIR, DST_DIR]:
    if not os.path.exists(dname):
        os.makedirs(dname)


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

    ep = ExecutePreprocessor(allow_errors=False,
                             timeout=timeout,
                             kernel_name=kernel_name)
    ep.preprocess(nb, {'metadata': {'path': SOURCE_DIR}})

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
    nbs = (os.path.join(directory, x)
           for x in os.listdir(directory)
           if x.endswith('.ipynb'))
    return nbs


def do_one(nb, to=None, execute=None, timeout=None, kernel_name=None,
           report_error=True, error_fail=False, skip_existing=False):
    from traitlets.traitlets import TraitError
    import jupyter_client

    os.chdir(SOURCE_DIR)
    name = os.path.basename(nb)
    dst = os.path.join(EXECUTED_DIR, name)
    update_needed = True
    if skip_existing and os.path.exists(dst):
        update_needed = (os.path.getmtime(dst) <= os.path.getmtime(nb))
        if not update_needed:
            print('Skipping {0}'.format(nb))
            return dst

    if execute and update_needed:
        print("Executing %s to %s" % (nb, dst))
        try:
            nb = execute_nb(nb, dst, timeout=timeout, kernel_name=kernel_name)
        except Exception as e:
            if report_error:
                print(Fore.RED + error_message.format(notebook=nb,
                                                      exception=str(e),
                                                      message=str(e.args[0])))
                print(Fore.RESET)
            if error_fail:
                raise

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


def do(fp=None, directory=None, to='html', execute=True, timeout=1000,
       kernel_name='', parallel=False, report_errors=True, error_fail=False,
       skip_existing=False):
    if fp is None:
        nbs = find_notebooks(directory)
    else:
        nbs = [fp]

    if kernel_name is None:
        kernel_name = find_kernel_name()

    func = partial(do_one, to=to,
                   execute=execute, timeout=timeout, kernel_name=kernel_name,
                   report_error=report_errors, error_fail=error_fail,
                   skip_existing=skip_existing)

    if parallel and has_futures:
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
parser.add_argument("--timeout", type=int, default=1000,
                    help="Seconds to allow for each cell before timing out")
parser.add_argument("--kernel_name", type=str, default=None,
                    help="Name of kernel to execute with")
parser.add_argument("--skip-execution", dest='skip_execution',
                    action='store_true',
                    help="Skip execution notebooks before converting")
parser.add_argument('--parallel', dest='parallel', action='store_true',
                    help='Execute notebooks in parallel')
parser.add_argument('--report-errors', dest='report_errors',
                    action='store_true',
                    help='Report errors that occur when executing notebooks')
parser.add_argument('--fail-on-error', dest='error_fail', action='store_true',
                    help='Fail when an error occurs when executing a cell '
                         'in a notebook.')
parser.add_argument('--skip-existing', dest='skip_existing',
                    action='store_true',
                    help='Skip execution of an executed file exists and '
                         'is newer than the notebook.')
parser.set_defaults(parallel=True, skip_execution=False,
                    report_errors=True, error_fail=False,
                    skip_existing=False)


def main():
    args = parser.parse_args()
    do(fp=args.fp,
       directory=args.directory,
       to=args.to,
       execute=not args.skip_execution,
       timeout=args.timeout,
       kernel_name=args.kernel_name,
       parallel=args.parallel,
       report_errors=args.report_errors,
       error_fail=args.error_fail,
       skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
