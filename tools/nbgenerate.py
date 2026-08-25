#!/usr/bin/env python3
import argparse
import asyncio
from concurrent import futures
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from colorama import Fore, init
from nbconvert import HTMLExporter, RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

if sys.platform == "win32" and sys.version_info < (3, 14):
    # Set the policy to prevent "Event loop is closed" error on Windows
    # https://github.com/encode/httpx/issues/914
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

init()

here = Path(__file__).parent
pkgdir = os.path.split(here)[0]
EXAMPLE_DIR = Path(pkgdir).joinpath("examples").resolve()
SOURCE_DIR = Path(EXAMPLE_DIR).joinpath("notebooks")
DOC_SRC_DIR = Path(pkgdir).joinpath("docs", "source")
DST_DIR = Path(DOC_SRC_DIR).joinpath("examples", "notebooks", "generated").resolve()
EXECUTED_DIR = DST_DIR

error_message = """
******************************************************************************
ERROR: Error occurred when running {notebook}
{exception}
{message}
******************************************************************************
"""
for dname in [EXECUTED_DIR, DST_DIR]:
    if not Path(dname).exists():
        Path(dname).mkdir(parents=True)


def execute_nb(src, dst, allow_errors=False, timeout=1000, kernel_name=None):
    """
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
    """
    with Path(src).open(encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(
        allow_errors=False, timeout=timeout, kernel_name=kernel_name, transport="ipc"
    )
    ep.preprocess(nb, {"metadata": {"path": SOURCE_DIR}})

    with Path(dst).open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return dst


def convert(src, dst, to="rst"):
    """
    Convert a notebook `src`.

    Parameters
    ----------
    src, dst: str
        filepaths
    to: {'rst', 'html'}
        format to export to
    """
    dispatch = {"rst": RSTExporter, "html": HTMLExporter}
    exporter = dispatch[to.lower()]()

    body, resources = exporter.from_filename(src)
    with Path(dst).open("w", encoding="utf-8") as f:
        f.write(body)
    return dst


def find_notebooks(directory=None):
    if directory is None:
        directory = SOURCE_DIR
    nbs = (p for p in Path(directory).iterdir() if p.suffix == ".ipynb")
    return nbs


def do_one(
    nb,
    to=None,
    execute=None,
    timeout=None,
    kernel_name=None,
    report_error=True,
    error_fail=False,
    skip_existing=False,
    execute_only=False,
):
    import jupyter_client
    from traitlets.traitlets import TraitError

    os.chdir(SOURCE_DIR)
    name = Path(nb).name
    dst = Path(EXECUTED_DIR).joinpath(name)
    hash_file = dst.with_suffix(".json")
    existing_hash = ""
    if Path(hash_file).exists():
        with Path(hash_file).open(encoding="utf-8") as hf:
            existing_hash = json.load(hf)
    with Path(nb).open(mode="rb") as f:
        current_hash = hashlib.sha512(f.read()).hexdigest()
    update_needed = existing_hash != current_hash
    # Update if dst missing
    update_needed = update_needed or not Path(dst).exists()
    update_needed = update_needed or not skip_existing
    if not update_needed:
        print(f"Skipping {nb}")

    if execute and update_needed:
        print(f"Executing {nb} to {dst}")
        try:
            nb = execute_nb(nb, dst, timeout=timeout, kernel_name=kernel_name)
        except Exception as e:
            if report_error:
                print(
                    Fore.RED
                    + error_message.format(
                        notebook=nb, exception=str(e), message=str(e.args[0])
                    )
                )
                print(Fore.RESET)
            if error_fail:
                raise
    elif not execute:
        print(f"Copying (without executing) {nb} to {dst}")
        shutil.copy(nb, dst)

    if execute_only:
        with Path(hash_file).open(encoding="utf-8", mode="w") as hf:
            json.dump(current_hash, hf)
        return dst

    dst = str(Path(DST_DIR).joinpath(name).with_suffix("." + to))
    print(f"Converting {nb} to {dst}")
    try:
        convert(nb, dst, to=to)
        with Path(hash_file).open(encoding="utf-8", mode="w") as hf:
            json.dump(current_hash, hf)
    except TraitError as exc:
        kernels = jupyter_client.kernelspec.find_kernel_specs()
        msg = f"Could not find kernel named `{kernel_name}`, Available kernels:\n {kernels}"
        raise ValueError(msg) from exc

    return dst


def do(
    fp=None,
    directory=None,
    to="html",
    execute=True,
    timeout=1000,
    kernel_name="",
    parallel=False,
    report_errors=True,
    error_fail=False,
    skip_existing=False,
    execute_only=False,
    skip_specific=(),
):
    if fp is None:
        nbs = find_notebooks(directory)
    else:
        nbs = [fp]

    nbs = list(nbs)
    skip = set()
    for nb in nbs:
        for skip_nb in skip_specific:
            if skip_nb in nb:
                skip.add(nb)
    nbs = [nb for nb in nbs if nb not in skip]

    if kernel_name is None:
        kernel_name = find_kernel_name()

    func = partial(
        do_one,
        to=to,
        execute=execute,
        timeout=timeout,
        kernel_name=kernel_name,
        report_error=report_errors,
        error_fail=error_fail,
        skip_existing=skip_existing,
        execute_only=execute_only,
    )

    if parallel:
        with futures.ProcessPoolExecutor() as pool:
            for dst in pool.map(func, nbs):
                print(f"Finished {dst}")
    else:
        for nb in nbs:
            func(nb)
            print(f"Finished {nb}")

    skip_func = partial(
        do_one,
        to=to,
        execute=False,
        timeout=timeout,
        kernel_name=kernel_name,
        report_error=report_errors,
        error_fail=error_fail,
        skip_existing=skip_existing,
        execute_only=execute_only,
    )
    for nb in skip:
        skip_func(nb)
        print(f"Finished (without execution) {nb}")


def find_kernel_name():
    import jupyter_client

    kernels = jupyter_client.kernelspec.find_kernel_specs()
    kernel_name = f"python{sys.version_info.major}"
    if kernel_name not in kernels:
        return ""
    return kernel_name


parser = argparse.ArgumentParser(description="Process example notebooks")
parser.add_argument(
    "--fp",
    type=str,
    default=None,
    help="Path to notebook to convert. Converts all notebooks "
    "in `directory` by default.",
)
parser.add_argument(
    "--directory",
    type=str,
    default=None,
    help="Path to notebook directory to convert",
)
parser.add_argument(
    "--to",
    type=str,
    default="html",
    help="Type to convert to. One of `{'html', 'rst'}`",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=1000,
    help="Seconds to allow for each cell before timing out",
)
parser.add_argument(
    "--kernel_name",
    type=str,
    default=None,
    help="Name of kernel to execute with",
)
parser.add_argument(
    "--skip-execution",
    dest="skip_execution",
    action="store_true",
    help="Skip execution notebooks before converting",
)
parser.add_argument(
    "--execute-only",
    dest="execute_only",
    action="store_true",
    help="Execute notebooks but do not convert to html",
)
parser.add_argument(
    "--parallel",
    dest="parallel",
    action="store_true",
    help="Execute notebooks in parallel",
)
parser.add_argument(
    "--report-errors",
    dest="report_errors",
    action="store_true",
    help="Report errors that occur when executing notebooks",
)
parser.add_argument(
    "--fail-on-error",
    dest="error_fail",
    action="store_true",
    help="Fail when an error occurs when executing a cell in a notebook.",
)
parser.add_argument(
    "--skip-existing",
    dest="skip_existing",
    action="store_true",
    help="Skip execution of an executed file exists and is newer than the notebook.",
)
parser.add_argument(
    "--execution-blacklist",
    type=str,
    default=None,
    help="Comma separated list of notebook names to skip, e.g,"
    "slow-notebook.ipynb,other-notebook.ipynb",
)

parser.set_defaults(
    parallel=True,
    skip_execution=False,
    report_errors=True,
    error_fail=False,
    skip_existing=False,
)


def main():
    args = parser.parse_args()
    skip_nb_exec = args.execution_blacklist
    skip_specific = skip_nb_exec.split(",") if skip_nb_exec else []
    do(
        fp=args.fp,
        directory=args.directory,
        to=args.to,
        execute=not args.skip_execution,
        timeout=args.timeout,
        kernel_name=args.kernel_name,
        parallel=args.parallel,
        report_errors=args.report_errors,
        error_fail=args.error_fail,
        skip_existing=args.skip_existing,
        execute_only=args.execute_only,
        skip_specific=skip_specific,
    )


if __name__ == "__main__":
    main()
