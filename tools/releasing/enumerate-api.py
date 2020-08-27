"""
This tool is used in generating meaningful release notes.

This tool simplifies reading the current statsmodels API to a JSON file. It
can also be used against any installed statsmodels package to read the API.
The typical use would be

source venv/legacy-statsmodels/activate
python enumerate-api.py -of statsmodels-legacy.json
source venv/legacy-statsmodels/deactivate
python enumerate-api.py --diff statsmodels-legacy.json

which produces a RST file that can be included in the docs or edited.
"""
from setuptools import find_packages

import argparse
import importlib
import inspect
import json
import logging
import os
from pkgutil import iter_modules
import sys


def find_modules(path):
    modules = set()
    for pkg in find_packages(path):
        modules.add(pkg)
        pkgpath = path + "/" + pkg.replace(".", "/")
        if sys.version_info.major == 2 or (
            sys.version_info.major == 3 and sys.version_info.minor < 6
        ):
            for _, name, ispkg in iter_modules([pkgpath]):
                if not ispkg:
                    modules.add(pkg + "." + name)
        else:
            for info in iter_modules([pkgpath]):
                if not info.ispkg:
                    modules.add(pkg + "." + info.name)
    return modules


def update_class(func, funcs, class_name, full_name):
    logger = logging.getLogger("enumerate-api")
    class_api = {}
    for v2 in dir(func):
        if v2.startswith("_") and v2 != "__init__":
            continue
        method = getattr(func, v2)
        if not (
            inspect.isfunction(method)
            or inspect.isclass(method)
            or inspect.ismethod(method)
            or isinstance(method, property)
        ):
            continue
        if isinstance(method, property):
            try:
                name = f"{method.fget.__module__}.{class_name}.{v2}"
                class_api[name] = tuple()
            except Exception:
                name = ""
        else:
            sig = inspect.signature(method)
            name = f"{method.__module__}.{class_name}.{v2}"
            class_api[name] = tuple(k for k in sig.parameters.keys())
        logger.info(name)
    funcs[full_name] = class_api


def walk_modules(path):
    logger = logging.getLogger("enumerate-api")
    modules = find_modules(path)
    api = {"functions": {}, "classes": {}}
    for mod in modules:
        module = f"statsmodels.{mod}"
        logger.info(module)
        if (
            ".sandbox" in module
            or module.endswith(".tests")
            or ".tests." in module
        ):
            continue
        try:
            lib = importlib.import_module(module)
        except (ImportError, OSError):
            lib = None
        if lib is None:
            continue
        for v in dir(lib):
            if v.startswith("_"):
                continue
            func = getattr(lib, v)
            if not (inspect.isfunction(func) or inspect.isclass(func)):
                continue
            if "statsmodels" not in func.__module__:
                continue
            name = f"{func.__module__}.{v}"
            try:
                if inspect.isfunction(func):
                    d = api["functions"]
                else:
                    d = api["classes"]
                sig = inspect.signature(func)
                d[name] = tuple(k for k in sig.parameters.keys())
            except Exception:
                d[name] = tuple()
            if inspect.isclass(func):
                update_class(func, api["classes"], v, name)
            logger.info(f"{module}.{v}")
    return api


def generate_diff(api, other):
    api_classes = set(api["classes"].keys())
    other_classes = set(other["classes"].keys())
    new_classes = api_classes.difference(other_classes)
    removed_classes = set(other_classes).difference(api_classes)
    new_methods = {}
    removed_methods = {}
    changed_methods = {}
    expanded_methods = {}
    expanded_funcs = {}
    changed_funcs = {}
    common = api_classes.intersection(other_classes)
    for key in common:
        current_class = api["classes"][key]
        other_class = other["classes"][key]
        new = set(current_class.keys()).difference(other_class.keys())
        for meth in new:
            new_methods[meth] = current_class[meth]
        removed = set(other_class.keys()).difference(current_class.keys())
        for meth in removed:
            removed_methods[meth] = tuple(other_class[meth])
        common_methods = set(other_class.keys()).intersection(
            current_class.keys()
        )
        for meth in common_methods:
            if current_class[meth] != tuple(other_class[meth]):
                if set(current_class[meth]).issuperset(other_class[meth]):
                    expanded_methods[key] = set(
                        current_class[meth]
                    ).difference(other_class[meth])
                else:
                    changed_methods[key] = {
                        "current": current_class[meth],
                        "other": tuple(other_class[meth]),
                    }

    api_funcs = set(api["functions"].keys())
    other_funcs = set(other["functions"].keys())
    new_funcs = api_funcs.difference(other_funcs)
    removed_funcs = set(other_funcs).difference(api_funcs)
    common_funcs = api_funcs.intersection(other_funcs)
    for key in common_funcs:
        current_func = api["functions"][key]
        other_func = other["functions"][key]
        if current_func == tuple(other_func):
            continue
        elif set(current_func).issuperset(other_func):
            expanded_funcs[key] = set(current_func).difference(other_func)
        else:
            changed_funcs[key] = {
                "current": current_func,
                "other": tuple(other_func),
            }

    def header(v, first=False):
        return (
            "\n\n" * (not first)
            + f"\n{v}\n"
            + "-" * len(v)
            + "\n"
        )

    with open("api-differences.rst", "w") as rst:
        rst.write(header("New Classes", first=True))
        for val in sorted(new_classes):
            rst.write(f"* :class:`{val}`\n")
        rst.write(header("Removed Classes"))
        for val in sorted(removed_classes):
            rst.write(f"* ``{val}``\n")

        rst.write(header("New Methods"))
        for val in sorted(new_methods):
            rst.write(f"* :meth:`{val}`\n")

        rst.write(header("Removed Methods"))
        for val in sorted(removed_methods):
            rst.write(f"* ``{val}``\n")

        rst.write(header("Methods with New Arguments"))
        for val in sorted(expanded_methods):
            args = map(lambda v: f"``{v}``", expanded_methods[val])
            rst.write(f"* :meth:`{val}`: " + ", ".join(args) + "\n")

        rst.write(header("Methods with Changed Arguments"))
        for val in sorted(changed_methods):
            rst.write(f"* :meth:`{val}`\n")
            name = val.split(".")[-1]
            args = ", ".join(changed_methods[val]["current"])
            if args.startswith("self"):
                args = args[4:]
                if args.startswith(", "):
                    args = args[2:]
            rst.write(f"   * New: ``{name}({args})``\n")
            args = ", ".join(changed_methods[val]["other"])
            if args.startswith("self"):
                args = args[4:]
                if args.startswith(", "):
                    args = args[2:]
            rst.write(f"   * Old: ``{name}({args})``\n")

        rst.write(header("New Functions"))
        for val in sorted(new_funcs):
            rst.write(f"* :func:`{val}`\n")
        rst.write(header("Removed Functions"))
        for val in sorted(removed_funcs):
            rst.write(f"* ``{val}``\n")

        rst.write(header("Functions with New Arguments"))
        for val in sorted(expanded_funcs):
            args = map(lambda v: f"``{v}``", expanded_funcs[val])
            rst.write(f"* :func:`{val}`: " + ", ".join(args) + "\n")

        rst.write(header("Functions with Changed Arguments"))
        for val in sorted(changed_funcs):
            rst.write(f"* :func:`{val}`\n")
            name = val.split(".")[-1]
            args = ", ".join(changed_funcs[val]["current"])
            rst.write(f"   * New: ``{name}({args})``\n")
            args = ", ".join(changed_funcs[val]["other"])
            rst.write(f"   * Old: ``{name}({args})``\n")


parser = argparse.ArgumentParser(
    description="""
Store the current visible API as json, or read the API of any version into a
JSON file, or compare the current API to a different version.
"""
)
parser.add_argument(
    "--file-path",
    "-fp",
    type=str,
    default=None,
    help="Path to the root directory. If not provided, assumed to be be"
    "the import location of statsmodels.",
)
parser.add_argument(
    "--out-file",
    "-of",
    type=str,
    default=None,
    help="Name of output json file. Default is statsmodels-{version}-api.json",
)
parser.add_argument(
    "--diff", "-d", type=str, default=None, help="json file to diff"
)


def main():
    args = parser.parse_args()

    logger = logging.getLogger("enumerate-api")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    file_path = args.file_path
    if file_path is None:
        import statsmodels

        file_path = os.path.dirname(statsmodels.__file__)
    current_api = walk_modules(file_path)
    out_file = args.out_file
    if out_file is None:
        import statsmodels

        out_file = f"statsmodels-{statsmodels.__version__}-api.json"
    with open(out_file, "w") as api:
        json.dump(current_api, api, indent=2, sort_keys=True)
    if args.diff is not None:
        with open(args.diff, "r") as other:
            other_api = json.load(other)
        generate_diff(current_api, other_api)


if __name__ == "__main__":
    main()
