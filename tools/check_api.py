"""
Recursively find all .api entry points and compares the list of available
functions to what is contained in __all__
"""

import importlib
import pkgutil
from types import ModuleType

import statsmodels

api_modules = []

# Do not check paths with .{blacklist item}. in them
BLACKLIST = ["tests", "sandbox", "libqsturng"]


def import_submodules(module: ModuleType):
    """Import all submodules of a module, recursively."""
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        module.__path__, module.__name__ + "."
    ):
        blacklisted = any([f".{bl}." in module_name for bl in BLACKLIST])
        if blacklisted:
            continue
        mod = importlib.import_module(module_name)
        if mod.__name__.endswith(".api"):
            api_modules.append(mod)


import_submodules(statsmodels)

missing = {}
for mod in api_modules:
    d = [v for v in dir(mod) if not v.startswith("_")]
    if "__all__" not in dir(mod):
        missing[mod.__name__] = d
        continue
    a = mod.__all__
    indiv = sorted(set(d).difference(a))
    if indiv:
        missing[mod.__name__] = indiv


for key in missing:
    print("-" * 60)
    print(key)
    print("-" * 60)
    print()
    for val in missing[key]:
        print(f'"{val}",')

if not missing:
    print("All api files are correct!")
