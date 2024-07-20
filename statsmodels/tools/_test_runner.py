"""
Pytest runner that allows test to be run within python
"""

import os
import sys


class PytestTester:
    def __init__(self, package_path=None):
        f = sys._getframe(1)
        if package_path is None:
            package_path = f.f_locals.get("__file__", None)
            if package_path is None:
                raise ValueError("Unable to determine path")
        self.package_path = os.path.dirname(package_path)
        self.package_name = f.f_locals.get("__name__", None)

    def __call__(self, extra_args=None, exit=False):
        import pytest

        if extra_args is None:
            extra_args = ["--tb=short", "--disable-pytest-warnings"]
        cmd = [self.package_path] + extra_args
        print("Running pytest " + " ".join(cmd))
        status = pytest.main(cmd)
        if exit:
            print(f"Exit status: {status}")
            sys.exit(status)

        return (status == 0)
