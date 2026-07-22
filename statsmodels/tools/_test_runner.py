"""Pytest runner that allows tests to be run within Python"""

import os
import sys


class PytestTester:
    """Run a package test suite using pytest"""

    def __init__(self, package_path=None):
        """
        Initialize the tester for the calling module's package

        Parameters
        ----------
        package_path : str, optional
            Path used to locate the test suite to run. If None, the
            path is inferred from the ``__file__`` of the module that
            instantiated this class.
        """
        f = sys._getframe(1)
        if package_path is None:
            package_path = f.f_locals.get("__file__", None)
            if package_path is None:
                raise ValueError("Unable to determine path")
        self.package_path = os.path.dirname(package_path)
        self.package_name = f.f_locals.get("__name__", None)

    def __call__(self, extra_args=None, exit=False):
        """
        Run the package test suite using pytest

        Parameters
        ----------
        extra_args : list[str], optional
            Command line arguments to pass to pytest. If None, defaults
            to ``["--tb=short", "--disable-pytest-warnings"]``.
        exit : bool, optional
            If True, call `sys.exit` with the pytest exit status after
            the run completes.

        Returns
        -------
        bool
            True if all tests passed.
        """
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
