"""Utility functions and testing helpers for statsmodels"""

from statsmodels.tools._test_runner import PytestTester

from .tools import add_constant

__all__ = ["add_constant", "test"]

test = PytestTester()
