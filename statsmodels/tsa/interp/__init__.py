__all__ = ["dentonm", "test"]
from statsmodels.tools._test_runner import PytestTester

from .denton import dentonm

test = PytestTester()
