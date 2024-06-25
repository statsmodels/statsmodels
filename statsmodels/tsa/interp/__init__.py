__all__ = ['dentonm', 'test']
from .denton import dentonm
from statsmodels.tools._test_runner import PytestTester

test = PytestTester()
