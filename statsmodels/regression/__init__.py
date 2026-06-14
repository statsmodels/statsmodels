from statsmodels.tools._test_runner import PytestTester

from .linear_model import yule_walker

__all__ = ["test", "yule_walker"]

test = PytestTester()
