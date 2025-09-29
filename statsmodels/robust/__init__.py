"""
Robust statistical models
"""
__all__ = ["Huber", "HuberScale", "hubers_scale", "mad", "norms", "test"]
from statsmodels.tools._test_runner import PytestTester

from . import norms
from .scale import Huber, HuberScale, hubers_scale, mad

test = PytestTester()
