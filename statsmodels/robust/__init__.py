"""
Robust statistical models
"""
__all__ = ["Huber", "HuberScale", "hubers_scale", "mad", "norms", "test"]
from . import norms
from .scale import mad, Huber, HuberScale, hubers_scale

from statsmodels.tools._test_runner import PytestTester

test = PytestTester()
