"""
Robust statistical models
"""
__all__ = ["norms", "mad", "Huber", "HuberScale", "hubers_scale", "test"]
from . import norms
from .scale import mad, Huber, HuberScale, hubers_scale

from statsmodels.tools._test_runner import PytestTester

test = PytestTester()
