"""
Robust statistical models
"""
__all__ = ["norms", "mad", "Huber", "HuberScale", "hubers_scale"]
from . import norms
from .scale import mad, Huber, HuberScale, hubers_scale

from statsmodels import PytestTester
test = PytestTester()
