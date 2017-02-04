"""
Robust statistical models
"""
from . import norms
from .scale import mad, stand_mad, Huber, HuberScale, hubers_scale

from statsmodels import PytestTester
test = PytestTester()

