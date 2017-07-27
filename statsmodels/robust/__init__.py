"""
Robust statistical models
"""
from . import norms
from .scale import mad, Huber, HuberScale, hubers_scale

from statsmodels import NoseWrapper as Tester
test = Tester().test
