"""
Robust statistical models
"""
import norms
from .scale import mad, stand_mad, Huber, HuberScale, hubers_scale

from scikits.statsmodels import NoseWrapper as Tester
test = Tester().test
