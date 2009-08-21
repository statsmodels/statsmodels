"""
Robust statistical models
"""
import numpy as np
import numpy.linalg as L

from scikits.statsmodels.robust import norms
from scikits.statsmodels.robust.scale import MAD, stand_MAD, Huber
