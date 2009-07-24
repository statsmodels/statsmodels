"""
Robust statistical models
"""
import numpy as np
import numpy.linalg as L

from models.robust import norms
from scale import MAD, stand_MAD, Huber
