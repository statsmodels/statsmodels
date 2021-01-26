"""
import numpy as np
from .factor_rotation import rotate_factors, promax
from statsmodels.tools.decorators import cache_readonly
from statsmodels.iolib import summary2
import pandas as pd

try:
    import matplotlib.pyplot
    missing_matplotlib = False
except ImportError:
    missing_matplotlib = True

if not missing_matplotlib:
    from .plots import plot_scree, plot_loadings
"""
