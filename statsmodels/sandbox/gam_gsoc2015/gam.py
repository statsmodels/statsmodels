import patsy
from patsy import dmatrices, dmatrix, demo_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame()
x = np.linspace(-1, 1, 200)
z = np.linspace(-2, 0, 200)
y = x * x +  x + z
d = {"x": x, "z": z}
dm = dmatrix("bs(x, df=6, degree=3, include_intercept=True) + z - 1", d)
