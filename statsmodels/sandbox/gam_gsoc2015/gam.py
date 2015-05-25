import patsy
from patsy import dmatrices, dmatrix, demo_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


n = 2000
data = pd.DataFrame()
x = np.linspace(-1, 1, n)
d = {"x": x}
dm = dmatrix("bs(x, df=20, degree=2, include_intercept=True)", d)


z = np.linspace(-2, 0, n)
d["z"] = z
y = x * x +  x + z



plt.plot(dm)
plt.show()
