from vbench.api import Benchmark
from datetime import datetime

common_setup = """from sm_vb_common import *
import pandas as pd
"""

setup = common_setup + """
N = 10000
np.random.seed(1)

X = np.arange(N)
Y = X + np.random.randn(N)
X = sm.add_constant(X)

df = pd.DataFrame(X, columns=['const', 'x'])
df['y'] = Y
"""

stmt1 = "sm.OLS(Y, X).fit()"
ols_fit = Benchmark(stmt1, setup, start_date=datetime(2013, 6, 1))

# from_forumula
stmt2 = "sm.OLS.from_formula('y ~ x', df).fit()"
ols_formula_fit = Benchmark(stmt2, setup, start_date=datetime(2013, 6, 1))
