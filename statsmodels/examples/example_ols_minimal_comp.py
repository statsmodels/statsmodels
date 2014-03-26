"""Example: minimal OLS

add example for new compare methods

"""

from __future__ import print_function
import numpy as np
import statsmodels.api as sm

np.random.seed(765367)
nsample = 100
x = np.linspace(0,10, 100)
X = sm.add_constant(np.column_stack((x, x**2)))
beta = np.array([10, 1, 0.01])
y = np.dot(X, beta) + np.random.normal(size=nsample)

results = sm.OLS(y, X).fit()
print(results.summary())

results2 = sm.OLS(y, X[:,:2]).fit()
print(results.compare_f_test(results2))
print(results.f_test([0,0,1]))

print(results.compare_lr_test(results2))

'''
(1.841903749875428, 0.1778775592033047)
<F test: F=array([[ 1.84190375]]), p=[[ 0.17787756]], df_denom=97, df_num=1>
(1.8810663357027693, 0.17021300121753191, 1.0)
'''





