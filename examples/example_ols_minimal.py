"""Minimal Ordinary Least Squares Examples
"""

import numpy as np
import statsmodels.api as sm

# Artificial data: :math:`y_i = 1 + .1 * x_i + 1 * x_i^2 + e_i`, with :math:`e\sim N(0,1)`
nsample = 100
x = np.linspace(0, 10, 100)
X = sm.add_constant(np.column_stack((x, x**2)))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
y = np.dot(X, beta) + e

# Inspect data
print X[:5,:]
print y[:5]

# Describe, fit, and summarize OLS model
model = sm.OLS(y, X)
results = model.fit()
print results.summary()

