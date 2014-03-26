"""
This script provides a tutorial on how to use estimate and conduct
inference in an accelerated failure time model using empirical likelihood.

We will be using the Stanford Heart Transplant data

"""

from __future__ import print_function
import statsmodels.api as sm
import numpy as np

data = sm.datasets.heart.load()
# Note this data has endog, exog and censors
# We will take the log (base 10) of the endogenous survival times

model = sm.emplike.emplikeAFT(np.log10(data.endog),
                              sm.add_constant(data.exog), data.censors)

# We need to fit the model to get the parameters
fitted = model.fit()
print(fitted.params())
test1 = fitted.test_beta([4],[0])  # Test that the intercept is 4
print(test1)
test2 = fitted.test_beta([-.05], [1]) # Test that the slope is -.05
print(test2)
ci_beta1 = fitted.ci_beta(1, .1, -.1)
print(ci_beta1)
