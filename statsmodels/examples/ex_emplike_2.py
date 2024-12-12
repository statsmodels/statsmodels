"""
This script is a basic tutorial on how to conduct empirical
likelihood estimation and inference in linear regression models.
"""

import numpy as np

import statsmodels.api as sm

# Let's generate some regression data
np.random.seed(100)  # no significance of the seed
X = np.random.standard_normal((40, 3))
X = sm.add_constant(X)
beta = np.arange(1,5)
y = np.dot(X, beta) + np.random.standard_normal(40)
# There are no distributional assumptions on the error.  I just chose
# normal errors to demonstrate.

print('Lets play with EL Regression')


# In a model with an intercept, access EL inference through OLS results.


elmodel = sm.OLS(y, X)
fitted = elmodel.fit()


# Let's test if the intercept is 0
print('Intercept test')
test0_1 = fitted.el_test(np.array([0]), np.array([0]))
print(test0_1)
#  Let's test if beta3 is 4
print('beta3 test')
test1 = fitted.el_test(np.array([4]), np.array([3]))
print(test1)
#  Lets test the hypothesis that beta3=4 and beta2=3
print('joint beta test')
test2 = fitted.el_test(np.array([3, 4]), np.array([2, 3]))
print(test2)

#  Let's get the confidence intervals for the parameters
print('Confidence Interval for Beta1')
ci_beta1 = fitted.conf_int_el(1)
print(ci_beta1)

# Of course, we can still see the rest of the RegressionResults
print('R-squared')
print(fitted.rsquared)
print('Params')
print(fitted.params)

#  Now lets check out regression through the origin
print('Origin Regression')
originx = np.random.standard_normal((30, 3))
originbeta = np.array([[1], [2], [3]])
originy = np.dot(originx, originbeta) + np.random.standard_normal((30, 1))

originmodel = sm.emplike.ELOriginRegress(originy, originx)
#  Since in this case, parameter estimates are different then in OLS,
#  we need to fit the model.

originfit = originmodel.fit()


print('The fitted parameters')
print(originfit.params)
print('The MSE')
print(originfit.mse_model)
print('The R-squared')
print(originfit.rsquared)

# Note that the first element of param is 0 and there are 4 params.  That is
# because the first param is the intercept term.  This is noted in the
# documentation.

#  Now that the model is fitted, we can do some inference.

print('Test beta1 =1')
test_beta1 = originfit.el_test([1], [1])
print(test_beta1)

#  A confidence interval for Beta1.
print('confidence interval for beta1')
ci_beta2 = originfit.conf_int_el(1)
print(ci_beta2)

# Finally, since we initiated an EL model, normal inference is not available
try:
    originfit.conf_int()
except Exception:
    print('No normal inference available')
