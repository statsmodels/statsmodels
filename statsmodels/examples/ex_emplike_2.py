"""
This script is a basic tutorial on how to conduct empirical
likelihood estimation and inference in linear regression models.

I suggest running this script line by line because there will be many
optimization results printed.  Nevertheless, it can be ran all at once.

"""
import numpy as np
import statsmodels.api as sm

# Let's generate some regression data
np.random.seed(100)  # no significance of the seed
X = np.random.standard_normal((40, 3))
X = sm.add_constant(X, prepend=1)
beta = np.array([[1], [2], [3], [4]])
y = np.dot(X, beta).reshape(40, 1) + np.random.standard_normal((40, 1))

# There are no distributional assumptions on the error.  I just chose
# normal errors to demonstrate.

print 'Lets play with EL Regression'

"""
In a model with an intercept, there are two ways to conduct
EL inference.  the first is to call the ElLinReg instance directly.
The second is to access EL inference through OLS results.  We will go
through both, first by directly using EL and then via OLS.
"""

elmodel = sm.emplike.ElLinReg(y, X)


# Let's test if the intercept is 0
print 'Intercept test'
test0_1 = elmodel.test_beta([0], [0])
print test0_1
# Let's test if beta3 is 4
print 'beta3 test'
test1 = elmodel.test_beta([4], [3])
print test1
# Lets test the hypothesis that beta3=4 and beta2=3
print 'joint beta test'
test2 = elmodel.test_beta([3, 4], [2, 3])
print test2

# Let's get the confidence intervals for the parameters
print 'Confidence Interval for Beta1'
ci_beta1 = elmodel.ci_beta(1)
print ci_beta1

"""
Now we'll see how to conduct EL inference through the OLS instance
"""

model = sm.OLS(y, X)
fitted = model.fit()
print 'The confidence interval for Beta 1 through OLS'
print fitted.conf_int_el(1)
print 'Test the hypothesis that Beta 3 = 3, it''s true value'
print fitted.eltest([3], [2])  # Remember 0 indexing so Beta3 is parameter2

#Now lets check out regression through the origin

originx = np.random.standard_normal((30, 3))
originbeta = np.array([[1], [2], [3]])
originy = np.dot(originx, originbeta) + np.random.standard_normal((30, 1))

originmodel = sm.emplike.ElOriginRegress(originy, originx)
# Since in this case, parameter estimates are different then in OLS,
# we need to fit the model.

originfit = originmodel.fit()
# Now, originmodel has new attributes

print 'The fitted parameters'
print originfit.params
print 'The MSE'
print originfit.mse_model
print 'The R-squared'
print originfit.rsquared

# Note that the first element of param is 0 and there are 4 params.  That is
# because the first param is the intercept term.  This is noted in the
# documentation.

# Now that the model is fitted, we can do some inference.

print 'Test beta1 =1'
test_beta1 = originfit.test_beta_origin([1], [1])
print test_beta1

# A confidence interval for Beta1. In this case, a user supplied minimum and
# maximum is required.  This is because the optimization can easily get stuck
# at very unrealistic values for the log-likelihood if the starting values for
# the brent solver are too extreme.
print 'confidence interval for beta2'
ci_beta2 = originfit.ci_beta_origin(1, 1.5, .2)
print ci_beta2
