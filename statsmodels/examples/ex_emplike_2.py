"""
Second test script.  Used to demonstrate EL regression.

I suggest running this script line by line because there will be many
optimization results printed.  Nevertheless, it can be ran all at once.

"""
import numpy as np
import statsmodels.api as sm

# Let's generate some regression data
np.random.seed(100)
X = np.random.standard_normal((40,3))
X = sm.add_constant(X, prepend=1)
beta = np.array([[1],[2],[3],[4]])
y= np.dot(X, beta).reshape(40,1) + np.random.standard_normal((40,1))

# There are no distributional assumptions on the error.  I just chose
# normal errors to demonstrate.

print 'Lets play with EL Regression'
elmodel= sm.emplike.ElLinReg(y, X)

# Let's test the hypothesis that beta1 (the first slope parameter) is 0
print 'Example of optimization difficulties'
opt_diff =  elmodel.hy_test_beta([0], [1])
print opt_diff
# Note a p value of 0 and an enormous LLR.  The p-value is close to accurate
# but the LLR is way to high.  This is because 0 is not in the convex hull of
# the data so the optimization cannot find the accurate llr.  I'm still
# struggling with the topology theory but hope to add some insightful
# documentation regarding this phenomenon.

# Let's test if the intercept is 0
print 'Intercept test'
test0_1 = elmodel.hy_test_beta([0], [0])
print test0_1
# Let's test if beta3 is 4
print 'beta3 test'
test1 = elmodel.hy_test_beta([4], [3])
print test1
# Lets test the hypothesis that beta3=4 and beta2=3
print 'joint beta test'
test2 = elmodel.hy_test_beta([3,4], [2,3])
print test2
# Although the log likelihoods are the same, the p-values are
# different due to the increased degree of freedom.

# Let's get the confidence intervals for the parameters
ci_beta1 = elmodel.ci_beta(1)
# The optimization results are the results of optimizing out the nuisance
# parameters at each various values of beta.  With a significance level of
# .05, the function is looking for the value of beta 1 that gives a llr
# of 3.84
print ci_beta1

#Now lets check out regression through the origin

originx= np.random.standard_normal((30,3))
originbeta=np.array([[1],[2],[3]])
originy= np.dot(originx, originbeta) + np.random.standard_normal((30,1))

originmodel= sm.emplike.ElOriginRegress(originy, originx)
# Since in this case, parameter estimates are different then in OLS,
# we need to fit the model.

originmodel.fit()
# Now, originmodel has new attributes
print 'Some attributes'

print originmodel.params
print originmodel.mse_model
print originmodel.rsquared

# Note that the first element of param is 0 and there are 4 params.  That is
# because the first param is the intercept term.  This is noted in the
# documentation.

# Now that the model is fitted, we can do some inference.

print 'Test beta1 =1'
hy_test_beta1= originmodel.hy_test_beta_origin(1,1)
print hy_test_beta1

# and a confidence interval
print 'confidence interval for beta2'
ci_beta2= originmodel.ci_beta_origin(1, 1.5, .2)
print ci_beta2
