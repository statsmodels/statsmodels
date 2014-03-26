'''Examples to compare MLE with OLS

TODO: compare standard error of parameter estimates
'''

from scipy import optimize
import numpy as np
import statsmodels.base.model as models

print('\nExample 1: Artificial Data')
print('--------------------------\n')
import statsmodels.api as sm

np.random.seed(54321)
X = np.random.rand(40,2)
X = sm.add_constant(X, prepend=False)
beta = np.array((3.5, 5.7, 150))
Y = np.dot(X,beta) + np.random.standard_normal(40)
mod2 = sm.OLS(Y,X)
res2 = mod2.fit()
f2 = lambda params: -1*mod2.loglike(params)
resfmin = optimize.fmin(f2, np.ones(3), ftol=1e-10)
print('OLS')
print(res2.params)
print('MLE')
print(resfmin)



print('\nExample 2: Longley Data, high multicollinearity')
print('-----------------------------------------------\n')

from statsmodels.datasets.longley import load
data = load()
data.exog = sm.add_constant(data.exog, prepend=False)
mod = sm.OLS(data.endog, data.exog)
f = lambda params: -1*mod.loglike(params)
score = lambda params: -1*mod.score(params)

#now you're set up to try and minimize or root find, but I couldn't get this one to work
#note that if you want to get the results, it's also a property of mod, so you can do

res = mod.fit()
#print mod.results.params
print('OLS')
print(res.params)
print('MLE')
#resfmin2 = optimize.fmin(f, mod.results.params*0.9, maxfun=5000, maxiter=5000, xtol=1e-10, ftol= 1e-10)
resfmin2 = optimize.fmin(f, np.ones(7), maxfun=5000, maxiter=5000, xtol=1e-10, ftol= 1e-10)
print(resfmin2)
# there isn't a unique solution?  Is this due to the multicollinearity? Improved with use of analytically
# defined score function?

#check X'X matrix
xtxi = np.linalg.inv(np.dot(data.exog.T,data.exog))
eval, evec = np.linalg.eig(xtxi)
print('Eigenvalues')
print(eval)
# look at correlation
print('correlation matrix')
print(np.corrcoef(data.exog[:,:-1], rowvar=0)) #exclude constant
# --> conclusion high multicollinearity

# compare
print('with matrix formula')
print(np.dot(xtxi,np.dot(data.exog.T, data.endog[:,np.newaxis])).ravel())
print('with pinv')
print(np.dot(np.linalg.pinv(data.exog), data.endog[:,np.newaxis]).ravel())
