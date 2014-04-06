"""examples for usage of F-test on linear restrictions in OLS

linear restriction is R \beta = 0
R is (nr,nk), beta is (nk,1) (in matrix notation)


TODO: clean this up for readability and explain

Notes
-----
This example was written mostly for cross-checks and refactoring.
"""

import numpy as np
import numpy.testing as npt
import statsmodels.api as sm

print('\n\n Example 1: Longley Data, high multicollinearity')

data = sm.datasets.longley.load()
data.exog = sm.add_constant(data.exog, prepend=False)
res = sm.OLS(data.endog, data.exog).fit()

# test pairwise equality of some coefficients
R2 = [[0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0]]
Ftest = res.f_test(R2)
print(repr((Ftest.fvalue, Ftest.pvalue)))  # use repr to get more digits
# 9.740461873303655 0.0056052885317360301

##Compare to R (after running R_lm.s in the longley folder)
##
##> library(car)
##> linear.hypothesis(m1, c("GNP = UNEMP","POP = YEAR"))
##Linear hypothesis test
##
##Hypothesis:
##GNP - UNEMP = 0
##POP - YEAR = 0
##
##Model 1: TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR
##Model 2: restricted model
##
## Res.Df      RSS Df Sum of Sq      F   Pr(>F)
##1      9   836424
##2     11  2646903 -2  -1810479 9.7405 0.005605 **

print('Regression Results Summary')
print(res.summary())


print('\n F-test whether all variables have zero effect')
R = np.eye(7)[:-1, :]
Ftest0 = res.f_test(R)
print(repr((Ftest0.fvalue, Ftest0.pvalue)))
print('%r' % res.fvalue)
npt.assert_almost_equal(res.fvalue, Ftest0.fvalue, decimal=9)

ttest0 = res.t_test(R[0, :])
print(repr((ttest0.tvalue, ttest0.pvalue)))

betatval = res.tvalues
betatval[0]
npt.assert_almost_equal(betatval[0], ttest0.tvalue, decimal=15)

"""
# several ttests at the same time
# currently not checked for this, but it (kind of) works
>>> ttest0 = res.t_test(R[:2,:])
>>> print(repr((ttest0.t, ttest0.pvalue))
(array([[ 0.17737603,         NaN],
       [        NaN, -1.06951632]]), array([[ 0.43157042,  1.        ],
       [ 1.        ,  0.84365947]]))

>>> ttest0 = res.t_test(R)
>>> ttest0.t
array([[  1.77376028e-01,              NaN,              NaN,
                     NaN,  -1.43660623e-02,   2.15494063e+01],
       [             NaN,  -1.06951632e+00,  -1.62440215e+01,
         -1.78173553e+01,              NaN,              NaN],
       [             NaN,  -2.88010561e-01,  -4.13642736e+00,
         -4.06097408e+00,              NaN,              NaN],
       [             NaN,  -6.17679489e-01,  -7.94027056e+00,
         -4.82198531e+00,              NaN,              NaN],
       [  4.23409809e+00,              NaN,              NaN,
                     NaN,  -2.26051145e-01,   2.89324928e+02],
       [  1.77445341e-01,              NaN,              NaN,
                     NaN,  -8.08336103e-03,   4.01588981e+00]])
>>> betatval
array([ 0.17737603, -1.06951632, -4.13642736, -4.82198531, -0.22605114,
        4.01588981, -3.91080292])
>>> ttest0.t
array([ 0.17737603, -1.06951632, -4.13642736, -4.82198531, -0.22605114,
        4.01588981])
"""

print('\nsimultaneous t-tests')
ttest0 = res.t_test(R2)

t2 = ttest0.tvalue
print(ttest0.tvalue)
print(t2)
t2a = np.r_[res.t_test(np.array(R2)[0, :]).tvalue,
            res.t_test(np.array(R2)[1, :]).tvalue]
print(t2 - t2a)
t2pval = ttest0.pvalue
print('%r' % t2pval)  # reject
# array([  9.33832896e-04,   9.98483623e-01])
print('reject')
print('%r' % (t2pval < 0.05))

# f_test needs 2-d currently
Ftest2a = res.f_test(np.asarray(R2)[:1, :])
print(repr((Ftest2a.fvalue, Ftest2a.pvalue)))
Ftest2b = res.f_test(np.asarray(R2)[1:2, :])
print(repr((Ftest2b.fvalue, Ftest2b.pvalue)))

print('\nequality of t-test and F-test')
print(t2a**2 - np.array((Ftest2a.fvalue, Ftest2b.fvalue)))
npt.assert_almost_equal(t2a**2, np.vstack((Ftest2a.fvalue, Ftest2b.fvalue)))
#npt.assert_almost_equal(t2pval, np.array((Ftest2a.pvalue, Ftest2b.pvalue)))
npt.assert_almost_equal(t2pval * 2, np.c_[Ftest2a.pvalue,
    Ftest2b.pvalue].squeeze())


print('\n\n Example 2: Artificial Data')

nsample = 100
ncat = 4
sigma = 2
xcat = np.linspace(0, ncat - 1, nsample).round()[:, np.newaxis]
dummyvar = (xcat == np.arange(ncat)).astype(float)

beta = np.array([0., 2, -2, 1])[:, np.newaxis]
ytrue = np.dot(dummyvar, beta)
X = sm.tools.add_constant(dummyvar[:, :-1], prepend=False)
y = ytrue + sigma * np.random.randn(nsample, 1)
mod2 = sm.OLS(y[:, 0], X)
res2 = mod2.fit()

print(res2.summary())

R3 = np.eye(ncat)[:-1, :]
Ftest = res2.f_test(R3)
print(repr((Ftest.fvalue, Ftest.pvalue)))
R3 = np.atleast_2d([0, 1, -1, 2])
Ftest = res2.f_test(R3)
print(repr((Ftest.fvalue, Ftest.pvalue)))

print('simultaneous t-test for zero effects')
R4 = np.eye(ncat)[:-1, :]
ttest = res2.t_test(R4)
print(repr((ttest.tvalue, ttest.pvalue)))


R5 = np.atleast_2d([0, 1, 1, 2])
np.dot(R5, res2.params)
Ftest = res2.f_test(R5)
print(repr((Ftest.fvalue, Ftest.pvalue)))
ttest = res2.t_test(R5)
#print(repr((ttest.t, ttest.pvalue))
print(repr((ttest.tvalue, ttest.pvalue)))

R6 = np.atleast_2d([1, -1, 0, 0])
np.dot(R6, res2.params)
Ftest = res2.f_test(R6)
print(repr((Ftest.fvalue, Ftest.pvalue)))
ttest = res2.t_test(R6)
#print(repr((ttest.t, ttest.pvalue))
print(repr((ttest.tvalue, ttest.pvalue)))

R7 = np.atleast_2d([1, 0, 0, 0])
np.dot(R7, res2.params)
Ftest = res2.f_test(R7)
print(repr((Ftest.fvalue, Ftest.pvalue)))
ttest = res2.t_test(R7)
#print(repr((ttest.t, ttest.pvalue))
print(repr((ttest.tvalue, ttest.pvalue)))


print('\nExample: 2 categories: replicate stats.glm and stats.ttest_ind')

mod2 = sm.OLS(y[xcat.flat < 2][:, 0], X[xcat.flat < 2, :][:, (0, -1)])
res2 = mod2.fit()

R8 = np.atleast_2d([1, 0])
np.dot(R8, res2.params)
Ftest = res2.f_test(R8)
print(repr((Ftest.fvalue, Ftest.pvalue)))
print(repr((np.sqrt(Ftest.fvalue), Ftest.pvalue)))
ttest = res2.t_test(R8)
#print(repr(ttest.t), ttest.pvalue)))
print(repr((ttest.tvalue, ttest.pvalue)))


from scipy import stats
print(stats.glm(y[xcat < 2].ravel(), xcat[xcat < 2].ravel()))
print(stats.ttest_ind(y[xcat == 0], y[xcat == 1]))

#TODO: compare with f_oneway
