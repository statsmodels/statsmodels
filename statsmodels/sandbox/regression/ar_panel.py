'''Paneldata model with fixed effect (constants) and AR(1) errors

checking fast evaluation of groupar1filter
quickly written to try out grouparfilter without python loops

maybe the example has MA(1) not AR(1) errors, I'm not sure and changed this.

results look good, I'm also differencing the dummy variable (constants) ???
e.g. nobs = 35
true 0.6, 10, 20, 30   (alpha, mean_0, mean_1, mean_2)
estimate 0.369453125 [ 10.14646929  19.87135086  30.12706505]

Currently minimizes ssr but could switch to minimize llf, i.e. conditional MLE.
This should correspond to iterative FGLS, where data are AR(1) transformed
similar to GLSAR ?
Result statistic from GLS return by OLS on transformed data should be
asymptotically correct (check)

Could be extended to AR(p) errors, but then requires panel with larger T

'''



import numpy as np
from scipy import optimize

from statsmodels.regression.linear_model import OLS


class PanelAR1(object):
    def __init__(self, endog, exog=None, groups=None):
        #take this from a super class, no checking is done here
        nobs = endog.shape[0]
        self.endog = endog
        if not exog is None:
            self.exog = exog

        self.groups_start = np.concatenate(([True], (np.diff(groups)!=0)))
        self.groups_valid = ~self.groups_start

    def ar1filter(self, xy, alpha):
        #print alpha,
        if hasattr(self, 'use_prais'):
            #print 'using prais'
            transformed = np.zeros_like(xy)
            transformed[1:] = (xy[1:] - alpha * xy[:-1])
            fac = np.sqrt(1.-min(alpha, 1)**2)
            transformed[self.groups_start] = xy[self.groups_start] * fac
            return transformed
        else:
            return (xy[1:] - alpha * xy[:-1])[self.groups_valid[1:]]

    def fit_conditional(self, alpha):
        y = self.ar1filter(self.endog, alpha)
        x = self.ar1filter(self.exog, alpha)
        res = OLS(y, x).fit()
        return res#-res.llf #res.ssr  #res.llf


    def fit(self):
        alpha0 = 0.1 #startvalue
        func = lambda alpha: -self.fit_conditional(alpha).llf
        fitres = optimize.fmin(func, alpha0)

        # fit_conditional only returns ssr for now
        alpha = fitres[0]
        y = self.ar1filter(self.endog, alpha)
        x = self.ar1filter(self.exog, alpha)
        reso = OLS(y, x).fit()

        return fitres, reso

if __name__ == '__main__':

    #------------ developement code for groupar1filter and example
    groups = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,
                       2,2,2,2,2,2,2,2])
    nobs = len(groups)
    data0 = np.arange(nobs)

    data = np.arange(1,nobs+1) - 0.5*np.arange(nobs) + 0.1*np.random.randn(nobs)

    y00 = 0.5*np.random.randn(nobs+1)

    # I don't think a trend is handled yet
    data = np.arange(nobs) + y00[1:] + 0.2*y00[:-1] + 0.1*np.random.randn(nobs)
    #Are these AR(1) or MA(1) errors ???
    data = y00[1:] + 0.6*y00[:-1] #+ 0.1*np.random.randn(nobs)

    group_codes = np.unique(groups)
    group_dummy = (groups[:,None] == group_codes).astype(int)

    groups_start = (np.diff(groups)!=0)
    groups_valid = (np.diff(groups)==0)  #this applies to y with length for AR(1)
    #could use np.nonzero for index instead

    y = data + np.dot(group_dummy, np.array([10, 20, 30]))
    y0 = data0 + np.dot(group_dummy, np.array([10, 20, 30]))

    print groups_valid
    print np.diff(y)[groups_valid]

    alpha = 1  #test with 1
    print (y0[1:] - alpha*y0[:-1])[groups_valid]
    alpha = 0.2  #test with 1
    print (y0[1:] - alpha*y0[:-1] + 0.001)[groups_valid]
    #this is now AR(1) for each group separately


    #------------

    #fitting the example

    exog = np.ones(nobs)
    exog = group_dummy
    mod = PanelAR1(y, exog, groups=groups)
    #mod = PanelAR1(data, exog, groups=groups) #data doesn't contain different means
    #print mod.ar1filter(mod.endog, 1)
    resa, reso = mod.fit()
    print resa[0], reso.params

