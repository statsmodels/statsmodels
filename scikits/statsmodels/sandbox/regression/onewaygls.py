# -*- coding: utf-8 -*-
"""
F test for null hypothesis that coefficients in several regressions are the same

* implemented by creating groupdummies*exog and testing appropriate contrast
  matrices
* similar to test for structural change in all variables at predefined break points
* allows only one group variable
* currently tests for change in all exog variables
* allows for heterogscedasticity, error variance varies across groups (not tried out)


TODO
----

* generalize anova structure,
  - structural break in only some variables
  - compare structural breaks in several exog versus constant only
  - fast way to construct comparisons
* print anova style results
* add all pairwise comparison tests (DONE) with and without Bonferroni correction
* add additional test, likelihood-ratio, lagrange-multiplier, wald ?
* test for heteroscedasticity, equality of variances
  - how?
  - like lagrange-multiplier in stattools heteroscedasticity tests
* permutation or bootstrap test statistic or pvalues


References
----------

Greene: section 7.4 Modeling and Testing for a Structural Break
   is not the same because I use a different normalization, which looks easier
   for more than 2 groups/subperiods

after looking at Greene:
* my version assumes that all groups are large enough to estimate the coefficients
* in sections 7.4.2 and 7.5.3, predictive tests can also be used when there are
  insufficient (nobs<nvars) observations in one group/subperiods
  question: can this be used to test structural change for last period?
        cusum test but only for current period,
        in general cusum is better done with recursive ols
        check other references again for this, there was one for non-recursive
        calculation of cusum (if I remember correctly)
* Greene 7.4.4: with unequal variances Greene mentions Wald test, but where
  size of test might not be very good
  no mention of F-test based on GLS, is there a reference for what I did?
  alternative: use Wald test with bootstrap pvalues?


Created on Sat Mar 27 01:48:01 2010
Author: josef-pktd
"""
import numpy as np
from scipy import stats
from scikits.statsmodels.regression import OLS, WLS


class OneWayLS(object):
    '''Class to test equality of regression coefficients across groups

    This class performs tests whether the linear regression coefficients are
    the same across pre-specified groups. This can be used to test for
    structural breaks at given change points, or for ANOVA style analysis of
    differences in the effect of explanatory variables across groups.

    Notes
    -----
    The test is implemented by regression on the original pooled exogenous
    variables and on group dummies times the exogenous regressors.

    y_i = X_i beta_i + u_i  for all groups i

    The test is for the null hypothesis: beta_i = beta for all i
    against the alternative that at least one beta_i is different.

    By default it is assumed that all u_i have the same variance. If the
    keyword option het is True, then it is assumed that the variance is
    group specific. This uses WLS with weights given by the standard errors
    from separate regressions for each group.
    Note: het=True is not sufficiently tested

    The F-test assumes that the errors are normally distributed.



    original question from mailing list for equality of coefficients
    across regressions, and example in Stata FAQ

    *testing*:

    * if constant is the only regressor then the result for the F-test is
      the same as scipy.stats.f_oneway
      (which in turn is verified against NIST for not badly scaled problems)
    * f-test for simple structural break is the same as in original script
    * power and size of test look ok in examples
    * not checked/verified for heteroscedastic case
      - for constant only: ftest result is the same with WLS as with OLS - check?

    check: I might be mixing up group names (unique)
           and group id (integers in arange(ngroups)
           not tested for groups that are not arange(ngroups)
           make sure groupnames are always consistently sorted/ordered
    '''
    def __init__(self, y, x, groups=None, het=False, data=None, meta=None):
        if groups is None:
            raise ValueError('use OLS if there are no groups')
            #maybe replace by dispatch to OLS
        if data:
            y = data[y]
            x = [data[v] for v in x]
            try:
                groups = data[groups]
            except [KeyError, ValueError]:
                pass
        self.endog = np.asarray(y)
        self.exog = np.asarray(x)
        if self.exog.ndim == 1:
            self.exog = self.exog[:,None]
        self.groups = np.asarray(groups)
        self.het = het

        self.groupsint = None
        if np.issubdtype(self.groups.dtype, int):
            self.unique = np.unique(self.groups)
            if (self.unique == np.arange(len(self.unique))).all():
                self.groupsint = self.groups

        if self.groupsint is None: # groups are not consecutive integers
            self.unique, self.groupsint = np.unique(self.groupsint, return_inverse=True)

    def fitbygroups(self):
        olsbygroup = {}
        sigmabygroup = []
        for gi, group in enumerate(self.unique):
            groupmask = self.groupsint == group
            res = OLS(self.endog[groupmask], self.exog[groupmask]).fit()
            olsbygroup[group] = res
            sigmabygroup.append(res.mse_resid)
        self.olsbygroup = olsbygroup
        self.sigmabygroup = np.array(sigmabygroup)
        self.weights = np.sqrt(self.sigmabygroup[self.groupsint]) #TODO:chk sqrt

    def fitjoint(self):
        if not hasattr(self, 'weights'):
            self.fitbygroups()
        groupdummy = (self.groupsint[:,None] == self.unique).astype(int)
        #order of dummy variables by variable - not used
        #dummyexog = self.exog[:,:,None]*groupdummy[:,None,1:]
        #order of dummy variables by grous - used
        dummyexog = self.exog[:,None,:]*groupdummy[:,1:,None]
        exog = np.c_[self.exog, dummyexog.reshape(self.exog.shape[0],-1)] #self.nobs ??
        #Notes: I changed to drop first group from dummy
        #instead I want one full set dummies
        if self.het:
            weights = self.weights
            res = WLS(self.endog, exog, weights=weights).fit()
        else:
            res = OLS(self.endog, exog).fit()
        self.lsjoint = res
        contrasts = {}
        nvars = self.exog.shape[1]
        nparams = exog.shape[1]
        ndummies = nparams - nvars
        contrasts['all'] = np.c_[np.zeros((ndummies, nvars)), np.eye(ndummies)]
        for groupind,group in enumerate(self.unique[1:]):  #need enumerate if groups != groupsint
            groupind = groupind + 1
            contr = np.zeros((nvars, nparams))
            contr[:,nvars*groupind:nvars*(groupind+1)] = np.eye(nvars)
            contrasts[group] = contr
            #save also for pairs, see next
            contrasts[(self.unique[0], group)] = contr

        #Note: I'm keeping some duplication for testing
        pairs = np.triu_indices(len(self.unique),1)
        for ind1,ind2 in zip(*pairs):  #replace with group1, group2 in sorted(keys)
            if ind1 == 0:
                continue # need comparison with benchmark/normalization group separate
            g1 = self.unique[ind1]
            g2 = self.unique[ind2]
            group = (g1, g2)
            contr = np.zeros((nvars, nparams))
            contr[:,nvars*ind1:nvars*(ind1+1)] = np.eye(nvars)
            contr[:,nvars*ind2:nvars*(ind2+1)] = -np.eye(nvars)
            contrasts[group] = contr


        self.contrasts = contrasts

    def fitpooled(self):
        if self.het:
            if not hasattr(self, 'weights'):
                self.fitbygroups()
            weights = self.weights
            res = WLS(self.endog, self.exog, weights=weights).fit()
        else:
            res = OLS(self.endog, self.exog).fit()
        self.lspooled = res

    def ftest_summary(self):
        if not hasattr(self, 'lsjoint'):
            self.fitjoint()
        txt = []
        summarytable = []

        txt.append('F-test for equality of coefficients across groups')
        fres = self.lsjoint.f_test(self.contrasts['all'])
        txt.append(fres.__str__())
        summarytable.append(('all',(fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))

#        for group in self.unique[1:]:  #replace with group1, group2 in sorted(keys)
#            txt.append('F-test for equality of coefficients between group'
#                       ' %s and group %s' % (group, '0'))
#            fres = self.lsjoint.f_test(self.contrasts[group])
#            txt.append(fres.__str__())
#            summarytable.append((group,(fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))
        pairs = np.triu_indices(len(self.unique),1)
        for ind1,ind2 in zip(*pairs):  #replace with group1, group2 in sorted(keys)
            g1 = self.unique[ind1]
            g2 = self.unique[ind2]
            txt.append('F-test for equality of coefficients between group'
                       ' %s and group %s' % (g1, g2))
            group = (g1, g2)
            fres = self.lsjoint.f_test(self.contrasts[group])
            txt.append(fres.__str__())
            summarytable.append((group,(fres.fvalue, fres.pvalue, fres.df_denom, fres.df_num)))

        return '\n'.join(txt), summarytable


    def lr_test(self):
        '''generic likelihood ration test between nested models

            \begin{align} D & = -2(\ln(\text{likelihood for null model}) - \ln(\text{likelihood for alternative model})) \\ & = -2\ln\left( \frac{\text{likelihood for null model}}{\text{likelihood for alternative model}} \right). \end{align}

            is distributed as chisquare with df equal to difference in number of parameters or equivalently
            difference in residual degrees of freedom  (sign?)

        TODO: put into separate function
        '''
        if not hasattr(self, 'lsjoint'):
            self.fitjoint()
        if not hasattr(self, 'lspooled'):
            self.fitpooled()
        loglikejoint = self.lsjoint.llf
        loglikepooled = self.lspooled.llf
        lrstat = -2*(loglikepooled - loglikejoint)   #??? check sign
        lrdf = self.lspooled.df_resid - self.lsjoint.df_resid
        lrpval = stats.chi2.sf(lrstat, lrdf)

        return lrstat, lrpval, lrdf





def linmod(y, x, **kwds):
    if 'weights' in kwds:
        return WLS(y, x, kwds)
    elif 'sigma' in kwds:
        return GLS(y, x,kwds)
    else:
        return OLS(y, x, kwds)



def recursive_olsresiduals(olsresults, skip):
    '''this is my original version based on Greene and references'''
    y = olsresults.model.endog
    x = olsresults.model.exog
    nobs, nvars = x.shape
    rparams = np.nan * np.zeros((nobs,nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))

    #XTX = np.zeros((nvars,nvars))
    #XTY = np.zeros((nvars))

    x0 = x[:skip]
    y0 = y[:skip]
    XTX = np.dot(x0.T, x0)
    XTY = np.dot(x0.T, y0) #xi * y   #np.dot(xi, y)
    beta = np.linalg.solve(XTX, XTY)
    rparams[skip-1] = beta
    yipred = np.dot(x[skip-1], beta)
    rypred[skip-1] = yipred
    rresid[skip-1] = y[skip-1] - yipred
    rvarraw[skip-1] = 1+np.dot(x[skip-1],np.dot(np.linalg.inv(XTX),x[skip-1]))
    for i in range(skip,nobs):
        xi = x[i:i+1,:]
        yi = y[i]
        xxT = np.dot(xi.T, xi)  #xi is 2d 1 row
        xy = (xi*yi).ravel() # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        print xy.shape, XTY.shape
        print XTX
        print XTY
        beta = np.linalg.solve(XTX, XTY)
        rparams[i-1] = beta  #this is beta based on info up to t-1
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        rresid[i] = yi - yipred
        rvarraw[i] = 1 + np.dot(xi,np.dot(np.linalg.inv(XTX),xi.T))
        XTX += xxT
        XTY += xy

    i = nobs
    beta = np.linalg.solve(XTX, XTY)
    rparams[i-1] = beta

    rresid_scaled = rresid/np.sqrt(rvarraw)   #this is N(0,sigma2) distributed
    nrr = nobs-skip
    sigma2 = rresid_scaled[skip-1:].var(ddof=1)
    rresid_standardized = rresid_scaled/np.sqrt(sigma2) #N(0,1) distributed
    rcusum = rresid_standardized[skip-1:].cumsum()
    #confidence interval points in Greene p136 looks strange?
    #this assumes sum of independent standard normal
    #rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    a = 1.143 #for alpha=0.99  =0.948 for alpha=0.95
    #following taken from Ploberger,
    crit = a*np.sqrt(nrr)
    rcusumci = (a*np.sqrt(nrr) + a*np.arange(0,nobs-skip)/np.sqrt(nrr)) * np.array([[-1.],[+1.]])
    return rresid, rparams, rypred, rresid_standardized, rresid_scaled, rcusum, rcusumci

'''jplv this follows Harvey, see BigJudge 5.5.2b for formula for inverse updating
[n k] = size(x);
rresid = zeros(n,1);
b = zeros(n,k);

% initial estimates based on first k observations
xxt1 = inv(x(1:k,1:k)'*x(1:k,1:k));
bt1 = xxt1*x(1:k,1:k)'*y(1:k,1);
b(k,:) = bt1';
j = k+1;

while j <= n
    xt = x(j,1:k); yt = y(j,1);
    ft = 1 + xt*xxt1*xt';
    xxt = xxt1 - xxt1*xt'*xt*xxt1/ft;
    vt = yt - xt*bt1;
    bt = bt1 + xxt1*xt'*vt/ft;
    % save results and prepare for next loop
    rresid(j,1) = vt/sqrt(ft);
    b(j,:) = bt';
    xxt1 = xxt;
    bt1 = bt;
    j = j+1;
end;
'''

def recursive_olsresiduals2(olsresults, skip=None):
    '''

    note: change to other version beta is now moved by 1 position
    produces same recursive residuals as other version

    References
    ----------
    jplv to check formulas, follows Harvey
    BigJudge 5.5.2b for formula for inverse(X'X) updating
    '''
    lamda = 0.0
    y = olsresults.model.endog
    x = olsresults.model.exog
    nobs, nvars = x.shape
    if skip is None:
        skip = nvars
    rparams = np.nan * np.zeros((nobs,nvars))
    rresid = np.nan * np.zeros((nobs))
    rypred = np.nan * np.zeros((nobs))
    rvarraw = np.nan * np.zeros((nobs))


    #intialize with skip observations
    x0 = x[:skip]
    y0 = y[:skip]
    #add Ridge to start (not in jplv
    XTXi = np.linalg.inv(np.dot(x0.T, x0)+lamda*np.eye(nvars))
    XTY = np.dot(x0.T, y0) #xi * y   #np.dot(xi, y)
    #beta = np.linalg.solve(XTX, XTY)
    beta = np.dot(XTXi, XTY)
    #print 'beta', beta
    rparams[skip-1] = beta
    yipred = np.dot(x[skip-1], beta)
    rypred[skip-1] = yipred
    rresid[skip-1] = y[skip-1] - yipred
    rvarraw[skip-1] = 1 + np.dot(x[skip-1],np.dot(XTXi, x[skip-1]))
    for i in range(skip,nobs):
        xi = x[i:i+1,:]
        yi = y[i]
        #xxT = np.dot(xi.T, xi)  #xi is 2d 1 row
        xy = (xi*yi).ravel() # XTY is 1d  #np.dot(xi, yi)   #np.dot(xi, y)
        #print xy.shape, XTY.shape
        #print XTX
        #print XTY

        # get prediction error with previous beta
        yipred = np.dot(xi, beta)
        rypred[i] = yipred
        residi = yi - yipred
        rresid[i] = residi

        #update beta and inverse(X'X)
        tmp = np.dot(XTXi, xi.T)
        ft = 1 + np.dot(xi, tmp)

        XTXi = XTXi - np.dot(tmp,tmp.T) / ft  #BigJudge equ 5.5.15

        #print 'beta', beta
        beta = beta + (tmp*residi / ft).ravel()  #BigJudge equ 5.5.14
#        #version for testing
#        XTY += xy
#        beta = np.dot(XTXi, XTY)
#        print (tmp*yipred / ft).shape
#        print 'tmp.shape, ft.shape, beta.shape', tmp.shape, ft.shape, beta.shape
        rparams[i] = beta
        rvarraw[i] = ft



    i = nobs
    #beta = np.linalg.solve(XTX, XTY)
    #rparams[i] = beta

    rresid_scaled = rresid/np.sqrt(rvarraw)   #this is N(0,sigma2) distributed
    nrr = nobs-skip
    sigma2 = rresid_scaled[skip-1:].var(ddof=1)
    rresid_standardized = rresid_scaled/np.sqrt(sigma2) #N(0,1) distributed
    rcusum = rresid_standardized[skip-1:].cumsum()
    #confidence interval points in Greene p136 looks strange?
    #this assumes sum of independent standard normal
    #rcusumci = np.sqrt(np.arange(skip,nobs+1))*np.array([[-1.],[+1.]])*stats.norm.sf(0.025)
    a = 1.143 #for alpha=0.99  =0.948 for alpha=0.95
    #following taken from Ploberger,
    crit = a*np.sqrt(nrr)
    rcusumci = (a*np.sqrt(nrr) + a*np.arange(0,nobs-skip)/np.sqrt(nrr)) * np.array([[-1.],[+1.]])
    return rresid, rparams, rypred, rresid_standardized, rresid_scaled, rcusum, rcusumci



