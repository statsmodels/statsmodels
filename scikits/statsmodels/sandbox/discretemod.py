"""
Limited dependent variable and qualitative variables.

Includes binary outcomes, count data, (ordered) ordinal data and limited
dependent variables.

References
-------

Cameron and Trivedi

Madalla

Greene

Davidson and MacKinnon
"""

from scikits.statsmodels.model import LikelihoodModel
from scikits.statsmodels.family import links
from scipy import stats, factorial, special, optimize # opt just for nbin
import numdifftools as nd

#TODO: is there not a logistic distribution or Weibull distribution in
#TODO: scipy.stats?

def add_factorial(X):
    """
    Returns a vector of descending numbers added sequential.

    For instance, if given [5, 4, 0, 2], returns [15, 10, 0, 3].
    """
    X = np.asarray(X)
    return X/2. * X + (1-X)
# or equivalently
#    return X*(X+1)/2.

class DiscreteModel(LikelihoodModel):
    """
    """
    def __init___(endog, exog):
        super(DiscreteModel, self).__init__(endog, exog)

    def initialize(self):
        pass

    def cdf(self, params):
        raise NotImplementedError

    def pdf(self, params):
        raise NotImplementedError


class Poisson(DiscreteModel):

    def initialize(self):
        pass

    def cdf(self, X):
        """
        Poisson model cumulataive distribution function
        """
# X should be XB, but cdf isn't used in algorithm
        y = self.endog
        L = np.exp(X)
        return (np.exp(-L)*L**y)/factorial(y)

    def loglike(self, params):
        """
        Loglikelihood of Poisson model
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        return np.sum(-np.exp(XB) +  endog*XB - np.log(factorial(endog)))

    def score(self, params):
        """
        Poisson model score function
        """
        X = self.exog
        L = np.exp(np.dot(X,params))
        return np.dot(self.endog - L,X)

    def hessian(self, params):
        X = self.exog
        L = np.exp(np.dot(X,params))
        return -np.dot(L*X.T, X)

class NbReg(DiscreteModel):
    pass

class mLogit(DiscreteModel):
    pass

class Logit(DiscreteModel):
    """
    Binary choice logit model
    """
    #TODO: should this be called distribution or something?

    def cdf(self, X):
        """
        The logistic cumulative distribution function

        Parameters
        ----------
        X : array-like

        Returns
        -------
        exp(X)/(1 + exp(X))
        """
        return 1/(1+np.exp(-X))

#    def pdf(self, X):
#        """
#        The logistic probability density function
#        """
#        return np.exp(-X)/((1+np.exp(-X)**2)

    def loglike(self, params):
        """
        Log likelihood of logit link.
        """
        q = 2*self.endog - 1
        X = self.exog
        return np.sum(np.log(self.cdf(q*np.dot(X,params))))

    def score(self, params):
        """
        Score vector of Logit model
        """
        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return np.dot(y - L,X)

    def hessian(self, params):
        """
        Hessian matrix of Logit model
        """
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return -np.dot(L*(1-L)*X.T,X)


class Probit(DiscreteModel):
    """
    Binary choice Probit model
    """
    def pdf(self, X):
        """
        Probit (Normal) probability density function
        """
        return stats.norm.pdf(X)

    def cdf(self, X):
        """
        Probit (Normal) cumulative distribution function
        """
        return stats.norm.cdf(X)  # need to supply loc and scale?

    def loglike(self, params):
        """
        Loglikelihood of probit (normal) distribution.
        """
        q = 2*self.endog - 1
        X = self.exog
        return np.sum(np.log(self.cdf(q*np.dot(X,params))))

    def score(self, params):
        """
        Score vector of Probit model
        """
        y = self.endog
        X = self.exog
        L = self.cdf(np.dot(X,params))
        return np.dot(y-L,X)

    def hessian(self, params):
        """
        Hessian matrix of Probit
        """
        X = self.exog
        XB = np.dot(X,params)
        q = 2*self.endog - 1
        L = q*self.pdf(q*XB)/self.cdf(q*XB)
        return np.dot(-L*(L+XB)*X.T,X)

class Weibull(DiscreteModel):
    """
    Binary choice Weibull model
    """
    def initialize(self):
        pass

    def cdf(self, X):
        """
        Gumbell (Log Weibull) cumulative distribution function
        """
#        return np.exp(-np.exp(-X))
        return stats.gumbel_r.cdf(X)
# these two are equivalent.
# Greene table and discussion is incorrect.

    def pdf(self, X):
        """
        Gumbell (LogWeibull) probability distribution function
        """
        return stats.gumbel_r.pdf(X)

    def loglike(self, params):
        """
        Loglikelihood of Weibull distribution
        """
        X = self.exog
        cdf = self.cdf(np.dot(X,params))
        y = self.endog
        return np.sum(y*np.log(cdf) + (1-y)*np.log(1-cdf))

    def score(self, params):
        y = self.endog
        X = self.exog
        F = self.cdf(np.dot(X,params))
        f = self.pdf(np.dot(X,params))
        term = (y*f/F + (1 - y)*-f/(1-F))
        return np.dot(term,X)

class NegBinTwo(DiscreteModel):
    """
    NB2 Negative Binomial model.
    """
#NOTE: to use this with the solvers, the likelihood fit will probably
# need to be amended to have args, so that we can pass the ancillary param
# if not we can just stick the alpha param on the end of the beta params and
# amend all the methods to reflect this
# if we try to keep them separate I think we'd have to use a callback...
# need to check variance function, then derive score vector, and hessian
# loglike should be fine...
# also, alpha should maybe always be lnalpha to contrain it to be positive

#    def pdf(self, X, alpha):
#        a1 = alpha**-1
#        term1 = special.gamma(X + a1)/(special.agamma(X+1)*special.gamma(a1))

    def loglike(self, params):
        """
        Loglikelihood for NB2 model

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """
        alpha = params[-1]
        params = params[:-1]
        a1 = alpha**-1
        y = self.endog
        J = special.gammaln(y+a1) - special.gammaln(a1)
# See Cameron and Trivedi 1998 for a simplification of the above
# writing a convenience function using the log summation, *might*
# be more accurate
        XB = np.dot(self.exog,params)
        return np.sum(J - np.log(factorial(y)) - \
                (y+a1)*np.log(1+alpha*np.exp(XB))+y*np.log(alpha)+y*XB)

    def score(self, params):
        """
        Score vector for NB2 model
        """
        y = self.endog
        X = self.exog
        jfun = nd.Jacobian(self.loglike)
        print params
        dLda2 = jfun(params)[-1]
        alpha = params[-1]
        params = params[:-1]
        XB = np.dot(X,params)
        mu = np.exp(XB)
        a1 = alpha**-1
        f1 = lambda x: 1./((x-1)*x/2. + x*a1)
        cond = y>0
        dJ = np.piecewise(y, cond, [f1,1./a1])
# if y_i < 1, this equals zero!  Not noted in C &T
        dLdB = np.dot((y-mu)/(1+alpha*mu),X)
        dLda = np.sum(1/alpha**2 * (np.log(1+alpha*mu) - dJ) + \
                (y-mu)/(alpha*(1+alpha*mu)))
        scorevec = np.zeros((len(dLdB)+1))
        scorevec[:-1] = dLdB
#        scorevec[-1] = dLda
        print dLda2[-1]
        scorevec[-1] = dLda2[-1]
        return scorevec

    def hessian(self, params):
        """
        Hessian of NB2 model.  Currently uses numdifftools
        """
#        d2dBdB =
#        d2da2 =
        print params
        Hfun = nd.Jacobian(self.score)
        return Hfun(params)

    def fit(self, start_params=None, maxiter=35, method='bfgs'):
#        start_params = [0]*(self.exog.shape[1])+[1]
# Use poisson fit as first guess.
        start_params = Poisson(self.endog, self.exog).fit().params
        start_params = np.roll(np.insert(start_params, 0, 1), -1)
        mlefit = super(NegBinTwo, self).fit(start_params=start_params,
                maxiter=maxiter, method=method)
        return mlefit

#    def fit(self, start_params=None, maxiter=35):
#        """
#        Temporary solution just to see mistake check.
#        """
#        start_params = ([0]*self.exog.shape[1]+1)
#Right now, the plus one above is pretty much the only diff.
#        f = lambda params, alpha: -self.loglike(params, alpha=alpha)
#        score = lambda params: -self.score(params, alpha=alpha)
#        xopt, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = \
#                optimize.fmin_bfgs(f, start_params, score, args=(alpha=alpha),
#                        full_output=1, maxiter=maxiter)
#        return xopt


if __name__=="__main__":
    from urllib2 import urlopen
    try:
        from scikits.statsmodels import lib
    except:
        raise ImportError, "I haven't distributed PyDTA until the license is \
changed."
    import numpy as np
    import scikits.statsmodels as sm
#    data = np.genfromtxt("http://pages.stern.nyu.edu/~wgreene/Text/Edition6/TableF16-1.txt", names=True)
    data = np.genfromtxt('./TableF16-1.txt', names=True)
    endog = data['GRADE']
    exog = data[['GPA','TUCE','PSI']].view(float).reshape(-1,3)
    exog = sm.add_constant(exog, prepend=True)
    lpm = sm.OLS(endog,exog)
    lmp_res = lpm.fit()
    logit_mod = Logit(endog, exog)
    logit_res = logit_mod.fit()
    probit_mod = Probit(endog, exog)
    probit_res = probit_mod.fit()
    weibull_mod = Weibull(endog, exog)
    weibull_res = weibull_mod.fit(method='ncg')
# The Weibull doesn't converge for bfgs?
#TODO: add hessian for Weibull
    print "This example is based on Greene Table 21.1 5th Edition"
    print lmp_res.params
    print logit_res.params
    print "The following probit parameters are a bit off. Not sure why."
    print probit_res.params
    print "Typo in Greene for Weibull, replaced with logWeibull or Gumbel"
    print "Errata doesn't note coeff. differences."
    print "But these look somewhat ok...?"
    print weibull_res.params
# dvisits was written using an R package, I can provide the dataset
# on request until the copyright is cleared up
    data2 = np.genfromtxt('./dvisits.txt', names=True)
# note that this has missing values for Accident
    endog = data2['doctorco']
    exog = data2[['sex','age','agesq','income','levyplus','freepoor',
            'freerepa','illness','actdays','hscore','chcond1',
            'chcond2']].view(float).reshape(len(data2),-1)
    exog = sm.add_constant(exog, prepend=True)
    poisson_mod = Poisson(endog, exog)
    poisson_res = poisson_mod.fit()
    nb2_mod = NegBinTwo(endog, exog)
#    nb2_res = nb2_mod.fit()
# solvers hang (with no error and no maxiter warn...)
# haven't derived hessian (though it will be block diagonal) to check
# newton
# appear to be something wrong with the score?
    nb2_params = [-2.190,.217,-.216,.609,-.142,.118,-.497,.145,.214,.144,
            .038,.099,.190,1.077] # alpha is last

   arr=np.array([  2.20160970e+02,   1.56881957e-01,   1.05629904e+00,  -8.48703607e-01,
  -2.05320578e-01,   1.23185435e-01,  -4.40060928e-01,   7.97984292e-02,
   1.86948430e-01,   1.26846479e-01,   3.00810049e-02,   1.14085308e-01,
   1.41158279e-01,   1.00000000e+00])
