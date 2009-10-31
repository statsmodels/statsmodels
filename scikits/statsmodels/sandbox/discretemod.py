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
from scipy import stats, factorial

#TODO: is there not a logistic distribution or Weibull distribution in
#TODO: scipy.stats?

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
    def initialize(self):
        pass

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
