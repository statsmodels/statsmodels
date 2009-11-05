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

def block_eye(N, k=(), dtype=float):
    """
    """
    m = np.zeros((N,N), dtype=dtype)

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

class MNLogit(DiscreteModel):
    def initialize(self):
#This is also a "whiten" method
        wendog, self.names = sm.tools.categorical(self.endog, drop=True,
                dictnames=True)
        self.wendog = wendog    # don't drop first category
        self.J = wendog.shape[1]
        self.K = self.exog.shape[1]

    def pdf(self, params):
        exog = self.exog
#        endog = self.endog
#        eXB = np.exp(np.dot(exog, params.reshape(exog.shape[1],-1)))
                # pred vals for each level except 0
#        eXB = np.column_stack((np.ones((self.nobs,1)), eXB))
                # add 1 for b0 = vec(0)

# change to using rows so that hessians, etc are easier
        eXB = np.exp(np.dot(params.reshape(-1, exog.shape[1]), exog.T))
        eXB = np.vstack((np.ones((1, self.nobs)), eXB))

        num = eXB
#        denom = 1 + eXB.sum(axis=1)
        denom = eXB.sum(axis=0)
#        return num/denom[:,None]
        return num/denom[None,:]

    def loglike(self, params):
        d = self.wendog
        logprob = np.log(self.pdf(params))
        return (d.T * logprob).sum()

    def score(self, params):
        """
        Score matrix for multinomial model

        In the multinomial model ths score matrix is K x J-1

        Returned as a flattened array to work with the solvers.
        """
#        firstterm = self.wendog[:,1:] - self.pdf(params)[:,1:]
        firstterm = self.wendog[:,1:].T - self.pdf(params)[1:,:]
#        return np.dot(self.exog.T, firstterm).flatten(1)
        return np.dot(firstterm, self.exog).flatten(0)

    def hessian(self, params):
        """
        Hessian matrix for multinomial model

        The hessian matrix has J**2 * K x K blocks.
        Note that ours will have this same number of elements but
        a different shape because of the shapes needed for solvers...
        """
#TODO: test this for a model where K != J-1
#        hess = nd.Jacobian(self.score)
#        h = hess(params)
        X = self.exog
        pr = self.pdf(params)
        partials = []
        J = self.wendog.shape[1] - 1
        K = self.exog.shape[1]
# This doesn't take advantage of symmetry, so computes upper and lower
        for i in range(J):
            for j in range(J): # this loop assumes we drop the first col.
                if i == j:
                    partials.append(\
                        -np.dot((pr[i+1,:]*(1-pr[j+1,:]))[None,:]*X.T,X))
                else:
                    partials.append(-np.dot(pr[i+1,:]*-pr[j+1,:][None,:]*X.T,X))
        H = np.array(partials)
# We now have a matrix that's J**2, K, K I believe, so we need to reshape this
# to be J*K, J*K as follows, see math note (once I've updated it)
# to clear this up.
# Test for other Js and Ks to make sure this is robust
        H = np.transpose(H.reshape(J,J,K,K), (0,2,1,3)).reshape(J*K,J*K)
#Also realize that once we only caclculate the J*(J-1)/2. this might be different

        return H
#        self.h = h
#        return h

    def fit(self, start_params=None, maxiter=35, method='newton',
            tol=1e-08):
        if start_params is None:
            start_params = np.zeros((self.exog.shape[1]*\
                    (self.wendog.shape[1]-1)))
        mlefit = super(MNLogit, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
#        mlefit.params = mlefit.params.reshape(self.exog.shape[1],-1)
        mlefit.params = mlefit.params.reshape(-1, self.exog.shape[1])
        return mlefit


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

    def hessian(self, params):
        hess = nd.Jacobian(self.score)
        return hess(params)

    def fit(self, start_params=None, method='newton', maxiter=35, tol=1e-08):
# The example had problems with all zero start values, Hessian = 0
        if start_params is None:
            start_params = sm.OLS(self.endog, self.exog).fit().params
        mlefit = super(Weibull, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, tol=tol)
        return mlefit

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
        return jfun(params)[-1]
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
        return dLdB
#
#        dLda = np.sum(1/alpha**2 * (np.log(1+alpha*mu) - dJ) + \
#                (y-mu)/(alpha*(1+alpha*mu)))
#        scorevec = np.zeros((len(dLdB)+1))
#        scorevec[:-1] = dLdB
#        scorevec[-1] = dLda
#        scorevec[-1] = dLda2[-1]
#        return scorevec

    def hessian(self, params):
        """
        Hessian of NB2 model.  Currently uses numdifftools
        """
#        d2dBdB =
#        d2da2 =
        Hfun = nd.Jacobian(self.score)
        return Hfun(params)[-1]
# is the numerical hessian block diagonal?  or is it block diagonal by assumption?

    def fit(self, start_params=None, maxiter=35, method='bfgs', tol=1e-08):
#        start_params = [0]*(self.exog.shape[1])+[1]
# Use poisson fit as first guess.
        start_params = Poisson(self.endog, self.exog).fit().params
        start_params = np.roll(np.insert(start_params, 0, 1), -1)
        mlefit = super(NegBinTwo, self).fit(start_params=start_params,
                maxiter=maxiter, method=method, tol=tol)
        return mlefit

if __name__=="__main__":
    from urllib2 import urlopen
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
    weibull_res = weibull_mod.fit(method='newton')
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
#    data2 = np.genfromtxt('./dvisits.txt', names=True)
# note that this has missing values for Accident
#    endog = data2['doctorco']
#    exog = data2[['sex','age','agesq','income','levyplus','freepoor',
#            'freerepa','illness','actdays','hscore','chcond1',
#            'chcond2']].view(float).reshape(len(data2),-1)
#    exog = sm.add_constant(exog, prepend=True)
#    poisson_mod = Poisson(endog, exog)
#    poisson_res = poisson_mod.fit()
#    nb2_mod = NegBinTwo(endog, exog)
#    nb2_res = nb2_mod.fit()
# solvers hang (with no error and no maxiter warn...)
# haven't derived hessian (though it will be block diagonal) to check
# newton, note that Lawless (1987) has the derivations
# appear to be something wrong with the score?
# according to Lawless, traditionally the likelihood is maximized wrt to B
# and a gridsearch on a to determin ahat?
# or the Breslow approach, which is 2 step iterative.
    nb2_params = [-2.190,.217,-.216,.609,-.142,.118,-.497,.145,.214,.144,
            .038,.099,.190,1.077] # alpha is last
    # taken from Cameron and Trivedi
# the below is from Cameron and Trivedi as well
#    endog2 = np.array(endog>=1, dtype=float)
# skipped for now, binary poisson results look off?

    # multinomial example from
# http://www.stat.washington.edu/quinn/classes/536/S/multinomexample.html
    mlogdata = np.genfromtxt("./nes96r.dat", names=True)
    mendog = mlogdata['PID']
    mexog = np.column_stack((np.log(mlogdata['popul']+.1),mlogdata[['selfLR',
                'age','educ','income']].view(float).reshape(-1,4)))
    mexog = sm.add_constant(mexog, prepend=True)
    mlogit_mod = MNLogit(mendog, mexog)
#    for PID 0-7 is
# results from R nnet package
    mlogit_arr = np.array([-0.373356261, -2.250934805, -3.665905084,
        -7.613694423, -7.060431370, -12.105193452, -0.011537359,
        -0.088750964, -0.105967684, -0.091555188, -0.093285749,
        -0.140879420,  0.297697981,  0.391662761,  0.573513420,
        1.278742543,  1.346939966,  2.069988287, -0.024944529,
        -0.022897526, -0.014851243, -0.008680754, -0.017903442,
        -0.009432601,  0.082487696, 0.181044184, -0.007131611,
        0.199828063,  0.216938699,  0.321923127,  0.005195818,
        0.047874118,  0.057577321,  0.084495215,  0.080958623, 0.108890412])
# the question is which is more accurate?  Our 3 agree more with each others..
    mlogit_arr = mlogit_arr.reshape(6,-1).T
# the rows are the different K coefs, and the cols are the J-1 responses
# the aboce comment is wrong now
    mlogit_res = mlogit_mod.fit(method = 'bfgs', maxiter=100)
#    mlogit_res2 = mlogit_mod.fit(method = 'ncg', maxiter=100)
    mlogit_res3 = mlogit_mod.fit(method = 'newton', maxiter=25)
#    np.testing.assert_almost_equal(mlogit_res.params, mlogit_arr, 3)
#    np.testing.assert_almost_equal(mlogit_res2.params, mlogit_arr, 3)

# this example taken from
# http://www.ats.ucla.edu/stat/r/dae/mlogit.htm
    mlogdta = np.genfromtxt('./mlogit.csv', delimiter=',', names=True)
    mend = mlogdta['brand']
    mex = mlogdta[['female','age']].view(float).reshape(-1,2)
    mex = sm.add_constant(mex, prepend=True)
    mlog = MNLogit(mend, mex)
    mlog_res = mlog.fit(method='newton')
#    marr = np.array([[22.721396, 10.946741],[-.465941,.057873],
#        [-.685908,-.317702]])
# The above are the results from R using Brand 3 as base outcome
    marr = np.array([[-11.77466, -22.7214],[.5238143, .4659414],
        [.3682065, .6859082]])
# The above results are from Stata using Brand 1 as base outcome
# we match these, but should provide a baseoutcome option


# The last ncg method for mlogit was slow on the last one
# Should have some kind of testing in mlefit to see which
# method will be the fastest
# the non-conjugate gradient methods are always going to be slower
# unless we provide the analytic hessian


