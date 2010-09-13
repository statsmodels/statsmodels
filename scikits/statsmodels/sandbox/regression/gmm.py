'''Generalized Method of Moments, GMM, and Two-Stage Least Squares for
instrumental variables IV2SLS



Issues
------
* number of parameters, nparams, and starting values for parameters
  Where to put them? start was taken from global scope (bug)
* When optimal weighting matrix cannot be calculated numerically
  In DistQuantilesGMM, we only have one row of moment conditions, not a
  moment condition for each observation, calculation for cov of moments
  breaks down. iter=1 works (weights is identity matrix)
  -> need method to do one iteration with an identity matrix or an
     analytical weighting matrix goven as parameter.
  -> add result statistics for this case, e.g. cov_params, I have it in the
     standalone function (and in calc_covparams which is a copy of it),
     but not tested yet.


Author: josef-pktd
License: BSD (3-clause)

'''




import numpy as np
from scipy import optimize
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime1, approx_hess
from scikits.statsmodels.model import LikelihoodModel, LikelihoodModelResults
from scikits.statsmodels.regression import RegressionResults, OLS


def maxabs(x):
    return np.abs(x).max()


class IV2SLS(LikelihoodModel):

    def __init__(self, endog, exog, instrument=None):
        self.instrument = instrument
        super(IV2SLS, self).__init__(endog, exog)
        # where is this supposed to be handled
        #Note: Greene p.77/78 dof correction is not necessary (because only
        #       asy results), but most packages do it anyway
        self.df_resid = exog.shape[0] - exog.shape[1] + 1

    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog

    def whiten(self, X):
        pass

    def fit(self):
        #Greene 5th edt., p.78 section 5.4
        #move this maybe
        y,x,z = self.endog, self.exog, self.instrument
        ztz = np.dot(z.T, z)
        ztx = np.dot(z.T, x)
        self.xhatparams = xhatparams = np.linalg.solve(ztz, ztx)
        print 'x.T.shape, xhatparams.shape', x.shape, xhatparams.shape
        F = xhat = np.dot(z, xhatparams)
        FtF = np.dot(F.T, F)
        self.xhatprod = FtF  #store for Housman specification test
        Ftx = np.dot(F.T, x)
        Fty = np.dot(F.T, y)
        params = np.linalg.solve(FtF, Fty)
        Ftxinv = np.linalg.inv(Ftx)
        self.normalized_cov_params = np.dot(Ftxinv.T, np.dot(FtF, Ftxinv))

        lfit = RegressionResults(self, params,
                       normalized_cov_params=self.normalized_cov_params)
        self._results = lfit
        return lfit

    #copied from GLS, because I subclass currently LikelihoodModel and not GLS
    def predict(self, exog, params=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        #JP: this doesn't look correct for GLMAR
        #SS: it needs its own predict method
        if self._results is None and params is None:
            raise ValueError, "If the model has not been fit, then you must specify the params argument."
        if self._results is not None:
            return np.dot(exog, self._results.params)
        else:
            return np.dot(exog, params)

    def spec_hausman(self, dof=None):
        '''Hausman's specification test


        See Also
        --------
        spec_hausman : generic function for Hausman's specification test

        '''
        #use normalized cov_params for OLS

        resols = OLS(endog, exog).fit()
        normalized_cov_params_ols = resols.model.normalized_cov_params
        se2 = resols.mse_resid

        params_diff = self._results.params - resols.params

        cov_diff = np.linalg.pinv(self.xhatprod) - normalized_cov_params_ols
        #TODO: the following is very inefficient, solves problem (svd) twice
        #use linalg.lstsq or svd directly
        #cov_diff will very often be in-definite (singular)
        if not dof:
            dof = sm.tools.rank(cov_diff)
        cov_diffpinv = np.linalg.pinv(cov_diff)
        H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff))/se2
        pval = stats.chi2.sf(H, dof)

        return H, pval, dof








###############  GMM with standalone functions, only for development

#copied from distributions estimation and not fully adjusted
def fitgmm(momcond, args, start, weights=None, fixed=None, weightsoptimal=True):
    '''estimate parameters of distribution function for binned data using GMM

    Parameters
    ----------
    momcond : function
        needs to return (nobs, nmoms) array


    Returns
    -------
    paramest : array
        estimated parameters

    Notes
    -----
    todo: add fixed parameter option

    added factorial

    '''
    if not fixed is None:
        raise NotImplementedError

    tmp = momcond(start, *args)
    nmoms = tmp.shape[-1]
    if weights is None:
        if weightsoptimal:
            raise NotImplementedError
            weights = freq/float(nobs)
        else:
            weights = np.eye(nmoms)
    # skip turning weights into matrix diag(freq/float(nobs))
    def gmmobjective(params):
        '''negative loglikelihood function of binned data

        corresponds to multinomial
        '''

        moms = momcond(params, *args)
        #return np.dot(moms*weights, moms)
        return np.dot(np.dot(moms.sum(0),weights), moms.sum(0))
    return optimize.fmin(gmmobjective, start)

def gmmiter(momcond, start, maxiter=2, args=None, centered_weights=True):
    '''iteration over gmm estimation with updating of optimal weighting matrix

    '''
    w = np.eye(len(start))
    for i in range(maxiter):
        resgmm = fitgmm(momcond, args, start, weights=w, fixed=None, weightsoptimal=False)
        moms = momcond(resgmm, *args)
        if centered_weights:
            w = np.cov(moms, rowvar=0) # note: I need this also for cov_params
        else:
            w = np.dot(moms.T, moms)
        start = resgmm
    return resgmm

def gmmcov_params(moms, gradmoms, weights=None, has_optimal_weights=True, centered_weights=True):
    '''calculate covariance of parameter estimates

    not all options tried out yet
    '''

    nobs = moms.shape[0]
    if centered_weights:
        # note: code duplication from gmmiter
        omegahat = np.cov(moms, rowvar=0)  # estimate of moment covariance
    else:
        omegahat = np.dot(moms.T, moms)/nobs
    #add other options, Barzen, ...  longrun var estimators
    if weights is None: #has_optimal_weights:
        cov = np.linalg.inv(np.dot(gradmoms.T,
                                   np.dot(np.linalg.inv(omegahat), gradmoms)))
    else:
        gw = np.dot(gradmoms.T, weights)
        gwginv = np.linalg.inv(np.dot(gw, gradmoms))
        cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)

    return cov/nobs


########## end standalone GMM functions


class GMM(object):
    '''
    Class for estimation by Generalized Method of Moments

    needs to be subclassed, where the subclass defined the moment conditions
    `momcond`

    Parameters
    ----------
    endog : array
        endogenous variable, see notes
    exog : array
        array of exogenous variables, see notes
    instrument : array
        array of instruments, see notes
    nmoms : None or int
        number of moment conditions, if None then it is set equal to the
        number of columns of instruments. Mainly needed to determin the shape
        or size of start parameters and starting weighting matrix.
    kwds : anything
        this is mainly if additional variables need to be stored for the
        calculations of the moment conditions

    Attributes
    ----------
    results : instance of GMMResults
        currently just a storage class for params and cov_params without it's
        own methods
    bse : property
        return bse


    Methods
    -------
    fit
    cov_params

    other methods

    fititer
    fitgmm
    calc_weightmatrix


    Notes
    -----
    The GMM class only uses the moment conditions and does not use any data
    directly. endog, exog, instrument and kwds in the creation of the class
    instance are only used to store them for access in the moment conditions.
    Which of this are required and how they are used depends on the moment
    conditions of the subclass.

    Warning:

    Options for various methods have not been fully implemented and
    are still missing in several methods.



    '''

    def __init__(self, endog, exog, instrument, nmoms=None, **kwds):
        '''
        maybe drop and use mixin instead

        GMM doesn't really care about the data, just the moment conditions
        '''
        self.endog = endog
        self.exog = exog
        self.instrument = instrument
        self.nmoms = nmoms or instrument.shape[1]
        self.results = GMMResults()
        self.__dict__.update(kwds)
        self.epsilon_iter = 1e-6

    def fit(self, start=None):
        '''
        Estimate the parameters using default settings.

        For estimation with more options use fititer method.

        Returns
        -------
        results : instance of GMMResults
            this is also attached as attribute results

        Notes
        -----
        this function attaches the estimated parameters, params, the
        weighting matrix of the final iteration, weights, and the value
        of the GMM objective function, jval to results


        '''
        #bug: where does start come from ???
        if start is None:
            start = self.fitstart() #TODO: temporary hack
        params, weights = self.fititer(start, maxiter=10, start_weights=None,
                                        weights_method='cov', wargs=())
        self.results.params = params
        self.results.weights = weights
        self.results.jval = self.gmmobjective(params, weights)

        return self.results


    def fitgmm(self, start, weights=None):
        '''estimate parameters using GMM

        Parameters
        ----------
        momcond : function
            needs to return (nobs, nmoms) array


        Returns
        -------
        paramest : array
            estimated parameters

        Notes
        -----
        todo: add fixed parameter option, not here ???

        added factorial

        '''
##        if not fixed is None:  #fixed not defined in this version
##            raise NotImplementedError

        #tmp = momcond(start, *args)  # forgott to delete this
        #nmoms = tmp.shape[-1]
        if weights is None:
            weights = np.eye(self.nmoms)


##        def gmmobjective(self, params):
##            '''
##            objective function for GMM minimization
##
##            Parameters
##            ----------
##            params : array
##               parameter values at which objective is evaluated
##
##            uses weights from outer scope
##
##            '''
##            moms = momcond(params, *args)
##            return np.dot(np.dot(moms.sum(0),weights), moms.sum(0))
        return optimize.fmin(self.gmmobjective, start, (weights,), disp=0)

    def gmmobjective(self, params, weights):
        '''
        objective function for GMM minimization

        Parameters
        ----------
        params : array
            parameter values at which objective is evaluated
        weights : array
            weighting matrix

        Returns
        -------
        jval : float
            value of objective function

        '''
        moms = self.momcond(params)
        return np.dot(np.dot(moms.sum(0),weights), moms.sum(0))


    def fititer(self, start, maxiter=2, start_weights=None,
                    weights_method='cov', wargs=()):
        '''iterative estimation with updating of optimal weighting matrix

        stopping criteria are maxiter or change in parameter estimate less
        than self.epsilon_iter, with default 1e-6.

        Parameters
        ----------
        start : array
            starting value for parameters
        maxiter : int
            maximum number of iterations
        start_weights : array (nmoms, nmoms)
            initial weighting matrix; if None, then the identity matrix
            is used
        weights_method : {'cov', ...}
            method to use to estimate the optimal weighting matrix,
            see calc_weightmatrix for details

        Returns
        -------
        params : array
            estimated parameters
        weights : array
            optimal weighting matrix calculated with final parameter
            estimates

        Notes
        -----




        '''
        momcond = self.momcond

        if start_weights is None:
            w = np.eye(self.nmoms)
        else:
            w = start_weights

        #call fitgmm function
        #args = (self.endog, self.exog, self.instrument)
        #args is not used in the method version
        for it in range(maxiter):
            winv = np.linalg.inv(w)
            #this is still calling function not method
##            resgmm = fitgmm(momcond, (), start, weights=winv, fixed=None,
##                            weightsoptimal=False)
            resgmm = self.fitgmm(start, weights=winv)

            moms = momcond(resgmm)
            w = self.calc_weightmatrix(moms, method='momcov', wargs=())

            if it > 2 and maxabs(resgmm - start) < self.epsilon_iter:
                #check rule for early stopping
                break
            start = resgmm
        return resgmm, w

    #todo: check if there is a matrix inverse missing somewhere, after
    #   converting to method
    def calc_weightmatrix(self, moms, method='momcov', wargs=()):
        '''calculate omega or the weighting matrix

        Parameters
        ----------

        moms : array, (nobs, nmoms)
            moment conditions for all observations evaluated at a parameter
            value
        method : 'momcov', anything else
            If method='momcov' is cov then the matrix is calculated as simple
            covariance of the moment conditions. For anything else, the
            uncentered moments are used. (The latter is not recommended.)
        wargs : tuple
            parameters that are required by some kernel methods to
            estimate the long-run covariance. Not used yet.

        Returns
        -------
        w : array (nmoms, nmoms)
            estimate for the weighting matrix or covariance of the moment
            condition



        TODO: implement long-run cov estimators, kernel-based

        Newey-West
        Andrews
        Andrews-Moy????

        References
        ----------
        Greene
        Hansen, Bruce

        '''
        nobs = moms.shape[0]
        if method == 'momcov':
            w = np.cov(moms, rowvar=0)
        elif method == 'fakekernel':
            #uniform cut-off window
            moms_centered = moms - moms.mean()
            maxlag = 5
            h = np.ones(maxlag)
            w = np.dot(moms.T, moms)/nobs
            for i in range(1,maxlag+1):
                w += (h * np.dot(moms_centered[i:].T, moms_centered[:-i]) /
                                                                  (nobs-i))
        else:
            w = np.dot(moms.T, moms)/nobs

        return w


    def momcond_mean(self, params):
        '''
        mean of moment conditions,

        '''

        #endog, exog = args
        return self.momcond(params).mean(0)

    def gradient_momcond(self, params, epsilon=1e-4, method='centered'):

        momcond = self.momcond_mean
        if method == 'centered':
            gradmoms = (approx_fprime1(params, momcond, epsilon=epsilon) +
                    approx_fprime1(params, momcond, epsilon=-epsilon))/2
        else:
            gradmoms = approx_fprime1(params, momcond, epsilon=epsilon)

        return gradmoms


    def cov_params(self):  #TODO add options ???
        if not hasattr(self.results, 'params'):
            raise ValueError('the model has to be fit first')

        if hasattr(self.results, '_cov_params'):
            #replace with decorator later
            return self.results._cov_params

        gradmoms = self.gradient_momcond(self.results.params)
        moms = self.momcond(self.results.params)
        covparams = self.calc_cov_params(moms, gradmoms)
        self.results._cov_params = covparams
        return self.results._cov_params



    #still needs to be fully converted to method
    def calc_cov_params(self, moms, gradmoms, weights=None, has_optimal_weights=True, centered_weights=True):
        '''calculate covariance of parameter estimates

        not all options tried out yet
        '''

        nobs = moms.shape[0]
        if centered_weights:
            # note: code duplication from gmmiter
            omegahat = np.cov(moms, rowvar=0)  # estimate of moment covariance
        else:
            omegahat = np.dot(moms.T, moms)/nobs
        #add other options, Barzen, ...  longrun var estimators
        if weights is None: #has_optimal_weights:
            cov = np.linalg.inv(np.dot(gradmoms.T,
                                       np.dot(np.linalg.inv(omegahat), gradmoms)))
        else:
            gw = np.dot(gradmoms.T, weights)
            gwginv = np.linalg.inv(np.dot(gw, gradmoms))
            cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)

        return cov/nobs

    @property
    def bse(self):
        return self.get_bse()

    def get_bse(self, method=None):
        '''

        method option not defined yet
        '''
        return np.sqrt(np.diag(self.cov_params()))

class GMMResults(object):
    '''just a storage class right now'''
    pass

class IVGMM(GMM):
    '''
    Class for linear instrumental variables estimation with homoscedastic
    errors

    currently mainly a test case, doesn't exploit linear structure

    '''

    def fitstart(self):
        return np.zeros(self.exog.shape[1])

    def momcond(self, params):
        endog, exog, instrum = self.endog, self.exog, self.instrument
        return instrum * (endog - np.dot(exog, params))[:,None]

#not tried out yet
class NonlinearIVGMM(GMM):
    '''
    Class for linear instrumental variables estimation with homoscedastic
    errors

    currently mainly a test case, doesn't exploit linear structure

    '''

    def fitstart(self):
        #might not make sense for more general functions
        return np.zeros(self.exog.shape[1])

    def __init__(self, endog, exog, instrument, **kwds):
        self.func = func

    def momcond(self, params):
        endog, exog, instrum = self.endog, self.exog, self.instrument
        return instrum * (endog - self.func(params, exog))[:,None]


def spec_hausman(params_e, params_i, cov_params_e, cov_params_i, dof=None):
    '''Hausmans specification test

    (params_e, cov_params_e) :
        efficient and consistent under Null hypothesis,
        inconsistent under alternative hypothesis

    params_i, cov_params_i
        consistent under Null hypothesis,
        consistent under alternative hypothesis

    example instrumental variables OLS estimator is `e`, IV estimator is `i`

    Todos,Issues
    - check dof calculations and verify for linear case
    - check one-sided hypothesis


    References
    ----------
    Greene section 5.5 p.82/83


    '''
    params_diff = (params_i - params_e)
    cov_diff = cov_params_i - cov_params_e
    #TODO: the following is very inefficient, solves problem (svd) twice
    #use linalg.lstsq or svd directly
    #cov_diff will very often be in-definite (singular)
    if not dof:
        dof = sm.tools.rank(cov_diff)
    cov_diffpinv = np.linalg.pinv(cov_diff)
    H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff))
    pval = stats.chi2.sf(H, dof)

    return H, pval, dof




###########

class DistQuantilesGMM(GMM):
    '''
    Estimate distribution parameters by GMM based on matching quantiles



    '''

    def __init__(self, endog, exog, instrument, **kwds):
        #TODO: something wrong with super
        #super(self.__class__).__init__(endog, exog, instrument) #, **kwds)
        #self.func = func
        self.epsilon_iter = 1e-5

        self.distfn = kwds['distfn']
        #done by super doesn't work yet
        #TypeError: super does not take keyword arguments
        self.endog = endog

        #make this optional for fit
        if not 'pquant' in kwds:
            self.pquant = pquant = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
        else:
            self.pquant = pquant = kwds['pquant']

        #TODO: vectorize this: use edf
        self.xquant = np.array([stats.scoreatpercentile(endog, p) for p
                                in pquant*100])
        self.nmoms = len(self.pquant)

        #TODOcopied from GMM, make super work
        self.endog = endog
        self.exog = exog
        self.instrument = instrument
        self.results = GMMResults()
        #self.__dict__.update(kwds)
        self.epsilon_iter = 1e-6

    def fitstart(self):
        #todo: replace with or add call to distfn._fitstart
        #      added but not used during testing, avoid Travis
        distfn = self.distfn
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(x)
        else:
            start = [1]*distfn.numargs + [0.,1.]

        return np.array([1]*self.distfn.numargs + [0,1])

    def momcond(self, params): #drop distfn as argument
        #, mom2, quantile=None, shape=None
        '''moment conditions for estimating distribution parameters by matching
        quantiles, defines as many moment conditions as quantiles.

        Returns
        -------
        difference : array
            difference between theoretical and empirical quantiles

        Notes
        -----
        This can be used for method of moments or for generalized method of
        moments.

        '''
        #this check looks redundant/unused know
        if len(params) == 2:
            loc, scale = params
        elif len(params) == 3:
            shape, loc, scale = params
        else:
            #raise NotImplementedError
            pass #see whether this might work, seems to work for beta with 2 shape args

        #mom2diff = np.array(distfn.stats(*params)) - mom2
        #if not quantile is None:
        pq, xq = self.pquant, self.xquant
        #ppfdiff = distfn.ppf(pq, alpha)
        cdfdiff = self.distfn.cdf(xq, *params) - pq
        #return np.concatenate([mom2diff, cdfdiff[:1]])
        return np.atleast_2d(cdfdiff    )

#######original version of GMM estimation of distribution parameters

from scipy import stats

def momentcondquant(distfn, params, mom2, quantile=None, shape=None):
    '''moment conditions for estimating distribution parameters by matching
    quantiles, defines as many moment conditions as quantiles.

    Returns
    -------
    difference : array
        difference between theoretical and empirical quantiles

    Notes
    -----
    This can be used for method of moments or for generalized method of
    moments.

    '''
    #this check looks redundant/unused know
    if len(params) == 2:
        loc, scale = params
    elif len(params) == 3:
        shape, loc, scale = params
    else:
        #raise NotImplementedError
        pass #see whether this might work, seems to work for beta with 2 shape args

    #mom2diff = np.array(distfn.stats(*params)) - mom2
    #if not quantile is None:
    pq, xq = quantile
    #ppfdiff = distfn.ppf(pq, alpha)
    cdfdiff = distfn.cdf(xq, *params) - pq
    #return np.concatenate([mom2diff, cdfdiff[:1]])
    return cdfdiff

def fitquantilesgmm(distfn, x, start=None, pquant=None, frozen=None):
    if pquant is None:
        pquant = np.array([0.01, 0.05,0.1,0.4,0.6,0.9,0.95,0.99])
    if start is None:
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(x)
        else:
            start = [1]*distfn.numargs + [0.,1.]
    #TODO: vectorize this:
    xqs = [stats.scoreatpercentile(x, p) for p in pquant*100]
    mom2s = None
    parest = optimize.fmin(lambda params:np.sum(
        momentcondquant(distfn, params, mom2s,(pquant,xqs), shape=None)**2), start)
    return parest

######## original examples of moment conditions:

def momcondOLS(params, endog, exog):
    #print exog.T.shape, endog.shape, params.shape
    #print np.dot(exog, params).shape
    #return np.dot(exog.T, endog - np.dot(exog, params))
    return exog * (endog - np.dot(exog, params))[:,None]

def momcondIVLS(params, endog, exog, instrum):
    return instrum * (endog - np.dot(exog, params))[:,None]


if __name__ == '__main__':

    exampledata = ['ols', 'iv', 'ivfake'][1]
    nobs = nsample = 500
    sige = 3
    corrfactor = 0.01


    x = np.linspace(0,10, nobs)
    X = sm.add_constant(np.column_stack((x, x**2)))
    beta = np.array([1, 0.1, 10])

    def sample_ols(exog):
        endog = np.dot(exog, beta) + sige*np.random.normal(size=nobs)
        return endog, exog, None

    def sample_iv(exog):
        print 'using iv example'
        X = exog.copy()
        e = sige * np.random.normal(size=nobs)
        endog = np.dot(X, beta) + e
        exog[:,0] = X[:,0] + corrfactor * e
        z0 = X[:,0] + np.random.normal(size=nobs)
        z1 = X.sum(1) + np.random.normal(size=nobs)
        z2 = X[:,1]
        z3 = (np.dot(X, np.array([2,1, 0])) +
                        sige/2. * np.random.normal(size=nobs))
        z4 = X[:,1] + np.random.normal(size=nobs)
        instrument = np.column_stack([z0, z1, z2, z3, z4, X[:,-1]])
        return endog, exog, instrument

    def sample_ivfake(exog):
        X = exog
        e = sige * np.random.normal(size=nobs)
        endog = np.dot(X, beta) + e
        #X[:,0] += 0.01 * e
        #z1 = X.sum(1) + np.random.normal(size=nobs)
        #z2 = X[:,1]
        z3 = (np.dot(X, np.array([2,1, 0])) +
                        sige/2. * np.random.normal(size=nobs))
        z4 = X[:,1] + np.random.normal(size=nobs)
        instrument = np.column_stack([X[:,:2], z3, z4, X[:,-1]]) #last is constant
        return endog, exog, instrument


    if exampledata == 'ols':
        endog, exog, _ = sample_ols(X)
        instrument = exog
    elif exampledata == 'iv':
        endog, exog, instrument = sample_iv(X)
    elif exampledata == 'ivfake':
        endog, exog, instrument = sample_ivfake(X)





    results = sm.OLS(endog, exog).fit()
    start = beta * 0.9
    resgmm = fitgmm(momcondOLS, (endog, exog), start, fixed=None, weightsoptimal=False)
    print resgmm
    print results.params

    print gmmiter(momcondOLS, start, maxiter=3, args=(endog, exog))
    ##w = np.eye(len(start))
    ##for i in range(10):
    ##    resgmm = fitgmm(momcondOLS, (y,X), start, weights=w, fixed=None, weightsoptimal=False)
    ##    moms = momcondOLS(resgmm, y, X)
    ##    w = np.dot(moms.T,moms)
    ##    start = resgmm
    ##
    ##    print resgmm


    # OLS using IV
    resgmmiv = fitgmm(momcondIVLS, (endog, exog, exog), start, fixed=None, weightsoptimal=False)
    print gmmiter(momcondIVLS, start, maxiter=3, args=(endog, exog, exog))

    #endog, exog = y, X
    def momcond(params):
        #endog, exog = args
        return momcondOLS(params, endog, exog).mean(0)

    #maybe grad first and then sum, need 3d return from approx_fprime1
    #   I don't think so, sum/mean and differentiation can be interchanged (?)
    gradgmm = approx_fprime1(resgmm, momcond)
    moms = momcondOLS(resgmm, endog, exog)
    w = np.dot(moms.T,moms)

    gradmoms2s = (approx_fprime1(resgmm, momcond, epsilon=1e-4)+approx_fprime1(resgmm, momcond, epsilon=-1e-4))/2

    covgmm = gmmcov_params(moms, gradmoms2s)
    print results.cov_params()
    print covgmm
    print 'percent difference'
    print (results.cov_params() - covgmm) / results.cov_params() *100
    print results.bse
    print np.sqrt(np.diag(covgmm))
    covgmmw = gmmcov_params(moms, gradmoms2s,
                            weights=np.linalg.inv(np.dot(moms.T, moms)))
    covgmmw2 = gmmcov_params(moms, gradmoms2s,
                             weights=np.linalg.inv(np.cov(moms, rowvar=0)))


    #using GMM and IV2SLS classes
    #----------------------------

    mod = IVGMM(endog, exog, instrument, nmoms=instrument.shape[1])
    res = mod.fit()
    modls = IV2SLS(endog, exog, instrument)
    resls = modls.fit()
    modols = OLS(endog, exog)
    resols = modols.fit()

    print '\nIV case'
    print 'params'
    print 'IV2SLS', resls.params
    print 'OLS   ', resols.params
    print 'GMM   ', res.params
    print 'diff  ', res.params - resls.params
    print '\nbse'
    print 'IV2SLS', resls.bse
    print 'OLS   ', resols.bse
    print 'GMM   ', mod.bse   #bse currently only attached to model not results
    print 'diff  ', mod.bse - resls.bse
    print '%-diff', resls.bse / mod.bse * 100 - 100

    print "Hausman's specification test"
    print modls.spec_hausman()
    print spec_hausman(resols.params, res.params, resols.cov_params(),
                       mod.cov_params())



    #estimating distribution parameters from quantiles
    #-------------------------------------------------

    #example taken from distribution_estimators.py
    gparrvs = stats.genpareto.rvs(2, size=500)
    x0p = [1., gparrvs.min()-5, 1]
    pfunc = fitquantilesgmm(stats.genpareto, gparrvs, start=x0p,
                          pquant=np.linspace(0.01,0.99,10), frozen=None)
    print pfunc

    moddist = DistQuantilesGMM(gparrvs, None, None, distfn=stats.genpareto)
    #produces non-sense because optimal weighting matrix calculations don't
    #apply to this case
    #resgp = moddist.fit() #now with 'cov': LinAlgError: Singular matrix
    pit1, wit1 = moddist.fititer([1.5,0,1.5], maxiter=1)
    print pit1
    p1 = moddist.fitgmm([1.5,0,1.5])
    print p1
    moddist2 = DistQuantilesGMM(gparrvs, None, None, distfn=stats.genpareto,
                                pquant=np.linspace(0.01,0.99,10))
    pit1a, wit1a = moddist2.fititer([1.5,0,1.5], maxiter=1)
    print pit1a
    p1a = moddist2.fitgmm([1.5,0,1.5])
    print p1a
    #Note: pit1a and p1a are the same and almost the same (1e-5) as
    #      fitquantilesgmm version (functions instead of class)
    print p1a - pfunc

