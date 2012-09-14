'''Generalized Method of Moments, GMM, and Two-Stage Least Squares for
instrumental variables IV2SLS



Issues
------
* number of parameters, nparams, and starting values for parameters
  Where to put them? start was initially taken from global scope (bug)
* When optimal weighting matrix cannot be calculated numerically
  In DistQuantilesGMM, we only have one row of moment conditions, not a
  moment condition for each observation, calculation for cov of moments
  breaks down. iter=1 works (weights is identity matrix)
  -> need method to do one iteration with an identity matrix or an
     analytical weighting matrix given as parameter.
  -> add result statistics for this case, e.g. cov_params, I have it in the
     standalone function (and in calc_covparams which is a copy of it),
     but not tested yet.
  DONE `fitonce` in DistQuantilesGMM, params are the same as in direct call to fitgmm
      move it to GMM class (once it's clearer for which cases I need this.)
* GMM doesn't know anything about the underlying model, e.g. y = X beta + u or panel
  data model. It would be good if we can reuse methods from regressions, e.g.
  predict, fitted values, calculating the error term, and some result statistics.
  What's the best way to do this, multiple inheritance, outsourcing the functions,
  mixins or delegation (a model creates a GMM instance just for estimation).


Unclear
-------
* dof in Hausman
  - based on rank
  - differs between IV2SLS method and function used with GMM or (IV2SLS)
  - with GMM, covariance matrix difference has negative eigenvalues in iv example, ???
* jtest/jval
  - I'm not sure about the normalization (multiply or divide by nobs) in jtest.
    need a test case. Scaling of jval is irrelevant for estimation.
    jval in jtest looks to large in example, but I have no idea about the size
* bse for fitonce look too large (no time for checking now)
    formula for calc_cov_params for the case without optimal weighting matrix
    is wrong. I don't have an estimate for omega in that case. And I'm confusing
    between weights and omega, which are *not* the same in this case.



Author: josef-pktd
License: BSD (3-clause)

'''




import numpy as np
from scipy import optimize, stats
from statsmodels.sandbox.regression.numdiff import approx_fprime1, approx_hess
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults
from statsmodels.regression.linear_model import RegressionResults, OLS
import statsmodels.tools.tools as tools


def maxabs(x):
    '''just a shortcut to np.abs(x).max()
    '''
    return np.abs(x).max()


class IV2SLS(LikelihoodModel):
    '''
    class for instrumental variables estimation using Two-Stage Least-Squares


    Parameters
    ----------
    endog: array 1d
       endogenous variable
    exog : array
       explanatory variables
    instruments : array
       instruments for explanatory variables, needs to contain those exog
       variables that are not instrumented out

    Notes
    -----
    All variables in exog are instrumented in the calculations. If variables
    in exog are not supposed to be instrumented out, then these variables
    need also to be included in the instrument array.


    '''

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
        '''estimate model using 2SLS IV regression

        Returns
        -------
        results : instance of RegressionResults
           regression result

        Notes
        -----
        This returns a generic RegressioResults instance as defined for the
        linear models.

        Parameter estimates and covariance are correct, but other results
        haven't been tested yet, to seee whether they apply without changes.

        '''
        #Greene 5th edt., p.78 section 5.4
        #move this maybe
        y,x,z = self.endog, self.exog, self.instrument
        ztz = np.dot(z.T, z)
        ztx = np.dot(z.T, x)
        self.xhatparams = xhatparams = np.linalg.solve(ztz, ztx)
        #print 'x.T.shape, xhatparams.shape', x.shape, xhatparams.shape
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
            dof = tools.rank(cov_diff)
        cov_diffpinv = np.linalg.pinv(cov_diff)
        H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff))/se2
        pval = stats.chi2.sf(H, dof)

        return H, pval, dof


############# classes for Generalized Method of Moments GMM

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

    Returns
    -------
    *Attributes*
    results : instance of GMMResults
        currently just a storage class for params and cov_params without it's
        own methods
    bse : property
        return bse



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

        Parameters
        ----------
        start : array (optional)
            starting value for parameters ub minimization. If None then
            fitstart method is called for the starting values

        Returns
        -------
        results : instance of GMMResults
            this is also attached as attribute results

        Notes
        -----
        This function attaches the estimated parameters, params, the
        weighting matrix of the final iteration, weights, and the value
        of the GMM objective function, jval to results. The results are
        attached to this instance and also returned.

        fititer is called with maxiter=10


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
        start : array_like
            starting values for minimization
        weights : array
            weighting matrix for moment conditions. If weights is None, then
            the identity matrix is used


        Returns
        -------
        paramest : array
            estimated parameters

        Notes
        -----
        todo: add fixed parameter option, not here ???

        uses scipy.optimize.fmin

        '''
##        if not fixed is None:  #fixed not defined in this version
##            raise NotImplementedError

        #tmp = momcond(start, *args)  # forgott to delete this
        #nmoms = tmp.shape[-1]
        if weights is None:
            weights = np.eye(self.nmoms)

        #TODO: add other optimization options and results
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
        return np.dot(np.dot(moms.mean(0),weights), moms.mean(0))


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


    def calc_weightmatrix(self, moms, method='momcov', wargs=()):
        '''calculate omega or the weighting matrix

        Parameters
        ----------

        moms : array, (nobs, nmoms)
            moment conditions for all observations evaluated at a parameter
            value
        method : 'momcov', anything else
            If method='momcov' is cov then the matrix is calculated as simple
            covariance of the moment conditions. For anything else, a
            constant cutoff window of length 5 is used.
        wargs : tuple
            parameters that are required by some kernel methods to
            estimate the long-run covariance. Not used yet.

        Returns
        -------
        w : array (nmoms, nmoms)
            estimate for the weighting matrix or covariance of the moment
            condition


        Notes
        -----

        currently a constant cutoff window is used
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


    def cov_params(self, **kwds):  #TODO add options ???
        if not hasattr(self.results, 'params'):
            raise ValueError('the model has to be fit first')

        if hasattr(self.results, '_cov_params'):
            #replace with decorator later
            return self.results._cov_params

        gradmoms = self.gradient_momcond(self.results.params)
        moms = self.momcond(self.results.params)
        covparams = self.calc_cov_params(moms, gradmoms, **kwds)
        self.results._cov_params = covparams
        return self.results._cov_params



    #still needs to be fully converted to method
    def calc_cov_params(self, moms, gradmoms, weights=None,
                                              has_optimal_weights=True,
                                              method='momcov', wargs=()):
        '''calculate covariance of parameter estimates

        not all options tried out yet

        If weights matrix is given, then the formula use to calculate cov_params
        depends on whether has_optimal_weights is true.
        If no weights are given, then the weight matrix is calculated with
        the given method, and has_optimal_weights is assumed to be true.

        (API Note: The latter assumption could be changed if we allow for
        has_optimal_weights=None.)

        '''

        nobs = moms.shape[0]
        if weights is None:
            omegahat = self.calc_weightmatrix(moms, method=method, wargs=wargs)
            has_optimal_weights = True
            #add other options, Barzen, ...  longrun var estimators
        else:
            omegahat = weights   #2 different names used,
            #TODO: this is wrong, I need an estimate for omega

        if has_optimal_weights: #has_optimal_weights:
            cov = np.linalg.inv(np.dot(gradmoms.T,
                                       np.dot(np.linalg.inv(omegahat), gradmoms)))
        else:
            gw = np.dot(gradmoms.T, weights)
            gwginv = np.linalg.inv(np.dot(gw, gradmoms))
            cov = np.dot(np.dot(gwginv, np.dot(np.dot(gw, omegahat), gw.T)), gwginv)
            cov = np.linalg.inv(cov)

        return cov/nobs

    @property
    def bse(self):
        '''standard error of the parameter estimates
        '''
        return self.get_bse()

    def get_bse(self, method=None):
        '''

        method option not defined yet
        '''
        return np.sqrt(np.diag(self.cov_params()))

    def jtest(self):
        '''overidentification test

        I guess this is missing a division by nobs,
        what's the normalization in jval ?
        '''

        jstat = self.results.jval
        nparams = self.results.params.size #self.nparams
        return jstat, stats.chi2.sf(jstat, self.nmoms - nparams)

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

    currently mainly a test case, not checked yet

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

    Parameters
    ----------
    params_e : array
        efficient and consistent under Null hypothesis,
        inconsistent under alternative hypothesis
    params_i: array
        consistent under Null hypothesis,
        consistent under alternative hypothesis
    cov_params_e : array, 2d
        covariance matrix of parameter estimates for params_e
    cov_params_i : array, 2d
        covariance matrix of parameter estimates for params_i

    example instrumental variables OLS estimator is `e`, IV estimator is `i`


    Notes
    -----

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
        dof = tools.rank(cov_diff)
    cov_diffpinv = np.linalg.pinv(cov_diff)
    H = np.dot(params_diff, np.dot(cov_diffpinv, params_diff))
    pval = stats.chi2.sf(H, dof)

    evals = np.linalg.eigvalsh(cov_diff)

    return H, pval, dof, evals




###########

class DistQuantilesGMM(GMM):
    '''
    Estimate distribution parameters by GMM based on matching quantiles

    Currently mainly to try out different requirements for GMM when we cannot
    calculate the optimal weighting matrix.

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
        return np.atleast_2d(cdfdiff)

    def fitonce(self, start=None, weights=None, has_optimal_weights=False):
        '''fit without estimating an optimal weighting matrix and return results

        This is a convenience function that calls fitgmm and covparams with
        a given weight matrix or the identity weight matrix.
        This is useful if the optimal weight matrix is know (or is analytically
        given) or if an optimal weight matrix cannot be calculated.

        (Developer Notes: this function could go into GMM, but is needed in this
        class, at least at the moment.)

        Parameters
        ----------


        Returns
        -------
        results : GMMResult instance
            result instance with params and _cov_params attached

        See Also
        --------
        fitgmm
        cov_params

        '''
        if weights is None:
            weights = np.eye(self.nmoms)
        params = self.fitgmm(start=start)
        self.results.params = params  #required before call to self.cov_params
        _cov_params = self.cov_params(weights=weights,
                                      has_optimal_weights=has_optimal_weights)


        self.results.weights = weights
        self.results.jval = self.gmmobjective(params, weights)
        return self.results





if __name__ == '__main__':
    import statsmodels.api as sm
    examples = ['ivols', 'distquant'][:]

    if 'ivols' in examples:
        exampledata = ['ols', 'iv', 'ivfake'][1]
        nobs = nsample = 500
        sige = 3
        corrfactor = 0.025


        x = np.linspace(0,10, nobs)
        X = tools.add_constant(np.column_stack((x, x**2)))
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


        #using GMM and IV2SLS classes
        #----------------------------

        mod = IVGMM(endog, exog, instrument, nmoms=instrument.shape[1])
        res = mod.fit()
        modgmmols = IVGMM(endog, exog, exog, nmoms=exog.shape[1])
        resgmmols = modgmmols.fit()
        #the next is the same as IV2SLS, (Z'Z)^{-1} as weighting matrix
        modgmmiv = IVGMM(endog, exog, instrument, nmoms=instrument.shape[1]) #same as mod
        resgmmiv = modgmmiv.fitgmm(np.ones(exog.shape[1], float),
                        weights=np.linalg.inv(np.dot(instrument.T, instrument)))
        modls = IV2SLS(endog, exog, instrument)
        resls = modls.fit()
        modols = OLS(endog, exog)
        resols = modols.fit()

        print '\nIV case'
        print 'params'
        print 'IV2SLS', resls.params
        print 'GMMIV ', resgmmiv # .params
        print 'GMM   ', res.params
        print 'diff  ', res.params - resls.params
        print 'OLS   ', resols.params
        print 'GMMOLS', resgmmols.params

        print '\nbse'
        print 'IV2SLS', resls.bse
        print 'GMM   ', mod.bse   #bse currently only attached to model not results
        print 'diff  ', mod.bse - resls.bse
        print '%-diff', resls.bse / mod.bse * 100 - 100
        print 'OLS   ', resols.bse
        print 'GMMOLS', modgmmols.bse
        #print 'GMMiv', modgmmiv.bse

        print "Hausman's specification test"
        print modls.spec_hausman()
        print spec_hausman(resols.params, res.params, resols.cov_params(),
                           mod.cov_params())
        print spec_hausman(resgmmols.params, res.params, modgmmols.cov_params(),
                           mod.cov_params())


    if 'distquant' in examples:


        #estimating distribution parameters from quantiles
        #-------------------------------------------------

        #example taken from distribution_estimators.py
        gparrvs = stats.genpareto.rvs(2, size=5000)
        x0p = [1., gparrvs.min()-5, 1]

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
        res1b = moddist2.fitonce([1.5,0,1.5])
        print res1b.params
        print moddist2.bse  #they look much too large
        print np.sqrt(np.diag(res1b._cov_params))



